"""NAM (Neural Additive Model) decoder for per-period demand decomposition.

Each demand component gets a dedicated sub-network that sees the FULL time series:
- Channel CNN: sees channel_spend + y + context + type_embedding → learns spend↔y relationship
- Baseline CNN: sees y + context → learns organic demand patterns
- Other CNN: sees y + context → learns price, competition, macro, noise

The key insight: the channel CNN processes spend AND y together across all timesteps,
so it can learn "when this channel spends more, y goes up proportionally" — i.e.,
it can estimate channel effectiveness (beta) from the observable data.

Architecture based on:
- Neural Additive Models (Agarwal et al., ICML 2021)
- N-BEATS additive decomposition (Oreshkin et al., ICLR 2020)
- Google NNN for MMM (April 2025)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from demantiq.neural.utils import NUM_CHANNEL_TYPES
from demantiq.orchestration.training_format import (
    DECOMP_COLS,
    DECOMP_IDX_BASELINE,
    DECOMP_IDX_CHANNELS_START,
    DECOMP_IDX_PRICE,
    DECOMP_IDX_DISTRIBUTION,
    DECOMP_IDX_COMPETITION,
    DECOMP_IDX_MACRO,
    DECOMP_IDX_NOISE,
    MAX_CONTEXT_COLS,
)


def _build_dilated_cnn(in_channels: int, out_channels: int, hidden: int = 64) -> nn.Sequential:
    """Build a dilated CNN that preserves temporal dimension.

    3 dilated layers (receptive field ~31 timesteps) + 1x1 projection.
    Same padding scheme as the proven DilatedConv1DEncoder.
    """
    return nn.Sequential(
        nn.Conv1d(in_channels, hidden, kernel_size=7, padding=3, dilation=1),
        nn.LeakyReLU(0.01),
        nn.Conv1d(hidden, hidden, kernel_size=5, padding=4, dilation=2),
        nn.LeakyReLU(0.01),
        nn.Conv1d(hidden, hidden // 2, kernel_size=3, padding=4, dilation=4),
        nn.LeakyReLU(0.01),
        nn.Conv1d(hidden // 2, out_channels, kernel_size=1),
    )


class NAMTemporalDecoder(nn.Module):
    """Neural Additive Model decoder for demand decomposition.

    Three sub-networks, each seeing the full time series:
        1. channel_cnn: per-channel contribution (shared weights, type-conditioned)
        2. baseline_cnn: organic baseline
        3. other_cnn: price, competition, macro, noise

    Args:
        n_context_cols: Number of business context columns.
        type_embed_dim: Dimension of channel-type embeddings.
        hidden_dim: Hidden dimension of CNN layers.
        n_channel_types: Number of distinct channel types.
    """

    def __init__(
        self,
        n_context_cols: int = MAX_CONTEXT_COLS,
        type_embed_dim: int = 16,
        hidden_dim: int = 64,
        n_channel_types: int = NUM_CHANNEL_TYPES,
    ):
        super().__init__()
        self.type_embed_dim = type_embed_dim
        self.n_context_cols = n_context_cols

        # Channel type embeddings (learned, conditions shared CNN per channel type)
        self.channel_type_embedding = nn.Embedding(n_channel_types, type_embed_dim)

        # Channel CNN: spend(1) + y(1) + context(n_ctx) + type_emb(16) → 1 share
        channel_in = 1 + 1 + n_context_cols + type_embed_dim
        self.channel_cnn = _build_dilated_cnn(channel_in, 1, hidden_dim)

        # Baseline CNN: y(1) + context(n_ctx) → 1 share
        baseline_in = 1 + n_context_cols
        self.baseline_cnn = _build_dilated_cnn(baseline_in, 1, hidden_dim)

        # Other CNN: y(1) + context(n_ctx) → 4 shares (price, competition, macro, noise)
        other_in = 1 + n_context_cols
        self.other_cnn = _build_dilated_cnn(other_in, 4, hidden_dim)

    def forward(
        self,
        spend_normed: Tensor,
        y_normed: Tensor,
        ctx_normed: Tensor,
        channel_type_ids: Tensor,
        channel_mask: Tensor,
        n_channels: Tensor,
    ) -> Tensor:
        """Predict per-period component shares using NAM architecture.

        Args:
            spend_normed: (B, T, max_ch) normalized spend matrix.
            y_normed: (B, T) normalized outcome.
            ctx_normed: (B, T, n_ctx) normalized context.
            channel_type_ids: (B, max_ch) channel type indices.
            channel_mask: (B, max_ch) True for active channels.
            n_channels: (B,) number of active channels.

        Returns:
            (B, T, DECOMP_COLS) predicted shares for all components.
        """
        B, T, max_ch = spend_normed.shape
        device = spend_normed.device

        # --- Channel contributions (batched across all channels) ---
        # Get type embeddings: (B, max_ch, type_embed_dim)
        type_emb = self.channel_type_embedding(channel_type_ids[:, :max_ch])

        # Build per-channel input: (B, max_ch, T, 1+1+n_ctx+16)
        # spend_i(T,1) + y(T,1) + context(T,n_ctx) + type_emb(T,16)
        y_expanded = y_normed.unsqueeze(2).expand(-1, -1, max_ch)  # (B, T, max_ch)
        # Rearrange: (B, max_ch, T)
        spend_ch = spend_normed.permute(0, 2, 1)  # (B, max_ch, T)
        y_ch = y_expanded.permute(0, 2, 1)  # (B, max_ch, T)
        ctx_ch = ctx_normed.unsqueeze(1).expand(-1, max_ch, -1, -1)  # (B, max_ch, T, n_ctx)
        ctx_ch = ctx_ch.permute(0, 1, 3, 2)  # (B, max_ch, n_ctx, T)
        type_emb_ch = type_emb.unsqueeze(-1).expand(-1, -1, -1, T)  # (B, max_ch, 16, T)

        # Concatenate along feature dim: (B, max_ch, 1+1+n_ctx+16, T)
        ch_input = torch.cat([
            spend_ch.unsqueeze(2),  # (B, max_ch, 1, T)
            y_ch.unsqueeze(2),      # (B, max_ch, 1, T)
            ctx_ch,                 # (B, max_ch, n_ctx, T)
            type_emb_ch,            # (B, max_ch, 16, T)
        ], dim=2)

        # Flatten batch and channel dims for batched CNN: (B*max_ch, features, T)
        ch_input_flat = ch_input.reshape(B * max_ch, -1, T)
        ch_shares_flat = self.channel_cnn(ch_input_flat)  # (B*max_ch, 1, T)
        ch_shares = ch_shares_flat.reshape(B, max_ch, T)  # (B, max_ch, T)

        # Mask inactive channels
        ch_shares = ch_shares * channel_mask.unsqueeze(-1).float()

        # --- Baseline ---
        # Input: y(1) + context(n_ctx) → (B, 1+n_ctx, T)
        base_input = torch.cat([
            y_normed.unsqueeze(1),           # (B, 1, T)
            ctx_normed.permute(0, 2, 1),     # (B, n_ctx, T)
        ], dim=1)
        baseline_share = self.baseline_cnn(base_input).squeeze(1)  # (B, T)

        # --- Other components (price, competition, macro, noise) ---
        other_shares = self.other_cnn(base_input)  # (B, 4, T)

        # --- Assemble output: (B, T, DECOMP_COLS) ---
        output = torch.zeros(B, T, DECOMP_COLS, device=device)
        output[:, :, DECOMP_IDX_BASELINE] = baseline_share
        output[:, :, DECOMP_IDX_CHANNELS_START:DECOMP_IDX_CHANNELS_START + max_ch] = ch_shares.permute(0, 2, 1)
        output[:, :, DECOMP_IDX_PRICE] = other_shares[:, 0, :]
        output[:, :, DECOMP_IDX_COMPETITION] = other_shares[:, 1, :]
        output[:, :, DECOMP_IDX_MACRO] = other_shares[:, 2, :]
        output[:, :, DECOMP_IDX_NOISE] = other_shares[:, 3, :]
        # Distribution cap stays 0 (multiplicative, excluded from training)

        return output
