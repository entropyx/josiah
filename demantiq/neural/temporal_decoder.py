"""Additive Neural Decomposition decoder.

Each demand component gets a dedicated CNN sub-network that predicts its
ABSOLUTE contribution (in demand units, not shares). Components are summed
to reconstruct y. Loss is on both individual components and reconstruction.

Key design: channel sub-networks see ONLY their own spend — not y.
This prevents the CNN from shortcutting by predicting y/n_channels.

Architecture:
- channel_cnn: spend_i + type_emb + global_stats → contribution_i(T)
- baseline_cnn: context → baseline(T)
- other_cnn: context → price_effect(T), competition_effect(T), macro_effect(T)

Based on: NAMs (ICML 2021), N-BEATS (ICLR 2020), Google NNN (2025)
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

# Number of global summary stats passed to channel CNN
N_GLOBAL_STATS = 4  # mean_y, mean_total_spend, y_spend_ratio, n_channels


def _build_dilated_cnn(in_channels: int, out_channels: int, hidden: int = 64) -> nn.Sequential:
    """Dilated CNN preserving temporal dimension. Receptive field ~31 timesteps."""
    return nn.Sequential(
        nn.Conv1d(in_channels, hidden, kernel_size=7, padding=3, dilation=1),
        nn.LeakyReLU(0.01),
        nn.Conv1d(hidden, hidden, kernel_size=5, padding=4, dilation=2),
        nn.LeakyReLU(0.01),
        nn.Conv1d(hidden, hidden // 2, kernel_size=3, padding=4, dilation=4),
        nn.LeakyReLU(0.01),
        nn.Conv1d(hidden // 2, out_channels, kernel_size=1),
    )


class AdditiveDecompositionDecoder(nn.Module):
    """Additive decoder: predict absolute contributions, sum to reconstruct y.

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

        # Channel type embeddings
        self.channel_type_embedding = nn.Embedding(n_channel_types, type_embed_dim)

        # Channel CNN: spend(1) + type_emb(16) + global_stats(4) → contribution(1)
        # NO y input — forces learning actual spend→contribution transformation
        channel_in = 1 + type_embed_dim + N_GLOBAL_STATS
        self.channel_cnn = _build_dilated_cnn(channel_in, 1, hidden_dim)

        # Baseline CNN: context(n_ctx) + time_index(1) → baseline(1)
        # NO y, NO spend — baseline is organic demand
        baseline_in = n_context_cols + 1
        self.baseline_cnn = _build_dilated_cnn(baseline_in, 1, hidden_dim)

        # Other CNN: context(n_ctx) + time_index(1) → price(1) + competition(1) + macro(1)
        other_in = n_context_cols + 1
        self.other_cnn = _build_dilated_cnn(other_in, 3, hidden_dim)

    def forward(
        self,
        spend: Tensor,
        context: Tensor,
        channel_type_ids: Tensor,
        channel_mask: Tensor,
        n_channels: Tensor,
        global_stats: Tensor,
    ) -> dict[str, Tensor]:
        """Predict absolute per-component contributions.

        Args:
            spend: (B, T, max_ch) raw spend (NOT normalized — we normalize here).
            context: (B, T, n_ctx) business context.
            channel_type_ids: (B, max_ch) channel type indices.
            channel_mask: (B, max_ch) True for active channels.
            n_channels: (B,) number of active channels.
            global_stats: (B, N_GLOBAL_STATS) scenario-level summary statistics.

        Returns:
            Dict with:
                contributions: (B, T, DECOMP_COLS) per-component absolute contributions
                y_pred: (B, T) reconstructed y = sum of all components
        """
        B, T, max_ch = spend.shape
        device = spend.device

        # Normalize spend per channel (by global max across all channels)
        spend_scale = spend.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1.0)
        spend_normed = spend / spend_scale  # (B, T, max_ch), in [-1, 1]

        # --- Channel contributions ---
        type_emb = self.channel_type_embedding(channel_type_ids[:, :max_ch])  # (B, max_ch, 16)

        # Build per-channel input: spend(1) + type_emb(16) + global_stats(4)
        spend_ch = spend_normed.permute(0, 2, 1)  # (B, max_ch, T)
        type_emb_t = type_emb.unsqueeze(-1).expand(-1, -1, -1, T)  # (B, max_ch, 16, T)
        global_t = global_stats.unsqueeze(-1).unsqueeze(1).expand(-1, max_ch, -1, T)  # (B, max_ch, 4, T)

        ch_input = torch.cat([
            spend_ch.unsqueeze(2),  # (B, max_ch, 1, T)
            type_emb_t,             # (B, max_ch, 16, T)
            global_t,               # (B, max_ch, 4, T)
        ], dim=2)  # (B, max_ch, 21, T)

        # Batch all channels: (B*max_ch, 21, T)
        ch_input_flat = ch_input.reshape(B * max_ch, -1, T)
        ch_contrib_flat = self.channel_cnn(ch_input_flat)  # (B*max_ch, 1, T)
        ch_contribs = ch_contrib_flat.reshape(B, max_ch, T)  # (B, max_ch, T)

        # Mask inactive channels
        ch_contribs = ch_contribs * channel_mask.unsqueeze(-1).float()

        # Scale contributions back to demand units
        # The CNN outputs in normalized space; multiply by spend_scale to get demand units
        # spend_scale is (B, 1, 1), squeeze to (B,) and broadcast
        ch_contribs = ch_contribs * spend_scale.reshape(B, 1, 1)  # (B, max_ch, T)

        # --- Baseline ---
        # time_index as a feature (normalized 0-1)
        time_idx = torch.linspace(0, 1, T, device=device).unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        time_idx = time_idx.expand(B, -1, -1)  # (B, 1, T)
        ctx_t = context.permute(0, 2, 1)  # (B, n_ctx, T)
        base_input = torch.cat([ctx_t, time_idx], dim=1)  # (B, n_ctx+1, T)

        baseline = self.baseline_cnn(base_input).squeeze(1)  # (B, T)

        # --- Other components (price, competition, macro) ---
        other = self.other_cnn(base_input)  # (B, 3, T)

        # --- Assemble contributions ---
        contributions = torch.zeros(B, T, DECOMP_COLS, device=device)
        contributions[:, :, DECOMP_IDX_BASELINE] = baseline
        contributions[:, :, DECOMP_IDX_CHANNELS_START:DECOMP_IDX_CHANNELS_START + max_ch] = ch_contribs.permute(0, 2, 1)
        contributions[:, :, DECOMP_IDX_PRICE] = other[:, 0, :]
        contributions[:, :, DECOMP_IDX_COMPETITION] = other[:, 1, :]
        contributions[:, :, DECOMP_IDX_MACRO] = other[:, 2, :]
        # Noise is the residual (not predicted)
        # Distribution cap stays 0

        # Reconstruct y
        y_pred = baseline + ch_contribs.sum(dim=1) + other.sum(dim=1)  # (B, T)

        return {
            "contributions": contributions,
            "y_pred": y_pred,
        }
