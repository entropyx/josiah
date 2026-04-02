"""Temporal decoder for per-period demand decomposition.

Per-channel architecture: processes each channel individually through a shared
MLP, so the decoder sees one channel's features at a time (no position ambiguity).

Two sub-decoders:
    1. Channel decoder: per-channel features (80) + global context (129) → 1 share
       Runs once per channel, same weights. Sees each channel's specific spend pattern.
    2. Global decoder: global context (129) → 6 shares
       Predicts baseline, price, distribution, competition, macro, noise.

Output: (batch, T, DECOMP_COLS) assembled from per-channel + global shares.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from demantiq.orchestration.training_format import (
    DECOMP_COLS,
    DECOMP_IDX_BASELINE,
    DECOMP_IDX_CHANNELS_START,
    DECOMP_IDX_PRICE,
    DECOMP_IDX_DISTRIBUTION,
    DECOMP_IDX_COMPETITION,
    DECOMP_IDX_MACRO,
    DECOMP_IDX_NOISE,
)

# Non-channel component indices in output order for the global decoder
_GLOBAL_OUTPUT_INDICES = [
    DECOMP_IDX_BASELINE,
    DECOMP_IDX_PRICE,
    DECOMP_IDX_DISTRIBUTION,
    DECOMP_IDX_COMPETITION,
    DECOMP_IDX_MACRO,
    DECOMP_IDX_NOISE,
]
_N_GLOBAL = len(_GLOBAL_OUTPUT_INDICES)


def _build_mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(n_layers):
        layers.extend([nn.Linear(d, hidden_dim), nn.LeakyReLU(0.01), nn.Dropout(dropout)])
        d = hidden_dim
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class PerChannelTemporalDecoder(nn.Module):
    """Decoder that processes each channel individually through a shared MLP.

    Solves the channel differentiation problem: instead of predicting all 26
    shares from one mixed vector, each channel's share is predicted from that
    channel's specific features. No position ambiguity.

    Args:
        channel_dim: Dimension of per-channel features (CNN + type embedding).
        global_dim: Dimension of global context (y + context + n_channels).
        hidden_dim: Hidden dimension of the MLPs.
        n_layers: Number of hidden layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        channel_dim: int = 80,
        global_dim: int = 129,  # y(64) + ctx(64) + n_ch(1)
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.channel_dim = channel_dim
        self.global_dim = global_dim

        # Channel decoder: per-channel features + global context → 1 share
        self.channel_mlp = _build_mlp(
            channel_dim + global_dim, hidden_dim, 1, n_layers, dropout,
        )

        # Global decoder: global context + channel summary → non-channel shares
        # Channel summary = masked mean of all channel features
        self.global_mlp = _build_mlp(
            global_dim + channel_dim, hidden_dim, _N_GLOBAL, n_layers, dropout,
        )

    def forward(
        self,
        ch_features: Tensor,
        global_features: Tensor,
        channel_mask: Tensor,
        n_channels: Tensor,
    ) -> Tensor:
        """Predict per-period component shares.

        Args:
            ch_features: (B, max_ch, channel_dim, T) per-channel temporal features.
            global_features: (B, global_dim, T) global context (y + ctx + n_ch).
            channel_mask: (B, max_ch) True for active channels.
            n_channels: (B,) number of active channels.

        Returns:
            (B, T, DECOMP_COLS) predicted shares for all components.
        """
        B, max_ch, C, T = ch_features.shape
        device = ch_features.device

        # Transpose time to batch-friendly dim: (B, T, ...)
        # ch_features: (B, max_ch, C, T) → (B, T, max_ch, C)
        ch_feat_t = ch_features.permute(0, 3, 1, 2)  # (B, T, max_ch, C)
        # global_features: (B, global_dim, T) → (B, T, global_dim)
        global_t = global_features.permute(0, 2, 1)  # (B, T, global_dim)

        # --- Channel shares: process each channel individually ---
        # Expand global to match channels: (B, T, 1, global_dim) → (B, T, max_ch, global_dim)
        global_expanded = global_t.unsqueeze(2).expand(-1, -1, max_ch, -1)
        # Concatenate: (B, T, max_ch, channel_dim + global_dim)
        ch_input = torch.cat([ch_feat_t, global_expanded], dim=-1)
        # Run shared MLP on all channels at once (batch dim includes channels)
        ch_input_flat = ch_input.reshape(B * T * max_ch, -1)
        ch_shares_flat = self.channel_mlp(ch_input_flat)  # (B*T*max_ch, 1)
        ch_shares = ch_shares_flat.reshape(B, T, max_ch)  # (B, T, max_ch)
        # Mask inactive channels to zero
        ch_shares = ch_shares * channel_mask.unsqueeze(1).float()  # (B, T, max_ch)

        # --- Global shares: baseline, price, distribution, competition, macro, noise ---
        # Channel summary: masked mean of channel features per timestep
        mask_3d = channel_mask.float().unsqueeze(-1).unsqueeze(-1)  # (B, max_ch, 1, 1)
        n_active = mask_3d.sum(dim=1).clamp(min=1)  # (B, 1, 1)
        ch_summary = (ch_features * mask_3d).sum(dim=1) / n_active  # (B, C, T)
        ch_summary_t = ch_summary.permute(0, 2, 1)  # (B, T, C)

        global_input = torch.cat([global_t, ch_summary_t], dim=-1)  # (B, T, global_dim + channel_dim)
        global_shares = self.global_mlp(global_input)  # (B, T, _N_GLOBAL)

        # --- Assemble output: (B, T, DECOMP_COLS) ---
        output = torch.zeros(B, T, DECOMP_COLS, device=device)
        # Place channel shares at indices 1..max_ch
        output[:, :, DECOMP_IDX_CHANNELS_START:DECOMP_IDX_CHANNELS_START + max_ch] = ch_shares
        # Place global shares at their indices
        for i, idx in enumerate(_GLOBAL_OUTPUT_INDICES):
            output[:, :, idx] = global_shares[:, :, i]

        return output
