"""PFN-inspired full-sequence transformer for Marketing Mix Model demand decomposition.

Each week is a token. Self-attention across ALL weeks captures temporal correlations
between channel spend and demand — the key signal for channel identification.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class PFNDecompositionModel(nn.Module):
    """PFN-inspired full-sequence transformer for demand decomposition.

    Each week is a token with features [spend_ch1..chC, y, context, flags, time].
    Self-attention across all weeks computes temporal correlations.
    Output per week: per-channel contributions + baseline + non_media.

    All outputs are in NORMALIZED space (divided by y_scale). The caller
    must multiply by y_scale to get absolute demand units.
    """

    def __init__(
        self,
        max_channels: int = 8,
        n_context_dims: int = 10,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 300,
    ):
        super().__init__()

        self.max_channels = max_channels
        self.n_context_dims = n_context_dims
        n_outputs = max_channels + 2  # channels + baseline + non_media

        # Input: spend (C) + y (1) + context (D) + flags (D) + time_idx (1) + sin (1) + cos (1)
        n_input_features = max_channels + 1 + 2 * n_context_dims + 3

        self.input_proj = nn.Linear(n_input_features, d_model)

        self.temporal_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_outputs),
        )

    def forward(
        self,
        x: Tensor,
        src_key_padding_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        Args:
            x: (B, T, n_input_features) per-week features.
            src_key_padding_mask: (B, T) True for padded timesteps.

        Returns:
            Dict with:
                'channel_contributions': (B, T, max_channels) — normalized by y_scale
                'baseline': (B, T) — normalized by y_scale
                'non_media': (B, T) — normalized by y_scale
                'raw_output': (B, T, n_outputs) — full raw output
        """
        B, T, _ = x.shape

        h = self.input_proj(x)
        h = h + self.temporal_encoding[:, :T, :]
        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)
        out = self.output_proj(h)  # (B, T, n_outputs)

        channel_contributions = out[:, :, : self.max_channels]
        baseline = out[:, :, self.max_channels]
        non_media = out[:, :, self.max_channels + 1]

        return {
            "channel_contributions": channel_contributions,
            "baseline": baseline,
            "non_media": non_media,
            "raw_output": out,
        }


def build_pfn_input(
    spend: np.ndarray,
    y: np.ndarray,
    context: np.ndarray,
    max_channels: int = 8,
    n_context_dims: int = 10,
) -> tuple[torch.Tensor, float]:
    """Prepare input features for PFNDecompositionModel.

    Normalizes inputs and builds the feature tensor.

    Args:
        spend: (T, n_channels) raw spend.
        y: (T,) observed demand.
        context: (T, n_context_dims) context variables.
        max_channels: Maximum channel slots; spend is zero-padded to this size.
        n_context_dims: Expected context dimension; context is zero-padded if needed.

    Returns:
        (input_tensor, y_scale) where input_tensor is (1, T, n_features)
        and y_scale is the normalization factor to recover absolute values.
    """
    T = len(y)

    # --- y ---
    y_scale = float(max(float(np.abs(y).mean()), 1.0))
    y_normed = (y / y_scale).astype(np.float32).reshape(T, 1)

    # --- spend ---
    n_channels = spend.shape[1] if spend.ndim == 2 else 1
    spend_padded = np.zeros((T, max_channels), dtype=np.float32)
    spend_padded[:, :min(n_channels, max_channels)] = spend[:, :min(n_channels, max_channels)]
    spend_scale = float(max(float(np.abs(spend_padded).max()), 1.0))
    spend_normed = (spend_padded / spend_scale).astype(np.float32)

    # --- context ---
    ctx_cols = context.shape[1] if context.ndim == 2 else 1
    ctx_padded = np.zeros((T, n_context_dims), dtype=np.float32)
    ctx_padded[:, :min(ctx_cols, n_context_dims)] = context[:, :min(ctx_cols, n_context_dims)]

    flags = (np.abs(ctx_padded).sum(axis=0) > 0).astype(np.float32)  # (n_context_dims,)
    flags_tiled = np.tile(flags, (T, 1))  # (T, n_context_dims)

    col_scales = np.maximum(np.abs(ctx_padded).max(axis=0), 1.0)  # (n_context_dims,)
    ctx_normed = (ctx_padded / col_scales).astype(np.float32)

    # --- temporal features ---
    time_index = np.linspace(0.0, 1.0, T, dtype=np.float32).reshape(T, 1)
    week_of_year = np.arange(T, dtype=np.float32) % 52.0
    sin_week = np.sin(2.0 * np.pi * week_of_year / 52.0).reshape(T, 1)
    cos_week = np.cos(2.0 * np.pi * week_of_year / 52.0).reshape(T, 1)

    # --- concatenate ---
    features = np.concatenate(
        [spend_normed, y_normed, ctx_normed, flags_tiled, time_index, sin_week, cos_week],
        axis=1,
    )  # (T, max_channels + 1 + 2*n_context_dims + 3)

    input_tensor = torch.from_numpy(features).unsqueeze(0)  # (1, T, n_features)
    return input_tensor, y_scale
