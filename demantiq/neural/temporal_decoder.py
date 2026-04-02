"""Temporal decoder for per-period demand decomposition.

Takes per-period embeddings from the encoder and predicts the fractional
share of each demand component for every timestep.

No softmax — shares are unconstrained so the model can represent:
    - Negative shares (noise, price effect)
    - Exact zeros (inactive channels)
    - Shares that don't sum to exactly 1.0

Architecture:
    1. Per-timestep MLP maps embedding → component shares
    2. Output: (batch, T, N_components)

The components are laid out to match the decomposition matrix format:
    [baseline, ch_0, ch_1, ..., ch_{MAX_CH-1}, price, distribution,
     competition, macro, noise]
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from demantiq.orchestration.training_format import DECOMP_COLS


class TemporalDecoder(nn.Module):
    """Decoder that predicts per-period demand decomposition shares.

    For each timestep, maps the embedding to component shares via a shared
    MLP + softmax. All timesteps share parameters (translation-equivariant).

    Args:
        input_dim: Dimension of per-period embeddings from the encoder.
        hidden_dim: Hidden dimension of the per-timestep MLP.
        n_components: Number of output components (default: DECOMP_COLS = 26).
        n_layers: Number of hidden layers in the MLP.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        n_components: int = DECOMP_COLS,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_components = n_components

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, n_components))
        self.mlp = nn.Sequential(*layers)

    def forward(self, embeddings: Tensor) -> Tensor:
        """Predict per-period component shares.

        Args:
            embeddings: (batch, T, input_dim) per-period embeddings.

        Returns:
            (batch, T, n_components) fractional shares (unconstrained — can be
            negative for noise/price, zero for inactive channels).
        """
        return self.mlp(embeddings)  # (batch, T, n_components)
