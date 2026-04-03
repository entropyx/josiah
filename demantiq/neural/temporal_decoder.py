"""Simple temporal decoder for per-period demand decomposition.

Per-timestep MLP: takes embedding(256) → predicts all 26 component shares.
This is the approach that achieved 4-8pp category accuracy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from demantiq.orchestration.training_format import DECOMP_COLS


class TemporalDecoder(nn.Module):
    """Per-timestep MLP decoder. Predicts all component shares from embedding.

    Args:
        input_dim: Dimension of per-period embeddings from the encoder.
        hidden_dim: Hidden dimension of the MLP.
        n_components: Number of output components (DECOMP_COLS = 26).
        n_layers: Number of hidden layers.
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
        layers: list[nn.Module] = []
        d = input_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(d, hidden_dim), nn.LeakyReLU(0.01), nn.Dropout(dropout)])
            d = hidden_dim
        layers.append(nn.Linear(hidden_dim, n_components))
        self.mlp = nn.Sequential(*layers)

    def forward(self, embeddings: Tensor) -> Tensor:
        """Predict per-period component shares.

        Args:
            embeddings: (batch, T, input_dim) per-period embeddings.

        Returns:
            (batch, T, n_components) unconstrained shares.
        """
        return self.mlp(embeddings)
