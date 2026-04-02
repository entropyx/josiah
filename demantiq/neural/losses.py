"""Dual loss for additive neural decomposition.

Two loss terms:
1. L_component: MSE between predicted and true per-component contributions
2. L_reconstruction: MSE between sum-of-components and actual y

Both in absolute demand units (not shares).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from demantiq.orchestration.training_format import (
    DECOMP_COLS,
    DECOMP_IDX_BASELINE,
    DECOMP_IDX_CHANNELS_START,
    DECOMP_IDX_CHANNELS_END,
    DECOMP_IDX_DISTRIBUTION,
    DECOMP_IDX_NOISE,
)


class AdditiveDecompositionLoss(nn.Module):
    """Dual loss: per-component accuracy + y reconstruction.

    Args:
        reconstruction_weight: Weight for L_reconstruction relative to L_component.
    """

    def __init__(self, reconstruction_weight: float = 1.0):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight

    def forward(
        self,
        pred_contributions: Tensor,
        true_contributions: Tensor,
        y_pred: Tensor,
        y_actual: Tensor,
        valid_mask: Tensor,
        n_channels: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute dual loss.

        Args:
            pred_contributions: (B, T, DECOMP_COLS) predicted absolute contributions.
            true_contributions: (B, T, DECOMP_COLS) true absolute contributions.
            y_pred: (B, T) reconstructed y from sum of components.
            y_actual: (B, T) actual observed y.
            valid_mask: (B, T) True for valid timesteps.
            n_channels: (B,) number of active channels.

        Returns:
            total_loss, loss_dict
        """
        B, T, C = pred_contributions.shape
        vm = valid_mask.float()  # (B, T)

        # Build active component mask (exclude inactive channels, distribution, noise)
        comp_mask = self._build_component_mask(n_channels, B, C, pred_contributions.device)
        full_mask = vm.unsqueeze(-1) * comp_mask.unsqueeze(1).float()  # (B, T, C)

        # --- L_component: per-component MSE in absolute units ---
        # Normalize by y_scale to make loss scale-invariant across scenarios
        y_scale = y_actual.abs().mean(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
        pred_normalized = pred_contributions / y_scale.unsqueeze(-1)
        true_normalized = true_contributions / y_scale.unsqueeze(-1)

        diff = (pred_normalized - true_normalized) * full_mask
        n_valid = full_mask.sum().clamp(min=1)
        l_component = (diff ** 2).sum() / n_valid

        # --- L_reconstruction: MSE(y_pred, y_actual) ---
        y_diff = (y_pred - y_actual) / y_scale * vm
        n_valid_t = vm.sum().clamp(min=1)
        l_reconstruction = (y_diff ** 2).sum() / n_valid_t

        total = l_component + self.reconstruction_weight * l_reconstruction

        loss_dict = {
            "L_component": l_component.detach(),
            "L_reconstruction": l_reconstruction.detach(),
        }

        return total, loss_dict

    def _build_component_mask(
        self, n_channels: Tensor, B: int, C: int, device: torch.device
    ) -> Tensor:
        """Build (B, C) boolean mask of active components."""
        mask = torch.zeros(B, C, device=device, dtype=torch.bool)
        mask[:, DECOMP_IDX_BASELINE] = True
        # Active channels
        ch_idx = torch.arange(DECOMP_IDX_CHANNELS_END - DECOMP_IDX_CHANNELS_START, device=device)
        active = ch_idx.unsqueeze(0) < n_channels.unsqueeze(1)
        mask[:, DECOMP_IDX_CHANNELS_START:DECOMP_IDX_CHANNELS_END] = active
        # Price, competition, macro (not distribution, not noise — noise is residual)
        from demantiq.orchestration.training_format import DECOMP_IDX_PRICE, DECOMP_IDX_COMPETITION, DECOMP_IDX_MACRO
        mask[:, DECOMP_IDX_PRICE] = True
        mask[:, DECOMP_IDX_COMPETITION] = True
        mask[:, DECOMP_IDX_MACRO] = True
        return mask
