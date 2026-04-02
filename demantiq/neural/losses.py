"""Composite loss function for demand decomposition.

Replaces simple MSE with a multi-term loss that explicitly measures:
1. Per-week share accuracy (on active components only)
2. Per-channel total contribution accuracy
3. Category-level accuracy (baseline vs media vs other)
4. Channel ranking accuracy (pairwise margin loss)

Uses learned uncertainty weighting (Kendall et al. 2018) for self-balancing
— no manual weight tuning required.
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
    DECOMP_IDX_PRICE,
    DECOMP_IDX_DISTRIBUTION,
    DECOMP_IDX_COMPETITION,
    DECOMP_IDX_MACRO,
    DECOMP_IDX_NOISE,
)

# Fixed component indices that are always active (besides channels)
_FIXED_ACTIVE = [
    DECOMP_IDX_BASELINE,
    DECOMP_IDX_PRICE,
    DECOMP_IDX_COMPETITION,
    DECOMP_IDX_MACRO,
    DECOMP_IDX_NOISE,
]


class DecompositionLoss(nn.Module):
    """Composite loss for demand decomposition with self-balancing.

    Four loss terms, each targeting a specific failure mode:
        L_share:    MSE on active component shares per timestep
        L_total:    Scale-normalized MSE on summed contributions per component
        L_category: MSE on baseline/media/other category means
        L_rank:     Pairwise margin loss on channel ranking

    Self-balancing via learned log-variance parameters (Kendall et al. 2018).

    Args:
        ranking_margin: Margin for pairwise ranking loss (0 = just get direction right).
        zero_reg_weight: Fixed weight for inactive channel zero-regularizer.
    """

    def __init__(self, ranking_margin: float = 0.0, zero_reg_weight: float = 0.01):
        super().__init__()
        self.ranking_margin = ranking_margin
        self.zero_reg_weight = zero_reg_weight
        # Running-mean normalization: each loss divided by its own EMA
        # so all terms contribute equally. NOT learnable — no gaming.
        self.register_buffer("ema_share", torch.tensor(1.0))
        self.register_buffer("ema_total", torch.tensor(1.0))
        self.register_buffer("ema_category", torch.tensor(1.0))
        self.register_buffer("ema_rank", torch.tensor(1.0))
        self._ema_initialized = False
        self._ema_decay = 0.99

    def forward(
        self,
        pred: Tensor,
        true: Tensor,
        valid_mask: Tensor,
        n_channels: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute composite loss.

        Args:
            pred: (B, T, DECOMP_COLS) predicted shares.
            true: (B, T, DECOMP_COLS) true shares.
            valid_mask: (B, T) True for valid timesteps.
            n_channels: (B,) number of active channels per sample.

        Returns:
            total_loss: scalar tensor for backprop.
            loss_dict: dict of individual loss terms for logging.
        """
        B, T, C = pred.shape
        comp_mask = self._build_component_mask(n_channels, B, C, pred.device)

        l_share = self._share_loss(pred, true, valid_mask, comp_mask)
        l_zero = self._zero_loss(pred, valid_mask, comp_mask)
        l_total = self._total_contribution_loss(pred, true, valid_mask, comp_mask)
        l_category = self._category_loss(pred, true, valid_mask, n_channels)
        l_rank = self._ranking_loss(pred, true, valid_mask, n_channels)
        l_sum = self._sum_loss(pred, valid_mask, comp_mask)

        # Running-mean normalization: each term ÷ its own EMA → all ≈ 1.0
        # Not learnable — model can't game this.
        if self.training:
            with torch.no_grad():
                if not self._ema_initialized:
                    # Initialize EMAs to first-batch values
                    self.ema_share.fill_(l_share.item() + 1e-8)
                    self.ema_total.fill_(l_total.item() + 1e-8)
                    self.ema_category.fill_(l_category.item() + 1e-8)
                    self.ema_rank.fill_(l_rank.item() + 1e-8)
                    self._ema_initialized = True
                else:
                    d = self._ema_decay
                    self.ema_share.mul_(d).add_(l_share.item() * (1 - d))
                    self.ema_total.mul_(d).add_(l_total.item() * (1 - d))
                    self.ema_category.mul_(d).add_(l_category.item() * (1 - d))
                    self.ema_rank.mul_(d).add_(l_rank.item() * (1 - d))

        # Normalize each term so they all contribute ~1.0
        norm_share = l_share / self.ema_share.clamp(min=1e-8)
        norm_total = l_total / self.ema_total.clamp(min=1e-8)
        norm_cat = l_category / self.ema_category.clamp(min=1e-8)
        norm_rank = l_rank / self.ema_rank.clamp(min=1e-8)

        total = norm_share + norm_total + norm_cat + norm_rank + self.zero_reg_weight * l_zero + 0.1 * l_sum

        loss_dict = {
            "L_share": l_share.detach(),
            "L_total": l_total.detach(),
            "L_category": l_category.detach(),
            "L_rank": l_rank.detach(),
            "L_zero": l_zero.detach(),
            "L_sum": l_sum.detach(),
            "ema_share": self.ema_share.clone(),
            "ema_total": self.ema_total.clone(),
            "ema_cat": self.ema_category.clone(),
            "ema_rank": self.ema_rank.clone(),
        }

        return total, loss_dict

    def _build_component_mask(
        self, n_channels: Tensor, B: int, C: int, device: torch.device
    ) -> Tensor:
        """Build (B, C) boolean mask of active components."""
        mask = torch.zeros(B, C, device=device, dtype=torch.bool)
        # Fixed active components
        for idx in _FIXED_ACTIVE:
            mask[:, idx] = True
        # Per-sample active channels
        ch_idx = torch.arange(DECOMP_IDX_CHANNELS_END - DECOMP_IDX_CHANNELS_START, device=device)
        ch_idx = ch_idx.unsqueeze(0)  # (1, 20)
        active = ch_idx < n_channels.unsqueeze(1)  # (B, 20)
        mask[:, DECOMP_IDX_CHANNELS_START:DECOMP_IDX_CHANNELS_END] = active
        # Distribution cap always excluded (already False)
        return mask

    def _share_loss(
        self, pred: Tensor, true: Tensor, valid_mask: Tensor, comp_mask: Tensor
    ) -> Tensor:
        """MSE on active component shares for valid timesteps."""
        # full_mask: (B, T, C) — both time-valid and component-active
        full_mask = valid_mask.unsqueeze(-1) & comp_mask.unsqueeze(1)
        fm = full_mask.float()
        diff = (pred - true) * fm
        n_valid = fm.sum().clamp(min=1)
        return (diff ** 2).sum() / n_valid

    def _zero_loss(
        self, pred: Tensor, valid_mask: Tensor, comp_mask: Tensor
    ) -> Tensor:
        """L2 penalty on inactive channel predictions (should be zero)."""
        inactive_mask = valid_mask.unsqueeze(-1) & ~comp_mask.unsqueeze(1)
        # Also exclude distribution cap (index 22) from penalty
        inactive_mask[:, :, DECOMP_IDX_DISTRIBUTION] = False
        im = inactive_mask.float()
        n_inactive = im.sum().clamp(min=1)
        return (pred ** 2 * im).sum() / n_inactive

    def _sum_loss(
        self, pred: Tensor, valid_mask: Tensor, comp_mask: Tensor
    ) -> Tensor:
        """Penalize when active shares don't sum to ~1.0 per timestep (additivity)."""
        # Sum active component shares per timestep
        full_mask = comp_mask.unsqueeze(1).float()  # (B, 1, C)
        active_sum = (pred * full_mask).sum(dim=-1)  # (B, T)
        # Should be ~1.0 for valid timesteps
        vm = valid_mask.float()
        diff = (active_sum - 1.0) * vm
        n_valid = vm.sum().clamp(min=1)
        return (diff ** 2).sum() / n_valid

    def _total_contribution_loss(
        self, pred: Tensor, true: Tensor, valid_mask: Tensor, comp_mask: Tensor
    ) -> Tensor:
        """Scale-normalized MSE on summed contributions per component."""
        vm = valid_mask.unsqueeze(-1).float()  # (B, T, 1)
        pred_totals = (pred * vm).sum(dim=1)  # (B, C)
        true_totals = (true * vm).sum(dim=1)  # (B, C)

        # Scale by true magnitude (prevents large channels from dominating)
        scale = true_totals.abs().clamp(min=0.1)  # (B, C)
        normalized_diff = (pred_totals - true_totals) / scale

        # Only active components
        cm = comp_mask.float()  # (B, C)
        n_active = cm.sum().clamp(min=1)
        return ((normalized_diff ** 2) * cm).sum() / n_active

    def _category_loss(
        self, pred: Tensor, true: Tensor, valid_mask: Tensor, n_channels: Tensor
    ) -> Tensor:
        """MSE on baseline/media/other category mean shares."""
        B = pred.shape[0]
        vm = valid_mask.float()  # (B, T)
        n_valid = vm.sum(dim=1).clamp(min=1)  # (B,)

        # Baseline category: index 0
        pred_baseline = (pred[:, :, DECOMP_IDX_BASELINE] * vm).sum(dim=1) / n_valid
        true_baseline = (true[:, :, DECOMP_IDX_BASELINE] * vm).sum(dim=1) / n_valid

        # Media category: sum of active channel shares
        pred_media = torch.zeros(B, device=pred.device)
        true_media = torch.zeros(B, device=pred.device)
        for b in range(B):
            n_ch = n_channels[b].item()
            ch_slice = slice(DECOMP_IDX_CHANNELS_START, DECOMP_IDX_CHANNELS_START + n_ch)
            pred_media[b] = (pred[b, :, ch_slice] * vm[b].unsqueeze(-1)).sum() / n_valid[b]
            true_media[b] = (true[b, :, ch_slice] * vm[b].unsqueeze(-1)).sum() / n_valid[b]

        # Other category: price + competition + macro + noise
        other_idx = [DECOMP_IDX_PRICE, DECOMP_IDX_COMPETITION, DECOMP_IDX_MACRO, DECOMP_IDX_NOISE]
        pred_other = sum(
            (pred[:, :, idx] * vm).sum(dim=1) / n_valid for idx in other_idx
        )
        true_other = sum(
            (true[:, :, idx] * vm).sum(dim=1) / n_valid for idx in other_idx
        )

        # MSE across 3 categories, averaged over batch
        cat_loss = (
            (pred_baseline - true_baseline) ** 2
            + (pred_media - true_media) ** 2
            + (pred_other - true_other) ** 2
        ) / 3.0
        return cat_loss.mean()

    def _ranking_loss(
        self, pred: Tensor, true: Tensor, valid_mask: Tensor, n_channels: Tensor
    ) -> Tensor:
        """Pairwise margin ranking loss on channel total contributions."""
        B = pred.shape[0]
        max_ch = DECOMP_IDX_CHANNELS_END - DECOMP_IDX_CHANNELS_START  # 20

        # Sum channel shares over valid timesteps
        vm = valid_mask.unsqueeze(-1).float()  # (B, T, 1)
        ch_pred = pred[:, :, DECOMP_IDX_CHANNELS_START:DECOMP_IDX_CHANNELS_END]
        ch_true = true[:, :, DECOMP_IDX_CHANNELS_START:DECOMP_IDX_CHANNELS_END]
        pred_totals = (ch_pred * vm).sum(dim=1)  # (B, 20)
        true_totals = (ch_true * vm).sum(dim=1)  # (B, 20)

        # Pairwise differences: (B, 20, 20)
        pred_diff = pred_totals.unsqueeze(2) - pred_totals.unsqueeze(1)
        true_diff = true_totals.unsqueeze(2) - true_totals.unsqueeze(1)
        true_sign = torch.sign(true_diff)

        # Pair validity mask: both channels must be active + upper triangle + no ties
        ch_active = torch.zeros(B, max_ch, device=pred.device, dtype=torch.bool)
        for b in range(B):
            ch_active[b, :n_channels[b].item()] = True
        pair_valid = ch_active.unsqueeze(2) & ch_active.unsqueeze(1)
        triu = torch.triu(torch.ones(max_ch, max_ch, device=pred.device, dtype=torch.bool), diagonal=1)
        pair_valid = pair_valid & triu.unsqueeze(0) & (true_sign != 0)

        # Pairwise hinge: ReLU(margin - true_sign * pred_diff)
        pair_loss = torch.relu(self.ranking_margin - true_sign * pred_diff)

        n_valid_pairs = pair_valid.float().sum().clamp(min=1)
        return (pair_loss * pair_valid.float()).sum() / n_valid_pairs
