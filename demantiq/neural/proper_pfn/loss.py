"""Weighted loss for proper PFN training.

Loss is computed on all valid weeks, but MASKED weeks get higher weight (5x)
than VISIBLE weeks (1x). Masked weighting forces the model to use context
from visible weeks. Non-zero visible weight ensures the model also learns
to produce correct outputs at inference (where all weeks are visible).

Pure held-out loss (zero weight on visible) causes a train/inference
distribution mismatch — the model only gets supervised on is_masked=1
inputs but is asked to predict on is_masked=0 inputs at inference.
"""

from __future__ import annotations

import torch


def held_out_loss(
    pred: dict,
    target: torch.Tensor,
    time_pad_mask: torch.Tensor,
    channel_pad_mask: torch.Tensor,
    week_is_masked: torch.Tensor,
    max_channels: int,
    masked_weight: float = 5.0,
    visible_weight: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Weighted MSE loss on masked AND visible weeks.

    Args:
        pred: dict with channel_shares (B,T,C), baseline_share (B,T), non_media_share (B,T).
        target: (B, T, max_channels + 2). Last two are [baseline, non_media].
        time_pad_mask: (B, T) bool, True=padded week.
        channel_pad_mask: (B, C) bool, True=padded channel.
        week_is_masked: (B, T) float, 1.0=held-out, 0.0=visible.
        max_channels: int, number of channel slots.
        masked_weight: loss weight on held-out weeks.
        visible_weight: loss weight on visible weeks.

    Returns:
        total_loss (scalar tensor), components (dict of floats).
    """
    ch_active = (~channel_pad_mask).float()  # (B, C)
    time_valid = (~time_pad_mask).float()    # (B, T)

    # Per-week weight: visible_weight on visible, masked_weight on masked
    weight = time_valid * (visible_weight + (masked_weight - visible_weight) * week_is_masked)
    masked_only = time_valid * week_is_masked
    n_total = weight.sum().clamp(min=1.0)
    n_masked = masked_only.sum().clamp(min=1.0)

    pred_ch = pred["channel_shares"]
    pred_base = pred["baseline_share"]
    pred_nm = pred["non_media_share"]

    true_ch = target[:, :, :max_channels]
    true_base = target[:, :, max_channels]
    true_nm = target[:, :, max_channels + 1]

    # --- Total loss: weighted MSE across all valid weeks ---
    ch_err_sq = (pred_ch - true_ch) ** 2
    ch_err_sq_w = ch_err_sq * ch_active.unsqueeze(1) * weight.unsqueeze(-1)
    denom_ch = (ch_active.sum(dim=1).unsqueeze(1) * weight).sum().clamp(min=1.0)
    l_ch = ch_err_sq_w.sum() / denom_ch

    l_base = ((pred_base - true_base) ** 2 * weight).sum() / n_total
    l_nm = ((pred_nm - true_nm) ** 2 * weight).sum() / n_total

    pred_sum = pred_base + (pred_ch * ch_active.unsqueeze(1)).sum(dim=-1) + pred_nm
    target_sum = torch.ones_like(pred_sum)
    l_recon = ((pred_sum - target_sum) ** 2 * weight).sum() / n_total

    total = l_ch + l_base + l_nm + l_recon

    # --- Diagnostics: masked-only losses for monitoring in-context learning ---
    ch_masked_sq = ch_err_sq * ch_active.unsqueeze(1) * masked_only.unsqueeze(-1)
    denom_ch_m = (ch_active.sum(dim=1).unsqueeze(1) * masked_only).sum().clamp(min=1.0)
    l_ch_masked = ch_masked_sq.sum() / denom_ch_m
    l_base_masked = ((pred_base - true_base) ** 2 * masked_only).sum() / n_masked

    return total, {
        "channels": l_ch.item(),
        "baseline": l_base.item(),
        "non_media": l_nm.item(),
        "reconstruction": l_recon.item(),
        "channels_masked": l_ch_masked.item(),
        "baseline_masked": l_base_masked.item(),
        "total": total.item(),
    }
