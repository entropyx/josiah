"""Held-out loss for proper PFN training.

Loss is computed ONLY on held-out (masked) weeks. Visible weeks contribute
zero gradient. This eliminates the averaging shortcut because the model
must use visible-week context to predict held-out weeks.
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
) -> tuple[torch.Tensor, dict]:
    """Compute MSE loss ONLY on held-out (masked) weeks.

    Args:
        pred: dict with channel_shares (B,T,C), baseline_share (B,T), non_media_share (B,T).
        target: (B, T, max_channels + 2). Last two are [baseline, non_media].
        time_pad_mask: (B, T) bool, True=padded week.
        channel_pad_mask: (B, C) bool, True=padded channel.
        week_is_masked: (B, T) float, 1.0=held-out, 0.0=visible.
        max_channels: int, number of channel slots.

    Returns:
        total_loss (scalar tensor), components (dict of floats).
    """
    ch_active = (~channel_pad_mask).float()  # (B, C)
    time_valid = (~time_pad_mask).float()    # (B, T)

    weight = time_valid * week_is_masked  # (B, T)
    n_valid = weight.sum().clamp(min=1.0)

    pred_ch = pred["channel_shares"]
    pred_base = pred["baseline_share"]
    pred_nm = pred["non_media_share"]

    true_ch = target[:, :, :max_channels]
    true_base = target[:, :, max_channels]
    true_nm = target[:, :, max_channels + 1]

    ch_err_sq = (pred_ch - true_ch) ** 2  # (B, T, C)
    ch_err_sq = ch_err_sq * ch_active.unsqueeze(1) * weight.unsqueeze(-1)
    denom_ch = (ch_active.sum(dim=1).unsqueeze(1) * weight).sum().clamp(min=1.0)
    l_ch = ch_err_sq.sum() / denom_ch

    l_base = ((pred_base - true_base) ** 2 * weight).sum() / n_valid
    l_nm = ((pred_nm - true_nm) ** 2 * weight).sum() / n_valid

    pred_sum = pred_base + (pred_ch * ch_active.unsqueeze(1)).sum(dim=-1) + pred_nm
    target_sum = torch.ones_like(pred_sum)
    l_recon = ((pred_sum - target_sum) ** 2 * weight).sum() / n_valid

    total = l_ch + l_base + l_nm + l_recon

    return total, {
        "channels_masked": l_ch.item(),
        "baseline_masked": l_base.item(),
        "non_media_masked": l_nm.item(),
        "reconstruction_masked": l_recon.item(),
        "total": total.item(),
    }
