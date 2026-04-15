"""Metrics for proper PFN evaluation.

Replaces the misleading "reconstruction R²" (which only measured whether
shares sum to 1) with metrics that actually measure decomposition quality.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def per_component_r2(true: np.ndarray, pred: np.ndarray) -> float:
    """R² of predicted time series against true time series."""
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    if ss_tot < 1e-10:
        return 0.0
    return 1.0 - ss_res / ss_tot


def channel_rank_correlation(true_totals: np.ndarray, pred_totals: np.ndarray) -> float:
    """Spearman rank correlation of per-channel total contributions."""
    if len(true_totals) < 2:
        return 0.0
    if len(true_totals) == 2:
        c = float(np.corrcoef(true_totals, pred_totals)[0, 1])
        if np.isnan(c):
            return 0.0
        return c
    corr, _ = spearmanr(true_totals, pred_totals)
    if np.isnan(corr):
        return 0.0
    return float(corr)


def scenario_level_metrics(
    pred_channels: np.ndarray,
    pred_baseline: np.ndarray,
    true_channels: np.ndarray,
    true_baseline: np.ndarray,
    y: np.ndarray,
) -> dict:
    """Compute all scenario-level evaluation metrics."""
    n_ch = pred_channels.shape[1]

    ch_r2s = []
    for i in range(n_ch):
        if np.std(true_channels[:, i]) > 1e-10:
            ch_r2s.append(per_component_r2(true_channels[:, i], pred_channels[:, i]))
    channel_r2_mean = float(np.mean(ch_r2s)) if ch_r2s else 0.0

    baseline_r2 = per_component_r2(true_baseline, pred_baseline)

    true_totals = true_channels.sum(axis=0)
    pred_totals = pred_channels.sum(axis=0)
    rank_corr = channel_rank_correlation(true_totals, pred_totals)

    per_ch_err = []
    for i in range(n_ch):
        if abs(true_totals[i]) > 1e-6:
            err = abs(pred_totals[i] - true_totals[i]) / abs(true_totals[i]) * 100.0
            per_ch_err.append(err)
    per_channel_error_pct = float(np.mean(per_ch_err)) if per_ch_err else 0.0

    total_y = float(y.sum())
    true_base_pct = float(true_baseline.sum() / total_y * 100) if total_y > 0 else 0.0
    pred_base_pct = float(pred_baseline.sum() / total_y * 100) if total_y > 0 else 0.0
    category_error_pp = abs(pred_base_pct - true_base_pct)

    return {
        "channel_r2_mean": channel_r2_mean,
        "baseline_r2": baseline_r2,
        "rank_corr": rank_corr,
        "per_channel_error_pct": per_channel_error_pct,
        "category_error_pp": category_error_pp,
        "true_baseline_pct": true_base_pct,
        "pred_baseline_pct": pred_base_pct,
        "per_channel_r2": ch_r2s,
    }
