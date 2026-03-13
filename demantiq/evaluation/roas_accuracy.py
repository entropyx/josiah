"""ROAS accuracy evaluation metrics."""

from dataclasses import dataclass, field
from scipy.stats import spearmanr
import numpy as np


@dataclass
class ROASResult:
    """Result of ROAS accuracy evaluation.

    Attributes:
        per_channel_mape: MAPE per channel.
        ranking_correlation: Spearman correlation of ROAS rankings.
        direction_accuracy: Fraction correctly identifying ROAS > 1 vs < 1.
    """

    per_channel_mape: dict = field(default_factory=dict)
    ranking_correlation: float = 0.0
    direction_accuracy: float = 0.0


class ROASAccuracy:
    """Evaluate accuracy of estimated ROAS values."""

    def evaluate(self, estimated_roas: dict, true_roas: dict) -> ROASResult:
        """Compare estimated vs true ROAS per channel.

        Args:
            estimated_roas: {channel_name: roas_value}
            true_roas: {channel_name: roas_value}

        Returns:
            ROASResult with error metrics.
        """
        common_keys = sorted(set(estimated_roas.keys()) & set(true_roas.keys()))

        # Per-channel MAPE
        per_channel_mape = {}
        for key in common_keys:
            est_val = float(estimated_roas[key])
            true_val = float(true_roas[key])
            if abs(true_val) > 0:
                per_channel_mape[key] = abs(est_val - true_val) / abs(true_val)
            else:
                per_channel_mape[key] = float("inf") if abs(est_val) > 0 else 0.0

        # Ranking correlation (Spearman)
        ranking_correlation = 0.0
        if len(common_keys) >= 2:
            est_vals = [float(estimated_roas[k]) for k in common_keys]
            true_vals = [float(true_roas[k]) for k in common_keys]
            corr, _ = spearmanr(est_vals, true_vals)
            ranking_correlation = float(corr) if not np.isnan(corr) else 0.0

        # Direction accuracy: correctly identifies ROAS > 1 vs < 1
        if common_keys:
            correct = sum(
                1
                for k in common_keys
                if (float(estimated_roas[k]) >= 1.0) == (float(true_roas[k]) >= 1.0)
            )
            direction_accuracy = correct / len(common_keys)
        else:
            direction_accuracy = 0.0

        return ROASResult(
            per_channel_mape=per_channel_mape,
            ranking_correlation=ranking_correlation,
            direction_accuracy=direction_accuracy,
        )
