"""Contribution accuracy evaluation metrics."""

from dataclasses import dataclass, field
from scipy.stats import kendalltau
import numpy as np


@dataclass
class ContribResult:
    """Result of contribution accuracy evaluation.

    Attributes:
        total_media_error: Absolute error in total media contribution.
        per_channel_mape: MAPE per channel.
        channel_ranking: Kendall's tau between true and estimated channel rankings.
        zero_detection_rate: Fraction of true-zero channels correctly identified as zero.
    """

    total_media_error: float = 0.0
    per_channel_mape: dict = field(default_factory=dict)
    channel_ranking: float = 0.0
    zero_detection_rate: float = 0.0


class ContributionAccuracy:
    """Evaluate accuracy of estimated channel contributions."""

    def evaluate(
        self, estimated_contributions: dict, true_contributions: dict
    ) -> ContribResult:
        """Compare estimated vs true per-channel total contributions.

        Args:
            estimated_contributions: {channel_name: total_contribution}
            true_contributions: {channel_name: total_contribution}

        Returns:
            ContribResult with error metrics.
        """
        common_keys = sorted(
            set(estimated_contributions.keys()) & set(true_contributions.keys())
        )

        # Total media error
        est_total = sum(float(estimated_contributions[k]) for k in common_keys)
        true_total = sum(float(true_contributions[k]) for k in common_keys)
        total_media_error = abs(est_total - true_total)

        # Per-channel MAPE
        per_channel_mape = {}
        for key in common_keys:
            est_val = float(estimated_contributions[key])
            true_val = float(true_contributions[key])
            if abs(true_val) > 0:
                per_channel_mape[key] = abs(est_val - true_val) / abs(true_val)
            else:
                per_channel_mape[key] = float("inf") if abs(est_val) > 0 else 0.0

        # Channel ranking (Kendall's tau)
        channel_ranking = 0.0
        if len(common_keys) >= 2:
            est_vals = [float(estimated_contributions[k]) for k in common_keys]
            true_vals = [float(true_contributions[k]) for k in common_keys]
            tau, _ = kendalltau(est_vals, true_vals)
            channel_ranking = float(tau) if not np.isnan(tau) else 0.0

        # Zero detection rate
        true_zeros = [k for k in common_keys if float(true_contributions[k]) == 0.0]
        if true_zeros:
            detected = sum(
                1
                for k in true_zeros
                if float(estimated_contributions[k]) == 0.0
            )
            zero_detection_rate = detected / len(true_zeros)
        else:
            zero_detection_rate = 1.0  # No zeros to detect, perfect by default

        return ContribResult(
            total_media_error=total_media_error,
            per_channel_mape=per_channel_mape,
            channel_ranking=channel_ranking,
            zero_detection_rate=zero_detection_rate,
        )
