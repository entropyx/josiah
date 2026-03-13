"""Parameter recovery evaluation metrics."""

from dataclasses import dataclass, field
import numpy as np
from scipy.stats import spearmanr


@dataclass
class RecoveryResult:
    """Result of parameter recovery evaluation.

    Attributes:
        bias: E[theta] - theta* per parameter.
        mape: |E[theta] - theta*| / |theta*| per parameter.
        coverage: P(theta* in 95% CI) per parameter (if posterior samples provided).
        interval_width: Width of 95% CI per parameter (if posterior samples provided).
        rank_correlation: Spearman correlation between true and estimated channel rankings.
    """

    bias: dict = field(default_factory=dict)
    mape: dict = field(default_factory=dict)
    coverage: dict = field(default_factory=dict)
    interval_width: dict = field(default_factory=dict)
    rank_correlation: float = 0.0


class ParameterRecovery:
    """Evaluate how well estimated parameters recover true values."""

    def evaluate(
        self,
        estimated: dict,
        truth: dict,
        posterior_samples: dict = None,
    ) -> RecoveryResult:
        """Compare estimated parameters to truth.

        Args:
            estimated: {param_name: point_estimate}
            truth: {param_name: true_value}
            posterior_samples: Optional {param_name: array of samples} for coverage.

        Returns:
            RecoveryResult with bias, MAPE, and optional coverage metrics.
        """
        common_keys = set(estimated.keys()) & set(truth.keys())

        bias = {}
        mape = {}
        for key in common_keys:
            est_val = float(estimated[key])
            true_val = float(truth[key])
            bias[key] = est_val - true_val
            if abs(true_val) > 0:
                mape[key] = abs(est_val - true_val) / abs(true_val)
            else:
                mape[key] = float("inf") if abs(est_val) > 0 else 0.0

        coverage = {}
        interval_width = {}
        if posterior_samples is not None:
            for key in common_keys:
                if key in posterior_samples:
                    samples = np.asarray(posterior_samples[key])
                    lo = np.percentile(samples, 2.5)
                    hi = np.percentile(samples, 97.5)
                    true_val = float(truth[key])
                    coverage[key] = 1.0 if lo <= true_val <= hi else 0.0
                    interval_width[key] = float(hi - lo)

        # Rank correlation on channel betas
        beta_keys = [k for k in common_keys if "beta" in k.lower()]
        rank_corr = 0.0
        if len(beta_keys) >= 2:
            est_vals = [float(estimated[k]) for k in beta_keys]
            true_vals = [float(truth[k]) for k in beta_keys]
            # Avoid warnings on constant input
            if len(set(est_vals)) > 1 and len(set(true_vals)) > 1:
                corr, _ = spearmanr(est_vals, true_vals)
                rank_corr = float(corr) if not np.isnan(corr) else 0.0

        return RecoveryResult(
            bias=bias,
            mape=mape,
            coverage=coverage,
            interval_width=interval_width,
            rank_correlation=rank_corr,
        )
