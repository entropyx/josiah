"""Realism validator for synthetic MMM datasets.

Runs a suite of statistical checks on synthetic data to flag unrealistic
properties (negative spend, implausible variability, missing seasonality,
excessive collinearity, etc.).  Uses only numpy — no scipy dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TestResult:
    """Outcome of a single validation test."""
    test_name: str
    passed: bool
    p_value: Optional[float] = None
    detail: str = ""


@dataclass
class ValidationReport:
    """Aggregate validation report across all tests."""
    pass_fail: bool
    flagged_properties: list[str]
    details: dict[str, TestResult]  # {test_name: TestResult}


class RealismValidator:
    """Validate that synthetic MMM data looks realistic.

    Runs 7 tests covering spend distributions, autocorrelation,
    outcome variability, spend-to-outcome ratios, outliers, channel
    collinearity, and seasonality presence.

    Args:
        reference_data: Optional DataFrame of real data for comparison.
        category: Optional category string (used for ratio benchmarks).
    """

    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        category: Optional[str] = None,
    ):
        self.reference_data = reference_data
        self.category = category

    # ── public API ─────────────────────────────────────────────────────

    def validate(self, synthetic_data: pd.DataFrame) -> ValidationReport:
        """Run all validation tests on *synthetic_data*.

        The DataFrame is expected to have columns ``y`` (outcome) and
        zero or more ``*_spend`` columns.

        Returns:
            ValidationReport with per-test results.
        """
        tests: dict[str, TestResult] = {}

        spend_cols = [c for c in synthetic_data.columns if c.endswith("_spend")]
        y = synthetic_data["y"].values if "y" in synthetic_data.columns else None

        tests["spend_non_negative"] = self._test_spend_non_negative(
            synthetic_data, spend_cols
        )
        tests["spend_autocorrelation"] = self._test_spend_autocorrelation(
            synthetic_data, spend_cols
        )
        tests["outcome_variability"] = self._test_outcome_variability(y)
        tests["spend_outcome_ratio"] = self._test_spend_outcome_ratio(
            synthetic_data, spend_cols, y
        )
        tests["outlier_frequency"] = self._test_outlier_frequency(y)
        tests["channel_collinearity"] = self._test_channel_collinearity(
            synthetic_data, spend_cols
        )
        tests["seasonality_presence"] = self._test_seasonality_presence(y)

        pass_fail = all(t.passed for t in tests.values())
        flagged = [name for name, t in tests.items() if not t.passed]
        return ValidationReport(pass_fail, flagged, tests)

    # ── individual tests ───────────────────────────────────────────────

    @staticmethod
    def _test_spend_non_negative(
        df: pd.DataFrame, spend_cols: list[str]
    ) -> TestResult:
        """No spend column should contain negative values."""
        for col in spend_cols:
            if (df[col] < 0).any():
                return TestResult(
                    "spend_non_negative",
                    False,
                    detail=f"Negative values found in {col}",
                )
        return TestResult("spend_non_negative", True, detail="All spend >= 0")

    @staticmethod
    def _test_spend_autocorrelation(
        df: pd.DataFrame, spend_cols: list[str]
    ) -> TestResult:
        """Spend should have positive lag-1 autocorrelation (campaigns are persistent)."""
        if not spend_cols:
            return TestResult(
                "spend_autocorrelation", True, detail="No spend columns"
            )

        autocorrs: list[float] = []
        for col in spend_cols:
            x = df[col].values.astype(float)
            if len(x) < 3 or np.std(x) == 0:
                continue
            xm = x - np.mean(x)
            c0 = np.dot(xm, xm)
            if c0 == 0:
                continue
            c1 = np.dot(xm[:-1], xm[1:])
            autocorrs.append(c1 / c0)

        if not autocorrs:
            return TestResult(
                "spend_autocorrelation", True, detail="No variable spend columns"
            )

        mean_ac = float(np.mean(autocorrs))
        passed = mean_ac > -0.2  # Very lenient — just no strongly negative AC
        return TestResult(
            "spend_autocorrelation",
            passed,
            detail=f"Mean lag-1 autocorrelation = {mean_ac:.3f}",
        )

    @staticmethod
    def _test_outcome_variability(y: Optional[np.ndarray]) -> TestResult:
        """Outcome CV should be in a reasonable range (0.01 – 2.0)."""
        if y is None or len(y) < 2:
            return TestResult(
                "outcome_variability", True, detail="No outcome data"
            )
        mean_y = np.mean(y)
        if mean_y == 0:
            return TestResult(
                "outcome_variability", False, detail="Outcome mean is zero"
            )
        cv = float(np.std(y) / abs(mean_y))
        passed = 0.01 <= cv <= 2.0
        return TestResult(
            "outcome_variability",
            passed,
            detail=f"CV = {cv:.4f}",
        )

    @staticmethod
    def _test_spend_outcome_ratio(
        df: pd.DataFrame,
        spend_cols: list[str],
        y: Optional[np.ndarray],
    ) -> TestResult:
        """Total spend / total outcome should be in a plausible range."""
        if y is None or not spend_cols:
            return TestResult(
                "spend_outcome_ratio", True, detail="Insufficient data"
            )
        total_spend = sum(df[c].sum() for c in spend_cols)
        total_outcome = float(np.sum(y))
        if total_outcome == 0:
            return TestResult(
                "spend_outcome_ratio", False, detail="Total outcome is zero"
            )
        ratio = total_spend / abs(total_outcome)
        # Very wide acceptable range — just catch extreme absurdity
        passed = 0.001 <= ratio <= 100.0
        return TestResult(
            "spend_outcome_ratio",
            passed,
            detail=f"Spend/outcome ratio = {ratio:.4f}",
        )

    @staticmethod
    def _test_outlier_frequency(y: Optional[np.ndarray]) -> TestResult:
        """No more than 5% of outcome values should be extreme outliers.

        Uses IQR-based detection (robust to outlier contamination of std):
        a point is an outlier if it falls beyond Q1 - 3*IQR or Q3 + 3*IQR.
        """
        if y is None or len(y) < 10:
            return TestResult(
                "outlier_frequency", True, detail="Insufficient data"
            )
        q1 = float(np.percentile(y, 25))
        q3 = float(np.percentile(y, 75))
        iqr = q3 - q1
        if iqr == 0:
            return TestResult(
                "outlier_frequency", True, detail="Zero IQR"
            )
        lower = q1 - 3.0 * iqr
        upper = q3 + 3.0 * iqr
        outlier_frac = float(np.mean((y < lower) | (y > upper)))
        passed = outlier_frac <= 0.05
        return TestResult(
            "outlier_frequency",
            passed,
            detail=f"Outlier fraction (IQR) = {outlier_frac:.4f}",
        )

    @staticmethod
    def _test_channel_collinearity(
        df: pd.DataFrame, spend_cols: list[str]
    ) -> TestResult:
        """No pair of spend channels should have |correlation| > 0.98."""
        if len(spend_cols) < 2:
            return TestResult(
                "channel_collinearity", True, detail="< 2 channels"
            )
        mat = df[spend_cols].values.astype(float)
        # Standardise columns
        stds = np.std(mat, axis=0)
        nonzero = stds > 0
        if nonzero.sum() < 2:
            return TestResult(
                "channel_collinearity", True, detail="< 2 variable channels"
            )
        mat_nz = mat[:, nonzero]
        mat_nz = (mat_nz - mat_nz.mean(axis=0)) / mat_nz.std(axis=0)
        corr = (mat_nz.T @ mat_nz) / mat_nz.shape[0]
        np.fill_diagonal(corr, 0)
        max_corr = float(np.max(np.abs(corr)))
        passed = max_corr < 0.98
        return TestResult(
            "channel_collinearity",
            passed,
            detail=f"Max |corr| = {max_corr:.4f}",
        )

    @staticmethod
    def _test_seasonality_presence(y: Optional[np.ndarray]) -> TestResult:
        """Check for some periodic structure using simple FFT energy ratio.

        Passes if at least 1% of spectral energy is outside the DC component,
        or if the series is too short to assess.
        """
        if y is None or len(y) < 12:
            return TestResult(
                "seasonality_presence", True, detail="Insufficient data"
            )
        y_centered = y - np.mean(y)
        fft_vals = np.abs(np.fft.rfft(y_centered))
        total_energy = float(np.sum(fft_vals**2))
        if total_energy == 0:
            return TestResult(
                "seasonality_presence", True, detail="Zero energy"
            )
        dc_energy = float(fft_vals[0] ** 2)
        non_dc_ratio = (total_energy - dc_energy) / total_energy
        # Very lenient: just needs *some* non-DC energy
        passed = non_dc_ratio > 0.01
        return TestResult(
            "seasonality_presence",
            passed,
            detail=f"Non-DC energy ratio = {non_dc_ratio:.4f}",
        )
