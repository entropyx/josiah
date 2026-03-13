"""Tests for RealismValidator."""

import numpy as np
import pandas as pd
import pytest

from demantiq.calibration.realism_validator import (
    RealismValidator,
    ValidationReport,
    TestResult,
)
from demantiq.scenarios.scenario_library import ScenarioLibrary
from demantiq.core.demand_kernel import simulate


class TestCleanRoomPasses:
    """A clean-room scenario should pass all validation tests."""

    @pytest.fixture(scope="class")
    def clean_data(self):
        config = ScenarioLibrary.clean_room()
        result = simulate(config)
        return result.observable_data

    def test_clean_room_passes_validation(self, clean_data):
        validator = RealismValidator()
        report = validator.validate(clean_data)
        assert report.pass_fail, f"Flagged: {report.flagged_properties}"

    def test_all_tests_ran(self, clean_data):
        validator = RealismValidator()
        report = validator.validate(clean_data)
        expected_tests = {
            "spend_non_negative",
            "spend_autocorrelation",
            "outcome_variability",
            "spend_outcome_ratio",
            "outlier_frequency",
            "channel_collinearity",
            "seasonality_presence",
        }
        assert set(report.details.keys()) == expected_tests

    def test_report_structure(self, clean_data):
        validator = RealismValidator()
        report = validator.validate(clean_data)
        assert isinstance(report, ValidationReport)
        assert isinstance(report.pass_fail, bool)
        assert isinstance(report.flagged_properties, list)
        for name, result in report.details.items():
            assert isinstance(result, TestResult)
            assert result.test_name == name


class TestBadDataFails:
    """Artificially bad data should fail specific tests."""

    def test_negative_spend_fails(self):
        df = pd.DataFrame({
            "y": np.random.default_rng(0).normal(1000, 100, 100),
            "facebook_spend": np.random.default_rng(1).normal(0, 100, 100),  # will have negatives
        })
        validator = RealismValidator()
        report = validator.validate(df)
        assert "spend_non_negative" in report.flagged_properties

    def test_zero_variance_outcome_fails(self):
        df = pd.DataFrame({
            "y": np.ones(100) * 1000,
            "facebook_spend": np.random.default_rng(0).uniform(100, 200, 100),
        })
        validator = RealismValidator()
        report = validator.validate(df)
        assert "outcome_variability" in report.flagged_properties

    def test_extreme_outliers_fail(self):
        # Create data with >5% of values beyond 3*IQR fences.
        # Tight normal cluster with 10% extreme spikes.
        rng = np.random.default_rng(42)
        n = 200
        y = rng.normal(1000, 1, n)  # IQR ~ 1.35
        # Replace 10% with values far beyond 3*IQR
        n_outliers = 20
        y[:n_outliers] = 1000 + 100 * np.sign(rng.standard_normal(n_outliers))
        df = pd.DataFrame({"y": y})
        validator = RealismValidator()
        report = validator.validate(df)
        assert "outlier_frequency" in report.flagged_properties

    def test_perfectly_collinear_channels_fail(self):
        x = np.arange(100, dtype=float)
        df = pd.DataFrame({
            "y": np.random.default_rng(0).normal(1000, 100, 100),
            "ch_a_spend": x,
            "ch_b_spend": x * 2 + 5,  # perfect linear correlation
        })
        validator = RealismValidator()
        report = validator.validate(df)
        assert "channel_collinearity" in report.flagged_properties

    def test_zero_outcome_fails_ratio(self):
        df = pd.DataFrame({
            "y": np.zeros(100),
            "facebook_spend": np.ones(100) * 1000,
        })
        validator = RealismValidator()
        report = validator.validate(df)
        assert "spend_outcome_ratio" in report.flagged_properties


class TestEdgeCases:
    """Edge cases: no spend columns, minimal data, etc."""

    def test_no_spend_columns(self):
        df = pd.DataFrame({
            "y": np.random.default_rng(0).normal(1000, 100, 50),
        })
        validator = RealismValidator()
        report = validator.validate(df)
        # Should still run without errors; spend-related tests pass vacuously
        assert isinstance(report, ValidationReport)

    def test_single_channel(self):
        rng = np.random.default_rng(99)
        df = pd.DataFrame({
            "y": rng.normal(1000, 100, 100),
            "google_spend": rng.uniform(500, 1500, 100),
        })
        validator = RealismValidator()
        report = validator.validate(df)
        assert isinstance(report, ValidationReport)

    def test_with_reference_data_and_category(self):
        """Constructor accepts optional reference_data and category."""
        ref = pd.DataFrame({"y": [1, 2, 3]})
        validator = RealismValidator(reference_data=ref, category="supplements")
        assert validator.reference_data is not None
        assert validator.category == "supplements"
