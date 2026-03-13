"""Tests for parameter recovery evaluation."""

import numpy as np
import pytest
from demantiq.evaluation.parameter_recovery import ParameterRecovery, RecoveryResult


class TestParameterRecovery:
    def setup_method(self):
        self.evaluator = ParameterRecovery()

    def test_perfect_estimates_zero_bias(self):
        truth = {"beta_fb": 1.0, "beta_google": 2.0, "alpha": 0.5}
        estimated = {"beta_fb": 1.0, "beta_google": 2.0, "alpha": 0.5}
        result = self.evaluator.evaluate(estimated, truth)
        for key in truth:
            assert result.bias[key] == 0.0
            assert result.mape[key] == 0.0

    def test_perfect_estimates_zero_mape(self):
        truth = {"beta_fb": 3.0, "beta_google": 5.0}
        result = self.evaluator.evaluate(truth.copy(), truth)
        for key in truth:
            assert result.mape[key] == 0.0

    def test_known_bias_correct_direction(self):
        truth = {"beta_fb": 1.0, "beta_google": 2.0}
        estimated = {"beta_fb": 1.5, "beta_google": 1.5}
        result = self.evaluator.evaluate(estimated, truth)
        assert result.bias["beta_fb"] > 0  # overestimate
        assert result.bias["beta_google"] < 0  # underestimate

    def test_mape_calculation(self):
        truth = {"beta_fb": 10.0}
        estimated = {"beta_fb": 12.0}
        result = self.evaluator.evaluate(estimated, truth)
        assert result.mape["beta_fb"] == pytest.approx(0.2)

    def test_rank_correlation_perfect(self):
        truth = {"beta_a": 1.0, "beta_b": 2.0, "beta_c": 3.0}
        estimated = {"beta_a": 10.0, "beta_b": 20.0, "beta_c": 30.0}
        result = self.evaluator.evaluate(estimated, truth)
        assert result.rank_correlation == pytest.approx(1.0)

    def test_rank_correlation_reversed(self):
        truth = {"beta_a": 1.0, "beta_b": 2.0, "beta_c": 3.0}
        estimated = {"beta_a": 30.0, "beta_b": 20.0, "beta_c": 10.0}
        result = self.evaluator.evaluate(estimated, truth)
        assert result.rank_correlation == pytest.approx(-1.0)

    def test_coverage_inside_interval(self):
        truth = {"beta_fb": 1.0}
        estimated = {"beta_fb": 1.1}
        # Samples centered around 1.0, truth should be inside 95% CI
        samples = {"beta_fb": np.random.normal(1.0, 0.1, 10000)}
        result = self.evaluator.evaluate(estimated, truth, posterior_samples=samples)
        assert result.coverage["beta_fb"] == 1.0

    def test_coverage_outside_interval(self):
        truth = {"beta_fb": 100.0}
        estimated = {"beta_fb": 0.0}
        # Samples far from truth
        samples = {"beta_fb": np.random.normal(0.0, 0.01, 10000)}
        result = self.evaluator.evaluate(estimated, truth, posterior_samples=samples)
        assert result.coverage["beta_fb"] == 0.0

    def test_interval_width(self):
        truth = {"beta_fb": 1.0}
        estimated = {"beta_fb": 1.0}
        samples = {"beta_fb": np.array([0.0, 0.5, 1.0, 1.5, 2.0] * 100)}
        result = self.evaluator.evaluate(estimated, truth, posterior_samples=samples)
        assert result.interval_width["beta_fb"] > 0

    def test_only_common_keys(self):
        truth = {"beta_fb": 1.0, "extra_true": 5.0}
        estimated = {"beta_fb": 1.0, "extra_est": 3.0}
        result = self.evaluator.evaluate(estimated, truth)
        assert "beta_fb" in result.bias
        assert "extra_true" not in result.bias
        assert "extra_est" not in result.bias

    def test_zero_true_value(self):
        truth = {"beta_fb": 0.0}
        estimated = {"beta_fb": 0.0}
        result = self.evaluator.evaluate(estimated, truth)
        assert result.mape["beta_fb"] == 0.0

    def test_zero_true_nonzero_estimate(self):
        truth = {"beta_fb": 0.0}
        estimated = {"beta_fb": 1.0}
        result = self.evaluator.evaluate(estimated, truth)
        assert result.mape["beta_fb"] == float("inf")

    def test_no_beta_keys_rank_correlation_zero(self):
        truth = {"alpha": 0.5, "lambda": 1.0}
        estimated = {"alpha": 0.5, "lambda": 1.0}
        result = self.evaluator.evaluate(estimated, truth)
        assert result.rank_correlation == 0.0

    def test_result_dataclass(self):
        result = RecoveryResult()
        assert result.bias == {}
        assert result.mape == {}
        assert result.coverage == {}
        assert result.interval_width == {}
        assert result.rank_correlation == 0.0
