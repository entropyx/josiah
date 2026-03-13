"""Tests for contribution accuracy evaluation."""

import pytest
from demantiq.evaluation.contribution_accuracy import ContributionAccuracy, ContribResult


class TestContributionAccuracy:
    def setup_method(self):
        self.evaluator = ContributionAccuracy()

    def test_identical_zero_error(self):
        contrib = {"fb": 100.0, "google": 200.0, "tiktok": 50.0}
        result = self.evaluator.evaluate(contrib, contrib)
        assert result.total_media_error == 0.0
        for v in result.per_channel_mape.values():
            assert v == 0.0

    def test_identical_perfect_ranking(self):
        contrib = {"fb": 100.0, "google": 200.0, "tiktok": 50.0}
        result = self.evaluator.evaluate(contrib, contrib)
        assert result.channel_ranking == pytest.approx(1.0)

    def test_shuffled_low_correlation(self):
        true = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0}
        estimated = {"a": 5.0, "b": 4.0, "c": 3.0, "d": 2.0, "e": 1.0}
        result = self.evaluator.evaluate(estimated, true)
        assert result.channel_ranking < 0  # reversed ranking

    def test_total_media_error(self):
        true = {"fb": 100.0, "google": 200.0}
        estimated = {"fb": 120.0, "google": 220.0}
        result = self.evaluator.evaluate(estimated, true)
        assert result.total_media_error == pytest.approx(40.0)

    def test_per_channel_mape(self):
        true = {"fb": 100.0}
        estimated = {"fb": 120.0}
        result = self.evaluator.evaluate(estimated, true)
        assert result.per_channel_mape["fb"] == pytest.approx(0.2)

    def test_zero_detection_all_zeros(self):
        true = {"fb": 0.0, "google": 0.0}
        estimated = {"fb": 0.0, "google": 0.0}
        result = self.evaluator.evaluate(estimated, true)
        assert result.zero_detection_rate == 1.0

    def test_zero_detection_missed(self):
        true = {"fb": 0.0, "google": 0.0}
        estimated = {"fb": 10.0, "google": 0.0}
        result = self.evaluator.evaluate(estimated, true)
        assert result.zero_detection_rate == 0.5

    def test_no_zeros_perfect_detection(self):
        true = {"fb": 100.0, "google": 200.0}
        estimated = {"fb": 100.0, "google": 200.0}
        result = self.evaluator.evaluate(estimated, true)
        assert result.zero_detection_rate == 1.0

    def test_only_common_keys(self):
        true = {"fb": 100.0, "extra": 50.0}
        estimated = {"fb": 100.0, "other": 30.0}
        result = self.evaluator.evaluate(estimated, true)
        assert "fb" in result.per_channel_mape
        assert "extra" not in result.per_channel_mape

    def test_result_dataclass_defaults(self):
        result = ContribResult()
        assert result.total_media_error == 0.0
        assert result.per_channel_mape == {}
        assert result.channel_ranking == 0.0
        assert result.zero_detection_rate == 0.0
