"""Tests for ROAS accuracy evaluation."""

import pytest
from demantiq.evaluation.roas_accuracy import ROASAccuracy, ROASResult


class TestROASAccuracy:
    def setup_method(self):
        self.evaluator = ROASAccuracy()

    def test_perfect_zero_mape(self):
        roas = {"fb": 2.5, "google": 1.8, "tiktok": 0.7}
        result = self.evaluator.evaluate(roas, roas)
        for v in result.per_channel_mape.values():
            assert v == 0.0

    def test_perfect_ranking(self):
        roas = {"fb": 2.5, "google": 1.8, "tiktok": 0.7}
        result = self.evaluator.evaluate(roas, roas)
        assert result.ranking_correlation == pytest.approx(1.0)

    def test_perfect_direction(self):
        roas = {"fb": 2.5, "google": 0.5}
        result = self.evaluator.evaluate(roas, roas)
        assert result.direction_accuracy == 1.0

    def test_direction_accuracy_correct(self):
        true = {"fb": 2.0, "google": 0.5}
        estimated = {"fb": 1.5, "google": 0.3}  # same direction
        result = self.evaluator.evaluate(estimated, true)
        assert result.direction_accuracy == 1.0

    def test_direction_accuracy_wrong(self):
        true = {"fb": 2.0, "google": 0.5}
        estimated = {"fb": 0.5, "google": 2.0}  # both wrong
        result = self.evaluator.evaluate(estimated, true)
        assert result.direction_accuracy == 0.0

    def test_direction_accuracy_partial(self):
        true = {"fb": 2.0, "google": 0.5}
        estimated = {"fb": 1.5, "google": 2.0}  # one right, one wrong
        result = self.evaluator.evaluate(estimated, true)
        assert result.direction_accuracy == 0.5

    def test_mape_calculation(self):
        true = {"fb": 2.0}
        estimated = {"fb": 2.4}
        result = self.evaluator.evaluate(estimated, true)
        assert result.per_channel_mape["fb"] == pytest.approx(0.2)

    def test_reversed_ranking(self):
        true = {"a": 1.0, "b": 2.0, "c": 3.0}
        estimated = {"a": 3.0, "b": 2.0, "c": 1.0}
        result = self.evaluator.evaluate(estimated, true)
        assert result.ranking_correlation < 0

    def test_empty_channels(self):
        result = self.evaluator.evaluate({}, {})
        assert result.per_channel_mape == {}
        assert result.direction_accuracy == 0.0

    def test_result_dataclass_defaults(self):
        result = ROASResult()
        assert result.per_channel_mape == {}
        assert result.ranking_correlation == 0.0
        assert result.direction_accuracy == 0.0
