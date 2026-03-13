"""Tests for interaction detection evaluation."""

import pytest
from demantiq.evaluation.interaction_detection import InteractionDetection, InteractionResult


class TestInteractionDetection:
    def setup_method(self):
        self.evaluator = InteractionDetection()

    def test_perfect_detection(self):
        true = {"fb_x_google": 0.5, "fb_x_tiktok": 0.3}
        estimated = {"fb_x_google": 0.5, "fb_x_tiktok": 0.3}
        result = self.evaluator.evaluate(estimated, true)
        assert result.detection_rate == 1.0
        assert result.false_positive_rate == 0.0

    def test_missed_interactions(self):
        true = {"fb_x_google": 0.5, "fb_x_tiktok": 0.3}
        estimated = {"fb_x_google": 0.5}  # missed fb_x_tiktok
        result = self.evaluator.evaluate(estimated, true)
        assert result.detection_rate == 0.5

    def test_false_positives(self):
        true = {"fb_x_google": 0.5}
        estimated = {"fb_x_google": 0.5, "fake_interaction": 0.2}
        result = self.evaluator.evaluate(estimated, true)
        assert result.detection_rate == 1.0
        assert result.false_positive_rate == 0.5

    def test_threshold_filters(self):
        true = {"fb_x_google": 0.5}
        estimated = {"fb_x_google": 0.01}  # below threshold
        result = self.evaluator.evaluate(estimated, true, threshold=0.05)
        assert result.detection_rate == 0.0

    def test_threshold_passes(self):
        true = {"fb_x_google": 0.5}
        estimated = {"fb_x_google": 0.1}  # above threshold
        result = self.evaluator.evaluate(estimated, true, threshold=0.05)
        assert result.detection_rate == 1.0

    def test_magnitude_mape(self):
        true = {"fb_x_google": 1.0}
        estimated = {"fb_x_google": 1.2}
        result = self.evaluator.evaluate(estimated, true)
        assert result.magnitude_mape["fb_x_google"] == pytest.approx(0.2)

    def test_no_true_interactions(self):
        true = {}
        estimated = {"fake": 0.5}
        result = self.evaluator.evaluate(estimated, true)
        assert result.detection_rate == 1.0  # nothing to detect
        assert result.false_positive_rate == 1.0  # all are false positives

    def test_no_estimated_interactions(self):
        true = {"fb_x_google": 0.5}
        estimated = {}
        result = self.evaluator.evaluate(estimated, true)
        assert result.detection_rate == 0.0
        assert result.false_positive_rate == 0.0

    def test_result_dataclass_defaults(self):
        result = InteractionResult()
        assert result.detection_rate == 0.0
        assert result.false_positive_rate == 0.0
        assert result.magnitude_mape == {}
