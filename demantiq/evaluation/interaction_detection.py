"""Interaction detection evaluation metrics."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class InteractionResult:
    """Result of interaction detection evaluation.

    Attributes:
        detection_rate: TP / total true interactions.
        false_positive_rate: FP / total estimated interactions not in truth.
        magnitude_mape: MAPE of interaction magnitudes for detected interactions.
    """

    detection_rate: float = 0.0
    false_positive_rate: float = 0.0
    magnitude_mape: dict = field(default_factory=dict)


class InteractionDetection:
    """Evaluate accuracy of detected interactions."""

    def evaluate(
        self,
        estimated_interactions: dict,
        true_interactions: dict,
        threshold: float = 0.0,
    ) -> InteractionResult:
        """Compare estimated vs true interactions.

        Args:
            estimated_interactions: {interaction_name: coefficient}
            true_interactions: {interaction_name: coefficient}
            threshold: Minimum absolute value to consider an interaction detected.

        Returns:
            InteractionResult with detection metrics.
        """
        # Detected interactions (above threshold)
        detected = {
            k: v
            for k, v in estimated_interactions.items()
            if abs(float(v)) > threshold
        }

        true_keys = set(true_interactions.keys())
        detected_keys = set(detected.keys())

        # Detection rate: TP / total true
        if true_keys:
            tp = len(true_keys & detected_keys)
            detection_rate = tp / len(true_keys)
        else:
            detection_rate = 1.0  # No true interactions, perfect by default

        # False positive rate: FP / total detected (or 0 if none detected)
        false_positives = detected_keys - true_keys
        if detected_keys:
            false_positive_rate = len(false_positives) / len(detected_keys)
        else:
            false_positive_rate = 0.0

        # Magnitude MAPE for correctly detected interactions
        magnitude_mape = {}
        common = true_keys & detected_keys
        for key in common:
            true_val = float(true_interactions[key])
            est_val = float(detected[key])
            if abs(true_val) > 0:
                magnitude_mape[key] = abs(est_val - true_val) / abs(true_val)
            else:
                magnitude_mape[key] = float("inf") if abs(est_val) > 0 else 0.0

        return InteractionResult(
            detection_rate=detection_rate,
            false_positive_rate=false_positive_rate,
            magnitude_mape=magnitude_mape,
        )
