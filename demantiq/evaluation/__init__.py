"""Evaluation framework for Demantiq synthetic MMM data."""

from demantiq.evaluation.parameter_recovery import ParameterRecovery, RecoveryResult
from demantiq.evaluation.contribution_accuracy import (
    ContributionAccuracy,
    ContribResult,
)
from demantiq.evaluation.roas_accuracy import ROASAccuracy, ROASResult
from demantiq.evaluation.interaction_detection import (
    InteractionDetection,
    InteractionResult,
)
from demantiq.evaluation.optimization_quality import OptimizationQuality, OptResult
from demantiq.evaluation.capability_surface import CapabilitySurface
from demantiq.evaluation.model_comparison import (
    ModelAdapter,
    ModelComparison,
    ComparisonReport,
)

__all__ = [
    "ParameterRecovery",
    "RecoveryResult",
    "ContributionAccuracy",
    "ContribResult",
    "ROASAccuracy",
    "ROASResult",
    "InteractionDetection",
    "InteractionResult",
    "OptimizationQuality",
    "OptResult",
    "CapabilitySurface",
    "ModelAdapter",
    "ModelComparison",
    "ComparisonReport",
]
