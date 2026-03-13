"""Calibration module for Demantiq.

Provides empirical distributions, public data adapters, and realism
validation for synthetic MMM datasets.
"""

from demantiq.calibration.empirical_distributions import (
    EmpiricalDistributions,
    Observation,
)
from demantiq.calibration.public_data_adapter import (
    PublicDataAdapter,
    CATEGORY_BENCHMARKS,
)
from demantiq.calibration.realism_validator import (
    RealismValidator,
    TestResult,
    ValidationReport,
)

__all__ = [
    "EmpiricalDistributions",
    "Observation",
    "PublicDataAdapter",
    "CATEGORY_BENCHMARKS",
    "RealismValidator",
    "TestResult",
    "ValidationReport",
]
