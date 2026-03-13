"""Endogeneity/feedback configuration."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EndogeneityConfig:
    """Configuration for endogeneity and confounding.

    Attributes:
        overall_strength: Global endogeneity intensity (0.0=exogenous, 1.0=fully endogenous).
        feedback_lag: How many periods back outcome feeds into spend.
        feedback_channels: Which channels react to outcome signals. Empty = all channels.
        seasonal_allocation_bias: How much spend increases during high-baseline periods.
        performance_chasing: How much spend chases recent performance.
        algorithmic_targeting_bias: Simulates platform optimization.
        omitted_variable_strength: Strength of hidden confounder (OVB).
        omitted_variable_ar: AR(1) coefficient for hidden confounder.
    """

    overall_strength: float = 0.3
    feedback_lag: int = 1
    feedback_channels: list = field(default_factory=list)
    seasonal_allocation_bias: float = 0.0
    performance_chasing: float = 0.0
    algorithmic_targeting_bias: float = 0.0
    omitted_variable_strength: float = 0.0
    omitted_variable_ar: float = 0.7

    def to_dict(self) -> dict:
        return {
            "overall_strength": self.overall_strength,
            "feedback_lag": self.feedback_lag,
            "feedback_channels": list(self.feedback_channels),
            "seasonal_allocation_bias": self.seasonal_allocation_bias,
            "performance_chasing": self.performance_chasing,
            "algorithmic_targeting_bias": self.algorithmic_targeting_bias,
            "omitted_variable_strength": self.omitted_variable_strength,
            "omitted_variable_ar": self.omitted_variable_ar,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EndogeneityConfig":
        return cls(**d)
