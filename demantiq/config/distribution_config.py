"""Distribution/availability configuration."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DistributionConfig:
    """Configuration for distribution/availability.

    Attributes:
        initial_distribution: Starting weighted distribution (0.0-1.0).
        distribution_trajectory: How distribution evolves ('stable', 'growing', 'declining', 'step_change').
        trajectory_params: Parameters for trajectory (growth_rate, step_period, step_magnitude).
        distribution_ceiling_effect: How strongly distribution caps media-driven demand.
        stockout_probability: Random stockout probability per period.
        stockout_demand_loss: Fraction of demand lost during stockout.
    """
    initial_distribution: float = 0.8
    distribution_trajectory: str = "stable"
    trajectory_params: dict = field(default_factory=dict)
    distribution_ceiling_effect: float = 0.0
    stockout_probability: float = 0.0
    stockout_demand_loss: float = 0.5

    def to_dict(self) -> dict:
        return {
            "initial_distribution": self.initial_distribution,
            "distribution_trajectory": self.distribution_trajectory,
            "trajectory_params": dict(self.trajectory_params),
            "distribution_ceiling_effect": self.distribution_ceiling_effect,
            "stockout_probability": self.stockout_probability,
            "stockout_demand_loss": self.stockout_demand_loss,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DistributionConfig":
        return cls(**d)
