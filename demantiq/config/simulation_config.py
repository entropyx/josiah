"""Master simulation configuration."""

from dataclasses import dataclass, field
from demantiq.config.channel_config import ChannelConfig
from demantiq.config.baseline_config import BaselineConfig
from demantiq.config.noise_config import NoiseConfig


@dataclass(frozen=True)
class SimulationConfig:
    """Master configuration for a Demantiq simulation.

    Optional configs (pricing, distribution, etc.) default to None.
    The demand kernel skips those pipeline steps when None.

    Attributes:
        n_periods: Number of time periods to simulate (26-520).
        granularity: 'weekly' or 'daily'.
        channels: List of channel configurations.
        noise: Noise/error term configuration.
        baseline: Baseline demand configuration.
        seed: Random seed for exact reproducibility.
        metadata: Optional tags (category, region, scenario name).
        pricing: PricingConfig (M2, None for M1).
        distribution: DistributionConfig (M2, None for M1).
        competition: CompetitionConfig (M3, None for M1).
        macro: MacroConfig (M3, None for M1).
        endogeneity: EndogeneityConfig (M3, None for M1).
        interactions: InteractionConfig (M4, None for M1).
    """
    n_periods: int = 104
    granularity: str = "weekly"
    channels: tuple[ChannelConfig, ...] = ()
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    seed: int = 42
    metadata: dict = field(default_factory=dict)
    # Future milestone configs — None means skip that pipeline step
    pricing: object = None
    distribution: object = None
    competition: object = None
    macro: object = None
    endogeneity: object = None
    interactions: object = None

    def __post_init__(self):
        # Convert list of channels to tuple for frozen dataclass
        if isinstance(self.channels, list):
            object.__setattr__(self, 'channels', tuple(self.channels))

    def to_dict(self) -> dict:
        d = {
            "n_periods": self.n_periods,
            "granularity": self.granularity,
            "channels": [c.to_dict() for c in self.channels],
            "noise": self.noise.to_dict(),
            "baseline": self.baseline.to_dict(),
            "seed": self.seed,
            "metadata": dict(self.metadata),
        }
        # Add optional configs if present
        for key in ("pricing", "distribution", "competition", "macro",
                     "endogeneity", "interactions"):
            val = getattr(self, key)
            if val is not None and hasattr(val, 'to_dict'):
                d[key] = val.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SimulationConfig":
        channels = [ChannelConfig.from_dict(c) for c in d.get("channels", [])]
        noise = NoiseConfig.from_dict(d["noise"]) if "noise" in d else NoiseConfig()
        baseline = BaselineConfig.from_dict(d["baseline"]) if "baseline" in d else BaselineConfig()
        return cls(
            n_periods=d.get("n_periods", 104),
            granularity=d.get("granularity", "weekly"),
            channels=channels,
            noise=noise,
            baseline=baseline,
            seed=d.get("seed", 42),
            metadata=d.get("metadata", {}),
        )
