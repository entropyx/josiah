"""Per-channel configuration for the Demantiq simulator."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChannelConfig:
    """Configuration for a single media channel.

    Attributes:
        name: Human-readable channel label (e.g., 'tv', 'search').
        beta: True contribution coefficient.
        saturation_fn: Saturation function name ('hill', 'logistic', 'power', 'piecewise_linear').
        saturation_params: Function-specific parameters.
        adstock_fn: Adstock function name ('geometric', 'weibull_cdf', 'weibull_pdf', 'delayed_geometric').
        adstock_params: Function-specific parameters.
        spend_pattern: Spend pattern type ('always_on', 'pulsed', 'seasonal', 'front_loaded').
        spend_mean: Average spend per period.
        spend_std: Spend standard deviation.
        spend_floor: Minimum spend (0 for channels that go dark).
        correlation_group: Channels in same group have correlated spend.
    """
    name: str
    beta: float = 0.1
    saturation_fn: str = "hill"
    saturation_params: dict = field(default_factory=lambda: {"K": 0.5, "S": 2.0})
    adstock_fn: str = "geometric"
    adstock_params: dict = field(default_factory=lambda: {"alpha": 0.5, "max_lag": 8})
    spend_pattern: str = "always_on"
    spend_mean: float = 10000.0
    spend_std: float = 3000.0
    spend_floor: float = 0.0
    correlation_group: str = "default"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "beta": self.beta,
            "saturation_fn": self.saturation_fn,
            "saturation_params": dict(self.saturation_params),
            "adstock_fn": self.adstock_fn,
            "adstock_params": dict(self.adstock_params),
            "spend_pattern": self.spend_pattern,
            "spend_mean": self.spend_mean,
            "spend_std": self.spend_std,
            "spend_floor": self.spend_floor,
            "correlation_group": self.correlation_group,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChannelConfig":
        return cls(**d)
