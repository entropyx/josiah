"""External/macro variable configuration."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MacroVariable:
    """Configuration for a single external variable."""

    name: str = "consumer_confidence"
    effect_on_demand: float = 50.0
    time_series_type: str = "mean_reverting"  # random_walk, mean_reverting, trending, seasonal
    params: dict = field(default_factory=dict)
    correlation_with_spend: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "effect_on_demand": self.effect_on_demand,
            "time_series_type": self.time_series_type,
            "params": dict(self.params),
            "correlation_with_spend": self.correlation_with_spend,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MacroVariable":
        return cls(**d)


@dataclass(frozen=True)
class RegimeChange:
    """A structural break in the simulation."""

    period: int = 52
    change_type: str = "level_shift"
    magnitude: float = -0.2
    affected_params: list = field(default_factory=lambda: ["baseline"])
    recovery: str = "permanent"
    recovery_periods: int = 0

    def to_dict(self) -> dict:
        return {
            "period": self.period,
            "change_type": self.change_type,
            "magnitude": self.magnitude,
            "affected_params": list(self.affected_params),
            "recovery": self.recovery,
            "recovery_periods": self.recovery_periods,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RegimeChange":
        return cls(**d)


@dataclass(frozen=True)
class MacroConfig:
    """Configuration for external/macro variables."""

    variables: tuple = ()
    regime_changes: tuple = ()

    def __post_init__(self):
        if isinstance(self.variables, list):
            object.__setattr__(self, "variables", tuple(self.variables))
        if isinstance(self.regime_changes, list):
            object.__setattr__(self, "regime_changes", tuple(self.regime_changes))

    def to_dict(self) -> dict:
        return {
            "variables": [v.to_dict() for v in self.variables],
            "regime_changes": [r.to_dict() for r in self.regime_changes],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MacroConfig":
        variables = [MacroVariable.from_dict(v) for v in d.get("variables", [])]
        regime_changes = [RegimeChange.from_dict(r) for r in d.get("regime_changes", [])]
        return cls(variables=variables, regime_changes=regime_changes)
