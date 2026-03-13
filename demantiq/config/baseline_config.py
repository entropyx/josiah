"""Baseline configuration: trend, seasonality, organic level."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BaselineConfig:
    """Configuration for baseline demand (trend + seasonality + organic).

    Attributes:
        trend_type: Type of trend ('linear', 'cube_root').
        trend_params: Parameters for trend function.
        seasonality_type: Type of seasonality ('fourier').
        seasonality_period: Length of one seasonal cycle in periods (52 for weekly/yearly).
        seasonality_n_terms: Number of Fourier terms.
        seasonality_coefficients: Fourier coefficients. If empty, generated randomly.
        organic_level: Constant organic demand level (intercept).
    """
    trend_type: str = "linear"
    trend_params: dict = field(default_factory=lambda: {"slope": 1.0})
    seasonality_type: str = "fourier"
    seasonality_period: float = 52.0
    seasonality_n_terms: int = 2
    seasonality_coefficients: list = field(default_factory=list)
    organic_level: float = 1000.0

    def to_dict(self) -> dict:
        return {
            "trend_type": self.trend_type,
            "trend_params": dict(self.trend_params),
            "seasonality_type": self.seasonality_type,
            "seasonality_period": self.seasonality_period,
            "seasonality_n_terms": self.seasonality_n_terms,
            "seasonality_coefficients": list(self.seasonality_coefficients),
            "organic_level": self.organic_level,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BaselineConfig":
        return cls(**d)
