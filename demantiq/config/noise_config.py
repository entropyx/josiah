"""Noise/error term configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NoiseConfig:
    """Configuration for the error term.

    Attributes:
        noise_type: Distribution type ('gaussian', 't_distributed', 'heteroscedastic', 'autocorrelated').
        noise_scale: Base noise standard deviation. Ignored if signal_to_noise_ratio is set.
        signal_to_noise_ratio: If set, compute noise_scale from demand variance. Overrides noise_scale.
        autocorrelation: AR(1) coefficient for autocorrelated noise.
        t_df: Degrees of freedom for t-distributed noise.
        heteroscedasticity_power: For heteroscedastic noise, variance scales as demand^power.
        outlier_probability: Probability of outlier per period (0-1).
        outlier_magnitude: Scale factor for outlier observations.
    """
    noise_type: str = "gaussian"
    noise_scale: float = 50.0
    signal_to_noise_ratio: float | None = None
    autocorrelation: float = 0.0
    t_df: float = 5.0
    heteroscedasticity_power: float = 0.5
    outlier_probability: float = 0.0
    outlier_magnitude: float = 3.0

    def to_dict(self) -> dict:
        return {
            "noise_type": self.noise_type,
            "noise_scale": self.noise_scale,
            "signal_to_noise_ratio": self.signal_to_noise_ratio,
            "autocorrelation": self.autocorrelation,
            "t_df": self.t_df,
            "heteroscedasticity_power": self.heteroscedasticity_power,
            "outlier_probability": self.outlier_probability,
            "outlier_magnitude": self.outlier_magnitude,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NoiseConfig":
        return cls(**d)
