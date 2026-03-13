"""Competitive dynamics configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CompetitionConfig:
    """Configuration for competitive dynamics.

    Attributes:
        n_competitors: Number of simulated competitors.
        competitor_sov_mean: Average competitor share of voice.
        competitor_sov_pattern: How SOV evolves ('stable', 'seasonal', 'reactive', 'random').
        sov_suppression_coefficient: How much competitor SOV reduces own media effectiveness.
        competitive_intensity_trend: Overall trend ('stable', 'increasing', 'decreasing').
    """

    n_competitors: int = 2
    competitor_sov_mean: float = 0.3
    competitor_sov_pattern: str = "stable"
    sov_suppression_coefficient: float = 0.1
    competitive_intensity_trend: str = "stable"

    def to_dict(self) -> dict:
        return {
            "n_competitors": self.n_competitors,
            "competitor_sov_mean": self.competitor_sov_mean,
            "competitor_sov_pattern": self.competitor_sov_pattern,
            "sov_suppression_coefficient": self.sov_suppression_coefficient,
            "competitive_intensity_trend": self.competitive_intensity_trend,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CompetitionConfig":
        return cls(**d)
