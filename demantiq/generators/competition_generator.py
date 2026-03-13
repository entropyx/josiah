"""Competitive dynamics simulation."""

import numpy as np
from numpy.random import Generator
from dataclasses import dataclass

from demantiq.config.competition_config import CompetitionConfig


@dataclass
class CompetitionResult:
    """Result of competition generation."""

    competitor_sov: np.ndarray  # competitor share of voice
    competition_effect: np.ndarray  # effect on demand (typically negative)


def generate_competition(
    config: CompetitionConfig, n_periods: int, rng: Generator
) -> CompetitionResult:
    """Generate competitive dynamics."""
    sov = _generate_sov(config, n_periods, rng)

    # Apply intensity trend
    if config.competitive_intensity_trend == "increasing":
        t = np.arange(n_periods, dtype=float)
        sov = sov * (1 + 0.5 * t / n_periods)
    elif config.competitive_intensity_trend == "decreasing":
        t = np.arange(n_periods, dtype=float)
        sov = sov * (1 - 0.3 * t / n_periods)

    sov = np.clip(sov, 0, 1)

    # Competition effect: suppresses demand
    competition_effect = -config.sov_suppression_coefficient * sov

    return CompetitionResult(competitor_sov=sov, competition_effect=competition_effect)


def _generate_sov(
    config: CompetitionConfig, n_periods: int, rng: Generator
) -> np.ndarray:
    """Generate competitor share of voice time series."""
    mean = config.competitor_sov_mean

    if config.competitor_sov_pattern == "stable":
        return np.full(n_periods, mean) + rng.normal(0, mean * 0.05, n_periods)

    elif config.competitor_sov_pattern == "seasonal":
        t = np.arange(n_periods, dtype=float)
        seasonal = mean * (1 + 0.3 * np.sin(2 * np.pi * t / 52))
        return seasonal + rng.normal(0, mean * 0.05, n_periods)

    elif config.competitor_sov_pattern == "reactive":
        # Random walk around mean
        sov = np.zeros(n_periods)
        sov[0] = mean
        for t in range(1, n_periods):
            sov[t] = sov[t - 1] + rng.normal(0, mean * 0.05)
        return np.clip(sov, mean * 0.5, mean * 1.5)

    elif config.competitor_sov_pattern == "random":
        return rng.uniform(mean * 0.5, mean * 1.5, n_periods)

    return np.full(n_periods, mean)
