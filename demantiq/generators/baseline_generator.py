"""Baseline demand generation: trend + seasonality + organic level."""

import numpy as np
from numpy.random import Generator
from demantiq.config.baseline_config import BaselineConfig
from demantiq.utils.time_series import fourier_seasonality, linear_trend, cube_root_trend


def generate_baseline(config: BaselineConfig, n_periods: int,
                      rng: Generator) -> np.ndarray:
    """Generate baseline demand = trend + seasonality + organic_level.

    Args:
        config: Baseline configuration.
        n_periods: Number of time periods.
        rng: numpy Generator.

    Returns:
        Baseline demand array of shape (n_periods,).
    """
    trend = _generate_trend(config, n_periods)
    seasonality = _generate_seasonality(config, n_periods, rng)
    return config.organic_level + trend + seasonality


def _generate_trend(config: BaselineConfig, n_periods: int) -> np.ndarray:
    """Generate trend component."""
    params = config.trend_params
    if config.trend_type == "linear":
        slope = params.get("slope", 1.0)
        return linear_trend(n_periods, slope=slope)
    elif config.trend_type == "cube_root":
        scale = params.get("scale", 1.0)
        return cube_root_trend(n_periods, scale=scale)
    else:
        return np.zeros(n_periods)


def _generate_seasonality(config: BaselineConfig, n_periods: int,
                          rng: Generator) -> np.ndarray:
    """Generate seasonality component."""
    if config.seasonality_type == "fourier":
        coefficients = config.seasonality_coefficients if config.seasonality_coefficients else None
        return fourier_seasonality(
            n_periods,
            period=config.seasonality_period,
            n_terms=config.seasonality_n_terms,
            coefficients=coefficients,
            rng=rng,
        )
    return np.zeros(n_periods)
