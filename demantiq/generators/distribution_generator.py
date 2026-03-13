"""Distribution/availability simulation."""

import numpy as np
from numpy.random import Generator
from dataclasses import dataclass
from demantiq.config.distribution_config import DistributionConfig


@dataclass
class DistributionResult:
    """Result of distribution generation."""
    distribution: np.ndarray     # weighted distribution (0-1)
    stockout_mask: np.ndarray    # boolean: True = stockout
    distribution_cap: np.ndarray # multiplier applied to demand


def generate_distribution(config: DistributionConfig, n_periods: int,
                          rng: Generator) -> DistributionResult:
    """Generate distribution/availability over time."""
    dist = _generate_trajectory(config, n_periods)

    # Stockouts
    stockout_mask = rng.random(n_periods) < config.stockout_probability

    # Distribution cap: how distribution limits demand
    # cap = 1 when ceiling_effect = 0 (no ceiling)
    # cap = distribution when ceiling_effect = 1 (hard ceiling)
    cap = 1.0 - config.distribution_ceiling_effect * (1.0 - dist)

    # Apply stockout losses
    cap[stockout_mask] *= (1.0 - config.stockout_demand_loss)

    return DistributionResult(distribution=dist, stockout_mask=stockout_mask,
                               distribution_cap=cap)


def _generate_trajectory(config: DistributionConfig, n_periods: int) -> np.ndarray:
    """Generate distribution trajectory."""
    t = np.arange(n_periods, dtype=float)
    init = config.initial_distribution
    params = config.trajectory_params

    if config.distribution_trajectory == "stable":
        return np.full(n_periods, init)

    elif config.distribution_trajectory == "growing":
        rate = params.get("growth_rate", 0.01)
        ceiling = params.get("ceiling", 1.0)
        # Logistic growth
        midpoint = n_periods / 2
        raw = ceiling / (1 + np.exp(-rate * (t - midpoint)))
        return np.clip(init + raw * (ceiling - init), 0, 1)

    elif config.distribution_trajectory == "declining":
        rate = params.get("decline_rate", 0.01)
        return np.clip(init * np.exp(-rate * t / n_periods), 0, 1)

    elif config.distribution_trajectory == "step_change":
        step_period = params.get("step_period", n_periods // 2)
        step_magnitude = params.get("step_magnitude", 0.2)
        dist = np.full(n_periods, init)
        dist[step_period:] += step_magnitude
        return np.clip(dist, 0, 1)

    return np.full(n_periods, init)
