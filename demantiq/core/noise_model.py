"""Noise generation for the Demantiq simulator."""

import numpy as np
from numpy.random import Generator
from demantiq.config.noise_config import NoiseConfig


def generate_noise(config: NoiseConfig, demand: np.ndarray,
                   rng: Generator) -> np.ndarray:
    """Generate noise/error term for the simulation.

    Args:
        config: Noise configuration.
        demand: True demand signal (used for SNR and heteroscedasticity).
        rng: numpy Generator.

    Returns:
        Noise array of shape matching demand.
    """
    n = len(demand)
    scale = _compute_scale(config, demand)

    # Base noise
    if config.noise_type == "gaussian":
        noise = rng.normal(0, scale, size=n)
    elif config.noise_type == "t_distributed":
        noise = rng.standard_t(config.t_df, size=n) * scale
    elif config.noise_type == "heteroscedastic":
        # Variance scales with demand level
        demand_abs = np.abs(demand) + 1e-6  # avoid zero
        demand_normalized = demand_abs / demand_abs.mean()
        local_scale = scale * demand_normalized ** config.heteroscedasticity_power
        noise = rng.normal(0, 1, size=n) * local_scale
    elif config.noise_type == "autocorrelated":
        noise = _generate_ar1_noise(rng, n, scale, config.autocorrelation)
    else:
        noise = rng.normal(0, scale, size=n)

    # Add outliers
    if config.outlier_probability > 0:
        outlier_mask = rng.random(n) < config.outlier_probability
        outlier_values = rng.normal(0, scale * config.outlier_magnitude, size=n)
        noise = np.where(outlier_mask, outlier_values, noise)

    return noise


def _compute_scale(config: NoiseConfig, demand: np.ndarray) -> float:
    """Compute noise scale, potentially from SNR."""
    if config.signal_to_noise_ratio is not None:
        demand_std = np.std(demand)
        if demand_std > 0:
            return demand_std / config.signal_to_noise_ratio
        return config.noise_scale
    return config.noise_scale


def _generate_ar1_noise(rng: Generator, n: int, scale: float,
                        rho: float) -> np.ndarray:
    """Generate AR(1) autocorrelated noise."""
    innovations = rng.normal(0, scale * np.sqrt(1 - rho ** 2), size=n)
    noise = np.zeros(n)
    noise[0] = innovations[0]
    for t in range(1, n):
        noise[t] = rho * noise[t - 1] + innovations[t]
    return noise
