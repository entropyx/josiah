"""Custom probability distributions for spend generation."""

import numpy as np
from numpy.random import Generator


def sample_lognormal(rng: Generator, mean: float, std: float, size: int) -> np.ndarray:
    """Sample from log-normal distribution parameterized by desired mean and std.

    Converts desired mean/std to log-normal mu/sigma parameters.
    """
    variance = std ** 2
    mu = np.log(mean ** 2 / np.sqrt(variance + mean ** 2))
    sigma = np.sqrt(np.log(1 + variance / mean ** 2))
    return rng.lognormal(mu, sigma, size=size)


def sample_gamma(rng: Generator, mean: float, std: float, size: int) -> np.ndarray:
    """Sample from gamma distribution parameterized by desired mean and std."""
    variance = std ** 2
    shape = mean ** 2 / variance
    scale = variance / mean
    return rng.gamma(shape, scale, size=size)


def sample_truncated_normal(rng: Generator, mean: float, std: float,
                            low: float, high: float, size: int) -> np.ndarray:
    """Sample from truncated normal distribution."""
    samples = rng.normal(mean, std, size=size * 3)  # oversample
    samples = samples[(samples >= low) & (samples <= high)]
    while len(samples) < size:
        extra = rng.normal(mean, std, size=size)
        extra = extra[(extra >= low) & (extra <= high)]
        samples = np.concatenate([samples, extra])
    return samples[:size]
