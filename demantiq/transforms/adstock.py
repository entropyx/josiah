"""Adstock/carryover functions for the Demantiq simulator.

All functions accept a 1D numpy array and return an array of the same length.
"""

import numpy as np


def geometric(x: np.ndarray, alpha: float = 0.5, max_lag: int = 8,
              normalize: bool = True) -> np.ndarray:
    """Geometric adstock (exponential decay).

    weights[t] = alpha^t for t in [0, max_lag).
    Result is convolution of x with these weights.

    Args:
        x: Input array (e.g., spend).
        alpha: Retention rate (0 < alpha < 1).
        max_lag: Maximum lag in periods.
        normalize: If True, normalize weights to sum to 1.
    """
    x = np.asarray(x, dtype=float)
    weights = np.array([alpha ** i for i in range(max_lag)])
    if normalize:
        weights = weights / weights.sum()
    return np.convolve(x, weights, mode='full')[:len(x)]


def weibull_cdf(x: np.ndarray, shape: float = 2.0, scale: float = 3.0,
                max_lag: int = 12, normalize: bool = True) -> np.ndarray:
    """Weibull CDF adstock.

    w(t) = 1 - exp(-(t/scale)^shape)
    Weights are the differences of the CDF values.

    k > 1 gives delayed peak then decay (useful for TV brand effects).
    k = 1 equivalent to geometric decay.
    k < 1 peak at t=0, faster-than-exponential decay.

    Args:
        x: Input array.
        shape: Weibull shape parameter (k).
        scale: Weibull scale parameter (lambda).
        max_lag: Maximum lag.
        normalize: If True, normalize weights to sum to 1.
    """
    x = np.asarray(x, dtype=float)
    t = np.arange(max_lag, dtype=float)
    cdf_vals = 1 - np.exp(-((t + 1) / scale) ** shape)
    cdf_prev = np.concatenate([[0], cdf_vals[:-1]])
    weights = cdf_vals - cdf_prev
    weights = np.maximum(weights, 0)
    if normalize and weights.sum() > 0:
        weights = weights / weights.sum()
    return np.convolve(x, weights, mode='full')[:len(x)]


def weibull_pdf(x: np.ndarray, shape: float = 2.0, scale: float = 3.0,
                max_lag: int = 12, normalize: bool = True) -> np.ndarray:
    """Weibull PDF adstock.

    w(t) = (shape/scale) * (t/scale)^(shape-1) * exp(-(t/scale)^shape)

    Alternative parameterization with sharper peak than CDF version.

    Args:
        x: Input array.
        shape: Weibull shape parameter.
        scale: Weibull scale parameter.
        max_lag: Maximum lag.
        normalize: If True, normalize weights to sum to 1.
    """
    x = np.asarray(x, dtype=float)
    t = np.arange(1, max_lag + 1, dtype=float)  # Start at 1 to avoid division by zero
    weights = (shape / scale) * (t / scale) ** (shape - 1) * np.exp(-(t / scale) ** shape)
    weights = np.maximum(weights, 0)
    if normalize and weights.sum() > 0:
        weights = weights / weights.sum()
    return np.convolve(x, weights, mode='full')[:len(x)]


def delayed_geometric(x: np.ndarray, alpha: float = 0.5, delay: int = 2,
                      max_lag: int = 12, normalize: bool = True) -> np.ndarray:
    """Delayed geometric adstock.

    Zero effect for `delay` periods, then geometric decay.

    Args:
        x: Input array.
        alpha: Retention rate after delay.
        delay: Number of zero-effect periods.
        max_lag: Maximum lag (includes delay).
        normalize: If True, normalize weights to sum to 1.
    """
    x = np.asarray(x, dtype=float)
    weights = np.zeros(max_lag)
    for i in range(delay, max_lag):
        weights[i] = alpha ** (i - delay)
    if normalize and weights.sum() > 0:
        weights = weights / weights.sum()
    return np.convolve(x, weights, mode='full')[:len(x)]


# Registry mapping names to functions
ADSTOCK_FNS = {
    "geometric": geometric,
    "weibull_cdf": weibull_cdf,
    "weibull_pdf": weibull_pdf,
    "delayed_geometric": delayed_geometric,
}


def get_adstock_fn(name: str):
    """Get adstock function by name."""
    if name not in ADSTOCK_FNS:
        raise ValueError(f"Unknown adstock function: {name}. Available: {list(ADSTOCK_FNS.keys())}")
    return ADSTOCK_FNS[name]
