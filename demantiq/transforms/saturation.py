"""Saturation functions for the Demantiq simulator.

All functions accept numpy arrays and return arrays of the same shape.
Values are typically in [0, 1] range after saturation.
"""

import numpy as np


def hill(x: np.ndarray, K: float = 0.5, S: float = 2.0) -> np.ndarray:
    """Hill saturation function.

    f(x) = x^S / (K^S + x^S)

    Properties: f(0)=0, f(K)=0.5, f(inf)->1, monotonically increasing.

    Args:
        x: Input array (normalized spend).
        K: Half-saturation point (spend level at which response = 50% of max).
        S: Shape/steepness parameter.
    """
    x = np.asarray(x, dtype=float)
    return np.where(x <= 0, 0.0, x ** S / (K ** S + x ** S))


def hill_inverse(y: np.ndarray, K: float = 0.5, S: float = 2.0) -> np.ndarray:
    """Inverse of Hill saturation: x = K * (y / (1 - y))^(1/S)."""
    y = np.asarray(y, dtype=float)
    y = np.clip(y, 1e-10, 1 - 1e-10)
    return K * (y / (1 - y)) ** (1 / S)


def logistic(x: np.ndarray, k: float = 3.0, x0: float = 0.5) -> np.ndarray:
    """Logistic saturation function.

    f(x) = 1 / (1 + exp(-k * (x - x0)))

    Args:
        x: Input array.
        k: Steepness parameter.
        x0: Midpoint (value at which f(x) = 0.5).
    """
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def logistic_inverse(y: np.ndarray, k: float = 3.0, x0: float = 0.5) -> np.ndarray:
    """Inverse of logistic saturation: x = x0 - ln((1-y)/y) / k."""
    y = np.asarray(y, dtype=float)
    y = np.clip(y, 1e-10, 1 - 1e-10)
    return x0 - np.log((1 - y) / y) / k


def power(x: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Power saturation function.

    f(x) = x^alpha where 0 < alpha < 1 gives diminishing returns.

    Args:
        x: Input array (must be non-negative).
        alpha: Power parameter (0 < alpha < 1 for diminishing returns).
    """
    x = np.asarray(x, dtype=float)
    return np.where(x <= 0, 0.0, x ** alpha)


def power_inverse(y: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Inverse of power saturation: x = y^(1/alpha)."""
    y = np.asarray(y, dtype=float)
    return np.where(y <= 0, 0.0, y ** (1 / alpha))


def piecewise_linear(x: np.ndarray, breakpoints: list[float] = None,
                     slopes: list[float] = None) -> np.ndarray:
    """Piecewise linear saturation function.

    Linear segments with decreasing slopes at breakpoints.

    Args:
        x: Input array.
        breakpoints: x-values where slope changes. Must be sorted ascending.
        slopes: Slope for each segment. len(slopes) == len(breakpoints) + 1.
    """
    if breakpoints is None:
        breakpoints = [0.3, 0.7]
    if slopes is None:
        slopes = [1.5, 0.8, 0.2]

    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)

    for i, xi in enumerate(x.flat):
        y = 0.0
        prev_bp = 0.0
        for j, bp in enumerate(breakpoints):
            if xi <= bp:
                y += slopes[j] * (xi - prev_bp)
                break
            else:
                y += slopes[j] * (bp - prev_bp)
                prev_bp = bp
        else:
            # Past all breakpoints
            y += slopes[-1] * (xi - breakpoints[-1])
        result.flat[i] = y

    return result


# Registry mapping names to functions
SATURATION_FNS = {
    "hill": hill,
    "logistic": logistic,
    "power": power,
    "piecewise_linear": piecewise_linear,
}

SATURATION_INVERSE_FNS = {
    "hill": hill_inverse,
    "logistic": logistic_inverse,
    "power": power_inverse,
}


def get_saturation_fn(name: str):
    """Get saturation function by name."""
    if name not in SATURATION_FNS:
        raise ValueError(f"Unknown saturation function: {name}. Available: {list(SATURATION_FNS.keys())}")
    return SATURATION_FNS[name]
