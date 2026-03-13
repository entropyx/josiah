"""Time series utilities: Fourier seasonality, trend, structural breaks."""

import numpy as np


def fourier_seasonality(n_periods: int, period: float, n_terms: int,
                        coefficients: list[float] | None = None,
                        rng=None) -> np.ndarray:
    """Generate Fourier-based seasonality.

    Args:
        n_periods: Number of time periods.
        period: Length of one seasonal cycle (e.g., 52 for weekly data with yearly seasonality).
        n_terms: Number of Fourier terms (pairs of sin/cos).
        coefficients: 2*n_terms coefficients [a1, b1, a2, b2, ...]. If None, random.
        rng: numpy Generator for random coefficients.

    Returns:
        Seasonality array of shape (n_periods,).
    """
    t = np.arange(n_periods)
    if coefficients is None and rng is not None:
        coefficients = rng.uniform(-1, 1, size=2 * n_terms).tolist()
    elif coefficients is None:
        raise ValueError("Must provide coefficients or rng")

    result = np.zeros(n_periods)
    for i in range(n_terms):
        freq = 2 * np.pi * (i + 1) / period
        result += coefficients[2 * i] * np.sin(freq * t)
        result += coefficients[2 * i + 1] * np.cos(freq * t)
    return result


def linear_trend(n_periods: int, slope: float, intercept: float = 0.0) -> np.ndarray:
    """Generate linear trend: intercept + slope * t / n_periods."""
    t = np.arange(n_periods, dtype=float)
    return intercept + slope * t / n_periods


def cube_root_trend(n_periods: int, scale: float, intercept: float = 0.0) -> np.ndarray:
    """Generate cube-root trend: intercept + scale * (t/n)^(1/3)."""
    t = np.arange(n_periods, dtype=float)
    return intercept + scale * np.cbrt(t / n_periods)


def apply_structural_break(series: np.ndarray, break_period: int,
                           magnitude: float, break_type: str = "level_shift",
                           recovery: str = "permanent",
                           recovery_periods: int = 0) -> np.ndarray:
    """Apply a structural break to a time series.

    Args:
        series: Input time series.
        break_period: Period at which break occurs.
        magnitude: Size of the shift.
        break_type: 'level_shift', 'trend_break', or 'variance_change'.
        recovery: 'permanent', 'gradual_recovery', or 'v_shaped'.
        recovery_periods: Periods to recover (if applicable).

    Returns:
        Modified time series.
    """
    result = series.copy()
    n = len(series)

    if break_type == "level_shift":
        shift = np.zeros(n)
        shift[break_period:] = magnitude

        if recovery == "gradual_recovery" and recovery_periods > 0:
            for t in range(break_period, min(break_period + recovery_periods, n)):
                progress = (t - break_period) / recovery_periods
                shift[t] = magnitude * (1 - progress)
            shift[break_period + recovery_periods:] = 0
        elif recovery == "v_shaped" and recovery_periods > 0:
            half = recovery_periods // 2
            for t in range(break_period, min(break_period + half, n)):
                progress = (t - break_period) / half
                shift[t] = magnitude * (1 - progress)
            for t in range(break_period + half, min(break_period + recovery_periods, n)):
                progress = (t - break_period - half) / half
                shift[t] = magnitude * progress
            shift[break_period + recovery_periods:] = magnitude

        result += shift

    elif break_type == "trend_break":
        for t in range(break_period, n):
            result[t] += magnitude * (t - break_period) / n

    return result
