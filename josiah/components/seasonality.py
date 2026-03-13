import numpy as np
import pandas as pd


def fourier_seasonality(dates, n_terms, coefficients=None, period=365.25):
    """Fourier-based seasonality using sin/cos pairs.

    For each term k in [1, n_terms]:
        sin(2*pi*k*t/period) * coeff[2*(k-1)]
        cos(2*pi*k*t/period) * coeff[2*(k-1)+1]

    Args:
        dates: DatetimeIndex or array of dates.
        n_terms: Number of Fourier pairs.
        coefficients: List of 2*n_terms floats. If None, random coefficients are generated.
        period: Period in days (default 365.25 for annual).

    Returns:
        Seasonality array of same length as dates.
    """
    dates = pd.DatetimeIndex(dates)
    t = (dates - dates[0]).days.values.astype(float)

    if coefficients is None:
        coefficients = np.random.uniform(-0.1, 0.1, size=2 * n_terms)
    coefficients = np.asarray(coefficients)

    result = np.zeros(len(t))
    for k in range(1, n_terms + 1):
        idx = 2 * (k - 1)
        result += coefficients[idx] * np.sin(2 * np.pi * k * t / period)
        result += coefficients[idx + 1] * np.cos(2 * np.pi * k * t / period)

    # Zero-center so seasonality doesn't shift the mean
    result -= result.mean()
    return result


def sine_seasonality(dates, annual=0.15, semiannual=0.08, quarterly=0.05):
    """Legacy sine-wave seasonality decomposition.

    Args:
        dates: DatetimeIndex or array of dates.
        annual: Amplitude of annual cycle.
        semiannual: Amplitude of semi-annual cycle.
        quarterly: Amplitude of quarterly cycle.

    Returns:
        Seasonality multiplier array (zero-centered).
    """
    dates = pd.DatetimeIndex(dates)
    days = (dates - dates[0]).days.values.astype(float)

    result = (
        annual * np.sin(2 * np.pi * days / 365)
        + semiannual * np.sin(4 * np.pi * days / 365)
        + quarterly * np.sin(8 * np.pi * days / 365)
    )
    result -= result.mean()
    return result
