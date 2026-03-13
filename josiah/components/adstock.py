import numpy as np


def geometric_adstock(x, alpha, l_max, normalize=True):
    """PyMC Marketing-compatible geometric adstock transformation.

    weights[t] = alpha^t for t in [0, l_max)
    Result is the convolution of x with these weights.

    Args:
        x: Input array (e.g. spend).
        alpha: Retention rate (0 < alpha < 1).
        l_max: Maximum lag (number of periods).
        normalize: If True, normalize weights to sum to 1.

    Returns:
        Transformed array of same length as x.
    """
    x = np.asarray(x, dtype=float)
    weights = np.array([alpha ** i for i in range(l_max)])
    if normalize:
        weights = weights / weights.sum()
    result = np.convolve(x, weights, mode='full')[:len(x)]
    return result


def exponential_adstock(x, half_life):
    """Legacy exponential decay adstock.

    weights[t] = exp(-ln2 / half_life * t)

    Args:
        x: Input array.
        half_life: Half-life in periods.

    Returns:
        Transformed array of same length as x.
    """
    x = np.asarray(x, dtype=float)
    decay_rate = np.log(2) / half_life
    weights = np.exp(-decay_rate * np.arange(len(x)))
    weights = weights / weights.sum()
    result = np.convolve(x, weights, mode='full')[:len(x)]
    return result
