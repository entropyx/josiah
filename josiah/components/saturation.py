import numpy as np


def logistic_saturation(x, lam):
    """PyMC Marketing-compatible logistic saturation.

    f(x) = (1 - exp(-lam * x)) / (1 + exp(-lam * x))

    Args:
        x: Input array (already adstocked).
        lam: Saturation parameter (higher = saturates faster).

    Returns:
        Saturated array with values in (0, 1).
    """
    x = np.asarray(x, dtype=float)
    return (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x))


def hill_saturation(x, n, K):
    """Legacy Hill saturation function.

    f(x) = x^n / (K^n + x^n)

    Args:
        x: Input array.
        n: Hill coefficient (steepness).
        K: Half-saturation point.

    Returns:
        Saturated array with values in (0, 1).
    """
    x = np.asarray(x, dtype=float)
    return x ** n / (K ** n + x ** n)
