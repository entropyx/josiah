import numpy as np


def linear_trend(n, slope):
    """Generate a linear trend array.

    Args:
        n: Number of periods.
        slope: Slope per period.

    Returns:
        Array of length n.
    """
    return slope * np.arange(n)


def cube_root_trend(n, max_val, offset=1.0):
    """Generate a cube-root trend (concave growth).

    f(t) = (linspace(0, max_val, n) + offset)^(1/3) - offset^(1/3)

    Args:
        n: Number of periods.
        max_val: Maximum value of the linspace input.
        offset: Offset to avoid zero at origin.

    Returns:
        Array of length n.
    """
    t = np.linspace(0, max_val, n)
    return np.cbrt(t + offset) - np.cbrt(offset)
