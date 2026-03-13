import numpy as np


def generate_controls(n, gamma_shape, gamma_scale, coefficient, seed=None):
    """Generate a gamma-distributed control variable and its revenue contribution.

    Args:
        n: Number of periods.
        gamma_shape: Shape parameter for gamma distribution.
        gamma_scale: Scale parameter for gamma distribution.
        coefficient: Linear coefficient applied to the control variable.
        seed: Random seed.

    Returns:
        Tuple of (control_values, contribution) arrays.
    """
    rng = np.random.default_rng(seed)
    values = rng.gamma(gamma_shape, gamma_scale, size=n)
    contribution = coefficient * values
    return values, contribution
