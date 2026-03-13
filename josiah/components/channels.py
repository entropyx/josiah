import numpy as np
from .adstock import geometric_adstock
from .saturation import logistic_saturation


def generate_spend(n, mean, std, seed=None):
    """Generate spend array using a gamma distribution (always positive).

    The gamma distribution is parameterized via mean and std:
        shape = (mean/std)^2
        scale = std^2 / mean

    Args:
        n: Number of periods.
        mean: Mean daily/weekly spend.
        std: Standard deviation of spend.
        seed: Random seed.

    Returns:
        Array of length n with positive spend values.
    """
    rng = np.random.default_rng(seed)
    if std <= 0 or mean <= 0:
        return np.full(n, max(mean, 0))
    shape = (mean / std) ** 2
    scale = (std ** 2) / mean
    return rng.gamma(shape, scale, size=n)


def channel_effect(spend, alpha, l_max, lam, beta):
    """Compute channel contribution: beta * logistic(geometric_adstock(normalized_spend)).

    Spend is normalized by max(abs(spend)) before saturation so that lambda
    operates on a [0, 1] scale, matching PyMC Marketing's default scaler
    (MaxAbsScaler).

    Args:
        spend: Spend array (raw, unnormalized).
        alpha: Adstock retention rate.
        l_max: Maximum adstock lag.
        lam: Logistic saturation parameter (operates on normalized spend).
        beta: Channel coefficient (scales the saturated adstocked spend).

    Returns:
        Tuple of (channel contribution array, spend_scale) where spend_scale
        is max(abs(spend)) used for normalization.
    """
    spend = np.asarray(spend, dtype=float)
    # Normalize spend by max(abs) so lambda works on [0, 1] scale (matches PyMC Marketing)
    spend_scale = np.max(np.abs(spend))
    if spend_scale > 0:
        normalized = spend / spend_scale
    else:
        normalized = spend
        spend_scale = 1.0
    adstocked = geometric_adstock(normalized, alpha, l_max)
    saturated = logistic_saturation(adstocked, lam)
    return beta * saturated, spend_scale
