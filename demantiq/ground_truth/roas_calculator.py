"""True ROAS computation."""

import numpy as np


def compute_roas(contributions: dict[str, np.ndarray],
                 spend: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute true ROAS per channel.

    ROAS = total_contribution / total_spend

    Args:
        contributions: Dict mapping channel name to contribution array.
        spend: Dict mapping channel name to spend array.

    Returns:
        Dict mapping channel name to ROAS value.
    """
    result = {}
    for name in contributions:
        total_contrib = float(np.sum(contributions[name]))
        total_spend = float(np.sum(spend.get(name, np.array([0]))))
        result[name] = total_contrib / total_spend if total_spend > 0 else 0.0
    return result
