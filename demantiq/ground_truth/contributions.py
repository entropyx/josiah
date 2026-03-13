"""True channel contribution computation."""

import numpy as np


def compute_contributions(channel_effects: dict[str, np.ndarray],
                         total_demand: np.ndarray) -> dict:
    """Compute true per-channel contributions.

    Args:
        channel_effects: Dict mapping channel name to contribution array.
        total_demand: Total demand (y) array.

    Returns:
        Dict with 'absolute' and 'percentage' contributions per channel.
    """
    result = {}
    for name, effect in channel_effects.items():
        total_contrib = float(np.sum(effect))
        total_y = float(np.sum(total_demand))
        result[name] = {
            "total_contribution": total_contrib,
            "contribution_pct": total_contrib / total_y if total_y != 0 else 0.0,
            "per_period": effect,
        }
    return result
