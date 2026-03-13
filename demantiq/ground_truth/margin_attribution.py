"""True margin decomposition."""

import numpy as np


def compute_margin(revenue: np.ndarray, outcome: np.ndarray,
                   cogs_per_unit: float, variable_cost_per_unit: float,
                   media_cost: np.ndarray) -> dict:
    """Compute true margin decomposition.

    margin = revenue - (outcome * cogs) - (outcome * variable_cost) - media_cost
    """
    total_cost = outcome * (cogs_per_unit + variable_cost_per_unit)
    margin = revenue - total_cost - media_cost

    return {
        "revenue": revenue,
        "total_cost": total_cost,
        "media_cost": media_cost,
        "margin": margin,
        "total_revenue": float(np.sum(revenue)),
        "total_margin": float(np.sum(margin)),
        "total_media_cost": float(np.sum(media_cost)),
    }
