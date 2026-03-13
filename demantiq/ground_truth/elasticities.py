"""True price elasticity computation."""

import numpy as np


def compute_price_elasticity(price: np.ndarray, demand: np.ndarray) -> float:
    """Compute empirical price elasticity from simulated data.

    elasticity = (% change in demand) / (% change in price)
    Computed as regression coefficient on log-log scale.
    """
    log_price = np.log(price)
    log_demand = np.log(np.maximum(demand, 1e-6))

    # Simple OLS on log-log
    X = np.column_stack([np.ones_like(log_price), log_price])
    beta = np.linalg.lstsq(X, log_demand, rcond=None)[0]
    return float(beta[1])
