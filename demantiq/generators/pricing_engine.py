"""Price and promotion simulation."""

import numpy as np
from numpy.random import Generator
from dataclasses import dataclass
from demantiq.config.pricing_config import PricingConfig


@dataclass
class PricingResult:
    """Result of pricing generation."""
    price: np.ndarray        # effective price per period
    is_promo: np.ndarray     # boolean array
    promo_depth: np.ndarray  # discount percentage per period
    price_effect: np.ndarray # demand effect from pricing


def generate_pricing(config: PricingConfig, n_periods: int,
                     rng: Generator) -> PricingResult:
    """Generate pricing and promotional effects.

    Price effect = price_elasticity * ln(effective_price / base_price).
    During promos, effective_price = base_price * (1 - depth).
    """
    # Generate promo schedule
    is_promo = np.zeros(n_periods, dtype=bool)
    promo_depth = np.zeros(n_periods)

    freq_map = {"weekly": 1, "biweekly": 2, "monthly": 4, "quarterly": 13}
    interval = freq_map.get(config.promo_frequency, 4)

    t = int(rng.integers(0, interval))  # random start offset
    while t < n_periods:
        depth = max(0.01, rng.normal(config.promo_depth_mean, config.promo_depth_std))
        depth = min(depth, 0.5)  # cap at 50% discount
        duration = max(1, int(rng.integers(1, 3)))
        for d in range(duration):
            if t + d < n_periods:
                is_promo[t + d] = True
                promo_depth[t + d] = depth
        t += interval + int(rng.integers(-1, 2))

    # Effective price
    price = np.full(n_periods, config.base_price)
    price[is_promo] = config.base_price * (1 - promo_depth[is_promo])

    # Price effect on demand
    log_price_ratio = np.log(price / config.base_price)
    price_effect = config.price_elasticity * log_price_ratio * config.base_price

    return PricingResult(price=price, is_promo=is_promo,
                         promo_depth=promo_depth, price_effect=price_effect)
