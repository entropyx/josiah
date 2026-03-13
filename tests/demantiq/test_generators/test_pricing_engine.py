import numpy as np
from demantiq.config.pricing_config import PricingConfig
from demantiq.generators.pricing_engine import generate_pricing
from demantiq.utils.random import create_rng


def test_pricing_shape():
    config = PricingConfig()
    result = generate_pricing(config, 104, create_rng(42))
    assert result.price.shape == (104,)
    assert result.is_promo.shape == (104,)
    assert result.price_effect.shape == (104,)


def test_promo_reduces_price():
    config = PricingConfig(base_price=25.0, promo_depth_mean=0.2)
    result = generate_pricing(config, 104, create_rng(42))
    promo_prices = result.price[result.is_promo]
    if len(promo_prices) > 0:
        assert np.all(promo_prices < 25.0)


def test_negative_elasticity_positive_promo_effect():
    """Negative elasticity + lower price during promo = positive demand effect."""
    config = PricingConfig(base_price=25.0, price_elasticity=-1.5, promo_depth_mean=0.2)
    result = generate_pricing(config, 104, create_rng(42))
    promo_effects = result.price_effect[result.is_promo]
    if len(promo_effects) > 0:
        assert np.all(promo_effects > 0)  # promos boost demand


def test_no_promo_no_effect():
    """Non-promo periods should have zero price effect."""
    config = PricingConfig(base_price=25.0)
    result = generate_pricing(config, 104, create_rng(42))
    non_promo_effects = result.price_effect[~result.is_promo]
    np.testing.assert_allclose(non_promo_effects, 0.0, atol=1e-10)


def test_deterministic():
    config = PricingConfig()
    r1 = generate_pricing(config, 104, create_rng(42))
    r2 = generate_pricing(config, 104, create_rng(42))
    np.testing.assert_allclose(r1.price, r2.price)
