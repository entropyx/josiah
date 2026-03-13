"""Test demand kernel with pricing and distribution (M2 features)."""

import numpy as np
from demantiq import Simulator, SimulationConfig, ChannelConfig, BaselineConfig, NoiseConfig
from demantiq.config.pricing_config import PricingConfig, CostConfig
from demantiq.config.distribution_config import DistributionConfig


def _m2_config():
    return SimulationConfig(
        n_periods=104,
        channels=[
            ChannelConfig(name="tv", beta=500.0, spend_mean=10000, spend_std=3000),
            ChannelConfig(name="search", beta=300.0, spend_mean=5000, spend_std=1500),
        ],
        pricing=PricingConfig(
            base_price=25.0, price_elasticity=-1.5,
            promo_depth_mean=0.15, promo_frequency="monthly",
            cost_structure=CostConfig(cogs_per_unit=5.0, variable_cost_per_unit=2.0),
        ),
        distribution=DistributionConfig(
            initial_distribution=0.8, distribution_trajectory="stable",
            distribution_ceiling_effect=0.3,
        ),
        noise=NoiseConfig(noise_scale=30.0),
        baseline=BaselineConfig(organic_level=1000.0),
        seed=42,
    )


def test_m2_runs():
    result = Simulator(_m2_config()).run()
    assert result.observable_data.shape[0] == 104


def test_m2_has_pricing_columns():
    result = Simulator(_m2_config()).run()
    obs = result.observable_data
    assert "price" in obs.columns
    assert "is_promo" in obs.columns


def test_m2_has_distribution_columns():
    result = Simulator(_m2_config()).run()
    obs = result.observable_data
    assert "distribution" in obs.columns


def test_m2_has_revenue():
    result = Simulator(_m2_config()).run()
    obs = result.observable_data
    assert "revenue" in obs.columns
    assert np.all(obs["revenue"].values > 0)


def test_m2_ground_truth_has_price_effect():
    result = Simulator(_m2_config()).run()
    gt = result.ground_truth
    assert "true_price_effect" in gt.columns


def test_m2_no_nan():
    result = Simulator(_m2_config()).run()
    assert not result.observable_data.isnull().any().any()
    assert not result.ground_truth.isnull().any().any()


def test_m2_backward_compatible():
    """Config without pricing/distribution should still work (M1 behavior)."""
    config = SimulationConfig(
        n_periods=52,
        channels=[ChannelConfig(name="tv", beta=100.0)],
        seed=42,
    )
    result = Simulator(config).run()
    assert result.observable_data.shape[0] == 52
    assert "price" not in result.observable_data.columns
