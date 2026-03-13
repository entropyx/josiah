import numpy as np
from demantiq.config.distribution_config import DistributionConfig
from demantiq.generators.distribution_generator import generate_distribution
from demantiq.utils.random import create_rng


def test_stable_distribution():
    config = DistributionConfig(initial_distribution=0.8, distribution_trajectory="stable")
    result = generate_distribution(config, 104, create_rng(42))
    np.testing.assert_allclose(result.distribution, 0.8)


def test_growing_distribution():
    config = DistributionConfig(initial_distribution=0.3, distribution_trajectory="growing",
                                 trajectory_params={"growth_rate": 0.05})
    result = generate_distribution(config, 104, create_rng(42))
    assert result.distribution[-1] > result.distribution[0]


def test_distribution_range():
    config = DistributionConfig(initial_distribution=0.5, distribution_trajectory="growing")
    result = generate_distribution(config, 104, create_rng(42))
    assert np.all(result.distribution >= 0)
    assert np.all(result.distribution <= 1)


def test_no_ceiling_effect():
    config = DistributionConfig(distribution_ceiling_effect=0.0)
    result = generate_distribution(config, 104, create_rng(42))
    np.testing.assert_allclose(result.distribution_cap, 1.0)


def test_stockout():
    config = DistributionConfig(stockout_probability=0.5, stockout_demand_loss=0.8)
    result = generate_distribution(config, 1000, create_rng(42))
    assert np.any(result.stockout_mask)
    # Cap should be lower during stockouts
    assert np.mean(result.distribution_cap[result.stockout_mask]) < np.mean(result.distribution_cap[~result.stockout_mask])


def test_step_change():
    config = DistributionConfig(initial_distribution=0.5, distribution_trajectory="step_change",
                                 trajectory_params={"step_period": 50, "step_magnitude": 0.3})
    result = generate_distribution(config, 104, create_rng(42))
    assert np.mean(result.distribution[50:]) > np.mean(result.distribution[:50])
