import numpy as np
from demantiq.config.endogeneity_config import EndogeneityConfig
from demantiq.core.endogeneity_layer import apply_endogeneity
from demantiq.utils.random import create_rng


def test_zero_strength_no_change():
    config = EndogeneityConfig(overall_strength=0.0)
    spend = {"tv": np.ones(100) * 1000, "search": np.ones(100) * 500}
    baseline = np.ones(100) * 5000
    result = apply_endogeneity(spend, baseline, config, ["tv", "search"], create_rng(42))
    np.testing.assert_allclose(result.spend_endogenous["tv"], spend["tv"])


def test_seasonal_bias_increases_spend():
    config = EndogeneityConfig(overall_strength=0.5, seasonal_allocation_bias=0.5)
    spend = {"tv": np.ones(100) * 1000}
    baseline = np.concatenate([np.ones(50) * 1000, np.ones(50) * 5000])
    result = apply_endogeneity(spend, baseline, config, ["tv"], create_rng(42))
    # Spend should be higher during high-baseline periods
    assert np.mean(result.spend_endogenous["tv"][50:]) > np.mean(
        result.spend_endogenous["tv"][:50]
    )


def test_confounder_generated():
    config = EndogeneityConfig(overall_strength=0.5, omitted_variable_strength=0.5)
    spend = {"tv": np.ones(100) * 1000}
    baseline = np.ones(100) * 5000
    result = apply_endogeneity(spend, baseline, config, ["tv"], create_rng(42))
    assert result.confounders is not None
    assert not np.allclose(result.confounders, 0)


def test_exogenous_spend_preserved():
    config = EndogeneityConfig(overall_strength=0.5, seasonal_allocation_bias=0.3)
    spend = {"tv": np.ones(100) * 1000}
    baseline = np.ones(100) * 5000
    result = apply_endogeneity(spend, baseline, config, ["tv"], create_rng(42))
    np.testing.assert_allclose(result.exogenous_spend["tv"], 1000.0)


def test_endogeneity_bias_tracked():
    config = EndogeneityConfig(overall_strength=0.5, seasonal_allocation_bias=0.5)
    baseline = np.concatenate([np.ones(50) * 1000, np.ones(50) * 5000])
    spend = {"tv": np.ones(100) * 1000}
    result = apply_endogeneity(spend, baseline, config, ["tv"], create_rng(42))
    bias = result.endogeneity_bias["tv"]
    assert not np.allclose(bias, 0)
