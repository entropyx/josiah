import numpy as np
import pandas as pd
from demantiq.config import SimulationConfig, ChannelConfig, BaselineConfig, NoiseConfig
from demantiq.core.demand_kernel import simulate


def _simple_config():
    """Create a simple config for testing."""
    return SimulationConfig(
        n_periods=104,
        channels=[
            ChannelConfig(name="tv", beta=500.0, saturation_fn="hill",
                          saturation_params={"K": 0.5, "S": 2.0},
                          adstock_fn="geometric", adstock_params={"alpha": 0.5, "max_lag": 8},
                          spend_mean=10000, spend_std=3000),
            ChannelConfig(name="search", beta=300.0, saturation_fn="hill",
                          saturation_params={"K": 0.3, "S": 3.0},
                          adstock_fn="geometric", adstock_params={"alpha": 0.2, "max_lag": 4},
                          spend_mean=5000, spend_std=1500),
        ],
        noise=NoiseConfig(noise_type="gaussian", noise_scale=50.0),
        baseline=BaselineConfig(organic_level=1000.0, trend_type="linear",
                                trend_params={"slope": 10.0},
                                seasonality_n_terms=2,
                                seasonality_coefficients=[50, 30, 20, 10]),
        seed=42,
    )


def test_simulate_returns_result():
    config = _simple_config()
    result = simulate(config)
    assert hasattr(result, 'observable_data')
    assert hasattr(result, 'ground_truth')
    assert hasattr(result, 'summary_truth')
    assert hasattr(result, 'config')


def test_observable_shape():
    config = _simple_config()
    result = simulate(config)
    assert result.observable_data.shape[0] == 104
    assert "date" in result.observable_data.columns
    assert "y" in result.observable_data.columns
    assert "tv_spend" in result.observable_data.columns
    assert "search_spend" in result.observable_data.columns


def test_ground_truth_shape():
    config = _simple_config()
    result = simulate(config)
    gt = result.ground_truth
    assert gt.shape[0] == 104
    assert "true_baseline" in gt.columns
    assert "true_tv_contribution" in gt.columns
    assert "true_search_contribution" in gt.columns
    assert "true_noise" in gt.columns


def test_contributions_sum():
    """Contributions + baseline + noise should sum to y."""
    config = _simple_config()
    result = simulate(config)
    gt = result.ground_truth

    reconstructed = gt["true_baseline"].values.copy()
    for ch in config.channels:
        reconstructed += gt[f"true_{ch.name}_contribution"].values
    reconstructed += gt["true_noise"].values

    np.testing.assert_allclose(reconstructed, gt["y"].values, rtol=1e-10)


def test_positive_roas():
    """ROAS should be positive for positive beta channels."""
    config = _simple_config()
    result = simulate(config)
    for ch in config.channels:
        assert result.summary_truth["true_roas"][ch.name] > 0


def test_no_nan():
    config = _simple_config()
    result = simulate(config)
    assert not result.observable_data.isnull().any().any()
    assert not result.ground_truth.isnull().any().any()


def test_deterministic():
    config = _simple_config()
    r1 = simulate(config)
    r2 = simulate(config)
    np.testing.assert_allclose(
        r1.observable_data["y"].values,
        r2.observable_data["y"].values
    )


def test_summary_truth_fields():
    config = _simple_config()
    result = simulate(config)
    s = result.summary_truth
    assert "true_betas" in s
    assert "true_roas" in s
    assert "true_total_media_contribution_pct" in s
    assert "channel_scales" in s
    assert "seed" in s
    assert s["seed"] == 42


def test_no_channels():
    """Should work with zero channels (baseline only)."""
    config = SimulationConfig(
        n_periods=52,
        channels=[],
        baseline=BaselineConfig(organic_level=1000.0),
        noise=NoiseConfig(noise_scale=10.0),
        seed=42,
    )
    result = simulate(config)
    assert result.observable_data.shape[0] == 52
    assert "y" in result.observable_data.columns


def test_simulator_class():
    """Test the public Simulator wrapper."""
    from demantiq import Simulator, SimulationConfig, ChannelConfig
    config = SimulationConfig(
        n_periods=52,
        channels=[ChannelConfig(name="tv", beta=100.0)],
        seed=123,
    )
    sim = Simulator(config)
    result = sim.run()
    assert result.observable_data.shape[0] == 52
