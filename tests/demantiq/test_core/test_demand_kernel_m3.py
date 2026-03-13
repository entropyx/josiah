"""Test demand kernel with M3 features (endogeneity, competition, macro)."""

import numpy as np
from demantiq import Simulator, SimulationConfig, ChannelConfig, BaselineConfig, NoiseConfig
from demantiq.config.endogeneity_config import EndogeneityConfig
from demantiq.config.competition_config import CompetitionConfig
from demantiq.config.macro_config import MacroConfig, MacroVariable, RegimeChange


def _m3_config():
    return SimulationConfig(
        n_periods=104,
        channels=[
            ChannelConfig(name="tv", beta=500.0, spend_mean=10000, spend_std=3000),
            ChannelConfig(name="search", beta=300.0, spend_mean=5000, spend_std=1500),
        ],
        endogeneity=EndogeneityConfig(
            overall_strength=0.3,
            seasonal_allocation_bias=0.2,
            omitted_variable_strength=0.3,
        ),
        competition=CompetitionConfig(
            competitor_sov_mean=0.3,
            sov_suppression_coefficient=0.1,
        ),
        macro=MacroConfig(
            variables=[
                MacroVariable(name="gdp", effect_on_demand=50.0,
                              time_series_type="mean_reverting",
                              params={"mean": 0.0, "phi": 0.8, "sigma": 0.1}),
            ],
            regime_changes=[
                RegimeChange(period=60, change_type="level_shift", magnitude=-0.1),
            ],
        ),
        noise=NoiseConfig(noise_scale=30.0),
        baseline=BaselineConfig(organic_level=5000.0, seasonality_coefficients=[100, 50, 30, 20]),
        seed=42,
    )


def test_m3_runs():
    result = Simulator(_m3_config()).run()
    assert result.observable_data.shape[0] == 104


def test_m3_has_competition_columns():
    result = Simulator(_m3_config()).run()
    assert "competitor_sov" in result.observable_data.columns


def test_m3_has_macro_columns():
    result = Simulator(_m3_config()).run()
    assert "gdp" in result.observable_data.columns


def test_m3_has_endogeneity_in_ground_truth():
    result = Simulator(_m3_config()).run()
    gt = result.ground_truth
    assert "confounders" in gt.columns
    assert "exogenous_tv_spend" in gt.columns


def test_m3_has_competition_in_ground_truth():
    result = Simulator(_m3_config()).run()
    gt = result.ground_truth
    assert "true_competition_effect" in gt.columns


def test_m3_has_macro_in_ground_truth():
    result = Simulator(_m3_config()).run()
    gt = result.ground_truth
    assert "true_macro_effect" in gt.columns
    assert "true_regime_effects" in gt.columns


def test_m3_no_nan():
    result = Simulator(_m3_config()).run()
    assert not result.observable_data.isnull().any().any()
    assert not result.ground_truth.isnull().any().any()


def test_m3_backward_compatible():
    """Config without M3 features should still work."""
    config = SimulationConfig(
        n_periods=52,
        channels=[ChannelConfig(name="tv", beta=100.0)],
        seed=42,
    )
    result = Simulator(config).run()
    assert result.observable_data.shape[0] == 52
    assert "competitor_sov" not in result.observable_data.columns


def test_m3_endogeneity_modifies_spend():
    """With endogeneity, observable spend differs from exogenous spend."""
    config = SimulationConfig(
        n_periods=104,
        channels=[ChannelConfig(name="tv", beta=500.0, spend_mean=10000, spend_std=3000)],
        endogeneity=EndogeneityConfig(
            overall_strength=0.5,
            seasonal_allocation_bias=0.5,
        ),
        baseline=BaselineConfig(
            organic_level=5000.0,
            seasonality_coefficients=[500, 300, 200, 100],
        ),
        noise=NoiseConfig(noise_scale=30.0),
        seed=42,
    )
    result = Simulator(config).run()
    # Should have ground truth info
    assert result.observable_data.shape[0] == 104
    # Endogeneity bias should be non-zero for at least some periods
    gt = result.ground_truth
    assert "endogeneity_bias_tv" in gt.columns
    bias = gt["endogeneity_bias_tv"].values
    assert np.any(bias != 0), "Endogeneity should produce non-zero bias"


def test_m3_competition_only():
    """Test with only competition config."""
    config = SimulationConfig(
        n_periods=52,
        channels=[ChannelConfig(name="tv", beta=100.0)],
        competition=CompetitionConfig(competitor_sov_mean=0.4, sov_suppression_coefficient=0.2),
        seed=42,
    )
    result = Simulator(config).run()
    assert "competitor_sov" in result.observable_data.columns
    assert "true_competition_effect" in result.ground_truth.columns
    assert not result.observable_data.isnull().any().any()


def test_m3_macro_only():
    """Test with only macro config."""
    config = SimulationConfig(
        n_periods=52,
        channels=[ChannelConfig(name="tv", beta=100.0)],
        macro=MacroConfig(
            variables=[MacroVariable(name="cpi", effect_on_demand=30.0)],
        ),
        seed=42,
    )
    result = Simulator(config).run()
    assert "cpi" in result.observable_data.columns
    assert "true_macro_effect" in result.ground_truth.columns
    assert not result.observable_data.isnull().any().any()


def test_m3_summary_truth():
    """Test summary truth includes M3 info."""
    result = Simulator(_m3_config()).run()
    summary = result.summary_truth
    assert "endogeneity" in summary
    assert "competition" in summary
    assert "macro" in summary
    assert summary["macro"]["variables"] == ["gdp"]
    assert summary["competition"]["sov_suppression_coefficient"] == 0.1


def test_m3_reproducible():
    """Same seed produces identical results."""
    r1 = Simulator(_m3_config()).run()
    r2 = Simulator(_m3_config()).run()
    np.testing.assert_array_equal(r1.observable_data["y"].values, r2.observable_data["y"].values)


def test_full_m2_m3_config():
    """Test with ALL optional configs (pricing + distribution + endogeneity + competition + macro)."""
    from demantiq.config.pricing_config import PricingConfig
    from demantiq.config.distribution_config import DistributionConfig

    config = SimulationConfig(
        n_periods=104,
        channels=[
            ChannelConfig(name="tv", beta=500.0, spend_mean=10000, spend_std=3000),
        ],
        pricing=PricingConfig(base_price=25.0, price_elasticity=-1.2),
        distribution=DistributionConfig(initial_distribution=0.8),
        endogeneity=EndogeneityConfig(overall_strength=0.2),
        competition=CompetitionConfig(competitor_sov_mean=0.3),
        macro=MacroConfig(variables=[MacroVariable(name="cpi", effect_on_demand=30.0)]),
        noise=NoiseConfig(noise_scale=30.0),
        baseline=BaselineConfig(organic_level=5000.0),
        seed=42,
    )
    result = Simulator(config).run()
    assert result.observable_data.shape[0] == 104
    assert not result.observable_data.isnull().any().any()
    assert "price" in result.observable_data.columns
    assert "competitor_sov" in result.observable_data.columns
    assert "cpi" in result.observable_data.columns
