"""Test demand kernel with M4 interactions (full 15-step pipeline test)."""

import numpy as np
from demantiq import Simulator, SimulationConfig, ChannelConfig, BaselineConfig, NoiseConfig
from demantiq.config.interaction_config import InteractionConfig
from demantiq.config.pricing_config import PricingConfig
from demantiq.config.distribution_config import DistributionConfig
from demantiq.config.competition_config import CompetitionConfig


def _m4_config():
    return SimulationConfig(
        n_periods=104,
        channels=[
            ChannelConfig(name="tv", beta=500.0, spend_mean=10000, spend_std=3000),
            ChannelConfig(name="search", beta=300.0, spend_mean=5000, spend_std=1500),
        ],
        pricing=PricingConfig(base_price=25.0, price_elasticity=-1.2),
        distribution=DistributionConfig(initial_distribution=0.8),
        competition=CompetitionConfig(competitor_sov_mean=0.3, sov_suppression_coefficient=0.1),
        interactions=InteractionConfig(
            price_x_media={"tv": 0.3},
            distribution_x_media={"search": 0.2},
            media_x_media={("tv", "search"): 0.15},
            competition_x_media={"tv": 0.25},
        ),
        noise=NoiseConfig(noise_scale=30.0),
        baseline=BaselineConfig(organic_level=5000.0, seasonality_coefficients=[100, 50, 30, 20]),
        seed=42,
    )


def test_m4_runs():
    result = Simulator(_m4_config()).run()
    assert result.observable_data.shape[0] == 104


def test_m4_no_nan():
    result = Simulator(_m4_config()).run()
    assert not result.observable_data.isnull().any().any()
    assert not result.ground_truth.isnull().any().any()


def test_m4_has_interaction_ground_truth():
    result = Simulator(_m4_config()).run()
    gt = result.ground_truth
    assert "true_interaction_price_x_tv" in gt.columns
    assert "true_interaction_distribution_x_search" in gt.columns
    assert "true_interaction_tv_x_search" in gt.columns
    assert "true_interaction_competition_x_tv" in gt.columns


def test_m4_has_interaction_summary():
    result = Simulator(_m4_config()).run()
    summary = result.summary_truth
    assert "interactions" in summary
    assert "price_x_tv" in summary["interactions"]
    assert "total_effect" in summary["interactions"]["price_x_tv"]
    assert "mean_effect" in summary["interactions"]["price_x_tv"]


def test_m4_interactions_modify_contributions():
    """With interactions, contributions should differ from no-interaction run."""
    config_with = _m4_config()
    config_without = SimulationConfig(
        n_periods=config_with.n_periods,
        channels=list(config_with.channels),
        pricing=config_with.pricing,
        distribution=config_with.distribution,
        competition=config_with.competition,
        interactions=None,  # no interactions
        noise=config_with.noise,
        baseline=config_with.baseline,
        seed=config_with.seed,
    )
    r_with = Simulator(config_with).run()
    r_without = Simulator(config_without).run()
    # TV contribution should differ due to interactions
    tv_with = r_with.ground_truth["true_tv_contribution"].values
    tv_without = r_without.ground_truth["true_tv_contribution"].values
    assert not np.allclose(tv_with, tv_without)


def test_m4_backward_compatible():
    """Config without interactions should still work."""
    config = SimulationConfig(
        n_periods=52,
        channels=[ChannelConfig(name="tv", beta=100.0)],
        seed=42,
    )
    result = Simulator(config).run()
    assert result.observable_data.shape[0] == 52
    assert "interactions" not in result.summary_truth


def test_m4_reproducible():
    """Same seed produces identical results."""
    r1 = Simulator(_m4_config()).run()
    r2 = Simulator(_m4_config()).run()
    np.testing.assert_array_equal(
        r1.observable_data["y"].values,
        r2.observable_data["y"].values,
    )


def test_m4_media_x_media_only():
    """Test with only media_x_media interaction."""
    config = SimulationConfig(
        n_periods=52,
        channels=[
            ChannelConfig(name="tv", beta=500.0, spend_mean=10000),
            ChannelConfig(name="search", beta=300.0, spend_mean=5000),
        ],
        interactions=InteractionConfig(
            media_x_media={("tv", "search"): 0.3},
        ),
        seed=42,
    )
    result = Simulator(config).run()
    assert not result.observable_data.isnull().any().any()
    assert "true_interaction_tv_x_search" in result.ground_truth.columns


def test_m4_config_serialization():
    """InteractionConfig should roundtrip through SimulationConfig.to_dict/from_dict."""
    config = _m4_config()
    d = config.to_dict()
    assert "interactions" in d
    restored = SimulationConfig.from_dict(d)
    assert restored.interactions is not None
    assert restored.interactions.price_x_media == {"tv": 0.3}
    assert restored.interactions.media_x_media == {("tv", "search"): 0.15}


def test_full_15_step_pipeline():
    """Full pipeline: M1 + M2 + M3 + M4 all together."""
    from demantiq.config.endogeneity_config import EndogeneityConfig
    from demantiq.config.macro_config import MacroConfig, MacroVariable, RegimeChange

    config = SimulationConfig(
        n_periods=104,
        channels=[
            ChannelConfig(name="tv", beta=500.0, spend_mean=10000, spend_std=3000),
            ChannelConfig(name="search", beta=300.0, spend_mean=5000, spend_std=1500),
        ],
        pricing=PricingConfig(base_price=25.0, price_elasticity=-1.2),
        distribution=DistributionConfig(initial_distribution=0.8),
        endogeneity=EndogeneityConfig(overall_strength=0.2),
        competition=CompetitionConfig(competitor_sov_mean=0.3, sov_suppression_coefficient=0.1),
        macro=MacroConfig(
            variables=[MacroVariable(name="gdp", effect_on_demand=50.0,
                                      time_series_type="mean_reverting",
                                      params={"mean": 0.0, "phi": 0.8, "sigma": 0.1})],
            regime_changes=[RegimeChange(period=60, change_type="level_shift", magnitude=-0.1)],
        ),
        interactions=InteractionConfig(
            price_x_media={"tv": 0.3},
            distribution_x_media={"search": 0.2},
            media_x_media={("tv", "search"): 0.15},
            competition_x_media={"tv": 0.25},
        ),
        noise=NoiseConfig(noise_scale=30.0),
        baseline=BaselineConfig(organic_level=5000.0, seasonality_coefficients=[100, 50, 30, 20]),
        seed=42,
    )
    result = Simulator(config).run()
    obs = result.observable_data
    gt = result.ground_truth
    summary = result.summary_truth

    # All 15 steps should be reflected
    assert obs.shape[0] == 104
    assert not obs.isnull().any().any()

    # M1 columns
    assert "y" in obs.columns
    assert "tv_spend" in obs.columns

    # M2 columns
    assert "price" in obs.columns
    assert "distribution" in obs.columns
    assert "revenue" in obs.columns

    # M3 columns
    assert "competitor_sov" in obs.columns
    assert "gdp" in obs.columns
    assert "confounders" in gt.columns

    # M4 columns
    assert "true_interaction_price_x_tv" in gt.columns
    assert "interactions" in summary

    # Summary should have all milestone info
    assert "true_betas" in summary
    assert "true_price_elasticity" in summary
    assert "competition" in summary
    assert "macro" in summary
    assert "interactions" in summary
