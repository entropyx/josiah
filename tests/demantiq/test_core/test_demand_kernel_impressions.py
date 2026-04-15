import numpy as np
from demantiq.scenarios.scenario_sampler import ScenarioSampler
from demantiq.core.demand_kernel import simulate


def test_simulate_returns_impressions_and_clicks():
    sampler = ScenarioSampler(seed=42, channel_range=(3, 5))
    config = sampler.sample(1)[0]
    result = simulate(config)
    for ch in config.channels:
        impressions_col = f"{ch.name}_impressions"
        clicks_col = f"{ch.name}_clicks"
        assert impressions_col in result.observable_data.columns, f"Missing {impressions_col}"
        assert clicks_col in result.observable_data.columns, f"Missing {clicks_col}"


def test_impressions_follow_spend_times_cpm():
    sampler = ScenarioSampler(seed=42, channel_range=(3, 5))
    config = sampler.sample(1)[0]
    result = simulate(config)
    for ch in config.channels:
        spend = result.observable_data[f"{ch.name}_spend"].values
        impressions = result.observable_data[f"{ch.name}_impressions"].values
        expected_ratio = 1000.0 / ch.cpm
        nonzero = spend > 0
        if nonzero.sum() > 10:
            empirical_ratio = (impressions[nonzero] / spend[nonzero]).mean()
            # Allow 30% slack due to noise
            assert 0.7 * expected_ratio < empirical_ratio < 1.3 * expected_ratio


def test_clicks_are_fraction_of_impressions():
    sampler = ScenarioSampler(seed=42, channel_range=(3, 5))
    config = sampler.sample(1)[0]
    result = simulate(config)
    for ch in config.channels:
        impressions = result.observable_data[f"{ch.name}_impressions"].values
        clicks = result.observable_data[f"{ch.name}_clicks"].values
        nonzero = impressions > 0
        if nonzero.sum() > 10:
            empirical_ctr = (clicks[nonzero] / impressions[nonzero]).mean()
            assert 0.5 * ch.ctr < empirical_ctr < 1.5 * ch.ctr
