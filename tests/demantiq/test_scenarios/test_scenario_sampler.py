"""Tests for the scenario sampler."""

import pytest
from demantiq.scenarios.scenario_sampler import ScenarioSampler
from demantiq.config.simulation_config import SimulationConfig


class TestScenarioSampler:
    """Test random scenario sampling."""

    def test_sample_returns_correct_count(self):
        sampler = ScenarioSampler(seed=42)
        configs = sampler.sample(5)
        assert len(configs) == 5

    def test_sample_returns_simulation_configs(self):
        sampler = ScenarioSampler(seed=42)
        configs = sampler.sample(3)
        for config in configs:
            assert isinstance(config, SimulationConfig)
            assert config.n_periods >= 26
            assert len(config.channels) >= 2

    def test_deterministic_with_seed(self):
        configs_a = ScenarioSampler(seed=123).sample(3)
        configs_b = ScenarioSampler(seed=123).sample(3)
        for a, b in zip(configs_a, configs_b):
            assert a.seed == b.seed
            assert a.n_periods == b.n_periods
            assert len(a.channels) == len(b.channels)

    def test_different_seeds_differ(self):
        configs_a = ScenarioSampler(seed=1).sample(3)
        configs_b = ScenarioSampler(seed=2).sample(3)
        # At least one config should differ
        any_diff = any(
            a.seed != b.seed or a.n_periods != b.n_periods
            for a, b in zip(configs_a, configs_b)
        )
        assert any_diff

    def test_sample_zero(self):
        sampler = ScenarioSampler(seed=42)
        configs = sampler.sample(0)
        assert configs == []

    def test_sampled_configs_have_valid_channels(self):
        sampler = ScenarioSampler(seed=42)
        configs = sampler.sample(5)
        for config in configs:
            for ch in config.channels:
                assert ch.name
                assert ch.beta > 0
                assert ch.spend_mean > 0
