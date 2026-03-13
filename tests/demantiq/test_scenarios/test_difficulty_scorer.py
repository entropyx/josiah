"""Tests for the difficulty scorer."""

import pytest
from demantiq.scenarios.difficulty_scorer import score_difficulty, difficulty_components
from demantiq.scenarios.scenario_library import ScenarioLibrary
from demantiq.config.simulation_config import SimulationConfig
from demantiq.config.channel_config import ChannelConfig
from demantiq.config.noise_config import NoiseConfig


class TestDifficultyScorer:
    """Test difficulty scoring produces sensible orderings."""

    def test_clean_room_less_than_real_world(self):
        clean = score_difficulty(ScenarioLibrary.clean_room())
        real = score_difficulty(ScenarioLibrary.real_world())
        assert clean < real

    def test_real_world_less_than_adversarial(self):
        real = score_difficulty(ScenarioLibrary.real_world())
        adv = score_difficulty(ScenarioLibrary.adversarial())
        assert real < adv

    def test_clean_room_less_than_adversarial(self):
        clean = score_difficulty(ScenarioLibrary.clean_room())
        adv = score_difficulty(ScenarioLibrary.adversarial())
        assert clean < adv

    def test_score_in_valid_range(self):
        for name in ScenarioLibrary.list_scenarios():
            config = ScenarioLibrary.get(name)
            score = score_difficulty(config)
            assert 0.0 <= score <= 1.0, f"{name}: score={score} out of range"

    def test_components_sum_to_score(self):
        config = ScenarioLibrary.real_world()
        components = difficulty_components(config)
        score = score_difficulty(config)
        weights = {
            "collinearity": 0.20,
            "endogeneity": 0.20,
            "signal_to_noise": 0.15,
            "data_length": 0.15,
            "channel_count": 0.10,
            "interaction_complexity": 0.10,
            "structural_breaks": 0.10,
        }
        expected = sum(weights[k] * components[k] for k in weights)
        assert abs(score - expected) < 1e-10

    def test_minimal_config_low_score(self):
        """A config with 1 channel, low noise, long data should be easy."""
        config = SimulationConfig(
            n_periods=520,
            channels=[ChannelConfig(name="tv", correlation_group="solo")],
            noise=NoiseConfig(noise_scale=1.0),
        )
        score = score_difficulty(config)
        assert score < 0.1

    def test_empty_channels_score(self):
        """Config with no channels should not crash."""
        config = SimulationConfig(n_periods=104, channels=[])
        score = score_difficulty(config)
        assert 0.0 <= score <= 1.0

    def test_components_returns_all_keys(self):
        config = ScenarioLibrary.clean_room()
        components = difficulty_components(config)
        expected_keys = {
            "collinearity", "endogeneity", "signal_to_noise",
            "data_length", "channel_count", "interaction_complexity",
            "structural_breaks",
        }
        assert set(components.keys()) == expected_keys
