"""Tests for the scenario library."""

import pytest
from demantiq.scenarios.scenario_library import ScenarioLibrary
from demantiq.config.simulation_config import SimulationConfig
from demantiq.core.demand_kernel import simulate


class TestScenarioLibrary:
    """Test that all 15 scenarios are loadable and generate valid output."""

    def test_list_scenarios_returns_15(self):
        names = ScenarioLibrary.list_scenarios()
        assert len(names) == 15

    def test_all_scenarios_returns_dict(self):
        scenarios = ScenarioLibrary.all_scenarios()
        assert len(scenarios) == 15
        for name, config in scenarios.items():
            assert isinstance(config, SimulationConfig)

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown scenario"):
            ScenarioLibrary.get("nonexistent_scenario")

    @pytest.mark.parametrize("name", ScenarioLibrary.list_scenarios())
    def test_scenario_is_valid_config(self, name):
        config = ScenarioLibrary.get(name)
        assert isinstance(config, SimulationConfig)
        assert config.n_periods > 0
        assert len(config.channels) > 0

    @pytest.mark.parametrize("name", ScenarioLibrary.list_scenarios())
    def test_scenario_generates_output(self, name):
        config = ScenarioLibrary.get(name)
        result = simulate(config)
        assert len(result.observable_data) == config.n_periods
        assert "y" in result.observable_data.columns

    def test_clean_room_specifics(self):
        config = ScenarioLibrary.clean_room()
        assert config.n_periods == 156
        assert config.seed == 1001
        assert len(config.channels) == 5
        assert config.endogeneity is None
        assert config.noise.noise_scale == 5.0
        # Each channel in its own group
        groups = {ch.correlation_group for ch in config.channels}
        assert len(groups) == 5

    def test_adversarial_specifics(self):
        config = ScenarioLibrary.adversarial()
        assert config.n_periods == 52
        assert config.seed == 1003
        assert len(config.channels) == 12
        assert config.endogeneity is not None
        assert config.endogeneity.overall_strength == 0.7
        assert config.noise.noise_scale == 50.0
        assert config.macro is not None

    def test_dtc_pure_play_daily(self):
        config = ScenarioLibrary.dtc_pure_play()
        assert config.granularity == "daily"
        assert config.n_periods == 365

    def test_omnichannel_has_all_features(self):
        config = ScenarioLibrary.omnichannel_retail()
        assert len(config.channels) == 20
        assert config.pricing is not None
        assert config.distribution is not None
        assert config.endogeneity is not None
        assert config.competition is not None
        assert config.macro is not None
        assert config.interactions is not None

    def test_unique_seeds(self):
        scenarios = ScenarioLibrary.all_scenarios()
        seeds = [config.seed for config in scenarios.values()]
        assert len(set(seeds)) == len(seeds), "All scenarios must have unique seeds"
