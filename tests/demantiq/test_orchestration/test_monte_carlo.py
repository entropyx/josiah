"""Tests for the Monte Carlo runner."""

import pytest
from demantiq.orchestration.monte_carlo import MonteCarloRunner, MonteCarloResults
from demantiq.scenarios.scenario_library import ScenarioLibrary


class TestMonteCarloRunner:
    """Test batch Monte Carlo execution."""

    def test_basic_run(self):
        configs = [ScenarioLibrary.clean_room()]
        runner = MonteCarloRunner(configs, n_seeds_per_scenario=3, n_workers=1)
        results = runner.run()
        assert isinstance(results, MonteCarloResults)
        assert results.n_success == 3
        assert results.n_failed == 0

    def test_correct_result_count(self):
        configs = [ScenarioLibrary.clean_room(), ScenarioLibrary.short_data()]
        runner = MonteCarloRunner(configs, n_seeds_per_scenario=2, n_workers=1)
        results = runner.run()
        assert results.n_success == 4  # 2 configs * 2 seeds

    def test_summary_dataframe(self):
        configs = [ScenarioLibrary.clean_room()]
        runner = MonteCarloRunner(configs, n_seeds_per_scenario=2, n_workers=1)
        results = runner.run()
        assert len(results.summary) == 2
        assert "y_mean" in results.summary.columns
        assert "scenario_index" in results.summary.columns

    def test_different_seeds_produce_different_y(self):
        configs = [ScenarioLibrary.clean_room()]
        runner = MonteCarloRunner(configs, n_seeds_per_scenario=3, n_workers=1)
        results = runner.run()
        y_means = results.summary["y_mean"].tolist()
        # With different seeds, y_mean should vary
        assert len(set(round(v, 2) for v in y_means)) > 1
