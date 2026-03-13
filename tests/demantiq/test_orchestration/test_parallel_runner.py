"""Tests for the parallel runner."""

import pytest
from demantiq.orchestration.parallel_runner import run_parallel
from demantiq.core.demand_kernel import SimulationResult, simulate
from demantiq.scenarios.scenario_library import ScenarioLibrary


class TestParallelRunner:
    """Test parallel execution matches sequential."""

    def test_parallel_matches_sequential(self):
        configs = [ScenarioLibrary.clean_room(), ScenarioLibrary.short_data()]
        # Sequential
        sequential = [simulate(c) for c in configs]
        # Parallel (single worker to ensure determinism)
        parallel = run_parallel(configs, n_workers=1)

        assert len(parallel) == len(sequential)
        for s, p in zip(sequential, parallel):
            assert isinstance(p, SimulationResult)
            assert len(s.observable_data) == len(p.observable_data)
            # Same config, same seed -> same output
            assert s.observable_data["y"].sum() == pytest.approx(
                p.observable_data["y"].sum(), rel=1e-10
            )

    def test_empty_list(self):
        results = run_parallel([], n_workers=1)
        assert results == []

    def test_preserves_order(self):
        configs = [
            ScenarioLibrary.clean_room(),
            ScenarioLibrary.short_data(),
            ScenarioLibrary.mature_market(),
        ]
        results = run_parallel(configs, n_workers=1)
        assert len(results) == 3
        for config, result in zip(configs, results):
            assert isinstance(result, SimulationResult)
            assert len(result.observable_data) == config.n_periods

    def test_failure_does_not_crash_batch(self):
        """A bad config should yield an Exception, not crash others."""
        from demantiq.config.simulation_config import SimulationConfig
        from demantiq.config.channel_config import ChannelConfig

        good = ScenarioLibrary.clean_room()
        # Create a config that will fail (invalid saturation fn)
        bad = SimulationConfig(
            n_periods=10,
            channels=[ChannelConfig(name="bad", saturation_fn="nonexistent_fn")],
            seed=999,
        )
        results = run_parallel([good, bad], n_workers=1)
        assert len(results) == 2
        assert isinstance(results[0], SimulationResult)
        assert isinstance(results[1], Exception)
