"""Monte Carlo runner for batch simulation experiments."""

from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import pandas as pd

from demantiq.config.simulation_config import SimulationConfig
from demantiq.core.demand_kernel import SimulationResult, simulate


@dataclass
class MonteCarloResults:
    """Aggregated results from a Monte Carlo batch run.

    Attributes:
        results: List of (scenario_index, seed, SimulationResult) tuples.
        failures: List of (scenario_index, seed, error_message) tuples.
        summary: DataFrame summarizing each run.
    """
    results: list[tuple[int, int, SimulationResult]] = field(default_factory=list)
    failures: list[tuple[int, int, str]] = field(default_factory=list)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def n_success(self) -> int:
        return len(self.results)

    @property
    def n_failed(self) -> int:
        return len(self.failures)


def _run_single(args: tuple[int, SimulationConfig]) -> tuple[int, SimulationResult]:
    """Worker function for process pool.  Must be top-level for pickling."""
    idx, config = args
    result = simulate(config)
    return idx, result


class MonteCarloRunner:
    """Run multiple SimulationConfigs with varied seeds.

    Args:
        configs: List of SimulationConfig objects.
        n_seeds_per_scenario: Number of seed variants per config.
        base_seed: Starting seed for generating per-run seeds.
        n_workers: Max parallel workers (None = CPU count).
    """

    def __init__(
        self,
        configs: list[SimulationConfig],
        n_seeds_per_scenario: int = 10,
        base_seed: int = 0,
        n_workers: int | None = None,
    ):
        self.configs = configs
        self.n_seeds_per_scenario = n_seeds_per_scenario
        self.base_seed = base_seed
        self.n_workers = n_workers

    def _make_variants(self) -> list[tuple[int, int, SimulationConfig]]:
        """Create (scenario_idx, seed, config) tuples with varied seeds."""
        variants = []
        for scenario_idx, config in enumerate(self.configs):
            for seed_offset in range(self.n_seeds_per_scenario):
                seed = self.base_seed + scenario_idx * 1000 + seed_offset
                # Create a new config with the varied seed
                variant = SimulationConfig(
                    n_periods=config.n_periods,
                    granularity=config.granularity,
                    channels=config.channels,
                    noise=config.noise,
                    baseline=config.baseline,
                    seed=seed,
                    metadata={**config.metadata, "monte_carlo_seed": seed,
                              "scenario_index": scenario_idx},
                    pricing=config.pricing,
                    distribution=config.distribution,
                    competition=config.competition,
                    macro=config.macro,
                    endogeneity=config.endogeneity,
                    interactions=config.interactions,
                )
                variants.append((scenario_idx, seed, variant))
        return variants

    def run(self) -> MonteCarloResults:
        """Execute all variants, optionally in parallel.

        Returns:
            MonteCarloResults with all successful results and any failures.
        """
        variants = self._make_variants()
        mc_results = MonteCarloResults()

        if self.n_workers == 1:
            # Sequential execution
            for scenario_idx, seed, config in variants:
                try:
                    result = simulate(config)
                    mc_results.results.append((scenario_idx, seed, result))
                except Exception as e:
                    mc_results.failures.append((scenario_idx, seed, str(e)))
        else:
            # Parallel execution
            work_items = {i: (scenario_idx, seed)
                          for i, (scenario_idx, seed, _) in enumerate(variants)}
            configs_for_pool = [(i, v[2]) for i, v in enumerate(variants)]

            with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
                futures = {pool.submit(_run_single, item): item[0]
                           for item in configs_for_pool}
                for future in as_completed(futures):
                    idx = futures[future]
                    scenario_idx, seed = work_items[idx]
                    try:
                        _, result = future.result()
                        mc_results.results.append((scenario_idx, seed, result))
                    except Exception as e:
                        mc_results.failures.append((scenario_idx, seed, str(e)))

        # Build summary DataFrame
        rows = []
        for scenario_idx, seed, result in mc_results.results:
            obs = result.observable_data
            rows.append({
                "scenario_index": scenario_idx,
                "seed": seed,
                "n_periods": len(obs),
                "y_mean": float(obs["y"].mean()),
                "y_std": float(obs["y"].std()),
                "y_min": float(obs["y"].min()),
                "y_max": float(obs["y"].max()),
            })
        if rows:
            mc_results.summary = pd.DataFrame(rows)

        return mc_results
