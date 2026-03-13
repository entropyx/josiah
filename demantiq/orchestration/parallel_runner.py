"""Simple parallel runner for batch simulations."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

from demantiq.config.simulation_config import SimulationConfig
from demantiq.core.demand_kernel import SimulationResult, simulate


def _run_one(config: SimulationConfig) -> SimulationResult:
    """Worker function — must be top-level for pickling."""
    return simulate(config)


def run_parallel(
    configs: list[SimulationConfig],
    n_workers: int | None = None,
) -> list[SimulationResult | Exception]:
    """Run a list of SimulationConfigs in parallel.

    Args:
        configs: List of configs to simulate.
        n_workers: Max parallel workers (None = CPU count).

    Returns:
        List of SimulationResult or Exception for each config,
        preserving input order.  Failures do not crash the batch.
    """
    if not configs:
        return []

    results: list[SimulationResult | Exception] = [None] * len(configs)  # type: ignore

    if n_workers == 1:
        for i, config in enumerate(configs):
            try:
                results[i] = _run_one(config)
            except Exception as e:
                results[i] = e
        return results

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        future_to_idx = {pool.submit(_run_one, cfg): i
                         for i, cfg in enumerate(configs)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = e

    return results
