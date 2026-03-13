"""Training data pipeline for generating batched .npz training data."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from demantiq.scenarios.scenario_sampler import ScenarioSampler

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Generate training data in streaming batches.

    Uses a ScenarioSampler to generate configs, runs simulations in parallel,
    converts results to training format, and saves as .npz batches.

    Args:
        sampler: ScenarioSampler instance for generating configs.
        output_dir: Directory to write batch files.
        batch_size: Number of simulations per batch file.
    """

    def __init__(
        self,
        sampler: ScenarioSampler,
        output_dir: str,
        batch_size: int = 1000,
    ):
        self.sampler = sampler
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size

    def generate(
        self,
        n_total: int,
        n_workers: int = 4,
        seed: int = 42,
    ) -> None:
        """Generate training data in streaming batches.

        Generates configs from sampler, runs simulations in parallel,
        converts to training format, and saves as .npz files.
        Supports resumability by skipping already-completed batches.

        Args:
            n_total: Total number of simulations to generate.
            n_workers: Number of parallel worker processes.
            seed: Master seed for batch-level reproducibility.
        """
        from demantiq.orchestration.parallel_runner import run_parallel
        from demantiq.orchestration.training_format import (
            config_to_vector,
            save_batch,
            summary_to_vector,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        completed = self._get_completed_batches()
        n_batches = (n_total + self.batch_size - 1) // self.batch_size
        rng = np.random.default_rng(seed)

        for batch_id in range(n_batches):
            if batch_id in completed:
                logger.info("Skipping batch %d (already complete)", batch_id)
                continue

            batch_seed = int(rng.integers(0, 2**31))
            n_in_batch = min(self.batch_size, n_total - batch_id * self.batch_size)

            # Generate configs using a fresh sampler seeded per batch
            batch_sampler = ScenarioSampler(seed=batch_seed)
            configs = batch_sampler.sample(n_in_batch)

            # Run simulations in parallel
            results = run_parallel(configs, n_workers=n_workers)

            # Convert to training format
            tuples = []
            for config, result in zip(configs, results):
                if isinstance(result, Exception):
                    logger.warning("Simulation failed: %s", result)
                    continue

                channel_names = [ch.name for ch in config.channels]
                n_ch = len(channel_names)

                if n_ch > 0:
                    spend_matrix = np.column_stack(
                        [
                            result.observable_data[f"{ch}_spend"].values
                            for ch in channel_names
                        ]
                    )
                else:
                    spend_matrix = np.zeros((config.n_periods, 0))

                tuples.append(
                    {
                        "config_vector": config_to_vector(config),
                        "y": result.observable_data["y"].values,
                        "spend_matrix": spend_matrix,
                        "truth_vector": summary_to_vector(result.summary_truth),
                        "channel_names": channel_names,
                    }
                )

            if tuples:
                save_batch(tuples, str(self.output_dir), batch_id)

            logger.info(
                "Batch %d/%d: %d tuples saved", batch_id, n_batches, len(tuples)
            )

    def _get_completed_batches(self) -> set[int]:
        """Check which batches are already complete on disk."""
        completed: set[int] = set()
        if self.output_dir.exists():
            for f in self.output_dir.glob("batch_*.npz"):
                try:
                    batch_id = int(f.stem.split("_")[1])
                    completed.add(batch_id)
                except (IndexError, ValueError):
                    pass
        return completed
