"""Tests for training data pipeline."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from demantiq.orchestration.training_format import load_batch
from demantiq.orchestration.training_pipeline import TrainingPipeline
from demantiq.scenarios.scenario_sampler import ScenarioSampler


class TestTrainingPipeline:
    def test_small_batch_generates_npz(self):
        """A small batch of simulations produces valid .npz files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = ScenarioSampler(seed=123)
            pipeline = TrainingPipeline(sampler, output_dir=tmpdir, batch_size=5)
            pipeline.generate(n_total=5, n_workers=1, seed=99)

            npz_files = list(Path(tmpdir).glob("batch_*.npz"))
            assert len(npz_files) == 1

            loaded = load_batch(str(npz_files[0]))
            assert loaded["config_vectors"].shape[0] > 0
            assert loaded["truth_vectors"].shape[0] > 0
            assert loaded["y"].shape[0] > 0

    def test_metadata_json_valid(self):
        """Metadata sidecar JSON is written and parseable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = ScenarioSampler(seed=456)
            pipeline = TrainingPipeline(sampler, output_dir=tmpdir, batch_size=3)
            pipeline.generate(n_total=3, n_workers=1, seed=77)

            meta_files = list(Path(tmpdir).glob("batch_*_meta.json"))
            assert len(meta_files) == 1

            meta = json.loads(meta_files[0].read_text())
            assert "batch_id" in meta
            assert "n_samples" in meta
            assert meta["n_samples"] > 0
            assert "channel_names" in meta
            assert "n_periods" in meta

    def test_multiple_batches(self):
        """Pipeline creates multiple batch files when n_total > batch_size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = ScenarioSampler(seed=789)
            pipeline = TrainingPipeline(sampler, output_dir=tmpdir, batch_size=3)
            pipeline.generate(n_total=7, n_workers=1, seed=55)

            npz_files = sorted(Path(tmpdir).glob("batch_*.npz"))
            assert len(npz_files) == 3  # ceil(7/3) = 3

    def test_resumability_skips_completed(self):
        """Pipeline skips batches whose .npz already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = ScenarioSampler(seed=111)
            pipeline = TrainingPipeline(sampler, output_dir=tmpdir, batch_size=3)

            # First run: generate all
            pipeline.generate(n_total=6, n_workers=1, seed=42)
            npz_files = sorted(Path(tmpdir).glob("batch_*.npz"))
            assert len(npz_files) == 2

            # Record modification times
            mtimes = {f.name: f.stat().st_mtime for f in npz_files}

            # Second run: should skip both batches (already complete)
            pipeline2 = TrainingPipeline(
                ScenarioSampler(seed=111), output_dir=tmpdir, batch_size=3
            )
            pipeline2.generate(n_total=6, n_workers=1, seed=42)

            # Files should not have been rewritten
            for f in Path(tmpdir).glob("batch_*.npz"):
                assert f.stat().st_mtime == mtimes[f.name]

    def test_output_arrays_have_correct_types(self):
        """Verify dtypes and shapes of output arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = ScenarioSampler(seed=222)
            pipeline = TrainingPipeline(sampler, output_dir=tmpdir, batch_size=4)
            pipeline.generate(n_total=4, n_workers=1, seed=33)

            npz_files = list(Path(tmpdir).glob("batch_*.npz"))
            loaded = load_batch(str(npz_files[0]))

            assert loaded["config_vectors"].dtype == np.float64
            assert loaded["truth_vectors"].dtype == np.float64
            assert loaded["y"].dtype == np.float64
            assert loaded["spend"].dtype == np.float64
