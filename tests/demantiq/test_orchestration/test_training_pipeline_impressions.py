import tempfile
import numpy as np
from pathlib import Path
from demantiq.scenarios.scenario_sampler import ScenarioSampler
from demantiq.orchestration.training_pipeline import TrainingPipeline


def test_pipeline_writes_impressions_and_clicks():
    with tempfile.TemporaryDirectory() as tmpdir:
        sampler = ScenarioSampler(seed=42, rich_context=True, channel_range=(3, 5))
        pipeline = TrainingPipeline(sampler, output_dir=tmpdir, batch_size=2)
        pipeline.generate(n_total=2, n_workers=1, seed=42)

        npz_files = list(Path(tmpdir).glob("batch_*.npz"))
        assert len(npz_files) == 1
        data = np.load(str(npz_files[0]), allow_pickle=False)
        assert "impressions" in data.files
        assert "clicks" in data.files
        assert data["impressions"].ndim == 3
        assert data["clicks"].ndim == 3
        assert data["impressions"].shape == data["spend"].shape
