"""Tests for training data format utilities."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from demantiq.config.simulation_config import SimulationConfig
from demantiq.config.channel_config import ChannelConfig
from demantiq.config.baseline_config import BaselineConfig
from demantiq.config.noise_config import NoiseConfig
from demantiq.core.demand_kernel import simulate
from demantiq.orchestration.training_format import (
    CONFIG_VECTOR_LEN,
    TRUTH_VECTOR_LEN,
    MAX_CHANNELS,
    config_to_vector,
    summary_to_vector,
    vector_to_summary,
    save_batch,
    load_batch,
)


def _make_config(n_channels: int = 3, seed: int = 42) -> SimulationConfig:
    """Helper to create a SimulationConfig with given number of channels."""
    names = ["facebook", "google", "tiktok", "pinterest", "email",
             "youtube", "snapchat", "linkedin", "twitter", "display"]
    channels = [
        ChannelConfig(
            name=names[i],
            beta=float(50 + i * 10),
            saturation_fn="hill",
            saturation_params={"K": 0.5, "S": 2.0},
            adstock_fn="geometric",
            adstock_params={"alpha": 0.5, "max_lag": 8},
            spend_mean=10000.0,
            spend_std=3000.0,
        )
        for i in range(n_channels)
    ]
    return SimulationConfig(
        n_periods=52,
        channels=channels,
        noise=NoiseConfig(noise_scale=10.0),
        baseline=BaselineConfig(organic_level=500.0, trend_params={"slope": 2.0}),
        seed=seed,
    )


class TestConfigToVector:
    def test_returns_fixed_length(self):
        config = _make_config(3)
        vec = config_to_vector(config)
        assert vec.shape == (CONFIG_VECTOR_LEN,)

    def test_consistent_length_across_channel_counts(self):
        vec2 = config_to_vector(_make_config(2))
        vec5 = config_to_vector(_make_config(5))
        vec10 = config_to_vector(_make_config(10))
        assert vec2.shape == vec5.shape == vec10.shape == (CONFIG_VECTOR_LEN,)

    def test_encodes_global_params(self):
        config = _make_config(3)
        vec = config_to_vector(config)
        assert vec[0] == 52.0  # n_periods
        assert vec[1] == 3.0   # n_channels
        assert vec[2] == 10.0  # noise_scale
        assert vec[4] == 500.0 # organic_level
        assert vec[5] == 2.0   # trend slope

    def test_encodes_channel_beta(self):
        config = _make_config(3)
        vec = config_to_vector(config)
        # First channel beta at offset 13
        assert vec[13] == 50.0  # first channel beta
        assert vec[13 + 12] == 60.0  # second channel beta
        assert vec[13 + 24] == 70.0  # third channel beta

    def test_unused_channels_are_zero(self):
        config = _make_config(2)
        vec = config_to_vector(config)
        # Third channel slot should be all zeros
        offset = 13 + 2 * 12
        assert np.all(vec[offset:offset + 12] == 0.0)

    def test_different_saturation_fns(self):
        ch_logistic = ChannelConfig(
            name="test",
            beta=100.0,
            saturation_fn="logistic",
            saturation_params={"k": 5.0, "x0": 0.5},
        )
        config = SimulationConfig(channels=[ch_logistic], seed=1)
        vec = config_to_vector(config)
        # sat_fn encoded as 1 (logistic)
        assert vec[13 + 1] == 1.0
        # sat params
        assert vec[13 + 2] == 5.0  # k
        assert vec[13 + 3] == 0.5  # x0

    def test_different_adstock_fns(self):
        ch_weibull = ChannelConfig(
            name="test",
            beta=100.0,
            adstock_fn="weibull_cdf",
            adstock_params={"shape": 2.0, "scale": 3.0, "max_lag": 10},
        )
        config = SimulationConfig(channels=[ch_weibull], seed=1)
        vec = config_to_vector(config)
        # ads_fn encoded as 1 (weibull_cdf)
        assert vec[13 + 6] == 1.0
        assert vec[13 + 7] == 2.0   # shape
        assert vec[13 + 8] == 3.0   # scale
        assert vec[13 + 9] == 10.0  # max_lag

    def test_optional_config_flags(self):
        config = _make_config(2)
        vec = config_to_vector(config)
        # All optional configs are None
        for i in range(7, 13):
            assert vec[i] == 0.0


class TestSummaryToVector:
    def test_returns_fixed_length(self):
        config = _make_config(3)
        result = simulate(config)
        vec = summary_to_vector(result.summary_truth)
        assert vec.shape == (TRUTH_VECTOR_LEN,)

    def test_consistent_length_across_channel_counts(self):
        r2 = simulate(_make_config(2, seed=10))
        r5 = simulate(_make_config(5, seed=20))
        v2 = summary_to_vector(r2.summary_truth)
        v5 = summary_to_vector(r5.summary_truth)
        assert v2.shape == v5.shape == (TRUTH_VECTOR_LEN,)

    def test_encodes_media_contribution_pct(self):
        config = _make_config(3)
        result = simulate(config)
        vec = summary_to_vector(result.summary_truth)
        expected = result.summary_truth["true_total_media_contribution_pct"]
        assert vec[0] == pytest.approx(expected)

    def test_encodes_channel_betas(self):
        config = _make_config(3)
        result = simulate(config)
        vec = summary_to_vector(result.summary_truth)
        betas = result.summary_truth["true_betas"]
        for i, name in enumerate(betas):
            offset = 2 + i * 3
            assert vec[offset] == pytest.approx(betas[name])


class TestVectorToSummary:
    def test_round_trip(self):
        config = _make_config(3)
        result = simulate(config)
        summary = result.summary_truth
        channel_names = [ch.name for ch in config.channels]

        vec = summary_to_vector(summary)
        recovered = vector_to_summary(vec, channel_names)

        assert recovered["true_total_media_contribution_pct"] == pytest.approx(
            summary["true_total_media_contribution_pct"]
        )
        for name in channel_names:
            assert recovered["true_betas"][name] == pytest.approx(
                summary["true_betas"][name]
            )
            assert recovered["true_roas"][name] == pytest.approx(
                summary["true_roas"][name]
            )
            assert recovered["true_total_contribution"][name] == pytest.approx(
                summary["true_total_contribution"][name]
            )


class TestSaveBatchLoadBatch:
    def test_round_trip(self):
        config = _make_config(3)
        result = simulate(config)
        channel_names = [ch.name for ch in config.channels]

        tuples = [
            {
                "config_vector": config_to_vector(config),
                "y": result.observable_data["y"].values,
                "spend_matrix": np.column_stack(
                    [result.observable_data[f"{ch}_spend"].values for ch in channel_names]
                ),
                "truth_vector": summary_to_vector(result.summary_truth),
                "channel_names": channel_names,
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            save_batch(tuples, tmpdir, batch_id=0)

            npz_path = Path(tmpdir) / "batch_0.npz"
            meta_path = Path(tmpdir) / "batch_0_meta.json"
            assert npz_path.exists()
            assert meta_path.exists()

            loaded = load_batch(str(npz_path))
            assert "config_vectors" in loaded
            assert "truth_vectors" in loaded
            assert "y" in loaded
            assert "spend" in loaded

            # Check shapes
            assert loaded["config_vectors"].shape == (1, CONFIG_VECTOR_LEN)
            assert loaded["truth_vectors"].shape == (1, TRUTH_VECTOR_LEN)
            assert loaded["y"].shape[0] == 1
            assert loaded["y"].shape[1] == 52  # n_periods

            # Check values round-trip
            np.testing.assert_allclose(
                loaded["config_vectors"][0], tuples[0]["config_vector"]
            )
            np.testing.assert_allclose(
                loaded["truth_vectors"][0], tuples[0]["truth_vector"]
            )

    def test_multiple_samples_different_periods(self):
        """Batch with varying n_periods should pad correctly."""
        config1 = SimulationConfig(
            n_periods=30,
            channels=[ChannelConfig(name="facebook", beta=100.0)],
            seed=1,
        )
        config2 = SimulationConfig(
            n_periods=50,
            channels=[ChannelConfig(name="google", beta=200.0)],
            seed=2,
        )
        r1 = simulate(config1)
        r2 = simulate(config2)

        tuples = []
        for cfg, res in [(config1, r1), (config2, r2)]:
            ch_names = [ch.name for ch in cfg.channels]
            tuples.append({
                "config_vector": config_to_vector(cfg),
                "y": res.observable_data["y"].values,
                "spend_matrix": np.column_stack(
                    [res.observable_data[f"{ch}_spend"].values for ch in ch_names]
                ),
                "truth_vector": summary_to_vector(res.summary_truth),
                "channel_names": ch_names,
            })

        with tempfile.TemporaryDirectory() as tmpdir:
            save_batch(tuples, tmpdir, batch_id=5)
            loaded = load_batch(str(Path(tmpdir) / "batch_5.npz"))

            assert loaded["y"].shape == (2, 50)  # padded to max periods
            assert loaded["spend"].shape == (2, 50, 1)

            # First sample's y should match for first 30, then zeros
            np.testing.assert_allclose(loaded["y"][0, :30], r1.observable_data["y"].values)
            np.testing.assert_allclose(loaded["y"][0, 30:], 0.0)

    def test_metadata_sidecar(self):
        config = _make_config(2)
        result = simulate(config)
        ch_names = [ch.name for ch in config.channels]

        tuples = [{
            "config_vector": config_to_vector(config),
            "y": result.observable_data["y"].values,
            "spend_matrix": np.column_stack(
                [result.observable_data[f"{ch}_spend"].values for ch in ch_names]
            ),
            "truth_vector": summary_to_vector(result.summary_truth),
            "channel_names": ch_names,
        }]

        with tempfile.TemporaryDirectory() as tmpdir:
            save_batch(tuples, tmpdir, batch_id=0)
            loaded = load_batch(str(Path(tmpdir) / "batch_0.npz"))

            meta = loaded["metadata"]
            assert meta["batch_id"] == 0
            assert meta["n_samples"] == 1
            assert meta["max_periods"] == 52
            assert meta["max_channels"] == 2
