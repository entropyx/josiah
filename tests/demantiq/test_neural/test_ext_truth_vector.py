"""Tests for the extended truth vector format."""

import numpy as np
import pytest

from demantiq.config.simulation_config import SimulationConfig
from demantiq.config.channel_config import ChannelConfig
from demantiq.config.baseline_config import BaselineConfig
from demantiq.config.noise_config import NoiseConfig
from demantiq.core.demand_kernel import simulate
from demantiq.orchestration.training_format import (
    EXT_TRUTH_VECTOR_LEN,
    MAX_CHANNELS,
    _GLOBAL_EXT_TRUTH_LEN,
    _PER_CHANNEL_EXT_TRUTH_LEN,
    summary_to_ext_vector,
    ext_vector_to_summary,
)


def _make_config(n_channels: int = 3, seed: int = 42) -> SimulationConfig:
    names = ["facebook", "google", "tiktok", "pinterest", "email",
             "youtube", "snapchat", "linkedin", "twitter", "display"]
    channels = [
        ChannelConfig(
            name=names[i],
            beta=float(50 + i * 10),
            saturation_fn="hill" if i % 2 == 0 else "logistic",
            saturation_params=(
                {"K": 0.5, "S": 2.0} if i % 2 == 0
                else {"k": 5.0, "x0": 0.5}
            ),
            adstock_fn="geometric" if i % 2 == 0 else "weibull_cdf",
            adstock_params=(
                {"alpha": 0.5, "max_lag": 8} if i % 2 == 0
                else {"shape": 2.0, "scale": 3.0, "max_lag": 10}
            ),
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


class TestExtTruthVector:
    def test_returns_fixed_length(self):
        config = _make_config(3)
        result = simulate(config)
        vec = summary_to_ext_vector(result.summary_truth, config)
        assert vec.shape == (EXT_TRUTH_VECTOR_LEN,)

    def test_consistent_length_across_channel_counts(self):
        for n in [2, 5, 10]:
            config = _make_config(n, seed=n * 10)
            result = simulate(config)
            vec = summary_to_ext_vector(result.summary_truth, config)
            assert vec.shape == (EXT_TRUTH_VECTOR_LEN,)

    def test_encodes_media_contribution_pct(self):
        config = _make_config(3)
        result = simulate(config)
        vec = summary_to_ext_vector(result.summary_truth, config)
        expected = result.summary_truth["true_total_media_contribution_pct"]
        assert vec[0] == pytest.approx(expected)

    def test_encodes_channel_betas(self):
        config = _make_config(3)
        result = simulate(config)
        vec = summary_to_ext_vector(result.summary_truth, config)
        for i, ch in enumerate(config.channels):
            offset = _GLOBAL_EXT_TRUTH_LEN + i * _PER_CHANNEL_EXT_TRUTH_LEN
            assert vec[offset + 0] == pytest.approx(ch.beta)

    def test_encodes_saturation_params(self):
        config = _make_config(3)
        result = simulate(config)
        vec = summary_to_ext_vector(result.summary_truth, config)

        # First channel: hill with K=0.5, S=2.0
        offset = _GLOBAL_EXT_TRUTH_LEN
        assert vec[offset + 4] == 0.0  # hill = 0
        assert vec[offset + 5] == pytest.approx(0.5)  # K
        assert vec[offset + 6] == pytest.approx(2.0)  # S

        # Second channel: logistic with k=5.0, x0=0.5
        offset = _GLOBAL_EXT_TRUTH_LEN + _PER_CHANNEL_EXT_TRUTH_LEN
        assert vec[offset + 4] == 1.0  # logistic = 1
        assert vec[offset + 5] == pytest.approx(5.0)  # k
        assert vec[offset + 6] == pytest.approx(0.5)  # x0

    def test_encodes_adstock_params(self):
        config = _make_config(3)
        result = simulate(config)
        vec = summary_to_ext_vector(result.summary_truth, config)

        # First channel: geometric with alpha=0.5, max_lag=8
        offset = _GLOBAL_EXT_TRUTH_LEN
        assert vec[offset + 9] == 0.0  # geometric = 0
        assert vec[offset + 10] == pytest.approx(0.5)  # alpha
        assert vec[offset + 11] == pytest.approx(8.0)  # max_lag

        # Second channel: weibull_cdf with shape=2.0, scale=3.0, max_lag=10
        offset = _GLOBAL_EXT_TRUTH_LEN + _PER_CHANNEL_EXT_TRUTH_LEN
        assert vec[offset + 9] == 1.0  # weibull_cdf = 1
        assert vec[offset + 10] == pytest.approx(2.0)  # shape
        assert vec[offset + 11] == pytest.approx(3.0)  # scale
        assert vec[offset + 12] == pytest.approx(10.0)  # max_lag

    def test_contribution_fractions_sum_to_one(self):
        config = _make_config(5)
        result = simulate(config)
        vec = summary_to_ext_vector(result.summary_truth, config)

        fracs = []
        for i in range(5):
            offset = _GLOBAL_EXT_TRUTH_LEN + i * _PER_CHANNEL_EXT_TRUTH_LEN
            fracs.append(vec[offset + 3])

        assert sum(fracs) == pytest.approx(1.0, abs=1e-6)

    def test_unused_channels_are_zero(self):
        config = _make_config(2)
        result = simulate(config)
        vec = summary_to_ext_vector(result.summary_truth, config)

        # Third channel slot should be all zeros
        offset = _GLOBAL_EXT_TRUTH_LEN + 2 * _PER_CHANNEL_EXT_TRUTH_LEN
        assert np.all(vec[offset:offset + _PER_CHANNEL_EXT_TRUTH_LEN] == 0.0)


class TestExtVectorRoundTrip:
    def test_round_trip_betas(self):
        config = _make_config(3)
        result = simulate(config)
        vec = summary_to_ext_vector(result.summary_truth, config)
        channel_names = [ch.name for ch in config.channels]
        recovered = ext_vector_to_summary(vec, channel_names)

        for ch in config.channels:
            assert recovered["true_betas"][ch.name] == pytest.approx(ch.beta)

    def test_round_trip_roas(self):
        config = _make_config(3)
        result = simulate(config)
        vec = summary_to_ext_vector(result.summary_truth, config)
        channel_names = [ch.name for ch in config.channels]
        recovered = ext_vector_to_summary(vec, channel_names)

        for name in channel_names:
            assert recovered["true_roas"][name] == pytest.approx(
                result.summary_truth["true_roas"][name]
            )

    def test_round_trip_saturation_fn_names(self):
        config = _make_config(3)
        result = simulate(config)
        vec = summary_to_ext_vector(result.summary_truth, config)
        channel_names = [ch.name for ch in config.channels]
        recovered = ext_vector_to_summary(vec, channel_names)

        for ch in config.channels:
            assert recovered["true_saturation_fns"][ch.name] == ch.saturation_fn

    def test_round_trip_adstock_fn_names(self):
        config = _make_config(3)
        result = simulate(config)
        vec = summary_to_ext_vector(result.summary_truth, config)
        channel_names = [ch.name for ch in config.channels]
        recovered = ext_vector_to_summary(vec, channel_names)

        for ch in config.channels:
            assert recovered["true_adstock_fns"][ch.name] == ch.adstock_fn

    def test_round_trip_contribution_fractions(self):
        config = _make_config(4)
        result = simulate(config)
        vec = summary_to_ext_vector(result.summary_truth, config)
        channel_names = [ch.name for ch in config.channels]
        recovered = ext_vector_to_summary(vec, channel_names)

        total = sum(result.summary_truth["true_total_contribution"].values())
        for name in channel_names:
            expected_frac = result.summary_truth["true_total_contribution"][name] / total
            assert recovered["true_contribution_fractions"][name] == pytest.approx(
                expected_frac, abs=1e-6
            )
