"""Tests for neural utils."""

import torch
import pytest

from demantiq.neural.utils import (
    make_channel_mask,
    normalize_time_series,
    get_channel_type_indices,
    extract_per_channel_truth,
    MAX_CHANNELS,
    CHANNEL_NAME_TO_IDX,
)


class TestMakeChannelMask:
    def test_scalar_input(self):
        mask = make_channel_mask(5)
        assert mask.shape == (MAX_CHANNELS,)
        assert mask[:5].all()
        assert not mask[5:].any()

    def test_batch_input(self):
        n_ch = torch.tensor([3, 7, 1])
        mask = make_channel_mask(n_ch)
        assert mask.shape == (3, MAX_CHANNELS)
        assert mask[0, :3].all()
        assert not mask[0, 3:].any()
        assert mask[1, :7].all()
        assert not mask[1, 7:].any()
        assert mask[2, :1].all()
        assert not mask[2, 1:].any()


class TestNormalizeTimeSeries:
    def test_basic_normalization(self):
        x = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
        norm, mean, std = normalize_time_series(x)
        assert norm.shape == x.shape
        assert mean.shape == (1,)
        assert norm.mean().abs() < 1e-5

    def test_with_mask(self):
        x = torch.tensor([[10.0, 20.0, 30.0, 0.0]])
        mask = torch.tensor([[True, True, True, False]])
        norm, mean, std = normalize_time_series(x, mask=mask)
        # Masked position should be handled
        assert norm.shape == x.shape


class TestGetChannelTypeIndices:
    def test_known_channels(self):
        batch = [["facebook", "google"], ["tiktok"]]
        indices = get_channel_type_indices(batch)
        assert indices.shape == (2, MAX_CHANNELS)
        assert indices[0, 0] == CHANNEL_NAME_TO_IDX["facebook"]
        assert indices[0, 1] == CHANNEL_NAME_TO_IDX["google"]
        assert indices[1, 0] == CHANNEL_NAME_TO_IDX["tiktok"]

    def test_padding_is_zero(self):
        batch = [["facebook"]]
        indices = get_channel_type_indices(batch)
        assert indices[0, 1:].sum() == 0


class TestExtractPerChannelTruth:
    def test_shapes(self):
        from demantiq.orchestration.training_format import (
            EXT_TRUTH_VECTOR_LEN,
            _GLOBAL_EXT_TRUTH_LEN,
            _PER_CHANNEL_EXT_TRUTH_LEN,
        )
        ext = torch.randn(4, EXT_TRUTH_VECTOR_LEN)
        n_ch = torch.tensor([3, 5, 2, 8])
        global_t, per_ch = extract_per_channel_truth(ext, n_ch)
        assert global_t.shape == (4, _GLOBAL_EXT_TRUTH_LEN)
        assert per_ch.shape == (4, MAX_CHANNELS, _PER_CHANNEL_EXT_TRUTH_LEN)
