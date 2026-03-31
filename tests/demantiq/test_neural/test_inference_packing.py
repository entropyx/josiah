"""Tests for observation packing and the embedding wrapper."""

import torch
import pytest

from demantiq.neural.inference import (
    pack_observations,
    DemantiqEmbeddingWrapper,
)
from demantiq.neural.encoders import EmbeddingNetwork
from demantiq.neural.utils import MAX_CHANNELS
from demantiq.orchestration.training_format import MAX_CONTEXT_COLS


class TestPackObservations:
    def test_output_shape(self):
        max_T = 104
        batch = 4
        obs = pack_observations(
            y=torch.randn(batch, 52),
            spend=torch.randn(batch, 52, 5),
            n_channels=torch.tensor([5, 5, 5, 5]),
            n_periods=torch.tensor([52, 52, 52, 52]),
            channel_type_ids=torch.zeros(batch, MAX_CHANNELS, dtype=torch.long),
            max_T=max_T,
        )
        expected_dim = max_T + max_T * MAX_CHANNELS + max_T * MAX_CONTEXT_COLS + 1 + 1 + MAX_CHANNELS
        assert obs.shape == (batch, expected_dim)

    def test_y_padded_correctly(self):
        max_T = 104
        y = torch.ones(1, 52) * 42.0
        obs = pack_observations(
            y=y,
            spend=torch.zeros(1, 52, 3),
            n_channels=torch.tensor([3]),
            n_periods=torch.tensor([52]),
            channel_type_ids=torch.zeros(1, MAX_CHANNELS, dtype=torch.long),
            max_T=max_T,
        )
        # First 52 values should be 42, rest should be 0
        assert (obs[0, :52] == 42.0).all()
        assert (obs[0, 52:max_T] == 0.0).all()

    def test_n_channels_encoded(self):
        max_T = 52
        obs = pack_observations(
            y=torch.randn(1, 52),
            spend=torch.randn(1, 52, 5),
            n_channels=torch.tensor([5]),
            n_periods=torch.tensor([52]),
            channel_type_ids=torch.zeros(1, MAX_CHANNELS, dtype=torch.long),
            max_T=max_T,
        )
        # n_channels is at index max_T + max_T * MAX_CHANNELS + max_T * MAX_CONTEXT_COLS
        idx = max_T + max_T * MAX_CHANNELS + max_T * MAX_CONTEXT_COLS
        assert obs[0, idx] == 5.0


class TestDemantiqEmbeddingWrapper:
    def test_forward_shape(self):
        net = EmbeddingNetwork(
            temporal_dim=16,
            type_embed_dim=4,
            n_attn_heads=2,
            n_attn_layers=1,
            global_summary_dim=32,
            dropout=0.0,
        )
        max_T = 52
        wrapper = DemantiqEmbeddingWrapper(net, max_T=max_T)

        # Create a flat observation
        obs = pack_observations(
            y=torch.randn(2, 52),
            spend=torch.randn(2, 52, 5),
            n_channels=torch.tensor([5, 3]),
            n_periods=torch.tensor([52, 52]),
            channel_type_ids=torch.zeros(2, MAX_CHANNELS, dtype=torch.long),
            max_T=max_T,
        )

        out = wrapper(obs)
        assert out.shape == (2, 32)

    def test_gradients_flow(self):
        net = EmbeddingNetwork(
            temporal_dim=16,
            type_embed_dim=4,
            n_attn_heads=2,
            n_attn_layers=1,
            global_summary_dim=32,
            dropout=0.0,
        )
        max_T = 52
        wrapper = DemantiqEmbeddingWrapper(net, max_T=max_T)

        obs = pack_observations(
            y=torch.randn(2, 52),
            spend=torch.randn(2, 52, 5),
            n_channels=torch.tensor([5, 3]),
            n_periods=torch.tensor([52, 52]),
            channel_type_ids=torch.zeros(2, MAX_CHANNELS, dtype=torch.long),
            max_T=max_T,
        )

        out = wrapper(obs)
        loss = out.sum()
        loss.backward()

        # Check that embedding net params have gradients
        for p in net.parameters():
            if p.requires_grad:
                assert p.grad is not None
                break
