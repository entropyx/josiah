"""Tests for neural encoder architectures."""

import pytest
import torch

from demantiq.neural.encoders import (
    DilatedConv1DEncoder,
    MultiheadSetAttention,
    PoolingByMultiheadAttention,
    SetTransformerEncoder,
    EmbeddingNetwork,
)
from demantiq.neural.utils import MAX_CHANNELS


class TestDilatedConv1DEncoder:
    def test_output_shape_2d(self):
        enc = DilatedConv1DEncoder(in_channels=1, hidden_dim=16, output_dim=32)
        x = torch.randn(4, 104)  # (batch, T)
        out = enc(x)
        assert out.shape == (4, 32)

    def test_output_shape_3d(self):
        enc = DilatedConv1DEncoder(in_channels=3, hidden_dim=16, output_dim=32)
        x = torch.randn(4, 104, 3)  # (batch, T, C)
        out = enc(x)
        assert out.shape == (4, 32)

    def test_variable_length(self):
        enc = DilatedConv1DEncoder(in_channels=1, hidden_dim=16, output_dim=32)
        for T in [26, 52, 104, 260]:
            x = torch.randn(2, T)
            out = enc(x)
            assert out.shape == (2, 32)

    def test_gradients_flow(self):
        enc = DilatedConv1DEncoder(in_channels=1, hidden_dim=16, output_dim=32)
        x = torch.randn(2, 52, requires_grad=True)
        out = enc(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestMultiheadSetAttention:
    def test_output_shape(self):
        attn = MultiheadSetAttention(d_model=64, n_heads=4)
        x = torch.randn(4, 10, 64)  # (batch, N, d)
        out = attn(x)
        assert out.shape == (4, 10, 64)

    def test_with_padding_mask(self):
        attn = MultiheadSetAttention(d_model=64, n_heads=4)
        x = torch.randn(4, 10, 64)
        # Mask out last 5 elements in each sample
        mask = torch.zeros(4, 10, dtype=torch.bool)
        mask[:, 5:] = True
        out = attn(x, key_padding_mask=mask)
        assert out.shape == (4, 10, 64)


class TestPoolingByMultiheadAttention:
    def test_output_shape_k1(self):
        pma = PoolingByMultiheadAttention(d_model=64, n_heads=4, k=1)
        x = torch.randn(4, 10, 64)
        out = pma(x)
        assert out.shape == (4, 1, 64)

    def test_output_shape_k4(self):
        pma = PoolingByMultiheadAttention(d_model=64, n_heads=4, k=4)
        x = torch.randn(4, 10, 64)
        out = pma(x)
        assert out.shape == (4, 4, 64)


class TestSetTransformerEncoder:
    def test_output_shapes(self):
        enc = SetTransformerEncoder(d_model=64, n_heads=4, n_layers=2)
        x = torch.randn(4, 10, 64)
        summary, per_element = enc(x)
        assert summary.shape == (4, 64)
        assert per_element.shape == (4, 10, 64)

    def test_with_padding_mask(self):
        enc = SetTransformerEncoder(d_model=64, n_heads=4, n_layers=2)
        x = torch.randn(4, 10, 64)
        mask = torch.zeros(4, 10, dtype=torch.bool)
        mask[:, 7:] = True
        summary, per_element = enc(x, key_padding_mask=mask)
        assert summary.shape == (4, 64)
        assert per_element.shape == (4, 10, 64)


class TestEmbeddingNetwork:
    @pytest.fixture
    def net(self):
        return EmbeddingNetwork(
            temporal_dim=32,
            type_embed_dim=8,
            n_attn_heads=2,
            n_attn_layers=1,
            global_summary_dim=64,
            dropout=0.0,
        )

    def test_output_shapes(self, net):
        batch = 4
        T = 52
        y = torch.randn(batch, T)
        spend = torch.randn(batch, T, MAX_CHANNELS)
        n_channels = torch.tensor([5, 3, 8, 2])
        type_ids = torch.zeros(batch, MAX_CHANNELS, dtype=torch.long)

        global_summary, per_channel, ch_mask = net(y, spend, n_channels, type_ids)

        assert global_summary.shape == (batch, 64)  # global_summary_dim
        assert per_channel.shape == (batch, MAX_CHANNELS, net.per_channel_dim)
        assert ch_mask.shape == (batch, MAX_CHANNELS)

    def test_channel_mask_correct(self, net):
        batch = 2
        y = torch.randn(batch, 52)
        spend = torch.randn(batch, 52, MAX_CHANNELS)
        n_channels = torch.tensor([3, 7])
        type_ids = torch.zeros(batch, MAX_CHANNELS, dtype=torch.long)

        _, _, ch_mask = net(y, spend, n_channels, type_ids)

        assert ch_mask[0, :3].all()
        assert not ch_mask[0, 3:].any()
        assert ch_mask[1, :7].all()
        assert not ch_mask[1, 7:].any()

    def test_inactive_channels_zeroed(self, net):
        batch = 2
        y = torch.randn(batch, 52)
        spend = torch.randn(batch, 52, MAX_CHANNELS)
        n_channels = torch.tensor([2, 2])
        type_ids = torch.zeros(batch, MAX_CHANNELS, dtype=torch.long)

        _, per_channel, _ = net(y, spend, n_channels, type_ids)

        # Inactive channel embeddings should be zero
        assert torch.all(per_channel[:, 2:, :] == 0.0)

    def test_gradients_flow(self, net):
        y = torch.randn(2, 52, requires_grad=True)
        spend = torch.randn(2, 52, MAX_CHANNELS)
        n_channels = torch.tensor([3, 5])
        type_ids = torch.zeros(2, MAX_CHANNELS, dtype=torch.long)

        global_summary, _, _ = net(y, spend, n_channels, type_ids)
        loss = global_summary.sum()
        loss.backward()
        assert y.grad is not None

    def test_variable_time_lengths(self, net):
        for T in [26, 52, 104, 260]:
            y = torch.randn(2, T)
            spend = torch.randn(2, T, MAX_CHANNELS)
            n_channels = torch.tensor([3, 5])
            type_ids = torch.zeros(2, MAX_CHANNELS, dtype=torch.long)

            global_summary, per_channel, _ = net(y, spend, n_channels, type_ids)
            assert global_summary.shape == (2, 64)
