"""Utility functions for the neural inference engine.

Normalization, masking, and vector conversion helpers.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from demantiq.orchestration.training_format import (
    MAX_CHANNELS,
    _GLOBAL_EXT_TRUTH_LEN,
    _PER_CHANNEL_EXT_TRUTH_LEN,
    EXT_TRUTH_VECTOR_LEN,
)

# Channel name to index mapping (matches scenario_sampler._CHANNEL_NAMES)
CHANNEL_NAMES = [
    "facebook", "google", "tiktok", "pinterest", "email",
    "youtube", "snapchat", "linkedin", "twitter", "display",
    "programmatic", "podcast", "influencer", "radio", "ctv",
    "affiliate", "sms", "direct_mail", "ooh", "print",
]
CHANNEL_NAME_TO_IDX = {name: i for i, name in enumerate(CHANNEL_NAMES)}
NUM_CHANNEL_TYPES = len(CHANNEL_NAMES)


def make_channel_mask(n_channels: int | Tensor, max_channels: int = MAX_CHANNELS) -> Tensor:
    """Create a boolean mask for active channels.

    Args:
        n_channels: Number of active channels (scalar or batch tensor).
        max_channels: Maximum number of channels.

    Returns:
        Boolean tensor of shape (max_channels,) or (batch, max_channels).
    """
    if isinstance(n_channels, (int, float)):
        mask = torch.zeros(max_channels, dtype=torch.bool)
        mask[:int(n_channels)] = True
        return mask

    # Batched
    batch_size = n_channels.shape[0]
    indices = torch.arange(max_channels, device=n_channels.device).unsqueeze(0)
    mask = indices < n_channels.unsqueeze(1).int()
    return mask


def normalize_time_series(x: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """Z-score normalize a time series per sample.

    Args:
        x: Tensor of shape (batch, T) or (batch, T, C).
        mask: Optional boolean tensor of shape (batch, T) indicating valid timesteps.

    Returns:
        Tuple of (normalized_x, mean, std).
    """
    if mask is not None:
        # Masked normalization
        mask_expanded = mask.unsqueeze(-1) if x.dim() == 3 else mask
        x_masked = x * mask_expanded.float()
        counts = mask_expanded.float().sum(dim=1, keepdim=True).clamp(min=1)
        mean = x_masked.sum(dim=1, keepdim=True) / counts
        var = ((x_masked - mean * mask_expanded.float()) ** 2).sum(dim=1, keepdim=True) / counts
        std = var.sqrt().clamp(min=1e-8)
        normalized = (x - mean) / std * mask_expanded.float()
    else:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp(min=1e-8)
        normalized = (x - mean) / std

    return normalized, mean.squeeze(1), std.squeeze(1)


def get_channel_type_indices(channel_names_batch: list[list[str]]) -> Tensor:
    """Convert channel name lists to type index tensors.

    Args:
        channel_names_batch: List of channel name lists, one per sample.

    Returns:
        Long tensor of shape (batch, max_channels) with channel type indices.
        Unknown channels get index 0.
    """
    batch_size = len(channel_names_batch)
    indices = torch.zeros(batch_size, MAX_CHANNELS, dtype=torch.long)
    for b, names in enumerate(channel_names_batch):
        for i, name in enumerate(names):
            if i >= MAX_CHANNELS:
                break
            indices[b, i] = CHANNEL_NAME_TO_IDX.get(name, 0)
    return indices


def extract_per_channel_truth(
    ext_truth: Tensor, n_channels: Tensor
) -> tuple[Tensor, Tensor]:
    """Extract per-channel and global truth from extended truth vector.

    Args:
        ext_truth: Extended truth tensor of shape (batch, EXT_TRUTH_VECTOR_LEN).
        n_channels: Integer tensor of shape (batch,) with active channel counts.

    Returns:
        Tuple of:
            global_truth: (batch, _GLOBAL_EXT_TRUTH_LEN)
            per_channel_truth: (batch, MAX_CHANNELS, _PER_CHANNEL_EXT_TRUTH_LEN)
    """
    batch_size = ext_truth.shape[0]
    global_truth = ext_truth[:, :_GLOBAL_EXT_TRUTH_LEN]

    per_channel = ext_truth[:, _GLOBAL_EXT_TRUTH_LEN:].reshape(
        batch_size, MAX_CHANNELS, _PER_CHANNEL_EXT_TRUTH_LEN
    )

    return global_truth, per_channel
