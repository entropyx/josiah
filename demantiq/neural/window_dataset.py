"""Windowed Dataset for Marketing Mix Model neural training.

Wraps a DemantiqDataset and yields fixed-size overlapping windows with per-channel
ground truth contributions. This allows the model to train on temporal subsequences
rather than full scenarios, which improves sample efficiency and enables batching
across scenarios with different lengths.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader

from demantiq.neural.data_loader import DemantiqDataset
from demantiq.orchestration.training_format import (
    DECOMP_IDX_BASELINE,
    DECOMP_IDX_CHANNELS_START,
    DECOMP_IDX_PRICE,
    DECOMP_IDX_COMPETITION,
    DECOMP_IDX_MACRO,
    MAX_CONTEXT_COLS,
)


class WindowDataset(Dataset):
    """Extracts overlapping windows from pre-loaded scenario data.

    Wraps a DemantiqDataset. Each __getitem__ returns a fixed-size window
    with per-channel ground truth contributions.

    The index is precomputed as a flat list of (scenario_idx, start_t) pairs
    so __getitem__ is O(1) with no scanning. All tensor slicing is done lazily
    from the backing numpy arrays — no data is copied at construction time.

    Args:
        backing_dataset: A loaded DemantiqDataset with arrays in memory.
        window_size: Number of timesteps per window (W).
        stride: Step between consecutive window start positions.
        max_channels: Maximum number of channels in any returned window.
            Scenarios with more active channels than this are excluded.
            Padded dimensions are filled with zeros.
        context_dropout: Probability of zeroing out a present context dimension
            during training. Applied per-dim independently.
        training: When True, context dropout is active.
    """

    def __init__(
        self,
        backing_dataset: DemantiqDataset,
        window_size: int = 16,
        stride: int = 8,
        max_channels: int = 8,
        context_dropout: float = 0.3,
        training: bool = True,
    ) -> None:
        self.window_size = window_size
        self.stride = stride
        self.max_channels = max_channels
        self.context_dropout = context_dropout
        self.training = training

        # Hold references only — no data copying
        self._y = backing_dataset.y
        self._spend = backing_dataset.spend
        self._context = backing_dataset.context
        self._decomposition = backing_dataset.decomposition
        self._n_periods = backing_dataset.n_periods
        self._channel_names = backing_dataset.channel_names

        self._index = self._build_index()

    def _build_index(self) -> list[tuple[int, int]]:
        """Precompute all valid (scenario_idx, start_t) window positions.

        A window is valid if:
          - The scenario has at least window_size valid periods.
          - The scenario has between 2 and max_channels active channels (inclusive).
        """
        index: list[tuple[int, int]] = []
        for scenario_idx in range(len(self._n_periods)):
            n_ch = len(self._channel_names[scenario_idx])
            n_periods = int(self._n_periods[scenario_idx])

            # Filter by channel count: need at least 2 channels, no more than max_channels
            if n_ch < 2 or n_ch > self.max_channels:
                continue

            # Skip scenarios that don't fill even one window
            if n_periods < self.window_size:
                continue

            n_windows = (n_periods - self.window_size) // self.stride + 1
            for w in range(n_windows):
                start_t = w * self.stride
                index.append((scenario_idx, start_t))

        return index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        scenario_idx, start_t = self._index[idx]
        end_t = start_t + self.window_size
        n_ch = len(self._channel_names[scenario_idx])

        # --- y window ---
        y_window = torch.from_numpy(self._y[scenario_idx, start_t:end_t].copy())

        # --- spend window: take first n_ch columns, pad to max_channels ---
        spend_raw = self._spend[scenario_idx, start_t:end_t, :n_ch]  # (W, n_ch)
        spend_window = torch.zeros(self.window_size, self.max_channels, dtype=torch.float32)
        spend_window[:, :n_ch] = torch.from_numpy(spend_raw.copy())

        # --- context window ---
        context_raw = self._context[scenario_idx, start_t:end_t, :]  # (W, 10)
        context_window = torch.from_numpy(context_raw.copy())

        # --- context mask: 1 if any nonzero value exists in the window ---
        # Shape: (10,) — one flag per context dimension
        context_mask = (context_window.abs().sum(dim=0) > 0).float()

        # Context dropout: randomly zero present dims during training
        if self.training and self.context_dropout > 0.0:
            present_indices = context_mask.nonzero(as_tuple=True)[0]
            if len(present_indices) > 0:
                # Independent Bernoulli drop per present dim
                drop_flags = torch.rand(len(present_indices)) < self.context_dropout
                # Ensure at least one dim is kept
                if drop_flags.all():
                    keep_one = torch.randint(len(present_indices), (1,)).item()
                    drop_flags[keep_one] = False
                for i, pidx in enumerate(present_indices):
                    if drop_flags[i]:
                        context_window[:, pidx] = 0.0
                        context_mask[pidx] = 0.0

        # --- true channel contributions: decomp cols 1..1+n_ch, padded ---
        ch_start = DECOMP_IDX_CHANNELS_START
        ch_raw = self._decomposition[
            scenario_idx, start_t:end_t, ch_start : ch_start + n_ch
        ]  # (W, n_ch)
        true_channel_contributions = torch.zeros(
            self.window_size, self.max_channels, dtype=torch.float32
        )
        true_channel_contributions[:, :n_ch] = torch.from_numpy(ch_raw.copy())

        # --- true baseline: decomp column 0 ---
        true_baseline = torch.from_numpy(
            self._decomposition[scenario_idx, start_t:end_t, DECOMP_IDX_BASELINE].copy()
        )

        # --- true non-media: price (21) + competition (23) + macro (24) ---
        # Skipping distribution_cap (22) and noise (25) as specified
        true_non_media = (
            torch.from_numpy(
                self._decomposition[scenario_idx, start_t:end_t, DECOMP_IDX_PRICE].copy()
            )
            + torch.from_numpy(
                self._decomposition[scenario_idx, start_t:end_t, DECOMP_IDX_COMPETITION].copy()
            )
            + torch.from_numpy(
                self._decomposition[scenario_idx, start_t:end_t, DECOMP_IDX_MACRO].copy()
            )
        )

        # --- channel pad mask: True = padded (inactive) position ---
        # Used as key_padding_mask in attention (True means "ignore this position")
        channel_pad_mask = torch.ones(self.max_channels, dtype=torch.bool)
        channel_pad_mask[:n_ch] = False

        return {
            "y_window": y_window,
            "spend_window": spend_window,
            "context_window": context_window,
            "context_mask": context_mask,
            "true_channel_contributions": true_channel_contributions,
            "true_baseline": true_baseline,
            "true_non_media": true_non_media,
            "n_channels": n_ch,
            "channel_pad_mask": channel_pad_mask,
        }


def create_window_dataloader(
    data_dir: str,
    window_size: int = 16,
    stride: int = 8,
    max_channels: int = 8,
    batch_size: int = 512,
    shuffle: bool = True,
    max_samples: int | None = None,
    context_dropout: float = 0.3,
    training: bool = True,
) -> DataLoader:
    """Create a DataLoader that yields window batches.

    Loads all scenarios from data_dir into a DemantiqDataset, wraps it in a
    WindowDataset to produce overlapping fixed-size windows, then returns a
    DataLoader over those windows.

    Args:
        data_dir: Directory containing batch_*.npz files.
        window_size: Number of timesteps per window.
        stride: Step between consecutive window start positions.
        max_channels: Maximum active channels per scenario (others excluded).
        batch_size: Number of windows per batch.
        shuffle: Whether to shuffle windows across batches.
        max_samples: Optional cap on scenarios loaded from disk.
        context_dropout: Probability of dropping a present context dim per window.
        training: When True, context dropout is applied in WindowDataset.

    Returns:
        A PyTorch DataLoader yielding dicts of batched window tensors.
    """
    backing = DemantiqDataset(data_dir, max_samples=max_samples)
    dataset = WindowDataset(
        backing_dataset=backing,
        window_size=window_size,
        stride=stride,
        max_channels=max_channels,
        context_dropout=context_dropout,
        training=training,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
