"""Full-scenario Dataset for PFN-style transformer training.

Each item is one complete scenario with all its timesteps. Variable-length
scenarios are returned as-is and padded to batch-max length by pfn_collate_fn.
"""

from __future__ import annotations

import numpy as np
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


class PFNScenarioDataset(Dataset):
    """Full-scenario dataset for PFN transformer training.

    Each item is one complete scenario with all timesteps.
    Variable-length scenarios are padded to max_T with a padding mask.

    Args:
        backing_dataset: A loaded DemantiqDataset with arrays in memory.
        max_channels: Maximum number of active channels. Scenarios with
            fewer than 2 or more than max_channels are excluded.
        n_context_dims: Number of context dimensions expected (must match
            MAX_CONTEXT_COLS from training_format).
        context_dropout: Probability of zeroing a present context dim during
            training. Applied independently per dim, never drops all dims.
        training: When True, context dropout is active.
    """

    def __init__(
        self,
        backing_dataset: DemantiqDataset,
        max_channels: int = 8,
        n_context_dims: int = MAX_CONTEXT_COLS,
        context_dropout: float = 0.3,
        week_mask_fraction: float = 0.3,
        training: bool = True,
    ) -> None:
        self.max_channels = max_channels
        self.n_context_dims = n_context_dims
        self.context_dropout = context_dropout
        self.week_mask_fraction = week_mask_fraction
        self.training = training

        # Hold references only — no data copying
        self._y = backing_dataset.y
        self._spend = backing_dataset.spend
        self._context = backing_dataset.context
        self._decomposition = backing_dataset.decomposition
        self._n_periods = backing_dataset.n_periods
        self._channel_names = backing_dataset.channel_names

        self._valid_indices = self._filter_scenarios()

    def _filter_scenarios(self) -> list[int]:
        valid: list[int] = []
        for i in range(len(self._n_periods)):
            n_ch = len(self._channel_names[i])
            if 2 <= n_ch <= self.max_channels:
                valid.append(i)
        return valid

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> dict:
        scenario_idx = self._valid_indices[idx]
        n_periods = int(self._n_periods[scenario_idx])
        n_ch = len(self._channel_names[scenario_idx])

        # --- y ---
        y_raw = self._y[scenario_idx, :n_periods]  # (T,)
        y_scale = float(max(np.mean(np.abs(y_raw)), 1.0))
        y_norm = torch.from_numpy(y_raw.copy() / y_scale)  # (T,)

        # --- spend: (T, max_channels), normalized globally ---
        spend_raw = self._spend[scenario_idx, :n_periods, :n_ch]  # (T, n_ch)
        spend_scale = float(max(np.abs(spend_raw).max(), 1.0))
        spend_norm = torch.zeros(n_periods, self.max_channels, dtype=torch.float32)
        spend_norm[:, :n_ch] = torch.from_numpy(spend_raw.copy() / spend_scale)

        # --- context + presence flags + dropout ---
        ctx_raw = torch.from_numpy(
            self._context[scenario_idx, :n_periods, :].copy()
        )  # (T, n_context_dims)

        context_mask = (ctx_raw.abs().sum(dim=0) > 0).float()  # (n_context_dims,)

        if self.training and self.context_dropout > 0.0:
            present_indices = context_mask.nonzero(as_tuple=True)[0]
            if len(present_indices) > 0:
                drop_flags = torch.rand(len(present_indices)) < self.context_dropout
                if drop_flags.all():
                    keep_one = int(torch.randint(len(present_indices), (1,)).item())
                    drop_flags[keep_one] = False
                for i, pidx in enumerate(present_indices):
                    if drop_flags[i]:
                        ctx_raw[:, pidx] = 0.0
                        context_mask[pidx] = 0.0

        # Presence flags broadcast across T: (T, n_context_dims)
        presence_flags = context_mask.unsqueeze(0).expand(n_periods, -1)

        # --- temporal features ---
        time_index = torch.linspace(0.0, 1.0, n_periods, dtype=torch.float32)  # (T,)
        week_of_year = torch.arange(n_periods, dtype=torch.float32) % 52.0
        sin_week = torch.sin(2.0 * torch.pi * week_of_year / 52.0)
        cos_week = torch.cos(2.0 * torch.pi * week_of_year / 52.0)

        # Normalize context per-column
        col_scales = ctx_raw.abs().amax(dim=0).clamp(min=1.0)
        ctx_normed = ctx_raw / col_scales

        # --- BERT-style week masking (PFN held-out prediction) ---
        # During training: randomly mask a fraction of weeks. Masked weeks
        # have their content features zeroed, forcing the model to infer
        # decomposition for these weeks from context (unmasked weeks).
        # At inference: no masking — the model applies its learned in-context
        # inference skill to all weeks.
        week_is_masked = torch.zeros(n_periods, dtype=torch.float32)
        if self.training and self.week_mask_fraction > 0.0:
            n_mask = max(1, int(n_periods * self.week_mask_fraction))
            mask_indices = torch.randperm(n_periods)[:n_mask]
            week_is_masked[mask_indices] = 1.0

            # Zero out content features for masked weeks
            # (channel spend, y, context, presence flags)
            visible = 1.0 - week_is_masked  # (T,)
            spend_norm = spend_norm * visible.unsqueeze(1)
            y_norm = y_norm * visible
            ctx_normed = ctx_normed * visible.unsqueeze(1)

        # --- assemble input_features: (T, n_features) ---
        # Layout must match PFNDecompositionModel:
        # spend(max_channels) + y(1) + context(n_context_dims) +
        # presence_flags(n_context_dims) + time_index(1) + sin_week(1) + cos_week(1) + is_masked(1)
        input_features = torch.cat(
            [
                spend_norm,                                     # (T, max_channels)
                y_norm.unsqueeze(1),                            # (T, 1)
                ctx_normed,                                     # (T, n_context_dims)
                presence_flags,                                 # (T, n_context_dims)
                time_index.unsqueeze(1),                        # (T, 1)
                sin_week.unsqueeze(1),                          # (T, 1)
                cos_week.unsqueeze(1),                          # (T, 1)
                week_is_masked.unsqueeze(1),                    # (T, 1)
            ],
            dim=1,
        )  # (T, max_channels + 1 + 2*n_context_dims + 4)

        # --- target: per-timestep SHARES (component / y[t]) ---
        # Shares carry per-timestep signal and naturally sum to ~1, giving
        # the model a hard reconstruction constraint that varies across
        # scenarios. Clamp y in denominator to avoid instability when y → 0.
        y_raw_t = torch.from_numpy(y_raw.copy())  # (T,) original scale
        y_denom = y_raw_t.abs().clamp(min=max(y_scale * 0.1, 1.0))  # (T,)

        ch_start = DECOMP_IDX_CHANNELS_START
        ch_raw = self._decomposition[
            scenario_idx, :n_periods, ch_start : ch_start + n_ch
        ]  # (T, n_ch)
        channel_contribs = torch.zeros(n_periods, self.max_channels, dtype=torch.float32)
        channel_contribs[:, :n_ch] = (
            torch.from_numpy(ch_raw.copy()) / y_denom.unsqueeze(1)
        )

        baseline = (
            torch.from_numpy(
                self._decomposition[scenario_idx, :n_periods, DECOMP_IDX_BASELINE].copy()
            )
            / y_denom
        )  # (T,) — baseline share per timestep

        non_media = (
            (
                torch.from_numpy(
                    self._decomposition[scenario_idx, :n_periods, DECOMP_IDX_PRICE].copy()
                )
                + torch.from_numpy(
                    self._decomposition[scenario_idx, :n_periods, DECOMP_IDX_COMPETITION].copy()
                )
                + torch.from_numpy(
                    self._decomposition[scenario_idx, :n_periods, DECOMP_IDX_MACRO].copy()
                )
            )
            / y_denom
        )  # (T,) — non-media share per timestep

        target = torch.cat(
            [
                channel_contribs,             # (T, max_channels)
                baseline.unsqueeze(1),        # (T, 1)
                non_media.unsqueeze(1),       # (T, 1)
            ],
            dim=1,
        )  # (T, max_channels + 2)

        # --- channel pad mask: True = padded (inactive) ---
        channel_pad_mask = torch.ones(self.max_channels, dtype=torch.bool)
        channel_pad_mask[:n_ch] = False

        return {
            "input_features": input_features.float(),
            "target": target.float(),
            "week_is_masked": week_is_masked.float(),  # (T,) 1=masked, 0=visible
            "y_scale": torch.tensor(y_scale, dtype=torch.float32),
            "n_channels": torch.tensor(n_ch, dtype=torch.int32),
            "n_periods": torch.tensor(n_periods, dtype=torch.int32),
            "channel_pad_mask": channel_pad_mask,
        }


def pfn_collate_fn(batch: list[dict]) -> dict:
    """Pad variable-length scenarios to batch max and create padding masks."""
    max_t = max(item["input_features"].shape[0] for item in batch)
    b = len(batch)

    n_features = batch[0]["input_features"].shape[1]
    n_outputs = batch[0]["target"].shape[1]
    max_channels = batch[0]["channel_pad_mask"].shape[0]

    input_features = torch.zeros(b, max_t, n_features, dtype=torch.float32)
    target = torch.zeros(b, max_t, n_outputs, dtype=torch.float32)
    time_pad_mask = torch.ones(b, max_t, dtype=torch.bool)  # True = padded
    week_is_masked = torch.zeros(b, max_t, dtype=torch.float32)  # 1 = masked week

    y_scales = torch.zeros(b, dtype=torch.float32)
    n_channels = torch.zeros(b, dtype=torch.int32)
    n_periods = torch.zeros(b, dtype=torch.int32)
    channel_pad_mask = torch.zeros(b, max_channels, dtype=torch.bool)

    for i, item in enumerate(batch):
        t = item["input_features"].shape[0]
        input_features[i, :t] = item["input_features"]
        target[i, :t] = item["target"]
        time_pad_mask[i, :t] = False  # valid timesteps
        week_is_masked[i, :t] = item["week_is_masked"]
        y_scales[i] = item["y_scale"]
        n_channels[i] = item["n_channels"]
        n_periods[i] = item["n_periods"]
        channel_pad_mask[i] = item["channel_pad_mask"]

    return {
        "input_features": input_features,
        "target": target,
        "time_pad_mask": time_pad_mask,
        "week_is_masked": week_is_masked,
        "y_scale": y_scales,
        "n_channels": n_channels,
        "n_periods": n_periods,
        "channel_pad_mask": channel_pad_mask,
    }


def create_pfn_dataloader(
    data_dir: str,
    max_channels: int = 8,
    batch_size: int = 32,
    shuffle: bool = True,
    max_samples: int | None = None,
    context_dropout: float = 0.3,
    training: bool = True,
) -> DataLoader:
    """Create a DataLoader that yields full-scenario batches for PFN training.

    Loads all scenarios from data_dir into a DemantiqDataset, wraps it in a
    PFNScenarioDataset, and returns a DataLoader using pfn_collate_fn for
    variable-length padding.

    Args:
        data_dir: Directory containing batch_*.npz files.
        max_channels: Maximum active channels per scenario (others excluded).
        batch_size: Number of scenarios per batch.
        shuffle: Whether to shuffle scenarios across batches.
        max_samples: Optional cap on scenarios loaded from disk.
        context_dropout: Probability of dropping a present context dim.
        training: When True, context dropout is applied.

    Returns:
        A PyTorch DataLoader yielding dicts of padded batched tensors.
    """
    backing = DemantiqDataset(data_dir, max_samples=max_samples)
    dataset = PFNScenarioDataset(
        backing_dataset=backing,
        max_channels=max_channels,
        context_dropout=context_dropout,
        training=training,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=pfn_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
