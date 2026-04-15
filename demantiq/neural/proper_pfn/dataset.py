"""Full-scenario dataset for the proper PFN architecture.

Produces:
  - channel_tokens: (T, max_channels, channel_feature_dim)
      Per-channel per-week features: [spend, impressions, clicks] normalized.
  - global_tokens: (T, global_feature_dim)
      Per-week features shared across channels: [y, context, presence_flags, time].
  - week_is_masked: (T,) float
      1.0 for held-out weeks during training, 0.0 otherwise.
  - channel_pad_mask: (max_channels,) bool
      True for padded (inactive) channels.
  - target: (T, max_channels + 2)
      Per-week shares: [ch1_share, ..., chC_share, baseline_share, non_media_share].

The masking zeros input observations for masked weeks (model can only use
time features and other weeks' observations to infer decomposition).
"""

from __future__ import annotations

import math
import numpy as np
import torch
from torch.utils.data import Dataset

from demantiq.neural.data_loader import DemantiqDataset
from demantiq.orchestration.training_format import (
    DECOMP_IDX_BASELINE,
    DECOMP_IDX_CHANNELS_START,
    DECOMP_IDX_PRICE,
    DECOMP_IDX_COMPETITION,
    DECOMP_IDX_MACRO,
    MAX_CONTEXT_COLS,
)


CHANNEL_FEATURE_DIM = 3  # spend, impressions, clicks (all normalized)


class ProperPFNDataset(Dataset):
    """Per-scenario dataset with channel-token structure and held-out masking."""

    def __init__(
        self,
        backing_dataset: DemantiqDataset,
        max_channels: int = 8,
        n_context_dims: int = MAX_CONTEXT_COLS,
        week_mask_fraction: float = 0.3,
        training: bool = True,
    ) -> None:
        self.backing = backing_dataset
        self.max_channels = max_channels
        self.n_context_dims = n_context_dims
        self.week_mask_fraction = week_mask_fraction
        self.training = training

        self._valid_indices = self._filter()

    def _filter(self) -> list[int]:
        valid: list[int] = []
        for i in range(len(self.backing.n_periods)):
            n_ch = len(self.backing.channel_names[i])
            if 2 <= n_ch <= self.max_channels:
                valid.append(i)
        return valid

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> dict:
        scenario_idx = self._valid_indices[idx]
        T = int(self.backing.n_periods[scenario_idx])
        n_ch = len(self.backing.channel_names[scenario_idx])

        # --- y and y_denom for share computation ---
        y_raw = self.backing.y[scenario_idx, :T]  # numpy
        y_scale = float(max(np.mean(np.abs(y_raw)), 1.0))
        y_denom = np.maximum(np.abs(y_raw), max(y_scale * 0.1, 1.0))

        # --- Channel tokens: (T, max_channels, 3) with [spend, impressions, clicks] ---
        spend_raw = self.backing.spend[scenario_idx, :T, :n_ch]
        imp_raw = self.backing.impressions[scenario_idx, :T, :n_ch]
        clk_raw = self.backing.clicks[scenario_idx, :T, :n_ch]

        def per_channel_normalize(arr: np.ndarray) -> np.ndarray:
            # arr shape: (T, n_ch)
            scales = np.maximum(np.abs(arr).max(axis=0, keepdims=True), 1.0)
            return arr / scales

        spend_norm = per_channel_normalize(spend_raw)
        imp_norm = per_channel_normalize(imp_raw)
        clk_norm = per_channel_normalize(clk_raw)

        channel_tokens = np.zeros((T, self.max_channels, CHANNEL_FEATURE_DIM), dtype=np.float32)
        channel_tokens[:, :n_ch, 0] = spend_norm
        channel_tokens[:, :n_ch, 1] = imp_norm
        channel_tokens[:, :n_ch, 2] = clk_norm

        # --- Context ---
        ctx_raw = self.backing.context[scenario_idx, :T, :].copy()  # (T, 10)
        context_present = (np.abs(ctx_raw).sum(axis=0) > 0).astype(np.float32)  # (10,)
        col_scales = np.maximum(np.abs(ctx_raw).max(axis=0, keepdims=True), 1.0)
        ctx_norm = ctx_raw / col_scales

        y_norm = (y_raw / y_scale).astype(np.float32)

        # --- Temporal features ---
        time_index = np.linspace(0.0, 1.0, T, dtype=np.float32)
        week_of_year = np.arange(T, dtype=np.float32) % 52.0
        sin_week = np.sin(2.0 * np.pi * week_of_year / 52.0)
        cos_week = np.cos(2.0 * np.pi * week_of_year / 52.0)

        # --- Week masking (training only) ---
        week_is_masked = np.zeros(T, dtype=np.float32)
        if self.training and self.week_mask_fraction > 0.0:
            n_mask = max(1, int(math.ceil(T * self.week_mask_fraction)))
            idxs = np.random.permutation(T)[:n_mask]
            week_is_masked[idxs] = 1.0

            # Zero out input observations for masked weeks. Time features remain visible.
            visible = 1.0 - week_is_masked
            channel_tokens = channel_tokens * visible[:, None, None]
            y_norm = y_norm * visible
            ctx_norm = ctx_norm * visible[:, None]

        # --- Assemble global tokens: (T, global_feature_dim) ---
        # [y(1), ctx(10), presence_flags(10), time_index(1), sin_week(1), cos_week(1), is_masked(1)] = 25
        presence_flags = np.tile(context_present, (T, 1))  # (T, 10)
        global_tokens = np.concatenate(
            [
                y_norm.reshape(T, 1),
                ctx_norm.astype(np.float32),
                presence_flags,
                time_index.reshape(T, 1),
                sin_week.reshape(T, 1),
                cos_week.reshape(T, 1),
                week_is_masked.reshape(T, 1),
            ],
            axis=1,
        )

        # --- Target: shares per timestep ---
        ch_start = DECOMP_IDX_CHANNELS_START
        ch_shares = np.zeros((T, self.max_channels), dtype=np.float32)
        ch_shares[:, :n_ch] = (
            self.backing.decomposition[scenario_idx, :T, ch_start : ch_start + n_ch]
            / y_denom[:, None]
        )

        base_share = (
            self.backing.decomposition[scenario_idx, :T, DECOMP_IDX_BASELINE] / y_denom
        ).astype(np.float32)

        nm_share = (
            (
                self.backing.decomposition[scenario_idx, :T, DECOMP_IDX_PRICE]
                + self.backing.decomposition[scenario_idx, :T, DECOMP_IDX_COMPETITION]
                + self.backing.decomposition[scenario_idx, :T, DECOMP_IDX_MACRO]
            )
            / y_denom
        ).astype(np.float32)

        target = np.concatenate(
            [ch_shares, base_share.reshape(T, 1), nm_share.reshape(T, 1)],
            axis=1,
        )  # (T, max_channels + 2)

        # --- Channel pad mask ---
        channel_pad_mask = np.ones(self.max_channels, dtype=bool)
        channel_pad_mask[:n_ch] = False

        return {
            "channel_tokens": torch.from_numpy(channel_tokens),
            "global_tokens": torch.from_numpy(global_tokens),
            "week_is_masked": torch.from_numpy(week_is_masked),
            "target": torch.from_numpy(target),
            "y_raw": torch.from_numpy(y_raw.astype(np.float32)),
            "y_denom": torch.from_numpy(y_denom.astype(np.float32)),
            "y_scale": torch.tensor(y_scale, dtype=torch.float32),
            "n_channels": torch.tensor(n_ch, dtype=torch.int32),
            "n_periods": torch.tensor(T, dtype=torch.int32),
            "channel_pad_mask": torch.from_numpy(channel_pad_mask),
        }


def proper_pfn_collate_fn(batch: list[dict]) -> dict:
    """Pad variable-length scenarios to batch max."""
    max_t = max(item["channel_tokens"].shape[0] for item in batch)
    b = len(batch)
    max_channels = batch[0]["channel_tokens"].shape[1]
    channel_feat = batch[0]["channel_tokens"].shape[2]
    global_feat = batch[0]["global_tokens"].shape[1]
    target_feat = batch[0]["target"].shape[1]

    channel_tokens = torch.zeros(b, max_t, max_channels, channel_feat, dtype=torch.float32)
    global_tokens = torch.zeros(b, max_t, global_feat, dtype=torch.float32)
    target = torch.zeros(b, max_t, target_feat, dtype=torch.float32)
    week_is_masked = torch.zeros(b, max_t, dtype=torch.float32)
    time_pad_mask = torch.ones(b, max_t, dtype=torch.bool)
    y_raw = torch.zeros(b, max_t, dtype=torch.float32)
    y_denom = torch.ones(b, max_t, dtype=torch.float32)
    y_scale = torch.zeros(b, dtype=torch.float32)
    n_channels = torch.zeros(b, dtype=torch.int32)
    n_periods = torch.zeros(b, dtype=torch.int32)
    channel_pad_mask = torch.zeros(b, max_channels, dtype=torch.bool)

    for i, item in enumerate(batch):
        t = item["channel_tokens"].shape[0]
        channel_tokens[i, :t] = item["channel_tokens"]
        global_tokens[i, :t] = item["global_tokens"]
        target[i, :t] = item["target"]
        week_is_masked[i, :t] = item["week_is_masked"]
        time_pad_mask[i, :t] = False  # valid
        y_raw[i, :t] = item["y_raw"]
        y_denom[i, :t] = item["y_denom"]
        y_scale[i] = item["y_scale"]
        n_channels[i] = item["n_channels"]
        n_periods[i] = item["n_periods"]
        channel_pad_mask[i] = item["channel_pad_mask"]

    return {
        "channel_tokens": channel_tokens,
        "global_tokens": global_tokens,
        "target": target,
        "week_is_masked": week_is_masked,
        "time_pad_mask": time_pad_mask,
        "y_raw": y_raw,
        "y_denom": y_denom,
        "y_scale": y_scale,
        "n_channels": n_channels,
        "n_periods": n_periods,
        "channel_pad_mask": channel_pad_mask,
    }
