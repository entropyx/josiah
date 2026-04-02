"""PyTorch Dataset and DataLoader for Demantiq training data.

Reads .npz batch files produced by the TrainingPipeline and yields
(observable, truth, metadata) tuples for training the neural inference engine.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from demantiq.orchestration.training_format import (
    MAX_CHANNELS,
    MAX_CONTEXT_COLS,
    EXT_TRUTH_VECTOR_LEN,
    CONFIG_VECTOR_LEN,
    DECOMP_COLS,
)


class DemantiqDataset(Dataset):
    """Dataset that reads pre-generated .npz training batches.

    Each item returns a dict with:
        - y: (T,) outcome time series (float32)
        - spend: (T, max_channels) spend matrix (float32)
        - context: (T, MAX_CONTEXT_COLS) business context matrix (float32)
        - n_periods: scalar int, actual number of valid timesteps
        - n_channels: scalar int, number of active channels
        - channel_type_ids: (max_channels,) long tensor of channel type indices
        - ext_truth: (EXT_TRUTH_VECTOR_LEN,) extended truth vector (float32)
        - config_vector: (CONFIG_VECTOR_LEN,) config vector (float32)

    Args:
        data_dir: Directory containing batch_*.npz files.
        max_samples: Optional limit on total samples loaded.
    """

    def __init__(self, data_dir: str, max_samples: int | None = None):
        self.data_dir = Path(data_dir)
        self._load_all_batches(max_samples)

    def _load_all_batches(self, max_samples: int | None) -> None:
        """Load all .npz files into memory as concatenated arrays."""
        npz_files = sorted(self.data_dir.glob("batch_*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No batch_*.npz files found in {self.data_dir}")

        all_y = []
        all_spend = []
        all_context = []
        all_decomp = []
        all_config = []
        all_truth = []
        all_ext_truth = []
        all_channel_names = []
        all_n_periods = []

        total_loaded = 0
        for npz_path in npz_files:
            if max_samples is not None and total_loaded >= max_samples:
                break

            data = np.load(str(npz_path), allow_pickle=False)

            # Load metadata for channel names and actual periods
            meta_path = npz_path.with_name(npz_path.stem + "_meta.json")
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                batch_channel_names = meta["channel_names"]
                batch_n_periods = meta["n_periods"]
            else:
                batch_size = data["y"].shape[0]
                batch_channel_names = [[] for _ in range(batch_size)]
                batch_n_periods = [data["y"].shape[1]] * batch_size

            n_in_batch = data["y"].shape[0]
            if max_samples is not None:
                n_in_batch = min(n_in_batch, max_samples - total_loaded)

            all_y.append(data["y"][:n_in_batch])
            all_spend.append(data["spend"][:n_in_batch])
            all_config.append(data["config_vectors"][:n_in_batch])
            all_truth.append(data["truth_vectors"][:n_in_batch])

            if "ext_truth_vectors" in data:
                all_ext_truth.append(data["ext_truth_vectors"][:n_in_batch])
            else:
                all_ext_truth.append(
                    np.zeros((n_in_batch, EXT_TRUTH_VECTOR_LEN), dtype=np.float64)
                )

            # Business context (backward compatible — zero if absent)
            if "context" in data:
                all_context.append(data["context"][:n_in_batch])
            else:
                t_dim = data["y"].shape[1]
                all_context.append(
                    np.zeros((n_in_batch, t_dim, MAX_CONTEXT_COLS), dtype=np.float64)
                )

            # Per-period decomposition (backward compatible — zero if absent)
            if "decomposition" in data:
                all_decomp.append(data["decomposition"][:n_in_batch])
            else:
                t_dim = data["y"].shape[1]
                all_decomp.append(
                    np.zeros((n_in_batch, t_dim, DECOMP_COLS), dtype=np.float64)
                )

            all_channel_names.extend(batch_channel_names[:n_in_batch])
            all_n_periods.extend(batch_n_periods[:n_in_batch])

            total_loaded += n_in_batch

        # Pad to uniform dimensions across batches
        max_t = max(a.shape[1] for a in all_y)
        max_c = max(a.shape[2] for a in all_spend) if all_spend else 1

        y_arrays = []
        spend_arrays = []
        context_arrays = []
        decomp_arrays = []
        for y_arr, sp_arr, ctx_arr, dec_arr in zip(all_y, all_spend, all_context, all_decomp):
            # Pad time dimension
            pad_t = max_t - y_arr.shape[1]
            if pad_t > 0:
                y_arr = np.pad(y_arr, ((0, 0), (0, pad_t)))
            y_arrays.append(y_arr)

            pad_t_sp = max_t - sp_arr.shape[1]
            pad_c = max_c - sp_arr.shape[2]
            if pad_t_sp > 0 or pad_c > 0:
                sp_arr = np.pad(sp_arr, ((0, 0), (0, pad_t_sp), (0, pad_c)))
            spend_arrays.append(sp_arr)

            pad_t_ctx = max_t - ctx_arr.shape[1]
            if pad_t_ctx > 0:
                ctx_arr = np.pad(ctx_arr, ((0, 0), (0, pad_t_ctx), (0, 0)))
            # Pad context columns if needed
            if ctx_arr.shape[2] < MAX_CONTEXT_COLS:
                ctx_arr = np.pad(
                    ctx_arr, ((0, 0), (0, 0), (0, MAX_CONTEXT_COLS - ctx_arr.shape[2]))
                )
            context_arrays.append(ctx_arr)

            pad_t_dec = max_t - dec_arr.shape[1]
            if pad_t_dec > 0:
                dec_arr = np.pad(dec_arr, ((0, 0), (0, pad_t_dec), (0, 0)))
            decomp_arrays.append(dec_arr)

        self.y = np.concatenate(y_arrays, axis=0).astype(np.float32)
        self.spend = np.concatenate(spend_arrays, axis=0).astype(np.float32)
        self.context = np.concatenate(context_arrays, axis=0).astype(np.float32)
        self.decomposition = np.concatenate(decomp_arrays, axis=0).astype(np.float32)
        self.config_vectors = np.concatenate(all_config, axis=0).astype(np.float32)
        self.truth_vectors = np.concatenate(all_truth, axis=0).astype(np.float32)
        self.ext_truth_vectors = np.concatenate(all_ext_truth, axis=0).astype(np.float32)
        self.channel_names = all_channel_names
        self.n_periods = np.array(all_n_periods, dtype=np.int32)

        # Precompute channel type indices
        from demantiq.neural.utils import CHANNEL_NAME_TO_IDX
        self.channel_type_ids = np.zeros(
            (len(self.channel_names), MAX_CHANNELS), dtype=np.int64
        )
        for b, names in enumerate(self.channel_names):
            for i, name in enumerate(names):
                if i < MAX_CHANNELS:
                    self.channel_type_ids[b, i] = CHANNEL_NAME_TO_IDX.get(name, 0)

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> dict:
        n_ch = len(self.channel_names[idx])
        return {
            "y": torch.from_numpy(self.y[idx]),
            "spend": torch.from_numpy(self.spend[idx]),
            "context": torch.from_numpy(self.context[idx]),
            "decomposition": torch.from_numpy(self.decomposition[idx]),
            "n_periods": self.n_periods[idx],
            "n_channels": n_ch,
            "channel_type_ids": torch.from_numpy(self.channel_type_ids[idx]),
            "ext_truth": torch.from_numpy(self.ext_truth_vectors[idx]),
            "config_vector": torch.from_numpy(self.config_vectors[idx]),
        }


def create_dataloader(
    data_dir: str,
    batch_size: int = 256,
    shuffle: bool = True,
    max_samples: int | None = None,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for Demantiq training data.

    Args:
        data_dir: Directory containing batch_*.npz files.
        batch_size: Batch size for training.
        shuffle: Whether to shuffle data.
        max_samples: Optional limit on total samples.
        num_workers: Number of data loading workers.

    Returns:
        PyTorch DataLoader instance.
    """
    dataset = DemantiqDataset(data_dir, max_samples=max_samples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
