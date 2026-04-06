"""Window-based contribution engine — training loop, loss, and inference.

Trains the WindowContributionModel on overlapping windows extracted from
simulated scenarios. Each window has stable parameters (betas, saturation,
adstock), enabling per-channel contribution prediction via cross-channel
attention.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from demantiq.neural.data_loader import DemantiqDataset
from demantiq.neural.window_dataset import WindowDataset, create_window_dataloader
from demantiq.neural.window_encoder import WindowContributionModel

logger = logging.getLogger(__name__)


@dataclass
class WindowContributionConfig:
    # Architecture
    feature_dim: int = 64
    n_attention_heads: int = 4
    n_attention_layers: int = 2
    max_channels: int = 8
    n_context_dims: int = 10
    dropout: float = 0.1

    # Window extraction
    window_size: int = 16
    stride: int = 8
    context_dropout: float = 0.3

    # Training
    batch_size: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    n_epochs: int = 100
    patience: int = 20
    grad_clip: float = 1.0
    val_fraction: float = 0.1

    # Loss weights
    reconstruction_weight_start: float = 0.1
    reconstruction_weight_end: float = 1.0
    reconstruction_anneal_epochs: int = 20

    # Data
    n_train: int = 100000
    data_dir: str = "training_data"


def _reconstruction_weight(epoch: int, config: WindowContributionConfig) -> float:
    """Anneal reconstruction weight from start to end over anneal_epochs."""
    if epoch >= config.reconstruction_anneal_epochs:
        return config.reconstruction_weight_end
    t = epoch / max(config.reconstruction_anneal_epochs, 1)
    return config.reconstruction_weight_start + t * (
        config.reconstruction_weight_end - config.reconstruction_weight_start
    )


def window_contribution_loss(
    pred_channels: torch.Tensor,    # (B, W, max_channels)
    pred_baseline: torch.Tensor,    # (B, W)
    pred_non_media: torch.Tensor,   # (B, W)
    true_channels: torch.Tensor,    # (B, W, max_channels)
    true_baseline: torch.Tensor,    # (B, W)
    true_non_media: torch.Tensor,   # (B, W)
    y: torch.Tensor,                # (B, W)
    channel_pad_mask: torch.Tensor, # (B, max_channels) True=padded
    recon_weight: float,
) -> tuple[torch.Tensor, dict]:
    """Compute combined loss for window contribution prediction.

    Returns total loss and a dict of individual loss components for logging.
    """
    B, W, C = pred_channels.shape
    active_mask = (~channel_pad_mask).float()  # (B, C) 1=active, 0=padded

    # Per-channel MSE, normalized by each channel's scale
    ch_scale = (
        true_channels.abs().mean(dim=1).clamp(min=1.0)  # (B, C)
    )
    ch_diff = (pred_channels - true_channels) / ch_scale.unsqueeze(1)  # (B, W, C)
    ch_diff = ch_diff * active_mask.unsqueeze(1)  # zero out padded
    n_active = active_mask.sum().clamp(min=1)
    l_channels = (ch_diff ** 2).sum() / (n_active * W)

    # Baseline MSE, normalized by scale
    base_scale = true_baseline.abs().mean(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
    l_baseline = (((pred_baseline - true_baseline) / base_scale) ** 2).mean()

    # Non-media MSE, normalized by scale
    nm_scale = true_non_media.abs().mean(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
    l_non_media = (((pred_non_media - true_non_media) / nm_scale) ** 2).mean()

    # Reconstruction: predicted sum should equal y
    active_contribs = pred_channels * active_mask.unsqueeze(1)  # (B, W, C)
    y_hat = pred_baseline + active_contribs.sum(dim=-1) + pred_non_media  # (B, W)
    y_scale = y.abs().mean(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
    l_recon = (((y_hat - y) / y_scale) ** 2).mean()

    total = l_channels + l_baseline + l_non_media + recon_weight * l_recon

    components = {
        "channels": l_channels.item(),
        "baseline": l_baseline.item(),
        "non_media": l_non_media.item(),
        "reconstruction": l_recon.item(),
        "total": total.item(),
    }
    return total, components


class WindowContributionEngine:
    """Training loop and inference for window-based contribution model."""

    def __init__(self, config: WindowContributionConfig | None = None):
        self.config = config or WindowContributionConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = WindowContributionModel(
            feature_dim=self.config.feature_dim,
            n_attention_heads=self.config.n_attention_heads,
            n_attention_layers=self.config.n_attention_layers,
            max_channels=self.config.max_channels,
            n_context_dims=self.config.n_context_dims,
            dropout=self.config.dropout,
        ).to(self.device)

        self.optimizer = None
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def train(self, data_dir: str | None = None, n_train: int | None = None) -> dict:
        """Train the window contribution model."""
        data_dir = data_dir or self.config.data_dir
        n_train = n_train or self.config.n_train

        logger.info("Loading training data from %s (max %d samples)", data_dir, n_train)
        backing = DemantiqDataset(data_dir, max_samples=n_train)
        logger.info("Loaded %d scenarios", len(backing))

        # Split by scenario index (not by window) to avoid leakage
        n_scenarios = len(backing)
        if n_scenarios == 1:
            # Special case: use the same scenario for train and val (overfit test)
            n_val = 0
            n_train_actual = 1
        else:
            n_val = max(1, int(n_scenarios * self.config.val_fraction))
            n_train_actual = n_scenarios - n_val

        indices = torch.randperm(n_scenarios, generator=torch.Generator().manual_seed(42))
        train_indices = indices[:n_train_actual].tolist()
        val_indices = indices[n_train_actual:].tolist()

        train_window_ds = _SubsetWindowDataset(
            backing, train_indices,
            window_size=self.config.window_size,
            stride=self.config.stride,
            max_channels=self.config.max_channels,
            context_dropout=self.config.context_dropout,
            training=True,
        )

        has_val = len(val_indices) > 0
        if has_val:
            val_window_ds = _SubsetWindowDataset(
                backing, val_indices,
                window_size=self.config.window_size,
                stride=self.config.stride,
                max_channels=self.config.max_channels,
                context_dropout=0.0,
                training=False,
            )
        else:
            val_window_ds = None

        logger.info("Windows: %d train, %d val",
                     len(train_window_ds), len(val_window_ds) if val_window_ds else 0)

        train_loader = DataLoader(
            train_window_ds, batch_size=self.config.batch_size, shuffle=True,
            num_workers=0, pin_memory=self.device.type == "cuda",
        )
        val_loader = None
        if val_window_ds and len(val_window_ds) > 0:
            val_loader = DataLoader(
                val_window_ds, batch_size=self.config.batch_size, shuffle=False,
                num_workers=0, pin_memory=self.device.type == "cuda",
            )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.n_epochs, eta_min=1e-6,
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            "Training: %d train windows / %d val windows, %d epochs, "
            "batch_size=%d, lr=%.1e, params=%d",
            len(train_window_ds), len(val_window_ds) if val_window_ds else 0,
            self.config.n_epochs,
            self.config.batch_size, self.config.learning_rate, n_params,
        )

        t_start = time.time()
        for epoch in range(self.config.n_epochs):
            recon_w = _reconstruction_weight(epoch, self.config)
            train_loss, train_components = self._train_epoch(train_loader, recon_w)

            if val_loader is not None:
                val_loss, val_components = self._validate(val_loader, recon_w)
            else:
                val_loss = train_loss
                val_components = train_components

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            scheduler.step()

            elapsed = time.time() - t_start
            lr = self.optimizer.param_groups[0]["lr"]

            if epoch % 5 == 0 or epoch == self.config.n_epochs - 1:
                logger.info(
                    "Epoch %3d/%d  train=%.4f (ch=%.4f bl=%.4f nm=%.4f rc=%.4f)  "
                    "val=%.4f  lr=%.1e  rw=%.2f  [%.0fs]",
                    epoch + 1, self.config.n_epochs,
                    train_loss, train_components["channels"],
                    train_components["baseline"], train_components["non_media"],
                    train_components["reconstruction"],
                    val_loss, lr, recon_w, elapsed,
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        if best_state:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        total_time = time.time() - t_start
        logger.info("Training complete: %.1f min, best_val=%.6f", total_time / 60, best_val_loss)
        return {
            "best_val_loss": best_val_loss,
            "final_train_loss": self.train_losses[-1],
            "n_epochs_trained": len(self.train_losses),
            "training_time_minutes": total_time / 60,
        }

    def _train_epoch(self, loader: DataLoader, recon_weight: float) -> tuple[float, dict]:
        self.model.train()
        total_loss = 0.0
        total_components = {}
        n_batches = 0

        for batch in loader:
            self.optimizer.zero_grad()
            loss, components = self._compute_loss(batch, recon_weight)
            loss.backward()

            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.optimizer.step()
            total_loss += loss.item()
            for k, v in components.items():
                total_components[k] = total_components.get(k, 0.0) + v
            n_batches += 1

        n = max(n_batches, 1)
        avg_components = {k: v / n for k, v in total_components.items()}
        return total_loss / n, avg_components

    @torch.no_grad()
    def _validate(self, loader: DataLoader, recon_weight: float) -> tuple[float, dict]:
        self.model.eval()
        total_loss = 0.0
        total_components = {}
        n_batches = 0

        for batch in loader:
            loss, components = self._compute_loss(batch, recon_weight)
            total_loss += loss.item()
            for k, v in components.items():
                total_components[k] = total_components.get(k, 0.0) + v
            n_batches += 1

        n = max(n_batches, 1)
        avg_components = {k: v / n for k, v in total_components.items()}
        return total_loss / n, avg_components

    def _compute_loss(self, batch: dict, recon_weight: float) -> tuple[torch.Tensor, dict]:
        spend = batch["spend_window"].to(self.device)
        y = batch["y_window"].to(self.device)
        context = batch["context_window"].to(self.device)
        context_mask = batch["context_mask"].to(self.device)
        channel_pad_mask = batch["channel_pad_mask"].to(self.device)
        true_channels = batch["true_channel_contributions"].to(self.device)
        true_baseline = batch["true_baseline"].to(self.device)
        true_non_media = batch["true_non_media"].to(self.device)

        out = self.model(spend, y, context, context_mask, channel_pad_mask)

        return window_contribution_loss(
            pred_channels=out["channel_contributions"],
            pred_baseline=out["baseline"],
            pred_non_media=out["non_media"],
            true_channels=true_channels,
            true_baseline=true_baseline,
            true_non_media=true_non_media,
            y=y,
            channel_pad_mask=channel_pad_mask,
            recon_weight=recon_weight,
        )

    @torch.no_grad()
    def infer(
        self,
        y: np.ndarray,
        spend: np.ndarray,
        context: np.ndarray,
        n_channels: int,
        context_present: np.ndarray | None = None,
    ) -> dict:
        """Run inference on a full scenario using sliding windows.

        Args:
            y: (T,) observed demand.
            spend: (T, n_channels) per-channel spend.
            context: (T, n_context_dims) context variables.
            n_channels: number of active channels.
            context_present: (n_context_dims,) presence flags. If None, inferred from data.

        Returns:
            Dict with per-channel contributions, baseline, non_media, y_hat.
        """
        self.model.eval()
        T = len(y)
        W = self.config.window_size
        C = self.config.max_channels

        if context_present is None:
            context_present = (np.abs(context).sum(axis=0) > 0).astype(np.float32)

        # Pad spend to max_channels
        if spend.shape[1] < C:
            spend_padded = np.zeros((T, C), dtype=np.float32)
            spend_padded[:, :n_channels] = spend[:, :n_channels]
        else:
            spend_padded = spend[:, :C].astype(np.float32)

        # Channel pad mask
        ch_pad = np.ones(C, dtype=bool)
        ch_pad[:n_channels] = False

        # Accumulate predictions with overlap averaging
        contrib_sum = np.zeros((T, C), dtype=np.float64)
        baseline_sum = np.zeros(T, dtype=np.float64)
        non_media_sum = np.zeros(T, dtype=np.float64)
        count = np.zeros(T, dtype=np.float64)

        # Sliding window with stride=1 for smooth inference
        for start in range(0, max(1, T - W + 1)):
            end = start + W
            if end > T:
                break

            # Build single-sample batch
            y_w = torch.from_numpy(y[start:end].astype(np.float32)).unsqueeze(0).to(self.device)
            sp_w = torch.from_numpy(spend_padded[start:end]).unsqueeze(0).to(self.device)
            ctx_w = torch.from_numpy(context[start:end].astype(np.float32)).unsqueeze(0).to(self.device)
            ctx_m = torch.from_numpy(context_present).unsqueeze(0).to(self.device)
            ch_m = torch.from_numpy(ch_pad).unsqueeze(0).to(self.device)

            out = self.model(sp_w, y_w, ctx_w, ctx_m, ch_m)

            contrib_sum[start:end] += out["channel_contributions"][0].cpu().numpy()
            baseline_sum[start:end] += out["baseline"][0].cpu().numpy()
            non_media_sum[start:end] += out["non_media"][0].cpu().numpy()
            count[start:end] += 1.0

        # Average overlapping predictions
        count = np.maximum(count, 1.0)
        channel_contributions = (contrib_sum / count[:, np.newaxis])[:, :n_channels]
        baseline = baseline_sum / count
        non_media = non_media_sum / count
        y_hat = baseline + channel_contributions.sum(axis=1) + non_media

        return {
            "channel_contributions": channel_contributions.astype(np.float32),
            "baseline": baseline.astype(np.float32),
            "non_media": non_media.astype(np.float32),
            "y_hat": y_hat.astype(np.float32),
        }

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out / "window_model.pt")
        (out / "window_config.json").write_text(json.dumps(asdict(self.config), indent=2))
        logger.info("Model saved to %s", path)

    def load(self, path: str | Path) -> None:
        out = Path(path)
        state = torch.load(out / "window_model.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        logger.info("Model loaded from %s", path)


class _SubsetWindowDataset(WindowDataset):
    """WindowDataset that only uses a subset of scenarios from the backing dataset."""

    def __init__(
        self,
        backing: DemantiqDataset,
        scenario_indices: list[int],
        window_size: int = 16,
        stride: int = 8,
        max_channels: int = 8,
        context_dropout: float = 0.3,
        training: bool = True,
    ):
        # Don't call super().__init__ — we build the window index ourselves
        # but set the same attributes that WindowDataset.__getitem__ expects
        self.window_size = window_size
        self.stride = stride
        self.max_channels = max_channels
        self.context_dropout = context_dropout
        self.training = training

        self._y = backing.y
        self._spend = backing.spend
        self._context = backing.context
        self._decomposition = backing.decomposition
        self._n_periods = backing.n_periods
        self._channel_names = backing.channel_names

        # Build window index from the subset of scenarios
        self._index: list[tuple[int, int]] = []
        for scenario_idx in scenario_indices:
            n_periods = int(backing.n_periods[scenario_idx])
            n_ch = len(backing.channel_names[scenario_idx])
            if n_periods < window_size or n_ch < 2 or n_ch > max_channels:
                continue
            n_windows = max(0, (n_periods - window_size) // stride + 1)
            for w in range(n_windows):
                start_t = w * stride
                self._index.append((scenario_idx, start_t))
