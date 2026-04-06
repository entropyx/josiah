"""PFN training engine — trains full-sequence transformer on multi-scenario data.

Each scenario is processed as a complete sequence (26-260 weeks). The transformer
learns in-context inference: given a scenario's observable data, it predicts
per-channel contributions, baseline, and non-media effects.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from demantiq.neural.data_loader import DemantiqDataset
from demantiq.neural.pfn_dataset import PFNScenarioDataset, pfn_collate_fn
from demantiq.neural.pfn_model import PFNDecompositionModel, build_pfn_input

logger = logging.getLogger(__name__)


@dataclass
class PFNConfig:
    # Architecture
    max_channels: int = 8
    n_context_dims: int = 10
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 6
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    n_epochs: int = 100
    patience: int = 20
    grad_clip: float = 1.0
    val_fraction: float = 0.1
    context_dropout: float = 0.3

    # Data
    n_train: int = 100000
    data_dir: str = "training_data"


def pfn_loss(
    pred: dict[str, torch.Tensor],
    target: torch.Tensor,
    time_pad_mask: torch.Tensor,
    channel_pad_mask: torch.Tensor,
    max_channels: int,
) -> tuple[torch.Tensor, dict]:
    """Compute loss for PFN decomposition predictions.

    All values are in normalized space (divided by y_scale).
    """
    B, T, _ = target.shape
    ch_active = (~channel_pad_mask).float()  # (B, C) 1=active

    pred_ch = pred["channel_contributions"]  # (B, T, C)
    pred_base = pred["baseline"]             # (B, T)
    pred_nm = pred["non_media"]              # (B, T)

    true_ch = target[:, :, :max_channels]
    true_base = target[:, :, max_channels]
    true_nm = target[:, :, max_channels + 1]

    # Time mask: (B, T) 1=valid, 0=padded
    time_valid = (~time_pad_mask).float()
    n_valid = time_valid.sum().clamp(min=1)

    # Per-channel MSE (normalized by channel scale, masked for active + valid timesteps)
    ch_scale = (true_ch.abs() * time_valid.unsqueeze(-1)).sum(dim=1) / time_valid.sum(dim=1, keepdim=True).clamp(min=1)  # (B, C)
    ch_scale = ch_scale.clamp(min=0.01)
    ch_diff = (pred_ch - true_ch) / ch_scale.unsqueeze(1)
    ch_diff = ch_diff * ch_active.unsqueeze(1) * time_valid.unsqueeze(-1)
    n_active_total = (ch_active.sum(dim=1) * time_valid.sum(dim=1)).sum().clamp(min=1)
    l_ch = (ch_diff ** 2).sum() / n_active_total

    # Baseline MSE
    base_scale = (true_base.abs() * time_valid).sum(dim=1) / time_valid.sum(dim=1).clamp(min=1)  # (B,)
    base_scale = base_scale.clamp(min=0.01).unsqueeze(1)
    l_base = (((pred_base - true_base) / base_scale) ** 2 * time_valid).sum() / n_valid

    # Non-media MSE
    nm_scale = (true_nm.abs() * time_valid).sum(dim=1) / time_valid.sum(dim=1).clamp(min=1)
    nm_scale = nm_scale.clamp(min=0.01).unsqueeze(1)
    l_nm = (((pred_nm - true_nm) / nm_scale) ** 2 * time_valid).sum() / n_valid

    # Reconstruction: channel + baseline + non_media should sum to y_normed
    # y_normed is at position max_channels in input features, but we don't have it here.
    # Instead, sum of true target components ≈ y_normed (by construction).
    # Use reconstruction against true total:
    true_sum = true_base + (true_ch * ch_active.unsqueeze(1)).sum(dim=-1) + true_nm
    pred_sum = pred_base + (pred_ch * ch_active.unsqueeze(1)).sum(dim=-1) + pred_nm
    l_recon = ((pred_sum - true_sum) ** 2 * time_valid).sum() / n_valid

    total = l_ch + l_base + l_nm + l_recon

    return total, {
        "channels": l_ch.item(),
        "baseline": l_base.item(),
        "non_media": l_nm.item(),
        "reconstruction": l_recon.item(),
        "total": total.item(),
    }


class PFNEngine:
    """Training loop and inference for PFN decomposition model."""

    def __init__(self, config: PFNConfig | None = None):
        self.config = config or PFNConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = PFNDecompositionModel(
            max_channels=self.config.max_channels,
            n_context_dims=self.config.n_context_dims,
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            dropout=self.config.dropout,
        ).to(self.device)

        self.optimizer = None
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def train(self, data_dir: str | None = None, n_train: int | None = None) -> dict:
        data_dir = data_dir or self.config.data_dir
        n_train = n_train or self.config.n_train

        logger.info("Loading training data from %s (max %d samples)", data_dir, n_train)
        backing = DemantiqDataset(data_dir, max_samples=n_train)
        logger.info("Loaded %d scenarios", len(backing))

        # Split by scenario
        n_scenarios = len(backing)
        if n_scenarios == 1:
            n_val = 0
            n_train_actual = 1
        else:
            n_val = max(1, int(n_scenarios * self.config.val_fraction))
            n_train_actual = n_scenarios - n_val

        indices = torch.randperm(n_scenarios, generator=torch.Generator().manual_seed(42))
        train_indices = indices[:n_train_actual].tolist()
        val_indices = indices[n_train_actual:].tolist()

        train_ds = _SubsetPFNDataset(
            backing, train_indices,
            max_channels=self.config.max_channels,
            context_dropout=self.config.context_dropout,
            training=True,
        )

        has_val = len(val_indices) > 0
        val_ds = _SubsetPFNDataset(
            backing, val_indices,
            max_channels=self.config.max_channels,
            context_dropout=0.0,
            training=False,
        ) if has_val else None

        logger.info("Scenarios: %d train, %d val", len(train_ds), len(val_ds) if val_ds else 0)

        train_loader = DataLoader(
            train_ds, batch_size=self.config.batch_size, shuffle=True,
            num_workers=0, pin_memory=self.device.type == "cuda",
            collate_fn=pfn_collate_fn,
        )
        val_loader = None
        if val_ds and len(val_ds) > 0:
            val_loader = DataLoader(
                val_ds, batch_size=self.config.batch_size, shuffle=False,
                num_workers=0, pin_memory=self.device.type == "cuda",
                collate_fn=pfn_collate_fn,
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
            "Training: %d scenarios, %d epochs, batch=%d, lr=%.1e, params=%d",
            len(train_ds), self.config.n_epochs,
            self.config.batch_size, self.config.learning_rate, n_params,
        )

        t_start = time.time()
        for epoch in range(self.config.n_epochs):
            train_loss, train_comp = self._train_epoch(train_loader)
            val_loss = train_loss
            if val_loader:
                val_loss, _ = self._validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            scheduler.step()

            elapsed = time.time() - t_start
            lr = self.optimizer.param_groups[0]["lr"]

            if epoch % 10 == 0 or epoch == self.config.n_epochs - 1:
                logger.info(
                    "Epoch %3d/%d  train=%.4f (ch=%.4f bl=%.4f nm=%.4f rc=%.4f)  "
                    "val=%.4f  lr=%.1e  [%.0fs]",
                    epoch + 1, self.config.n_epochs,
                    train_loss, train_comp["channels"], train_comp["baseline"],
                    train_comp["non_media"], train_comp["reconstruction"],
                    val_loss, lr, elapsed,
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
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

    def _train_epoch(self, loader: DataLoader) -> tuple[float, dict]:
        self.model.train()
        total_loss = 0.0
        total_comp = {}
        n = 0

        for batch in loader:
            self.optimizer.zero_grad()
            loss, comp = self._compute_loss(batch)
            loss.backward()
            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            for k, v in comp.items():
                total_comp[k] = total_comp.get(k, 0) + v
            n += 1

        n = max(n, 1)
        return total_loss / n, {k: v / n for k, v in total_comp.items()}

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> tuple[float, dict]:
        self.model.eval()
        total_loss = 0.0
        total_comp = {}
        n = 0

        for batch in loader:
            loss, comp = self._compute_loss(batch)
            total_loss += loss.item()
            for k, v in comp.items():
                total_comp[k] = total_comp.get(k, 0) + v
            n += 1

        n = max(n, 1)
        return total_loss / n, {k: v / n for k, v in total_comp.items()}

    def _compute_loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        x = batch["input_features"].to(self.device)
        target = batch["target"].to(self.device)
        time_mask = batch["time_pad_mask"].to(self.device)
        ch_mask = batch["channel_pad_mask"].to(self.device)

        pred = self.model(x, src_key_padding_mask=time_mask)
        return pfn_loss(pred, target, time_mask, ch_mask, self.config.max_channels)

    @torch.no_grad()
    def infer(
        self,
        y: np.ndarray,
        spend: np.ndarray,
        context: np.ndarray,
        n_channels: int,
    ) -> dict:
        """Run inference on a single scenario.

        Args:
            y: (T,) observed demand.
            spend: (T, n_channels) per-channel spend.
            context: (T, n_context_dims) context variables.
            n_channels: number of active channels.

        Returns:
            Dict with channel_contributions (T, n_ch), baseline (T,),
            non_media (T,), y_hat (T,) — all in absolute demand units.
        """
        self.model.eval()

        x, y_scale = build_pfn_input(
            spend, y, context,
            max_channels=self.config.max_channels,
            n_context_dims=self.config.n_context_dims,
        )
        x = x.to(self.device)

        pred = self.model(x)

        C = self.config.max_channels
        pred_ch = pred["channel_contributions"][0].cpu().numpy() * y_scale  # (T, C)
        pred_base = pred["baseline"][0].cpu().numpy() * y_scale              # (T,)
        pred_nm = pred["non_media"][0].cpu().numpy() * y_scale               # (T,)

        ch_contribs = pred_ch[:, :n_channels]
        y_hat = pred_base + ch_contribs.sum(axis=1) + pred_nm

        return {
            "channel_contributions": ch_contribs.astype(np.float32),
            "baseline": pred_base.astype(np.float32),
            "non_media": pred_nm.astype(np.float32),
            "y_hat": y_hat.astype(np.float32),
        }

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out / "pfn_model.pt")
        (out / "pfn_config.json").write_text(json.dumps(asdict(self.config), indent=2))
        logger.info("Model saved to %s", path)

    def load(self, path: str | Path) -> None:
        out = Path(path)
        state = torch.load(out / "pfn_model.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        logger.info("Model loaded from %s", path)


class _SubsetPFNDataset(PFNScenarioDataset):
    """PFNScenarioDataset that uses a subset of scenarios."""

    def __init__(
        self,
        backing: DemantiqDataset,
        scenario_indices: list[int],
        max_channels: int = 8,
        n_context_dims: int = 10,
        context_dropout: float = 0.3,
        training: bool = True,
    ):
        self.max_channels = max_channels
        self.n_context_dims = n_context_dims
        self.context_dropout = context_dropout
        self.training = training

        self._y = backing.y
        self._spend = backing.spend
        self._context = backing.context
        self._decomposition = backing.decomposition
        self._n_periods = backing.n_periods
        self._channel_names = backing.channel_names

        # Filter valid scenarios from the subset
        self._valid_indices = []
        for idx in scenario_indices:
            n_ch = len(backing.channel_names[idx])
            if 2 <= n_ch <= max_channels:
                self._valid_indices.append(idx)
