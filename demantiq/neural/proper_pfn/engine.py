"""Training loop and inference for proper PFN."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from demantiq.neural.data_loader import DemantiqDataset
from demantiq.neural.proper_pfn.dataset import (
    ProperPFNDataset, proper_pfn_collate_fn, CHANNEL_FEATURE_DIM,
)
from demantiq.neural.proper_pfn.model import ProperPFNModel
from demantiq.neural.proper_pfn.loss import held_out_loss

logger = logging.getLogger(__name__)


@dataclass
class ProperPFNConfig:
    # Architecture
    max_channels: int = 8
    n_context_dims: int = 10
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1

    # Training
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    n_epochs: int = 100
    patience: int = 20
    grad_clip: float = 1.0
    val_fraction: float = 0.1
    week_mask_fraction: float = 0.3

    # Data
    n_train: int = 50000
    data_dir: str = "training_data"


# Global feature dim = y(1) + ctx(10) + flags(10) + time_idx(1) + sin(1) + cos(1) + is_masked(1) = 25
GLOBAL_FEATURE_DIM = 25


class ProperPFNEngine:
    def __init__(self, config: ProperPFNConfig | None = None):
        self.config = config or ProperPFNConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ProperPFNModel(
            channel_feat_dim=CHANNEL_FEATURE_DIM,
            global_feat_dim=GLOBAL_FEATURE_DIM,
            max_channels=self.config.max_channels,
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

        logger.info("Loading training data from %s (max %d)", data_dir, n_train)
        backing = DemantiqDataset(data_dir, max_samples=n_train)
        logger.info("Loaded %d scenarios", len(backing))

        n = len(backing)
        if n == 1:
            train_idx, val_idx = [0], []
        else:
            n_val = max(1, int(n * self.config.val_fraction))
            g = torch.Generator().manual_seed(42)
            perm = torch.randperm(n, generator=g).tolist()
            train_idx, val_idx = perm[:n - n_val], perm[n - n_val:]

        train_ds = _SubsetDataset(
            backing, train_idx,
            max_channels=self.config.max_channels,
            week_mask_fraction=self.config.week_mask_fraction,
            training=True,
        )
        val_ds = None
        if val_idx:
            val_ds = _SubsetDataset(
                backing, val_idx,
                max_channels=self.config.max_channels,
                week_mask_fraction=self.config.week_mask_fraction,
                training=True,  # apply mask so val measures held-out performance
            )

        train_loader = DataLoader(
            train_ds, batch_size=self.config.batch_size, shuffle=True,
            num_workers=0, pin_memory=self.device.type == "cuda",
            collate_fn=proper_pfn_collate_fn,
        )
        val_loader = None
        if val_ds and len(val_ds) > 0:
            val_loader = DataLoader(
                val_ds, batch_size=self.config.batch_size, shuffle=False,
                num_workers=0, pin_memory=self.device.type == "cuda",
                collate_fn=proper_pfn_collate_fn,
            )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.n_epochs, eta_min=1e-6,
        )

        best_val = float("inf")
        best_state = None
        patience = 0

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            "Training: %d scenarios, %d epochs, batch=%d, lr=%.1e, params=%d",
            len(train_ds), self.config.n_epochs, self.config.batch_size,
            self.config.learning_rate, n_params,
        )

        t0 = time.time()
        for epoch in range(self.config.n_epochs):
            train_loss, train_comp = self._train_epoch(train_loader)
            val_loss = train_loss
            if val_loader:
                val_loss, _ = self._validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            scheduler.step()

            if epoch % 5 == 0 or epoch == self.config.n_epochs - 1:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch %3d/%d  train=%.4f (ch=%.4f bl=%.4f nm=%.4f rc=%.4f "
                    "ch_m=%.4f bl_m=%.4f) val=%.4f  lr=%.1e  [%.0fs]",
                    epoch + 1, self.config.n_epochs, train_loss,
                    train_comp.get("channels", 0.0), train_comp.get("baseline", 0.0),
                    train_comp.get("non_media", 0.0), train_comp.get("reconstruction", 0.0),
                    train_comp.get("channels_masked", 0.0), train_comp.get("baseline_masked", 0.0),
                    val_loss, lr, time.time() - t0,
                )

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= self.config.patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        if best_state:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        return {
            "best_val_loss": best_val,
            "final_train_loss": self.train_losses[-1] if self.train_losses else 0.0,
            "n_epochs_trained": len(self.train_losses),
            "training_time_minutes": (time.time() - t0) / 60,
        }

    def _train_epoch(self, loader: DataLoader) -> tuple[float, dict]:
        self.model.train()
        total = 0.0
        total_comp = {}
        n = 0
        for batch in loader:
            self.optimizer.zero_grad()
            loss, comp = self._compute_loss(batch)
            loss.backward()
            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            total += loss.item()
            for k, v in comp.items():
                total_comp[k] = total_comp.get(k, 0.0) + v
            n += 1
        n = max(n, 1)
        return total / n, {k: v / n for k, v in total_comp.items()}

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> tuple[float, dict]:
        self.model.eval()
        total = 0.0
        total_comp = {}
        n = 0
        for batch in loader:
            loss, comp = self._compute_loss(batch)
            total += loss.item()
            for k, v in comp.items():
                total_comp[k] = total_comp.get(k, 0.0) + v
            n += 1
        n = max(n, 1)
        return total / n, {k: v / n for k, v in total_comp.items()}

    def _compute_loss(self, batch: dict):
        channel_tokens = batch["channel_tokens"].to(self.device)
        global_tokens = batch["global_tokens"].to(self.device)
        target = batch["target"].to(self.device)
        time_pad = batch["time_pad_mask"].to(self.device)
        ch_pad = batch["channel_pad_mask"].to(self.device)
        week_masked = batch["week_is_masked"].to(self.device)

        pred = self.model(channel_tokens, global_tokens, ch_pad, time_pad)
        return held_out_loss(
            pred, target, time_pad, ch_pad, week_masked, self.config.max_channels,
        )

    @torch.no_grad()
    def infer(
        self,
        y: np.ndarray,
        spend: np.ndarray,
        impressions: np.ndarray,
        clicks: np.ndarray,
        context: np.ndarray,
        n_channels: int,
    ) -> dict:
        """Inference on a single scenario (no masking — all weeks visible)."""
        self.model.eval()
        T = len(y)
        C = self.config.max_channels

        def pad_channel_arr(arr):
            out = np.zeros((T, C), dtype=np.float32)
            out[:, :n_channels] = arr[:, :n_channels]
            return out

        spend_p = pad_channel_arr(spend)
        imp_p = pad_channel_arr(impressions)
        clk_p = pad_channel_arr(clicks)

        def per_channel_normalize(arr):
            scales = np.maximum(np.abs(arr).max(axis=0, keepdims=True), 1.0)
            return arr / scales

        spend_n = per_channel_normalize(spend_p)
        imp_n = per_channel_normalize(imp_p)
        clk_n = per_channel_normalize(clk_p)

        channel_tokens = np.stack([spend_n, imp_n, clk_n], axis=-1).astype(np.float32)  # (T, C, 3)

        y_scale = max(float(np.abs(y).mean()), 1.0)
        y_norm = (y / y_scale).astype(np.float32)

        ctx_present = (np.abs(context).sum(axis=0) > 0).astype(np.float32)
        col_scales = np.maximum(np.abs(context).max(axis=0, keepdims=True), 1.0)
        ctx_norm = (context / col_scales).astype(np.float32)
        presence = np.tile(ctx_present, (T, 1))

        time_idx = np.linspace(0.0, 1.0, T, dtype=np.float32)
        woy = np.arange(T, dtype=np.float32) % 52.0
        sin_w = np.sin(2.0 * np.pi * woy / 52.0)
        cos_w = np.cos(2.0 * np.pi * woy / 52.0)
        is_masked = np.zeros(T, dtype=np.float32)

        global_tokens = np.concatenate(
            [
                y_norm.reshape(T, 1),
                ctx_norm,
                presence,
                time_idx.reshape(T, 1),
                sin_w.reshape(T, 1),
                cos_w.reshape(T, 1),
                is_masked.reshape(T, 1),
            ],
            axis=1,
        ).astype(np.float32)

        ch_tok = torch.from_numpy(channel_tokens).unsqueeze(0).to(self.device)
        gl_tok = torch.from_numpy(global_tokens).unsqueeze(0).to(self.device)
        ch_pad = torch.ones(1, C, dtype=torch.bool, device=self.device)
        ch_pad[0, :n_channels] = False
        t_pad = torch.zeros(1, T, dtype=torch.bool, device=self.device)

        pred = self.model(ch_tok, gl_tok, ch_pad, t_pad)

        ch_shares = pred["channel_shares"][0].cpu().numpy()
        base_share = pred["baseline_share"][0].cpu().numpy()
        nm_share = pred["non_media_share"][0].cpu().numpy()

        y_f = y.astype(np.float32)
        ch_contrib = ch_shares[:, :n_channels] * y_f[:, None]
        base_contrib = base_share * y_f
        nm_contrib = nm_share * y_f
        y_hat = base_contrib + ch_contrib.sum(axis=1) + nm_contrib

        return {
            "channel_contributions": ch_contrib.astype(np.float32),
            "baseline": base_contrib.astype(np.float32),
            "non_media": nm_contrib.astype(np.float32),
            "y_hat": y_hat.astype(np.float32),
        }

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out / "proper_pfn.pt")
        (out / "proper_pfn_config.json").write_text(json.dumps(asdict(self.config), indent=2))

    def load(self, path: str | Path) -> None:
        state = torch.load(Path(path) / "proper_pfn.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)


class _SubsetDataset(ProperPFNDataset):
    """ProperPFNDataset limited to a list of scenario indices."""

    def __init__(self, backing, scenario_indices, **kwargs):
        super().__init__(backing, **kwargs)
        self._valid_indices = [
            i for i in scenario_indices
            if 2 <= len(backing.channel_names[i]) <= self.max_channels
        ]
