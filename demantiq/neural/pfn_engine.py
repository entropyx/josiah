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
    week_mask_fraction: float = 0.3  # fraction of weeks masked per scenario during training
    masked_loss_weight: float = 5.0  # loss weight for masked weeks (vs 1.0 for visible)

    # Data
    n_train: int = 100000
    data_dir: str = "training_data"


def pfn_loss(
    pred: dict[str, torch.Tensor],
    target: torch.Tensor,
    time_pad_mask: torch.Tensor,
    channel_pad_mask: torch.Tensor,
    week_is_masked: torch.Tensor,
    max_channels: int,
    masked_loss_weight: float = 5.0,
) -> tuple[torch.Tensor, dict]:
    """PFN loss on share targets (component / y[t]).

    Targets are shares — the fraction of each week's y attributable to each
    component. Per-scenario normalization is gone because shares are naturally
    comparable. Reconstruction: sum of shares should equal 1.0 per week.

    Masked weeks get higher loss weight (forcing in-context inference).
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

    # Per-week weight: masked weeks get higher weight
    week_weight = time_valid * (1.0 + (masked_loss_weight - 1.0) * week_is_masked)
    total_weight = week_weight.sum().clamp(min=1)

    # Per-channel MSE — NO per-scenario normalization (shares already comparable)
    ch_diff_sq = (pred_ch - true_ch) ** 2  # (B, T, C)
    ch_diff_sq = ch_diff_sq * ch_active.unsqueeze(1) * week_weight.unsqueeze(-1)
    active_week_weight = (ch_active.sum(dim=1).unsqueeze(1) * week_weight).sum().clamp(min=1)
    l_ch = ch_diff_sq.sum() / active_week_weight

    # Baseline MSE (share)
    l_base = (((pred_base - true_base) ** 2) * week_weight).sum() / total_weight

    # Non-media MSE (share)
    l_nm = (((pred_nm - true_nm) ** 2) * week_weight).sum() / total_weight

    # Reconstruction: shares should sum to ~1 per week
    # (pred_base + sum(pred_channels) + pred_nm - 1.0) ^ 2
    pred_sum = pred_base + (pred_ch * ch_active.unsqueeze(1)).sum(dim=-1) + pred_nm
    target_sum = torch.ones_like(pred_sum)  # shares sum to 1 ideally
    l_recon = (((pred_sum - target_sum) ** 2) * week_weight).sum() / total_weight

    total = l_ch + l_base + l_nm + l_recon

    # Track masked-week metrics (critical signal for in-context learning)
    masked_weight = time_valid * week_is_masked
    n_masked = masked_weight.sum().clamp(min=1)
    if n_masked > 0:
        ch_diff_masked = (pred_ch - true_ch) ** 2
        ch_diff_masked = ch_diff_masked * ch_active.unsqueeze(1) * masked_weight.unsqueeze(-1)
        l_ch_masked = ch_diff_masked.sum() / (ch_active.sum(dim=1).unsqueeze(1) * masked_weight).sum().clamp(min=1)
        l_base_masked = (((pred_base - true_base) ** 2) * masked_weight).sum() / n_masked
    else:
        l_ch_masked = torch.tensor(0.0, device=total.device)
        l_base_masked = torch.tensor(0.0, device=total.device)

    return total, {
        "channels": l_ch.item(),
        "baseline": l_base.item(),
        "non_media": l_nm.item(),
        "reconstruction": l_recon.item(),
        "ch_masked": l_ch_masked.item(),
        "bl_masked": l_base_masked.item(),
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
            week_mask_fraction=self.config.week_mask_fraction,
            training=True,
        )

        has_val = len(val_indices) > 0
        val_ds = _SubsetPFNDataset(
            backing, val_indices,
            max_channels=self.config.max_channels,
            context_dropout=0.0,
            week_mask_fraction=0.0,  # val measures inference-time performance (no masking)
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

            if epoch % 5 == 0 or epoch == self.config.n_epochs - 1:
                logger.info(
                    "Epoch %3d/%d  train=%.4f (ch=%.4f bl=%.4f nm=%.4f rc=%.4f "
                    "ch_m=%.4f bl_m=%.4f)  val=%.4f  lr=%.1e  [%.0fs]",
                    epoch + 1, self.config.n_epochs,
                    train_loss, train_comp["channels"], train_comp["baseline"],
                    train_comp["non_media"], train_comp["reconstruction"],
                    train_comp.get("ch_masked", 0), train_comp.get("bl_masked", 0),
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
        week_is_masked = batch["week_is_masked"].to(self.device)

        pred = self.model(x, src_key_padding_mask=time_mask)
        return pfn_loss(
            pred, target, time_mask, ch_mask, week_is_masked,
            max_channels=self.config.max_channels,
            masked_loss_weight=self.config.masked_loss_weight,
        )

    @torch.no_grad()
    def infer(
        self,
        y: np.ndarray,
        spend: np.ndarray,
        context: np.ndarray,
        n_channels: int,
    ) -> dict:
        """Run inference on a single scenario.

        The model outputs per-timestep SHARES (component / y). We multiply by
        y[t] to recover absolute demand units.

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

        x, _ = build_pfn_input(
            spend, y, context,
            max_channels=self.config.max_channels,
            n_context_dims=self.config.n_context_dims,
        )
        x = x.to(self.device)

        pred = self.model(x)

        C = self.config.max_channels
        # Model outputs shares; multiply by y[t] to get absolute values
        pred_ch_shares = pred["channel_contributions"][0].cpu().numpy()  # (T, C)
        pred_base_share = pred["baseline"][0].cpu().numpy()              # (T,)
        pred_nm_share = pred["non_media"][0].cpu().numpy()               # (T,)

        y_t = y.astype(np.float32)  # (T,)
        pred_ch = pred_ch_shares * y_t[:, np.newaxis]
        pred_base = pred_base_share * y_t
        pred_nm = pred_nm_share * y_t

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
        week_mask_fraction: float = 0.3,
        training: bool = True,
    ):
        self.max_channels = max_channels
        self.n_context_dims = n_context_dims
        self.context_dropout = context_dropout
        self.week_mask_fraction = week_mask_fraction
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
