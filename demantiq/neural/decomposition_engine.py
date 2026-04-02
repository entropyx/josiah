"""Additive decomposition engine — predict absolute contributions, sum to reconstruct y.

Each component sub-network sees only its own inputs (spend for channels,
context for baseline). No y pollution. Components are summed and compared
to actual y. Loss is on both per-component accuracy and reconstruction.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from demantiq.neural.data_loader import DemantiqDataset
from demantiq.neural.losses import AdditiveDecompositionLoss
from demantiq.neural.temporal_decoder import AdditiveDecompositionDecoder, N_GLOBAL_STATS
from demantiq.orchestration.training_format import (
    DECOMP_COLS,
    DECOMP_IDX_BASELINE,
    DECOMP_IDX_CHANNELS_START,
    DECOMP_IDX_DISTRIBUTION,
    DECOMP_IDX_NOISE,
)

logger = logging.getLogger(__name__)


@dataclass
class DecompositionConfig:
    """Configuration for the additive decomposition engine."""
    # Decoder
    type_embed_dim: int = 16
    decoder_hidden_dim: int = 64
    decoder_dropout: float = 0.1

    # Training
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    n_epochs: int = 50
    patience: int = 15
    val_fraction: float = 0.1
    grad_clip: float = 1.0

    # Data
    n_train: int = 50000
    data_dir: str = "training_data"
    n_fixed_channels: int | None = None

    # Legacy fields (kept for config compatibility)
    temporal_dim: int = 64
    n_attn_heads: int = 4
    n_attn_layers: int = 2
    embedding_dim: int = 256
    decoder_n_layers: int = 2
    ranking_margin: float = 0.0


def _compute_global_stats(y: torch.Tensor, spend: torch.Tensor, n_channels: torch.Tensor) -> torch.Tensor:
    """Compute scenario-level summary statistics for the channel CNN.

    Args:
        y: (B, T) demand.
        spend: (B, T, max_ch) spend.
        n_channels: (B,) active channel count.

    Returns:
        (B, N_GLOBAL_STATS) summary stats.
    """
    B = y.shape[0]
    stats = torch.zeros(B, N_GLOBAL_STATS, device=y.device)

    mean_y = y.mean(dim=1)  # (B,)
    total_spend = spend.sum(dim=2)  # (B, T) — sum across channels
    mean_total_spend = total_spend.mean(dim=1)  # (B,)

    stats[:, 0] = mean_y / mean_y.abs().max().clamp(min=1)  # normalized mean_y
    stats[:, 1] = mean_total_spend / mean_total_spend.abs().max().clamp(min=1)  # normalized mean_spend
    stats[:, 2] = (mean_y / mean_total_spend.clamp(min=1)).clamp(-10, 10)  # y/spend ratio
    stats[:, 3] = n_channels.float() / 20.0  # normalized n_channels

    return stats


class DecompositionEngine:
    """Training and inference engine for additive demand decomposition."""

    def __init__(self, config: DecompositionConfig | None = None):
        self.config = config or DecompositionConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.decoder = AdditiveDecompositionDecoder(
            type_embed_dim=self.config.type_embed_dim,
            hidden_dim=self.config.decoder_hidden_dim,
        ).to(self.device)

        self.loss_fn = AdditiveDecompositionLoss().to(self.device)

        self.optimizer: torch.optim.Optimizer | None = None
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def train(self, data_dir: str | None = None, n_train: int | None = None) -> dict:
        """Train the additive decomposition model."""
        data_dir = data_dir or self.config.data_dir
        n_train = n_train or self.config.n_train

        logger.info("Loading training data from %s (max %d samples)", data_dir, n_train)
        dataset = DemantiqDataset(data_dir, max_samples=n_train)
        logger.info("Loaded %d samples", len(dataset))

        # Check decomposition data exists
        sample = dataset[0]
        if sample["decomposition"].abs().sum() == 0:
            raise ValueError("Decomposition data is all zeros. Regenerate training data.")

        # Split train/val
        n_val = max(1, int(len(dataset) * self.config.val_fraction))
        n_train_actual = len(dataset) - n_val
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [n_train_actual, n_val],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = DataLoader(
            train_ds, batch_size=self.config.batch_size, shuffle=True,
            num_workers=0, pin_memory=self.device.type == "cuda",
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.config.batch_size, shuffle=False,
            num_workers=0, pin_memory=self.device.type == "cuda",
        )

        self.optimizer = torch.optim.AdamW(
            self.decoder.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        logger.info(
            "Training: %d train / %d val, %d epochs, batch_size=%d, lr=%.1e",
            n_train_actual, n_val, self.config.n_epochs,
            self.config.batch_size, self.config.learning_rate,
        )

        t_start = time.time()
        for epoch in range(self.config.n_epochs):
            self.decoder.train()
            epoch_loss = 0.0
            epoch_terms: dict[str, float] = {}
            n_batches = 0

            for batch in train_loader:
                loss, loss_terms = self._train_step(batch)
                epoch_loss += loss
                for k, v in loss_terms.items():
                    epoch_terms[k] = epoch_terms.get(k, 0.0) + v.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            avg_terms = {k: v / max(n_batches, 1) for k, v in epoch_terms.items()}
            self.train_losses.append(avg_train_loss)

            val_loss = self._validate(val_loader)
            self.val_losses.append(val_loss)
            scheduler.step(val_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t_start

            if epoch % 5 == 0 or epoch == self.config.n_epochs - 1:
                lc = avg_terms.get("L_component", 0)
                lr_val = avg_terms.get("L_reconstruction", 0)
                logger.info(
                    "Epoch %3d/%d  loss=%.4f  val=%.4f  "
                    "L_comp=%.4f L_recon=%.4f  lr=%.1e  [%.0fs]",
                    epoch + 1, self.config.n_epochs, avg_train_loss, val_loss,
                    lc, lr_val, current_lr, elapsed,
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    "decoder": {k: v.cpu().clone() for k, v in self.decoder.state_dict().items()},
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, self.config.patience)
                    break

        if best_state is not None:
            self.decoder.load_state_dict(best_state["decoder"])
            self.decoder.to(self.device)

        total_time = time.time() - t_start
        logger.info("Training complete: %.1f minutes, best_val_loss=%.6f", total_time / 60, best_val_loss)

        return {
            "best_val_loss": best_val_loss,
            "final_train_loss": self.train_losses[-1],
            "n_epochs_trained": len(self.train_losses),
            "training_time_minutes": total_time / 60,
        }

    def _train_step(self, batch: dict) -> tuple[float, dict]:
        self.optimizer.zero_grad()

        y = batch["y"].to(self.device)
        spend = batch["spend"].to(self.device)
        context = batch["context"].to(self.device)
        n_channels = batch["n_channels"].to(self.device)
        channel_type_ids = batch["channel_type_ids"].to(self.device)
        decomposition = batch["decomposition"].to(self.device)  # true absolute contributions
        n_periods = batch["n_periods"].to(self.device)

        # Build valid mask
        T = y.shape[1]
        t_idx = torch.arange(T, device=self.device).unsqueeze(0)
        valid_mask = t_idx < n_periods.unsqueeze(1)

        # Build channel mask
        max_ch = spend.shape[2]
        ch_idx = torch.arange(max_ch, device=self.device).unsqueeze(0)
        channel_mask = ch_idx < n_channels.unsqueeze(1)

        # Compute global stats
        global_stats = _compute_global_stats(y, spend, n_channels)

        # Forward
        result = self.decoder(
            spend, context, channel_type_ids[:, :max_ch],
            channel_mask, n_channels, global_stats,
        )

        # Loss
        loss, loss_terms = self.loss_fn(
            result["contributions"], decomposition,
            result["y_pred"], y,
            valid_mask, n_channels,
        )

        loss.backward()
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.decoder.parameters(), self.config.grad_clip)
        self.optimizer.step()

        return loss.item(), loss_terms

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> float:
        self.decoder.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in val_loader:
            y = batch["y"].to(self.device)
            spend = batch["spend"].to(self.device)
            context = batch["context"].to(self.device)
            n_channels = batch["n_channels"].to(self.device)
            channel_type_ids = batch["channel_type_ids"].to(self.device)
            decomposition = batch["decomposition"].to(self.device)
            n_periods = batch["n_periods"].to(self.device)

            T = y.shape[1]
            t_idx = torch.arange(T, device=self.device).unsqueeze(0)
            valid_mask = t_idx < n_periods.unsqueeze(1)
            max_ch = spend.shape[2]
            ch_idx = torch.arange(max_ch, device=self.device).unsqueeze(0)
            channel_mask = ch_idx < n_channels.unsqueeze(1)
            global_stats = _compute_global_stats(y, spend, n_channels)

            result = self.decoder(
                spend, context, channel_type_ids[:, :max_ch],
                channel_mask, n_channels, global_stats,
            )
            loss, _ = self.loss_fn(
                result["contributions"], decomposition,
                result["y_pred"], y,
                valid_mask, n_channels,
            )
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def infer(
        self, y: np.ndarray, spend: np.ndarray, context: np.ndarray,
        n_channels: int, channel_type_ids: np.ndarray, n_periods: int,
    ) -> dict:
        """Run inference on a single scenario.

        Returns absolute contributions (not shares).
        """
        self.decoder.eval()

        y_t = torch.from_numpy(y.copy()).float().unsqueeze(0).to(self.device)
        spend_t = torch.from_numpy(spend.copy()).float().unsqueeze(0).to(self.device)
        ctx_t = torch.from_numpy(context.copy()).float().unsqueeze(0).to(self.device)
        n_ch_t = torch.tensor([n_channels], device=self.device)
        type_ids_t = torch.from_numpy(channel_type_ids.copy()).long().unsqueeze(0).to(self.device)

        max_ch = spend_t.shape[2]
        ch_idx = torch.arange(max_ch, device=self.device).unsqueeze(0)
        channel_mask = ch_idx < n_ch_t.unsqueeze(1)
        global_stats = _compute_global_stats(y_t, spend_t, n_ch_t)

        result = self.decoder(
            spend_t, ctx_t, type_ids_t[:, :max_ch],
            channel_mask, n_ch_t, global_stats,
        )

        contributions = result["contributions"].squeeze(0).cpu().numpy()  # (T, DECOMP_COLS)
        y_pred = result["y_pred"].squeeze(0).cpu().numpy()  # (T,)

        # Compute noise as residual
        contributions[:, DECOMP_IDX_NOISE] = y[:contributions.shape[0]] - y_pred

        # Compute shares for backward compatibility
        y_safe = np.maximum(np.abs(y[:n_periods]), 1.0)
        shares = contributions[:n_periods] / y_safe[:, np.newaxis]

        return {
            "contributions": contributions[:n_periods],
            "y_pred": y_pred[:n_periods],
            "shares": shares,
            "y_reconstructed": y_pred[:n_periods],
        }

    def save(self, path: str) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.decoder.state_dict(), out / "decoder.pt")
        import json
        config_dict = {k: v for k, v in self.config.__dict__.items()}
        (out / "decomp_config.json").write_text(json.dumps(config_dict, indent=2))
        logger.info("Model saved to %s", path)

    def load(self, path: str) -> None:
        out = Path(path)
        self.decoder.load_state_dict(
            torch.load(out / "decoder.pt", map_location=self.device, weights_only=True)
        )
        logger.info("Model loaded from %s", path)
