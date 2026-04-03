"""Decomposition engine — encoder-decoder for per-period demand decomposition.

Simple approach that achieved 4-8pp category accuracy:
- EmbeddingNetwork.forward_temporal() → per-period embeddings (B, T, 256)
- TemporalDecoder (MLP) → per-period shares (B, T, 26)
- Loss: simple MSE on shares

Testing whether category-level accuracy generalizes across diverse baseline levels.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from demantiq.neural.data_loader import DemantiqDataset
from demantiq.neural.encoders import EmbeddingNetwork
from demantiq.neural.temporal_decoder import TemporalDecoder
from demantiq.orchestration.training_format import (
    DECOMP_COLS,
    DECOMP_IDX_DISTRIBUTION,
    DECOMP_IDX_NOISE,
)

logger = logging.getLogger(__name__)


@dataclass
class DecompositionConfig:
    temporal_dim: int = 64
    type_embed_dim: int = 16
    n_attn_heads: int = 4
    n_attn_layers: int = 2
    embedding_dim: int = 256
    decoder_hidden_dim: int = 128
    decoder_n_layers: int = 2
    decoder_dropout: float = 0.1
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    n_epochs: int = 50
    patience: int = 15
    val_fraction: float = 0.1
    grad_clip: float = 1.0
    n_train: int = 50000
    data_dir: str = "training_data"
    n_fixed_channels: int | None = None
    ranking_margin: float = 0.0


def _compute_true_shares(decomposition, y, n_periods):
    """Convert absolute decomposition to fractional shares."""
    B, T, C = decomposition.shape
    t_idx = torch.arange(T, device=y.device).unsqueeze(0)
    valid_mask = t_idx < n_periods.unsqueeze(1)

    y_safe = y.unsqueeze(-1).clamp(min=1.0)
    shares = decomposition / y_safe
    shares = shares.clamp(-2.0, 2.0)
    shares[:, :, DECOMP_IDX_DISTRIBUTION] = 0.0
    shares = shares * valid_mask.unsqueeze(-1).float()

    return shares, valid_mask


class DecompositionEngine:
    """Simple encoder-decoder for demand decomposition."""

    def __init__(self, config: DecompositionConfig | None = None):
        self.config = config or DecompositionConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = EmbeddingNetwork(
            temporal_dim=self.config.temporal_dim,
            type_embed_dim=self.config.type_embed_dim,
            n_attn_heads=self.config.n_attn_heads,
            n_attn_layers=self.config.n_attn_layers,
            global_summary_dim=self.config.embedding_dim,
            dropout=self.config.decoder_dropout,
            n_fixed_channels=self.config.n_fixed_channels,
        ).to(self.device)

        self.decoder = TemporalDecoder(
            input_dim=self.config.embedding_dim,
            hidden_dim=self.config.decoder_hidden_dim,
            n_components=DECOMP_COLS,
            n_layers=self.config.decoder_n_layers,
            dropout=self.config.decoder_dropout,
        ).to(self.device)

        self.optimizer = None
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def train(self, data_dir: str | None = None, n_train: int | None = None) -> dict:
        data_dir = data_dir or self.config.data_dir
        n_train = n_train or self.config.n_train

        logger.info("Loading training data from %s (max %d samples)", data_dir, n_train)
        dataset = DemantiqDataset(data_dir, max_samples=n_train)
        logger.info("Loaded %d samples", len(dataset))

        if dataset[0]["decomposition"].abs().sum() == 0:
            raise ValueError("Decomposition data is all zeros. Regenerate training data.")

        n_val = max(1, int(len(dataset) * self.config.val_fraction))
        n_train_actual = len(dataset) - n_val
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [n_train_actual, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=self.device.type == "cuda")
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False,
                                num_workers=0, pin_memory=self.device.type == "cuda")

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate,
                                            weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        logger.info("Training: %d train / %d val, %d epochs, batch_size=%d, lr=%.1e",
                     n_train_actual, n_val, self.config.n_epochs,
                     self.config.batch_size, self.config.learning_rate)

        t_start = time.time()
        for epoch in range(self.config.n_epochs):
            self.encoder.train()
            self.decoder.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                loss = self._train_step(batch)
                epoch_loss += loss
                n_batches += 1

            avg_train = epoch_loss / max(n_batches, 1)
            self.train_losses.append(avg_train)

            val_loss = self._validate(val_loader)
            self.val_losses.append(val_loss)
            scheduler.step(val_loss)

            elapsed = time.time() - t_start
            lr = self.optimizer.param_groups[0]["lr"]

            if epoch % 5 == 0 or epoch == self.config.n_epochs - 1:
                logger.info("Epoch %3d/%d  train=%.6f  val=%.6f  lr=%.1e  [%.0fs]",
                            epoch + 1, self.config.n_epochs, avg_train, val_loss, lr, elapsed)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    "encoder": {k: v.cpu().clone() for k, v in self.encoder.state_dict().items()},
                    "decoder": {k: v.cpu().clone() for k, v in self.decoder.state_dict().items()},
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        if best_state:
            self.encoder.load_state_dict(best_state["encoder"])
            self.decoder.load_state_dict(best_state["decoder"])
            self.encoder.to(self.device)
            self.decoder.to(self.device)

        total_time = time.time() - t_start
        logger.info("Training complete: %.1f min, best_val=%.6f", total_time / 60, best_val_loss)
        return {"best_val_loss": best_val_loss, "final_train_loss": self.train_losses[-1],
                "n_epochs_trained": len(self.train_losses), "training_time_minutes": total_time / 60}

    def _train_step(self, batch):
        self.optimizer.zero_grad()

        y = batch["y"].to(self.device)
        spend = batch["spend"].to(self.device)
        context = batch["context"].to(self.device)
        n_channels = batch["n_channels"].to(self.device)
        channel_type_ids = batch["channel_type_ids"].to(self.device)
        decomposition = batch["decomposition"].to(self.device)
        n_periods = batch["n_periods"].to(self.device)

        # Encoder → per-period embeddings
        temporal_emb = self.encoder.forward_temporal(y, spend, n_channels, channel_type_ids, context)

        # Decoder → shares
        pred_shares = self.decoder(temporal_emb)

        # True shares
        true_shares, valid_mask = _compute_true_shares(decomposition, y, n_periods)

        # Simple MSE loss on valid timesteps
        mask = valid_mask.unsqueeze(-1).float()
        diff = (pred_shares - true_shares) * mask
        loss = (diff ** 2).sum() / mask.sum().clamp(min=1) / DECOMP_COLS

        loss.backward()
        if self.config.grad_clip > 0:
            all_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
            nn.utils.clip_grad_norm_(all_params, self.config.grad_clip)
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def _validate(self, val_loader):
        self.encoder.eval()
        self.decoder.eval()
        total = 0.0
        n = 0
        for batch in val_loader:
            y = batch["y"].to(self.device)
            spend = batch["spend"].to(self.device)
            context = batch["context"].to(self.device)
            n_channels = batch["n_channels"].to(self.device)
            channel_type_ids = batch["channel_type_ids"].to(self.device)
            decomposition = batch["decomposition"].to(self.device)
            n_periods = batch["n_periods"].to(self.device)

            emb = self.encoder.forward_temporal(y, spend, n_channels, channel_type_ids, context)
            pred = self.decoder(emb)
            true_shares, valid_mask = _compute_true_shares(decomposition, y, n_periods)
            mask = valid_mask.unsqueeze(-1).float()
            diff = (pred - true_shares) * mask
            loss = (diff ** 2).sum() / mask.sum().clamp(min=1) / DECOMP_COLS
            total += loss.item()
            n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def infer(self, y, spend, context, n_channels, channel_type_ids, n_periods):
        self.encoder.eval()
        self.decoder.eval()

        y_t = torch.from_numpy(y.copy()).float().unsqueeze(0).to(self.device)
        spend_t = torch.from_numpy(spend.copy()).float().unsqueeze(0).to(self.device)
        ctx_t = torch.from_numpy(context.copy()).float().unsqueeze(0).to(self.device)
        n_ch_t = torch.tensor([n_channels], device=self.device)
        type_ids_t = torch.from_numpy(channel_type_ids.copy()).long().unsqueeze(0).to(self.device)

        emb = self.encoder.forward_temporal(y_t, spend_t, n_ch_t, type_ids_t, ctx_t)
        pred_shares = self.decoder(emb).squeeze(0).cpu().numpy()

        contributions = pred_shares * y[:pred_shares.shape[0], np.newaxis]
        additive_mask = np.ones(DECOMP_COLS, dtype=bool)
        additive_mask[DECOMP_IDX_DISTRIBUTION] = False
        y_recon = contributions[:, additive_mask].sum(axis=1)

        return {
            "shares": pred_shares[:n_periods],
            "contributions": contributions[:n_periods],
            "y_reconstructed": y_recon[:n_periods],
        }

    def save(self, path):
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.encoder.state_dict(), out / "encoder.pt")
        torch.save(self.decoder.state_dict(), out / "decoder.pt")
        import json
        (out / "decomp_config.json").write_text(json.dumps(self.config.__dict__, indent=2))
        logger.info("Model saved to %s", path)

    def load(self, path):
        out = Path(path)
        self.encoder.load_state_dict(torch.load(out / "encoder.pt", map_location=self.device, weights_only=True))
        self.decoder.load_state_dict(torch.load(out / "decoder.pt", map_location=self.device, weights_only=True))
        logger.info("Model loaded from %s", path)
