"""Decomposition inference engine — supervised regression for per-period demand decomposition.

Replaces the SBI/NSF approach with a standard PyTorch encoder-decoder that predicts
weekly demand component shares directly. Trained on simulator-generated per-period
ground truth.

Pipeline:
    1. EmbeddingNetwork.forward_temporal() → per-period embeddings (B, T, 256)
    2. TemporalDecoder → per-period component shares (B, T, 26)
    3. Loss: MSE between predicted shares and true shares from ground_truth
    4. Inference: multiply predicted shares × observed y → absolute contributions
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

from demantiq.neural.data_loader import DemantiqDataset, create_dataloader
from demantiq.neural.encoders import EmbeddingNetwork
from demantiq.neural.losses import DecompositionLoss
from demantiq.neural.temporal_decoder import TemporalDecoder
from demantiq.orchestration.training_format import (
    DECOMP_COLS,
    DECOMP_IDX_BASELINE,
    DECOMP_IDX_CHANNELS_START,
    DECOMP_IDX_CHANNELS_END,
    DECOMP_IDX_DISTRIBUTION,
    DECOMP_IDX_NOISE,
)

logger = logging.getLogger(__name__)


@dataclass
class DecompositionConfig:
    """Configuration for the decomposition engine."""

    # Encoder
    temporal_dim: int = 64
    type_embed_dim: int = 16
    n_attn_heads: int = 4
    n_attn_layers: int = 2
    embedding_dim: int = 256

    # Decoder
    decoder_hidden_dim: int = 128
    decoder_n_layers: int = 2
    decoder_dropout: float = 0.1

    # Training
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    n_epochs: int = 50
    patience: int = 10
    val_fraction: float = 0.1
    grad_clip: float = 1.0

    # Data
    n_train: int = 50000
    data_dir: str = "training_data"
    n_fixed_channels: int | None = None

    # Composite loss
    ranking_margin: float = 0.0


def _compute_true_shares(
    decomposition: torch.Tensor,
    y: torch.Tensor,
    n_periods: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert absolute decomposition values to fractional shares.

    Handles the distribution_cap column specially since it's multiplicative
    (stored as raw multiplier ~0.8-1.0), not an additive component.

    Args:
        decomposition: (B, T, DECOMP_COLS) raw component values.
        y: (B, T) observed demand.
        n_periods: (B,) valid period counts.

    Returns:
        shares: (B, T, DECOMP_COLS) fractional shares.
        valid_mask: (B, T) True for valid (non-padded) timesteps.
    """
    B, T, C = decomposition.shape

    # Build valid timestep mask
    t_idx = torch.arange(T, device=y.device).unsqueeze(0)  # (1, T)
    valid_mask = t_idx < n_periods.unsqueeze(1)  # (B, T)

    # Compute shares: component / y (where y != 0)
    y_safe = y.unsqueeze(-1).clamp(min=1.0)  # (B, T, 1) — avoid div by zero
    shares = decomposition / y_safe  # (B, T, C)

    # Clip extreme share values — outlier samples with small y produce huge shares
    # that dominate the MSE loss. Valid shares are typically in [-0.5, 1.5].
    shares = shares.clamp(-2.0, 2.0)

    # Distribution cap is multiplicative, not additive — set its share to 0
    # (the decoder should learn to ignore this slot, or we mask it in the loss)
    shares[:, :, DECOMP_IDX_DISTRIBUTION] = 0.0

    # Zero out padded timesteps
    shares = shares * valid_mask.unsqueeze(-1).float()

    return shares, valid_mask


class DecompositionEngine:
    """Training and inference engine for per-period demand decomposition.

    Uses the EmbeddingNetwork (temporal mode) + TemporalDecoder to predict
    weekly demand component shares from observable data.

    Args:
        config: DecompositionConfig with all hyperparameters.
    """

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

        self.loss_fn = DecompositionLoss(
            ranking_margin=self.config.ranking_margin,
        ).to(self.device)

        self.optimizer: torch.optim.Optimizer | None = None
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def train(
        self,
        data_dir: str | None = None,
        n_train: int | None = None,
    ) -> dict:
        """Train the encoder-decoder on per-period decomposition data.

        Args:
            data_dir: Directory containing .npz training batches.
            n_train: Max number of training samples to use.

        Returns:
            Dict with training metrics (final loss, best val loss, etc.)
        """
        data_dir = data_dir or self.config.data_dir
        n_train = n_train or self.config.n_train

        logger.info("Loading training data from %s (max %d samples)", data_dir, n_train)
        dataset = DemantiqDataset(data_dir, max_samples=n_train)
        logger.info("Loaded %d samples", len(dataset))

        # Check that decomposition data exists
        sample = dataset[0]
        if sample["decomposition"].abs().sum() == 0:
            raise ValueError(
                "Decomposition data is all zeros. Regenerate training data "
                "with the updated pipeline that saves per-period ground truth."
            )

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

        # Optimizer (loss_fn has no learnable params — uses running-mean normalization)
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = torch.optim.AdamW(
            params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay,
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
            # --- Train ---
            self.encoder.train()
            self.decoder.train()
            self.loss_fn.train()
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

            # --- Validate ---
            val_loss = self._validate(val_loader)
            self.val_losses.append(val_loss)
            scheduler.step(val_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t_start

            if epoch % 5 == 0 or epoch == self.config.n_epochs - 1:
                ls = avg_terms.get("L_share", 0)
                lt = avg_terms.get("L_total", 0)
                lc = avg_terms.get("L_category", 0)
                lr_val = avg_terms.get("L_rank", 0)
                logger.info(
                    "Epoch %3d/%d  loss=%.4f  val=%.4f  "
                    "L_sh=%.4f L_tot=%.4f L_cat=%.4f L_rk=%.4f  "
                    "lr=%.1e  [%.0fs]",
                    epoch + 1, self.config.n_epochs, avg_train_loss, val_loss,
                    ls, lt, lc, lr_val,
                    current_lr, elapsed,
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    "encoder": {k: v.cpu().clone() for k, v in self.encoder.state_dict().items()},
                    "decoder": {k: v.cpu().clone() for k, v in self.decoder.state_dict().items()},
                    "loss_fn": {k: v.cpu().clone() for k, v in self.loss_fn.state_dict().items()},
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, self.config.patience)
                    break

        # Restore best model
        if best_state is not None:
            self.encoder.load_state_dict(best_state["encoder"])
            self.decoder.load_state_dict(best_state["decoder"])
            if "loss_fn" in best_state:
                self.loss_fn.load_state_dict(best_state["loss_fn"])
            self.encoder.to(self.device)
            self.decoder.to(self.device)
            self.loss_fn.to(self.device)

        total_time = time.time() - t_start
        logger.info(
            "Training complete: %.1f minutes, best_val_loss=%.6f",
            total_time / 60, best_val_loss,
        )

        return {
            "best_val_loss": best_val_loss,
            "final_train_loss": self.train_losses[-1],
            "n_epochs_trained": len(self.train_losses),
            "training_time_minutes": total_time / 60,
        }

    def _train_step(self, batch: dict) -> float:
        """Single training step on a batch."""
        self.optimizer.zero_grad()

        y = batch["y"].to(self.device)
        spend = batch["spend"].to(self.device)
        context = batch["context"].to(self.device)
        n_channels = batch["n_channels"].to(self.device)
        channel_type_ids = batch["channel_type_ids"].to(self.device)
        decomposition = batch["decomposition"].to(self.device)
        n_periods = batch["n_periods"].to(self.device)

        # Forward pass
        temporal_emb = self.encoder.forward_temporal(
            y, spend, n_channels, channel_type_ids, context,
        )

        # Handle potential T mismatch (conv padding may change length)
        T_emb = temporal_emb.shape[1]
        T_data = y.shape[1]
        if T_emb != T_data:
            # Interpolate to match data length
            temporal_emb = temporal_emb.permute(0, 2, 1)  # (B, D, T_emb)
            temporal_emb = nn.functional.interpolate(
                temporal_emb, size=T_data, mode="linear", align_corners=False,
            )
            temporal_emb = temporal_emb.permute(0, 2, 1)  # (B, T_data, D)

        pred_shares = self.decoder(temporal_emb)  # (B, T, DECOMP_COLS)

        # Compute true shares from decomposition and y
        true_shares, valid_mask = _compute_true_shares(decomposition, y, n_periods)

        # Composite loss
        loss, loss_terms = self.loss_fn(pred_shares, true_shares, valid_mask, n_channels)

        loss.backward()
        if self.config.grad_clip > 0:
            all_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
            nn.utils.clip_grad_norm_(all_params, self.config.grad_clip)
        self.optimizer.step()

        return loss.item(), loss_terms

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> float:
        """Compute validation loss."""
        self.encoder.eval()
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

            temporal_emb = self.encoder.forward_temporal(
                y, spend, n_channels, channel_type_ids, context,
            )
            T_emb = temporal_emb.shape[1]
            T_data = y.shape[1]
            if T_emb != T_data:
                temporal_emb = temporal_emb.permute(0, 2, 1)
                temporal_emb = nn.functional.interpolate(
                    temporal_emb, size=T_data, mode="linear", align_corners=False,
                )
                temporal_emb = temporal_emb.permute(0, 2, 1)

            pred_shares = self.decoder(temporal_emb)
            true_shares, valid_mask = _compute_true_shares(decomposition, y, n_periods)

            loss, _ = self.loss_fn(pred_shares, true_shares, valid_mask, n_channels)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def infer(
        self,
        y: np.ndarray,
        spend: np.ndarray,
        context: np.ndarray,
        n_channels: int,
        channel_type_ids: np.ndarray,
        n_periods: int,
    ) -> dict:
        """Run inference on a single scenario.

        Args:
            y: (T,) observed demand.
            spend: (T, max_channels) spend matrix.
            context: (T, n_context_cols) context matrix.
            n_channels: Number of active channels.
            channel_type_ids: (max_channels,) channel type indices.
            n_periods: Number of valid periods.

        Returns:
            Dict with:
                shares: (T, DECOMP_COLS) predicted fractional shares.
                contributions: (T, DECOMP_COLS) absolute contributions (shares × y).
                y_reconstructed: (T,) sum of additive contributions.
        """
        self.encoder.eval()
        self.decoder.eval()

        # Convert to tensors with batch dim
        y_t = torch.from_numpy(y).float().unsqueeze(0).to(self.device)
        spend_t = torch.from_numpy(spend).float().unsqueeze(0).to(self.device)
        ctx_t = torch.from_numpy(context).float().unsqueeze(0).to(self.device)
        n_ch_t = torch.tensor([n_channels], device=self.device)
        type_ids_t = torch.from_numpy(channel_type_ids).long().unsqueeze(0).to(self.device)

        # Forward
        temporal_emb = self.encoder.forward_temporal(
            y_t, spend_t, n_ch_t, type_ids_t, ctx_t,
        )
        T_emb = temporal_emb.shape[1]
        T_data = y_t.shape[1]
        if T_emb != T_data:
            temporal_emb = temporal_emb.permute(0, 2, 1)
            temporal_emb = nn.functional.interpolate(
                temporal_emb, size=T_data, mode="linear", align_corners=False,
            )
            temporal_emb = temporal_emb.permute(0, 2, 1)

        pred_shares = self.decoder(temporal_emb)  # (1, T, DECOMP_COLS)
        shares = pred_shares.squeeze(0).cpu().numpy()  # (T, DECOMP_COLS)

        # Compute absolute contributions
        contributions = shares * y[:, np.newaxis]  # (T, DECOMP_COLS)

        # Reconstruct y from additive components (skip distribution_cap)
        additive_mask = np.ones(DECOMP_COLS, dtype=bool)
        additive_mask[DECOMP_IDX_DISTRIBUTION] = False
        y_reconstructed = contributions[:, additive_mask].sum(axis=1)

        return {
            "shares": shares[:n_periods],
            "contributions": contributions[:n_periods],
            "y_reconstructed": y_reconstructed[:n_periods],
        }

    def save(self, path: str) -> None:
        """Save model weights and config."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.encoder.state_dict(), out / "encoder.pt")
        torch.save(self.decoder.state_dict(), out / "decoder.pt")
        torch.save(self.loss_fn.state_dict(), out / "loss_fn.pt")

        import json
        config_dict = {k: v for k, v in self.config.__dict__.items()}
        (out / "decomp_config.json").write_text(json.dumps(config_dict, indent=2))
        logger.info("Model saved to %s", path)

    def load(self, path: str) -> None:
        """Load model weights."""
        out = Path(path)
        self.encoder.load_state_dict(
            torch.load(out / "encoder.pt", map_location=self.device, weights_only=True)
        )
        self.decoder.load_state_dict(
            torch.load(out / "decoder.pt", map_location=self.device, weights_only=True)
        )
        loss_path = out / "loss_fn.pt"
        if loss_path.exists():
            self.loss_fn.load_state_dict(
                torch.load(loss_path, map_location=self.device, weights_only=True)
            )
        logger.info("Model loaded from %s", path)
