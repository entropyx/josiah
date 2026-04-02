"""Compositional inference engine for Demantiq.

Splits inference into two focused density estimators:
  - Global NSF: 2 dims (media_contribution_pct, price_elasticity)
  - Per-channel NSF: 5 dims (beta, roas, frac, price_x_media, dist_x_media)

The per-channel NSF is trained on "unrolled" data — each N-channel simulation
becomes N per-channel training examples, each conditioned on its channel-specific
embedding + the global context. This gives 5x more training data for a 5-dim
problem instead of a 27-dim one.

The embedding network is pre-trained on a regression objective (predict aggregate
metrics from raw data), then frozen. The NSFs train on the pre-computed embeddings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from demantiq.neural.encoders import EmbeddingNetwork
from demantiq.neural.utils import MAX_CHANNELS, NUM_CHANNEL_TYPES
from demantiq.orchestration.training_format import (
    MAX_CONTEXT_COLS,
    _GLOBAL_EXT_TRUTH_LEN,
    _PER_CHANNEL_EXT_TRUTH_LEN,
)

logger = logging.getLogger(__name__)

_GLOBAL_THETA_DIM = 2   # media_pct, elasticity_norm
_CHANNEL_THETA_DIM = 5  # beta_norm, roas_norm, frac, price_x_media, dist_x_media


@dataclass
class CompositionalConfig:
    """Configuration for compositional inference training."""
    # Embedding pre-training
    embed_epochs: int = 50
    embed_lr: float = 1e-3
    embed_batch_size: int = 256
    # NSF training
    nsf_epochs: int = 100
    nsf_patience: int = 30
    nsf_batch_size: int = 256
    nsf_global_transforms: int = 5
    nsf_global_hidden: int = 64
    nsf_channel_transforms: int = 5
    nsf_channel_hidden: int = 128
    # Architecture
    global_summary_dim: int = 256
    temporal_dim: int = 64
    type_embed_dim: int = 16
    n_attn_heads: int = 4
    n_attn_layers: int = 2
    dropout: float = 0.1
    device: str = "cpu"


class CompositionalInferenceEngine:
    """Compositional inference engine with separate global and per-channel posteriors.

    Architecture:
        1. Pre-train EmbeddingNetwork on regression objective (predict aggregate metrics)
        2. Freeze embedding, extract all embeddings for training data
        3. Train Global NSF: 2-dim posterior over (media_pct, elasticity)
        4. Train Per-channel NSF: 5-dim posterior over (beta, roas, frac, px, dx)
           with 5x unrolled data (each sample × N channels)
        5. At inference: one embedding pass + two NSF passes = full posterior

    Args:
        config: CompositionalConfig with hyperparameters.
    """

    def __init__(self, config: CompositionalConfig | None = None):
        self.config = config or CompositionalConfig()
        self.device = torch.device(self.config.device)
        self.embedding_net: EmbeddingNetwork | None = None
        self.global_posterior = None
        self.channel_posterior = None
        self._is_trained = False
        self._beta_scale = 1.0
        self._roas_scale = 1.0
        self._elasticity_scale = 1.0
        self._n_theta_channels: int = MAX_CHANNELS
        self.max_T: int = 260

    def train(
        self,
        data_dir: str,
        max_samples: int | None = None,
        save_dir: str | None = None,
        n_fixed_channels: int | None = None,
    ) -> dict[str, Any]:
        """Train the compositional inference engine.

        Steps:
            1. Load data
            2. Pre-train embedding network
            3. Extract embeddings (frozen)
            4. Train global NSF on global_summary → (media_pct, elasticity)
            5. Train per-channel NSF on (per_ch_summary + global) → (beta, roas, frac, px, dx)

        Args:
            data_dir: Directory with .npz training batches.
            max_samples: Optional sample limit.
            save_dir: Optional directory to save model.
            n_fixed_channels: If set, only use this many channels per sample.

        Returns:
            Dict with training metrics.
        """
        from demantiq.neural.data_loader import DemantiqDataset
        from demantiq.neural.inference import pack_observations

        # Step 1: Load data
        logger.info("Loading training data from %s", data_dir)
        dataset = DemantiqDataset(data_dir, max_samples=max_samples)
        self.max_T = dataset.y.shape[1]
        self._n_theta_channels = n_fixed_channels if n_fixed_channels else MAX_CHANNELS
        n_total = len(dataset)

        logger.info("Loaded %d samples, max_T=%d, theta_channels=%d",
                     n_total, self.max_T, self._n_theta_channels)

        # Step 2: Pre-train embedding network
        logger.info("=== Phase 1: Pre-training embedding network ===")
        self.embedding_net = EmbeddingNetwork(
            temporal_dim=self.config.temporal_dim,
            type_embed_dim=self.config.type_embed_dim,
            n_attn_heads=self.config.n_attn_heads,
            n_attn_layers=self.config.n_attn_layers,
            global_summary_dim=self.config.global_summary_dim,
            dropout=self.config.dropout,
        ).to(self.device)

        self._pretrain_embedding(dataset)

        # Step 3: Extract all embeddings (frozen)
        logger.info("=== Phase 2: Extracting embeddings ===")
        self.embedding_net.eval()
        global_embs, per_ch_embs = self._extract_embeddings(dataset)
        logger.info("Extracted embeddings: global(%s), per_channel(%s)",
                     global_embs.shape, per_ch_embs.shape)

        # Compute normalization scales
        self._compute_scales(dataset)

        # Step 4: Build global training data
        logger.info("=== Phase 3: Training global NSF (2-dim) ===")
        theta_global = torch.zeros(n_total, _GLOBAL_THETA_DIM)
        for i in range(n_total):
            ext = dataset[i]["ext_truth"]
            theta_global[i, 0] = ext[0]  # media_contribution_pct
            theta_global[i, 1] = ext[1] / self._elasticity_scale  # elasticity normalized

        self.global_posterior = self._train_nsf(
            theta=theta_global,
            obs=global_embs,
            prior_low=torch.tensor([-0.1, -1.1]),
            prior_high=torch.tensor([1.1, 0.1]),
            n_transforms=self.config.nsf_global_transforms,
            hidden_features=self.config.nsf_global_hidden,
            label="global",
        )

        # Step 5: Build unrolled per-channel training data
        logger.info("=== Phase 4: Training per-channel NSF (5-dim, unrolled) ===")
        per_ch_dim = self.embedding_net.per_channel_dim  # 336
        global_dim = self.config.global_summary_dim       # 256
        obs_dim = per_ch_dim + global_dim                 # 592

        theta_channel_list = []
        obs_channel_list = []

        for i in range(n_total):
            ext = dataset[i]["ext_truth"]
            n_ch = min(dataset[i]["n_channels"], self._n_theta_channels)

            for ch_idx in range(n_ch):
                base_ext = _GLOBAL_EXT_TRUTH_LEN + ch_idx * _PER_CHANNEL_EXT_TRUTH_LEN

                if base_ext + 14 >= len(ext):
                    continue

                theta_ch = torch.zeros(_CHANNEL_THETA_DIM)
                theta_ch[0] = ext[base_ext + 0] / self._beta_scale
                theta_ch[1] = ext[base_ext + 1] / self._roas_scale
                theta_ch[2] = ext[base_ext + 3]   # contribution_frac
                theta_ch[3] = ext[base_ext + 13]  # price_x_media
                theta_ch[4] = ext[base_ext + 14]  # distribution_x_media
                theta_channel_list.append(theta_ch)

                # Observation: per_channel_embedding + global_embedding
                obs_ch = torch.cat([per_ch_embs[i, ch_idx], global_embs[i]])
                obs_channel_list.append(obs_ch)

        theta_channel = torch.stack(theta_channel_list)
        obs_channel = torch.stack(obs_channel_list)
        logger.info("Unrolled per-channel data: %d examples (%.1fx amplification)",
                     len(theta_channel), len(theta_channel) / n_total)

        # Per-channel prior bounds
        prior_low_ch = torch.tensor([-0.1, -0.1, -0.1, -0.15, -0.15])
        prior_high_ch = torch.tensor([1.1, 1.1, 1.1, 0.35, 0.35])

        self.channel_posterior = self._train_nsf(
            theta=theta_channel,
            obs=obs_channel,
            prior_low=prior_low_ch,
            prior_high=prior_high_ch,
            n_transforms=self.config.nsf_channel_transforms,
            hidden_features=self.config.nsf_channel_hidden,
            label="per-channel",
        )

        self._is_trained = True

        if save_dir:
            self.save(save_dir)

        return {"status": "trained"}

    def infer(
        self,
        y: np.ndarray,
        spend_matrix: np.ndarray,
        channel_names: list[str],
        n_samples: int = 10000,
        context_matrix: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run compositional inference on a single observation.

        Args:
            y: (T,) outcome time series.
            spend_matrix: (T, n_channels) spend matrix.
            channel_names: List of channel names.
            n_samples: Number of posterior samples.
            context_matrix: (T, n_context_cols) business context (optional).

        Returns:
            Dict with posterior samples and per-channel summaries.
        """
        if not self._is_trained:
            raise RuntimeError("Engine not trained. Call train() first.")

        from demantiq.neural.utils import CHANNEL_NAME_TO_IDX

        n_channels = len(channel_names)
        T = len(y)

        # Build tensors
        type_ids = torch.zeros(1, MAX_CHANNELS, dtype=torch.long)
        for i, name in enumerate(channel_names):
            if i < MAX_CHANNELS:
                type_ids[0, i] = CHANNEL_NAME_TO_IDX.get(name, 0)

        spend_padded = np.zeros((T, MAX_CHANNELS), dtype=np.float32)
        spend_padded[:, :spend_matrix.shape[1]] = spend_matrix

        ctx = None
        if context_matrix is not None:
            ctx_padded = np.zeros((T, MAX_CONTEXT_COLS), dtype=np.float32)
            ctx_padded[:, :context_matrix.shape[1]] = context_matrix
            ctx = torch.tensor(ctx_padded, dtype=torch.float32).unsqueeze(0).to(self.device)

        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(self.device)
        spend_t = torch.tensor(spend_padded, dtype=torch.float32).unsqueeze(0).to(self.device)
        n_ch_t = torch.tensor([n_channels], device=self.device)
        type_ids_t = type_ids.to(self.device)

        # Run embedding network
        self.embedding_net.eval()
        with torch.no_grad():
            global_emb, per_ch_emb, ch_mask = self.embedding_net(
                y=y_t, spend=spend_t, n_channels=n_ch_t,
                channel_type_ids=type_ids_t, context=ctx,
            )

        # Sample from posteriors using the underlying flow directly
        # (bypasses sbi's rejection sampling which crashes on spline edge cases)
        global_samples = _sample_direct(
            self.global_posterior, n_samples, global_emb.squeeze(0)
        ).cpu().numpy()

        result = {
            "total_media_contribution_pct": _summarize(global_samples[:, 0]),
            "price_elasticity": _summarize(global_samples[:, 1] * self._elasticity_scale),
            "raw_global_samples": global_samples,
        }

        # Sample from per-channel posterior for each channel
        all_ch_samples = {}
        for ch_idx, name in enumerate(channel_names):
            if ch_idx >= self._n_theta_channels:
                break

            x_ch = torch.cat([
                per_ch_emb[0, ch_idx],
                global_emb.squeeze(0),
            ]).to(self.device)

            ch_samples = _sample_direct(
                self.channel_posterior, n_samples, x_ch
            ).cpu().numpy()

            all_ch_samples[name] = ch_samples

            beta_samples = ch_samples[:, 0] * self._beta_scale
            roas_samples = ch_samples[:, 1] * self._roas_scale

            ch_result = {
                "beta": _summarize(beta_samples),
                "roas": _summarize(roas_samples),
                "contribution_fraction": _summarize(ch_samples[:, 2]),
                "price_x_media": _summarize(ch_samples[:, 3]),
                "distribution_x_media": _summarize(ch_samples[:, 4]),
            }
            result[name] = ch_result

        result["raw_channel_samples"] = all_ch_samples
        return result

    def _pretrain_embedding(self, dataset) -> None:
        """Pre-train embedding network with both global and per-channel objectives.

        Two prediction heads trained simultaneously:
        - Global head: global_summary(256) → [media_pct, elasticity, mean_beta, mean_roas]
        - Per-channel head: per_channel_summary(336) → [beta, roas, contrib_frac, px_media, dx_media]

        The per-channel head forces the Set Transformer to produce DIFFERENTIATED
        per-channel embeddings, not identical ones.
        """
        self.embedding_net.train()

        per_ch_dim = self.embedding_net.per_channel_dim  # 336

        # Two prediction heads
        global_head = nn.Linear(self.config.global_summary_dim, 4).to(self.device)
        channel_head = nn.Linear(per_ch_dim, 5).to(self.device)

        optimizer = torch.optim.Adam(
            list(self.embedding_net.parameters())
            + list(global_head.parameters())
            + list(channel_head.parameters()),
            lr=self.config.embed_lr,
        )

        n_total = len(dataset)
        indices = torch.randperm(n_total)
        bs = self.config.embed_batch_size

        for epoch in range(self.config.embed_epochs):
            total_loss = 0.0
            global_loss_sum = 0.0
            channel_loss_sum = 0.0
            n_batches = 0

            for start in range(0, n_total, bs):
                batch_idx = indices[start:start + bs]
                batch_items = [dataset[int(i)] for i in batch_idx]
                batch_size_actual = len(batch_items)

                y_batch = torch.stack([item["y"] for item in batch_items]).to(self.device)
                spend_raw = torch.stack([item["spend"] for item in batch_items])
                if spend_raw.shape[2] < MAX_CHANNELS:
                    pad = torch.zeros(spend_raw.shape[0], spend_raw.shape[1],
                                      MAX_CHANNELS - spend_raw.shape[2])
                    spend_raw = torch.cat([spend_raw, pad], dim=2)
                spend_batch = spend_raw.to(self.device)
                n_ch_batch = torch.tensor([item["n_channels"] for item in batch_items],
                                          device=self.device)
                type_ids_batch = torch.stack([item["channel_type_ids"]
                                              for item in batch_items]).to(self.device)
                ctx_batch = torch.stack([item["context"] for item in batch_items]).to(self.device)

                # Global targets
                global_targets = torch.zeros(batch_size_actual, 4, device=self.device)
                for j, item in enumerate(batch_items):
                    ext = item["ext_truth"]
                    global_targets[j, 0] = ext[0]  # media_pct
                    global_targets[j, 1] = ext[1]  # elasticity
                    n_ch = item["n_channels"]
                    beta_sum, roas_sum = 0.0, 0.0
                    for ch in range(min(n_ch, self._n_theta_channels)):
                        base = _GLOBAL_EXT_TRUTH_LEN + ch * _PER_CHANNEL_EXT_TRUTH_LEN
                        if base < len(ext):
                            beta_sum += ext[base + 0].item()
                            roas_sum += ext[base + 1].item()
                    global_targets[j, 2] = beta_sum / max(n_ch, 1) / 500.0
                    global_targets[j, 3] = roas_sum / max(n_ch, 1)

                # Per-channel targets: collect (sample_idx, ch_idx, target_vector)
                ch_targets_list = []
                ch_indices = []  # (sample_idx, ch_idx) pairs
                for j, item in enumerate(batch_items):
                    ext = item["ext_truth"]
                    n_ch = min(item["n_channels"], self._n_theta_channels)
                    for ch_idx in range(n_ch):
                        base = _GLOBAL_EXT_TRUTH_LEN + ch_idx * _PER_CHANNEL_EXT_TRUTH_LEN
                        if base + 14 < len(ext):
                            target = torch.zeros(5)
                            target[0] = ext[base + 0] / 500.0   # beta normalized
                            target[1] = ext[base + 1]            # roas (raw, small)
                            target[2] = ext[base + 3]            # contribution_frac
                            target[3] = ext[base + 13]           # price_x_media
                            target[4] = ext[base + 14]           # dist_x_media
                            ch_targets_list.append(target)
                            ch_indices.append((j, ch_idx))

                # Forward pass
                global_emb, per_ch_emb, ch_mask = self.embedding_net(
                    y=y_batch, spend=spend_batch, n_channels=n_ch_batch,
                    channel_type_ids=type_ids_batch, context=ctx_batch,
                )

                # Global loss
                global_preds = global_head(global_emb)
                loss_global = nn.functional.mse_loss(global_preds, global_targets)

                # Per-channel loss
                if ch_targets_list:
                    ch_targets = torch.stack(ch_targets_list).to(self.device)
                    # Gather the corresponding per-channel embeddings
                    ch_embs = torch.stack([
                        per_ch_emb[sample_idx, ch_idx]
                        for sample_idx, ch_idx in ch_indices
                    ])
                    ch_preds = channel_head(ch_embs)
                    loss_channel = nn.functional.mse_loss(ch_preds, ch_targets)
                else:
                    loss_channel = torch.tensor(0.0, device=self.device)

                # Combined loss (weight channel loss higher to force differentiation)
                loss = loss_global + 2.0 * loss_channel

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                global_loss_sum += loss_global.item()
                channel_loss_sum += loss_channel.item()
                n_batches += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info("Embedding epoch %d/%d: total=%.4f (global=%.4f, channel=%.4f)",
                             epoch + 1, self.config.embed_epochs,
                             total_loss / max(n_batches, 1),
                             global_loss_sum / max(n_batches, 1),
                             channel_loss_sum / max(n_batches, 1))

    def _extract_embeddings(self, dataset) -> tuple[Tensor, Tensor]:
        """Extract global and per-channel embeddings for all samples (no gradient)."""
        n_total = len(dataset)
        global_dim = self.config.global_summary_dim
        per_ch_dim = self.embedding_net.per_channel_dim
        bs = self.config.embed_batch_size

        global_embs = torch.zeros(n_total, global_dim)
        per_ch_embs = torch.zeros(n_total, MAX_CHANNELS, per_ch_dim)

        with torch.no_grad():
            for start in range(0, n_total, bs):
                end = min(start + bs, n_total)
                batch_items = [dataset[i] for i in range(start, end)]

                y_batch = torch.stack([item["y"] for item in batch_items]).to(self.device)
                spend_raw = torch.stack([item["spend"] for item in batch_items])
                # Pad spend to MAX_CHANNELS to match type_ids dimension
                if spend_raw.shape[2] < MAX_CHANNELS:
                    pad = torch.zeros(spend_raw.shape[0], spend_raw.shape[1],
                                      MAX_CHANNELS - spend_raw.shape[2])
                    spend_raw = torch.cat([spend_raw, pad], dim=2)
                spend_batch = spend_raw.to(self.device)
                n_ch_batch = torch.tensor([item["n_channels"] for item in batch_items],
                                          device=self.device)
                type_ids_batch = torch.stack([item["channel_type_ids"]
                                              for item in batch_items]).to(self.device)
                ctx_batch = torch.stack([item["context"] for item in batch_items]).to(self.device)

                g, pc, _ = self.embedding_net(
                    y=y_batch, spend=spend_batch, n_channels=n_ch_batch,
                    channel_type_ids=type_ids_batch, context=ctx_batch,
                )

                global_embs[start:end] = g.cpu()
                per_ch_embs[start:end] = pc.cpu()

        return global_embs, per_ch_embs

    def _compute_scales(self, dataset) -> None:
        """Compute normalization scales from training data."""
        beta_max = 0.0
        roas_max = 0.0
        elast_max = 0.0

        for i in range(len(dataset)):
            ext = dataset[i]["ext_truth"]
            elast_max = max(elast_max, abs(float(ext[1])))
            for ch in range(self._n_theta_channels):
                base = _GLOBAL_EXT_TRUTH_LEN + ch * _PER_CHANNEL_EXT_TRUTH_LEN
                if base + 1 < len(ext):
                    beta_max = max(beta_max, abs(float(ext[base + 0])))
                    roas_max = max(roas_max, abs(float(ext[base + 1])))

        # Ensure non-zero scales
        self._beta_scale = max(beta_max, 1.0)
        self._roas_scale = max(roas_max, 0.001)
        self._elasticity_scale = max(elast_max, 0.1)
        logger.info("Normalization scales: beta=%.1f, roas=%.3f, elasticity=%.3f",
                     beta_max, roas_max, elast_max)

    def _train_nsf(
        self,
        theta: Tensor,
        obs: Tensor,
        prior_low: Tensor,
        prior_high: Tensor,
        n_transforms: int,
        hidden_features: int,
        label: str,
    ):
        """Train a single SNPE_C + NSF density estimator."""
        from sbi.inference import SNPE_C
        from sbi.neural_nets import posterior_nn
        from sbi.utils import BoxUniform

        theta = theta.to(self.device)
        obs = obs.to(self.device)

        # Clamp theta to prior
        prior_low = prior_low.to(self.device)
        prior_high = prior_high.to(self.device)
        theta = theta.clamp(prior_low.unsqueeze(0), prior_high.unsqueeze(0))

        prior = BoxUniform(low=prior_low, high=prior_high)

        density_builder = posterior_nn(
            "nsf",
            hidden_features=hidden_features,
            num_transforms=n_transforms,
            z_score_theta="independent",
            z_score_x="independent",
        )

        trainer = SNPE_C(
            prior=prior,
            density_estimator=density_builder,
            device=str(self.device),
        )

        # Split train/val
        n = theta.shape[0]
        n_val = max(1, int(n * 0.1))
        n_train = n - n_val
        perm = torch.randperm(n)
        train_theta = theta[perm[:n_train]]
        train_obs = obs[perm[:n_train]]

        logger.info("Training %s NSF: %d dims, %d samples, %d transforms",
                     label, theta.shape[1], n_train, n_transforms)

        trainer.append_simulations(train_theta, train_obs)

        density_estimator = trainer.train(
            training_batch_size=min(self.config.nsf_batch_size, n_train),
            learning_rate=5e-4,
            max_num_epochs=self.config.nsf_epochs,
            stop_after_epochs=self.config.nsf_patience,
            show_train_summary=True,
        )

        posterior = trainer.build_posterior(density_estimator, sample_with="direct")
        logger.info("%s NSF training complete.", label.capitalize())
        return posterior

    def save(self, save_dir: str) -> None:
        """Save trained compositional model."""
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)

        torch.save({
            "embedding_state_dict": self.embedding_net.state_dict(),
            "global_posterior": self.global_posterior,
            "channel_posterior": self.channel_posterior,
            "max_T": self.max_T,
            "config": self.config,
            "is_trained": self._is_trained,
            "beta_scale": self._beta_scale,
            "roas_scale": self._roas_scale,
            "elasticity_scale": self._elasticity_scale,
            "n_theta_channels": self._n_theta_channels,
            "engine_type": "compositional",
        }, str(path / "demantiq_neural.pt"))
        logger.info("Compositional model saved to %s", path / "demantiq_neural.pt")

    def load(self, save_dir: str) -> None:
        """Load trained compositional model."""
        path = Path(save_dir) / "demantiq_neural.pt"
        checkpoint = torch.load(str(path), map_location=self.device, weights_only=False)

        if checkpoint.get("engine_type") != "compositional":
            raise ValueError("Checkpoint is not a compositional model. Use DemantiqInferenceEngine instead.")

        self.config = checkpoint["config"]
        self.max_T = checkpoint["max_T"]
        self._beta_scale = checkpoint["beta_scale"]
        self._roas_scale = checkpoint["roas_scale"]
        self._elasticity_scale = checkpoint["elasticity_scale"]
        self._n_theta_channels = checkpoint["n_theta_channels"]

        self.embedding_net = EmbeddingNetwork(
            temporal_dim=self.config.temporal_dim,
            type_embed_dim=self.config.type_embed_dim,
            n_attn_heads=self.config.n_attn_heads,
            n_attn_layers=self.config.n_attn_layers,
            global_summary_dim=self.config.global_summary_dim,
            dropout=self.config.dropout,
        ).to(self.device)
        self.embedding_net.load_state_dict(checkpoint["embedding_state_dict"])
        self.embedding_net.eval()

        self.global_posterior = checkpoint["global_posterior"]
        self.channel_posterior = checkpoint["channel_posterior"]
        self._is_trained = checkpoint["is_trained"]

        # Force direct sampling (skip rejection against prior to avoid spline crashes)
        if hasattr(self.global_posterior, '_sample_with'):
            self.global_posterior._sample_with = "direct"
        if hasattr(self.channel_posterior, '_sample_with'):
            self.channel_posterior._sample_with = "direct"

        logger.info("Compositional model loaded from %s", path)


def _sample_direct(posterior, n_samples: int, x: Tensor) -> Tensor:
    """Sample directly from the posterior's underlying flow.

    Bypasses sbi's rejection sampling which hangs/crashes when the NSF
    produces samples outside the prior bounds.
    """
    net = posterior.posterior_estimator
    x_context = x.unsqueeze(0) if x.dim() == 1 else x
    with torch.no_grad():
        samples = net.sample(torch.Size((n_samples,)), condition=x_context)
    if samples.dim() == 3:
        samples = samples.squeeze(1)
    return samples.detach()


def _summarize(samples: np.ndarray) -> dict[str, float]:
    """Compute summary statistics for posterior samples."""
    return {
        "mean": float(samples.mean()),
        "std": float(samples.std()),
        "q05": float(np.percentile(samples, 5)),
        "q95": float(np.percentile(samples, 95)),
    }
