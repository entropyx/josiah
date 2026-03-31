"""FMPE training and inference wrapper for Demantiq.

Uses sbi's Flow Matching Posterior Estimation with a custom embedding network
built from the Demantiq encoders. Supports both monolithic (Phase 1) and
compositional (Phase 2) inference.

Phase 1 (this file): Train a single FMPE model that maps
    (y, spend, channel_metadata) → posterior over all parameters.

The embedding network produces a global summary that conditions the flow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from demantiq.neural.encoders import EmbeddingNetwork
from demantiq.neural.utils import (
    MAX_CHANNELS,
    NUM_CHANNEL_TYPES,
    make_channel_mask,
    extract_per_channel_truth,
)
from demantiq.orchestration.training_format import (
    _GLOBAL_EXT_TRUTH_LEN,
    _PER_CHANNEL_EXT_TRUTH_LEN,
    EXT_TRUTH_VECTOR_LEN,
    MAX_CONTEXT_COLS,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for neural inference training.

    Args:
        learning_rate: Learning rate for the optimizer.
        n_epochs: Number of training epochs.
        batch_size: Training batch size.
        val_fraction: Fraction of data used for validation.
        patience: Early stopping patience (epochs without improvement).
        global_summary_dim: Dimension of the global embedding.
        temporal_dim: Dimension of the temporal encoder output.
        type_embed_dim: Dimension of channel-type embeddings.
        n_attn_heads: Number of attention heads in Set Transformer.
        n_attn_layers: Number of Set Transformer layers.
        n_flow_transforms: Number of flow transforms in the density estimator.
        dropout: Dropout rate in the embedding network.
        device: Device to train on ('cpu', 'cuda', 'mps').
    """
    learning_rate: float = 5e-4
    n_epochs: int = 200
    batch_size: int = 256
    val_fraction: float = 0.1
    patience: int = 20
    global_summary_dim: int = 256
    temporal_dim: int = 64
    type_embed_dim: int = 16
    n_attn_heads: int = 4
    n_attn_layers: int = 2
    simple_embedding: bool = False  # Use plain MLP instead of full embedding network
    n_flow_transforms: int = 5
    dropout: float = 0.1
    device: str = "cpu"


class DemantiqEmbeddingWrapper(nn.Module):
    """Wraps the EmbeddingNetwork to produce a flat embedding from a flat input.

    sbi expects embedding_net to take a single tensor and return a single tensor.
    This wrapper unpacks the flat observation tensor, runs the EmbeddingNetwork,
    and returns the global summary.

    The flat observation tensor layout:
        [y (max_T), spend_flat (max_T * max_ch), n_channels (1), n_periods (1),
         channel_type_ids (max_ch)]

    Args:
        embedding_net: The Demantiq EmbeddingNetwork instance.
        max_T: Maximum time series length in the dataset.
        max_ch: Maximum number of channels.
    """

    def __init__(self, embedding_net: EmbeddingNetwork, max_T: int, max_ch: int = MAX_CHANNELS):
        super().__init__()
        self.embedding_net = embedding_net
        self.max_T = max_T
        self.max_ch = max_ch
        self.n_ctx = MAX_CONTEXT_COLS
        # Total flat observation size:
        # y(max_T) + spend(max_T * max_ch) + context(max_T * n_ctx) + n_channels(1) + n_periods(1) + type_ids(max_ch)
        self.flat_obs_dim = max_T + max_T * max_ch + max_T * self.n_ctx + 1 + 1 + max_ch

    def forward(self, x: Tensor) -> Tensor:
        """Unpack flat observation and compute embedding.

        Args:
            x: (batch, flat_obs_dim) flat observation tensor.

        Returns:
            (batch, global_summary_dim) embedding.
        """
        idx = 0
        y = x[:, idx:idx + self.max_T]
        idx += self.max_T

        spend_flat = x[:, idx:idx + self.max_T * self.max_ch]
        spend = spend_flat.reshape(-1, self.max_T, self.max_ch)
        idx += self.max_T * self.max_ch

        context_flat = x[:, idx:idx + self.max_T * self.n_ctx]
        context = context_flat.reshape(-1, self.max_T, self.n_ctx)
        idx += self.max_T * self.n_ctx

        n_channels = x[:, idx]
        idx += 1

        n_periods = x[:, idx]
        idx += 1

        channel_type_ids = x[:, idx:idx + self.max_ch]

        # Recover integer metadata from potentially z-scored values
        n_channels_int = n_channels.round().long().clamp(1, self.max_ch)
        channel_type_ids_int = channel_type_ids.round().long().clamp(0, NUM_CHANNEL_TYPES - 1)

        global_summary, _, _ = self.embedding_net(
            y=y,
            spend=spend,
            n_channels=n_channels_int,
            channel_type_ids=channel_type_ids_int,
            context=context,
        )
        return global_summary


def pack_observations(
    y: Tensor,
    spend: Tensor,
    n_channels: Tensor,
    n_periods: Tensor,
    channel_type_ids: Tensor,
    max_T: int,
    context: Tensor | None = None,
) -> Tensor:
    """Pack observation components into a flat tensor for sbi.

    Args:
        y: (batch, T) outcome time series.
        spend: (batch, T, max_ch) spend matrix.
        n_channels: (batch,) number of active channels.
        n_periods: (batch,) number of valid timesteps.
        channel_type_ids: (batch, max_ch) channel type indices.
        max_T: Maximum time series length to pad to.
        context: (batch, T, n_context_cols) business context matrix (optional).

    Returns:
        (batch, flat_obs_dim) packed tensor.
    """
    batch_size = y.shape[0]
    device = y.device

    # Pad y to max_T if needed
    T = y.shape[1]
    if T < max_T:
        y_padded = torch.zeros(batch_size, max_T, device=device)
        y_padded[:, :T] = y
    else:
        y_padded = y[:, :max_T]

    # Pad spend to (max_T, MAX_CHANNELS)
    spend_padded = torch.zeros(batch_size, max_T, MAX_CHANNELS, device=device)
    t_end = min(spend.shape[1], max_T)
    c_end = min(spend.shape[2], MAX_CHANNELS)
    spend_padded[:, :t_end, :c_end] = spend[:, :t_end, :c_end]
    spend_flat = spend_padded.reshape(batch_size, -1)

    # Pad context to (max_T, MAX_CONTEXT_COLS)
    context_padded = torch.zeros(batch_size, max_T, MAX_CONTEXT_COLS, device=device)
    if context is not None:
        ct_end = min(context.shape[1], max_T)
        cc_end = min(context.shape[2], MAX_CONTEXT_COLS)
        context_padded[:, :ct_end, :cc_end] = context[:, :ct_end, :cc_end]
    context_flat = context_padded.reshape(batch_size, -1)

    return torch.cat([
        y_padded,
        spend_flat,
        context_flat,
        n_channels.float().unsqueeze(-1),
        n_periods.float().unsqueeze(-1),
        channel_type_ids.float(),
    ], dim=-1)


class DemantiqInferenceEngine:
    """Main inference engine for Demantiq.

    Wraps the training loop, model management, and inference methods.

    Args:
        config: Training configuration.
    """

    def __init__(self, config: TrainingConfig | None = None):
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device)
        self.embedding_net: EmbeddingNetwork | None = None
        self.posterior = None
        self.max_T: int = 260  # Will be set from data
        self._is_trained = False

    def train(
        self,
        data_dir: str,
        max_samples: int | None = None,
        save_dir: str | None = None,
        n_fixed_channels: int | None = None,
    ) -> dict[str, list[float]]:
        """Train the neural inference engine on pre-generated data.

        Args:
            data_dir: Directory containing batch_*.npz training data.
            max_samples: Optional limit on training samples.
            save_dir: Optional directory to save trained model.

        Returns:
            Dict with training history (train_loss, val_loss per epoch).
        """
        from demantiq.neural.data_loader import DemantiqDataset

        logger.info("Loading training data from %s", data_dir)
        dataset = DemantiqDataset(data_dir, max_samples=max_samples)
        self.max_T = dataset.y.shape[1]
        n_total = len(dataset)

        # Split train/val
        n_val = max(1, int(n_total * self.config.val_fraction))
        n_train = n_total - n_val
        indices = torch.randperm(n_total)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        logger.info("Training: %d samples, Validation: %d samples", n_train, n_val)

        # First pass: collect ext_truth for normalization
        all_ext_truth = []
        for i in range(n_total):
            item = dataset[i]
            all_ext_truth.append(item["ext_truth"])
        ext_stack = torch.stack(all_ext_truth)

        # Build reduced theta: [media_pct, price_elasticity_norm, ch0_beta_norm, ch0_roas, ch0_frac, ...]
        _REDUCED_PER_CH = 5  # beta_norm, roas_norm, contrib_frac, price_x_media, dist_x_media
        _GLOBAL_REDUCED = 2  # media_pct, price_elasticity_normalized

        # Dynamic sizing: use fixed channel count if specified, otherwise MAX_CHANNELS
        self._n_theta_channels = n_fixed_channels if n_fixed_channels is not None else MAX_CHANNELS
        self._n_reduced_params = _GLOBAL_REDUCED + self._n_theta_channels * _REDUCED_PER_CH
        self._reduced_per_ch = _REDUCED_PER_CH
        self._global_reduced = _GLOBAL_REDUCED
        logger.info("Theta dimensions: %d (%d global + %d channels × %d per-channel)",
                     self._n_reduced_params, _GLOBAL_REDUCED, self._n_theta_channels, _REDUCED_PER_CH)

        # Compute normalization scales from training data
        beta_max = 1.0
        roas_max = 1.0
        elasticity_vals = ext_stack[:, 1]  # price_elasticity at index 1
        elasticity_max = max(float(elasticity_vals.abs().max()), 1.0)

        for ch_idx in range(self._n_theta_channels):
            base = _GLOBAL_EXT_TRUTH_LEN + ch_idx * _PER_CHANNEL_EXT_TRUTH_LEN
            if base + 1 >= ext_stack.shape[1]:
                break
            betas = ext_stack[:, base + 0]
            roas_vals = ext_stack[:, base + 1]
            if betas.abs().max() > beta_max:
                beta_max = float(betas.abs().max())
            if roas_vals.abs().max() > roas_max:
                roas_max = float(roas_vals.abs().max())

        self._beta_scale = beta_max
        self._roas_scale = roas_max
        self._elasticity_scale = elasticity_max
        logger.info("Normalization scales: beta_max=%.1f, roas_max=%.3f, elasticity_max=%.3f",
                     beta_max, roas_max, elasticity_max)

        # Build observations and theta
        logger.info("Packing observations and building theta vectors...")
        all_obs = []
        all_theta = []

        if self.config.simple_embedding:
            # Simple path: extract hand-crafted summary statistics per sample
            # instead of raw time series. This gives the NSF a much smaller,
            # cleaner input that sbi's default z-scoring handles well.
            logger.info("Using SIMPLE embedding (summary statistics)")
            self._use_simple = True

            for i in range(n_total):
                item = dataset[i]
                obs = _extract_summary_stats(
                    item["y"], item["spend"], item.get("context"),
                    item["n_channels"], self._n_theta_channels,
                )
                all_obs.append(obs)

                ext = item["ext_truth"]
                theta = torch.zeros(self._n_reduced_params)
                theta[0] = ext[0]
                theta[1] = ext[1] / elasticity_max

                for ch_idx in range(self._n_theta_channels):
                    base_ext = _GLOBAL_EXT_TRUTH_LEN + ch_idx * _PER_CHANNEL_EXT_TRUTH_LEN
                    base_red = _GLOBAL_REDUCED + ch_idx * _REDUCED_PER_CH
                    if base_ext + 14 < len(ext):
                        theta[base_red + 0] = ext[base_ext + 0] / beta_max
                        theta[base_red + 1] = ext[base_ext + 1] / roas_max
                        theta[base_red + 2] = ext[base_ext + 3]   # contribution_frac
                        theta[base_red + 3] = ext[base_ext + 13]  # price_x_media
                        theta[base_red + 4] = ext[base_ext + 14]  # distribution_x_media
                all_theta.append(theta)
        else:
            # Full embedding path: pack raw time series for custom embedding network
            logger.info("Using FULL embedding network")
            self._use_simple = False

            self.embedding_net = EmbeddingNetwork(
                temporal_dim=self.config.temporal_dim,
                type_embed_dim=self.config.type_embed_dim,
                n_attn_heads=self.config.n_attn_heads,
                n_attn_layers=self.config.n_attn_layers,
                global_summary_dim=self.config.global_summary_dim,
                dropout=self.config.dropout,
            ).to(self.device)

            embedding_wrapper = DemantiqEmbeddingWrapper(
                self.embedding_net, max_T=self.max_T
            ).to(self.device)

            for i in range(n_total):
                item = dataset[i]
                ctx = item.get("context")
                if ctx is not None:
                    ctx = ctx.unsqueeze(0)
                obs = pack_observations(
                    y=item["y"].unsqueeze(0),
                    spend=item["spend"].unsqueeze(0),
                    n_channels=torch.tensor([item["n_channels"]]),
                    n_periods=torch.tensor([item["n_periods"]]),
                    channel_type_ids=item["channel_type_ids"].unsqueeze(0),
                    max_T=self.max_T,
                    context=ctx,
                ).squeeze(0)
                all_obs.append(obs)

                ext = item["ext_truth"]
                theta = torch.zeros(self._n_reduced_params)
                theta[0] = ext[0]
                theta[1] = ext[1] / elasticity_max

                for ch_idx in range(self._n_theta_channels):
                    base_ext = _GLOBAL_EXT_TRUTH_LEN + ch_idx * _PER_CHANNEL_EXT_TRUTH_LEN
                    base_red = _GLOBAL_REDUCED + ch_idx * _REDUCED_PER_CH
                    if base_ext + 14 < len(ext):
                        theta[base_red + 0] = ext[base_ext + 0] / beta_max
                        theta[base_red + 1] = ext[base_ext + 1] / roas_max
                        theta[base_red + 2] = ext[base_ext + 3]   # contribution_frac
                        theta[base_red + 3] = ext[base_ext + 13]  # price_x_media
                        theta[base_red + 4] = ext[base_ext + 14]  # distribution_x_media
                all_theta.append(theta)

        obs_tensor = torch.stack(all_obs).to(self.device)
        theta_tensor = torch.stack(all_theta).to(self.device)

        # Build the sbi inference object
        from sbi.inference import SNPE_C
        from sbi.neural_nets import posterior_nn
        from sbi.utils import BoxUniform

        logger.info("Using sbi SNPE_C with Neural Spline Flows")
        logger.info("Reduced parameter space: %d dims, observation space: %d dims",
                     self._n_reduced_params, obs_tensor.shape[1])

        # Prior bounds per parameter type
        prior_low = torch.full((self._n_reduced_params,), -0.1, device=self.device)
        prior_high = torch.full((self._n_reduced_params,), 1.1, device=self.device)
        # Price elasticity (index 1) is NEGATIVE after normalization (e.g., -1.5/3.0 = -0.5)
        prior_low[1] = -1.1
        prior_high[1] = 0.1
        # Interaction coefficients: sampled from uniform(-0.1, 0.3), give slight margin
        for ch_idx in range(self._n_theta_channels):
            base = _GLOBAL_REDUCED + ch_idx * _REDUCED_PER_CH
            prior_low[base + 3] = -0.15   # price_x_media
            prior_high[base + 3] = 0.35
            prior_low[base + 4] = -0.15   # distribution_x_media
            prior_high[base + 4] = 0.35
        prior = BoxUniform(low=prior_low, high=prior_high)

        # Clamp theta to prior bounds to avoid NaN in NSF training
        theta_tensor = theta_tensor.clamp(prior_low.unsqueeze(0), prior_high.unsqueeze(0))

        if self.config.simple_embedding:
            # Let sbi build its own default MLP embedding + z-score everything
            density_builder = posterior_nn(
                "nsf",
                hidden_features=128,
                num_transforms=self.config.n_flow_transforms,
                z_score_theta="independent",
                z_score_x="independent",
            )
        else:
            density_builder = posterior_nn(
                "nsf",
                embedding_net=embedding_wrapper,
                hidden_features=128,
                num_transforms=self.config.n_flow_transforms,
                z_score_theta="independent",
            )
        inference = SNPE_C(
            prior=prior,
            density_estimator=density_builder,
            device=str(self.device),
        )

        # Append training data
        train_obs = obs_tensor[train_idx]
        train_theta = theta_tensor[train_idx]
        inference.append_simulations(train_theta, train_obs)

        # Train
        logger.info("Starting training for %d epochs...", self.config.n_epochs)
        density_estimator = inference.train(
            training_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            max_num_epochs=self.config.n_epochs,
            stop_after_epochs=self.config.patience,
            show_train_summary=True,
        )

        self.posterior = inference.build_posterior(density_estimator)
        self._is_trained = True
        self._sbi_inference = inference  # Keep for sequential rounds
        self._sbi_prior = prior

        # Evaluate on validation set
        val_obs = obs_tensor[val_idx]
        val_theta = theta_tensor[val_idx]
        val_metrics = self._evaluate_subset(val_obs, val_theta)

        logger.info("Validation metrics: %s", val_metrics)

        # Save if requested
        if save_dir:
            self.save(save_dir)

        return {"val_metrics": val_metrics}

    def train_sequential(
        self,
        x_obs: np.ndarray,
        spend_obs: np.ndarray,
        channel_names: list[str],
        context_obs: np.ndarray | None = None,
        n_rounds: int = 2,
        n_sims_per_round: int = 2000,
        n_workers: int = 4,
    ) -> None:
        """Run sequential SNPE rounds focused on a specific observation.

        After the initial amortized training (train()), this method runs
        additional rounds where new simulations are generated from the
        posterior proposal, concentrating training data around the likely
        parameter region for the target observation.

        Args:
            x_obs: (T,) outcome time series for the target observation.
            spend_obs: (T, n_channels) spend matrix for the target observation.
            channel_names: List of channel names.
            context_obs: (T, n_context_cols) business context matrix (optional).
            n_rounds: Number of sequential rounds (default 2).
            n_sims_per_round: Simulations per round (default 2000).
            n_workers: Parallel workers for simulation.
        """
        if not self._is_trained or self.posterior is None:
            raise RuntimeError("Run train() first before sequential rounds.")
        if not hasattr(self, '_sbi_inference'):
            raise RuntimeError("No sbi inference object. Re-run train().")

        from demantiq.neural.utils import CHANNEL_NAME_TO_IDX

        # Pack the target observation
        n_ch = len(channel_names)
        n_periods = len(x_obs)
        type_ids = torch.zeros(1, MAX_CHANNELS, dtype=torch.long)
        for i, name in enumerate(channel_names):
            if i < MAX_CHANNELS:
                type_ids[0, i] = CHANNEL_NAME_TO_IDX.get(name, 0)

        T = spend_obs.shape[0]
        spend_padded = np.zeros((T, MAX_CHANNELS), dtype=np.float32)
        spend_padded[:, :spend_obs.shape[1]] = spend_obs

        ctx_tensor = None
        if context_obs is not None:
            ctx_tensor = torch.tensor(context_obs, dtype=torch.float32).unsqueeze(0)

        obs_packed = pack_observations(
            y=torch.tensor(x_obs, dtype=torch.float32).unsqueeze(0),
            spend=torch.tensor(spend_padded, dtype=torch.float32).unsqueeze(0),
            n_channels=torch.tensor([n_ch]),
            n_periods=torch.tensor([n_periods]),
            channel_type_ids=type_ids,
            max_T=self.max_T,
            context=ctx_tensor,
        ).squeeze(0).to(self.device)

        for round_idx in range(n_rounds):
            logger.info("=== Sequential round %d/%d ===", round_idx + 1, n_rounds)

            # Sample params from current posterior conditioned on target obs
            logger.info("Sampling %d params from current posterior...", n_sims_per_round)
            self.posterior.set_default_x(obs_packed)
            try:
                proposal_samples = self.posterior.sample(
                    (n_sims_per_round,), x=obs_packed
                )
            except (AssertionError, RuntimeError, TypeError) as e:
                logger.warning("Posterior sampling failed (%s), using perturbed prior...", e)
                # The NSF is numerically unstable. Fall back to sampling
                # from a Gaussian centered on the posterior mean (estimated
                # from a small successful sample or the prior center).
                try:
                    small_sample = self.posterior.sample((50,), x=obs_packed)
                    center = small_sample.mean(dim=0)
                    spread = small_sample.std(dim=0).clamp(min=0.02)
                except Exception:
                    center = torch.full((self._n_reduced_params,), 0.5, device=self.device)
                    spread = torch.full((self._n_reduced_params,), 0.2, device=self.device)

                proposal_samples = center.unsqueeze(0) + spread.unsqueeze(0) * torch.randn(
                    n_sims_per_round, self._n_reduced_params, device=self.device
                )
                proposal_samples = proposal_samples.clamp(-0.1, 1.1)

            # Convert reduced theta back to SimulationConfigs and simulate
            logger.info("Simulating %d datasets from proposal...", n_sims_per_round)
            new_theta_list = []
            new_obs_list = []

            for j in range(n_sims_per_round):
                theta_j = proposal_samples[j].cpu().numpy()
                try:
                    config = _theta_to_config(
                        theta_j, channel_names,
                        self._beta_scale, self._roas_scale, self._elasticity_scale,
                        self._global_reduced, self._reduced_per_ch,
                    )
                    from demantiq.core.demand_kernel import simulate
                    result = simulate(config)

                    # Pack the simulated observation
                    ch_names_sim = [ch.name for ch in config.channels]
                    y_sim = result.observable_data["y"].values
                    spend_sim = np.column_stack([
                        result.observable_data[f"{ch}_spend"].values
                        for ch in ch_names_sim
                    ])
                    spend_sim_pad = np.zeros((len(y_sim), MAX_CHANNELS), dtype=np.float32)
                    spend_sim_pad[:, :spend_sim.shape[1]] = spend_sim

                    from demantiq.orchestration.training_format import extract_context_matrix
                    ctx_sim = extract_context_matrix(result.observable_data, config.n_periods)
                    ctx_sim_t = torch.tensor(ctx_sim, dtype=torch.float32).unsqueeze(0)

                    type_ids_sim = torch.zeros(1, MAX_CHANNELS, dtype=torch.long)
                    for k, nm in enumerate(ch_names_sim):
                        if k < MAX_CHANNELS:
                            type_ids_sim[0, k] = CHANNEL_NAME_TO_IDX.get(nm, 0)

                    obs_sim = pack_observations(
                        y=torch.tensor(y_sim, dtype=torch.float32).unsqueeze(0),
                        spend=torch.tensor(spend_sim_pad, dtype=torch.float32).unsqueeze(0),
                        n_channels=torch.tensor([len(ch_names_sim)]),
                        n_periods=torch.tensor([config.n_periods]),
                        channel_type_ids=type_ids_sim,
                        max_T=self.max_T,
                        context=ctx_sim_t,
                    ).squeeze(0)

                    new_obs_list.append(obs_sim)
                    new_theta_list.append(proposal_samples[j])
                except Exception as e:
                    logger.debug("Simulation %d failed: %s", j, e)
                    continue

            if len(new_theta_list) < 10:
                logger.warning("Only %d successful simulations, skipping round", len(new_theta_list))
                continue

            new_theta = torch.stack(new_theta_list).to(self.device)
            new_obs = torch.stack(new_obs_list).to(self.device)

            logger.info("Appending %d proposal simulations for round %d",
                        len(new_theta_list), round_idx + 1)

            # Build a fresh SNPE_C for this round with the new data
            from sbi.inference import SNPE_C
            from sbi.neural_nets import posterior_nn
            from sbi.utils import BoxUniform

            prior_low = torch.full((self._n_reduced_params,), -0.1, device=self.device)
            prior_high = torch.full((self._n_reduced_params,), 1.1, device=self.device)
            prior = BoxUniform(low=prior_low, high=prior_high)

            embedding_wrapper = DemantiqEmbeddingWrapper(
                self.embedding_net, max_T=self.max_T
            ).to(self.device)

            # Use a simpler flow for sequential rounds to avoid numerical instability
            # Fewer transforms + fewer hidden features = more stable on small data
            density_builder = posterior_nn(
                "maf",  # Masked Autoregressive Flow — more stable than NSF on small data
                embedding_net=embedding_wrapper,
                hidden_features=64,
                num_transforms=3,
                z_score_theta="independent",
            )
            seq_inference = SNPE_C(
                prior=prior,
                density_estimator=density_builder,
                device=str(self.device),
            )

            seq_inference.append_simulations(new_theta, new_obs)

            # Train on focused data with strict epoch cap to prevent overfitting
            max_seq_epochs = min(50, self.config.n_epochs)
            logger.info("Training on %d focused simulations (max %d epochs)...",
                        len(new_theta_list), max_seq_epochs)
            density_estimator = seq_inference.train(
                training_batch_size=min(self.config.batch_size, len(new_theta_list)),
                learning_rate=self.config.learning_rate,
                max_num_epochs=max_seq_epochs,
                stop_after_epochs=10,  # Aggressive early stopping
                show_train_summary=True,
            )

            # Build posterior without prior constraint to avoid rejection sampling issues
            self.posterior = seq_inference.build_posterior(
                density_estimator, sample_with="direct"
            )
            self._sbi_inference = seq_inference
            logger.info("Round %d complete.", round_idx + 1)

    def _evaluate_subset(
        self, obs: Tensor, theta_true: Tensor, n_posterior_samples: int = 1000
    ) -> dict[str, float]:
        """Evaluate posterior quality on a subset of data.

        Args:
            obs: (N, flat_obs_dim) packed observations.
            theta_true: (N, EXT_TRUTH_VECTOR_LEN) true parameters.
            n_posterior_samples: Number of posterior samples per observation.

        Returns:
            Dict with evaluation metrics.
        """
        if self.posterior is None:
            return {}

        n_eval = min(len(obs), 100)  # Evaluate on subset for speed
        mape_list = []

        for i in range(n_eval):
            samples = self.posterior.sample(
                (n_posterior_samples,), x=obs[i]
            )
            posterior_mean = samples.mean(dim=0)
            true = theta_true[i]

            # MAPE on non-zero true values
            nonzero = true.abs() > 1e-6
            if nonzero.any():
                mape = ((posterior_mean[nonzero] - true[nonzero]).abs() / true[nonzero].abs()).mean()
                mape_list.append(mape.item())

        return {
            "mean_mape": float(np.mean(mape_list)) if mape_list else float("inf"),
            "median_mape": float(np.median(mape_list)) if mape_list else float("inf"),
            "n_evaluated": n_eval,
        }

    def infer(
        self,
        y: np.ndarray,
        spend_matrix: np.ndarray,
        channel_names: list[str],
        n_samples: int = 10000,
        context_matrix: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run inference on a single observation.

        Args:
            y: (T,) outcome time series.
            spend_matrix: (T, n_channels) spend matrix.
            channel_names: List of channel names.
            n_samples: Number of posterior samples.
            context_matrix: (T, n_context_cols) business context matrix (optional).
                           Contains price, promo, distribution, competition, macro.

        Returns:
            Dict with posterior samples and summary statistics.
        """
        if not self._is_trained or self.posterior is None:
            raise RuntimeError("Engine not trained. Call train() first.")

        from demantiq.neural.utils import CHANNEL_NAME_TO_IDX
        from demantiq.orchestration.training_format import (
            _GLOBAL_EXT_TRUTH_LEN,
            _PER_CHANNEL_EXT_TRUTH_LEN,
        )

        n_channels = len(channel_names)
        n_periods = len(y)

        use_simple = getattr(self, '_use_simple', False)

        if use_simple:
            obs = _extract_summary_stats_np(
                y, spend_matrix, context_matrix,
                n_channels, self._n_theta_channels,
            ).to(self.device)
        else:
            from demantiq.neural.utils import CHANNEL_NAME_TO_IDX

            type_ids = torch.zeros(1, MAX_CHANNELS, dtype=torch.long)
            for i, name in enumerate(channel_names):
                if i < MAX_CHANNELS:
                    type_ids[0, i] = CHANNEL_NAME_TO_IDX.get(name, 0)

            T = spend_matrix.shape[0]
            spend_padded = np.zeros((T, MAX_CHANNELS), dtype=np.float32)
            spend_padded[:, :spend_matrix.shape[1]] = spend_matrix

            ctx_tensor = None
            if context_matrix is not None:
                ctx_tensor = torch.tensor(context_matrix, dtype=torch.float32).unsqueeze(0)

            obs = pack_observations(
                y=torch.tensor(y, dtype=torch.float32).unsqueeze(0),
                spend=torch.tensor(spend_padded, dtype=torch.float32).unsqueeze(0),
                n_channels=torch.tensor([n_channels]),
                n_periods=torch.tensor([n_periods]),
                channel_type_ids=type_ids,
                max_T=self.max_T,
                context=ctx_tensor,
            ).squeeze(0).to(self.device)

        # Sample from posterior
        self.posterior.set_default_x(obs)
        try:
            samples = self.posterior.sample((n_samples,), x=obs)
        except (AssertionError, RuntimeError) as e:
            logger.warning("Posterior sampling failed (%s), rebuilding with direct sampling", e)
            if hasattr(self, '_sbi_inference') and self._sbi_inference is not None:
                self.posterior = self._sbi_inference.build_posterior(
                    self._sbi_inference._neural_net, sample_with="direct"
                )
                self.posterior.set_default_x(obs)
                samples = self.posterior.sample((n_samples,), x=obs)
            else:
                logger.warning("No inference object, returning prior samples")
                samples = torch.rand(n_samples, self._n_reduced_params, device=self.device)
        samples_np = samples.cpu().numpy()

        _REDUCED_PER_CH = self._reduced_per_ch
        _GLOBAL_RED = getattr(self, '_global_reduced', 2)

        # Parse samples into named parameters, denormalizing
        elasticity_samples = samples_np[:, 1] * self._elasticity_scale
        result = {
            "raw_samples": samples_np,
            "total_media_contribution_pct": _summarize(samples_np[:, 0]),
            "price_elasticity": _summarize(elasticity_samples),
        }

        # Per-channel parameters
        for i, name in enumerate(channel_names):
            if i >= MAX_CHANNELS:
                break
            base = _GLOBAL_RED + i * _REDUCED_PER_CH
            beta_samples = samples_np[:, base + 0] * self._beta_scale
            roas_samples = samples_np[:, base + 1] * self._roas_scale
            frac_samples = samples_np[:, base + 2]

            ch_result = {
                "beta": _summarize(beta_samples),
                "roas": _summarize(roas_samples),
                "contribution_fraction": _summarize(frac_samples),
            }

            # Interaction coefficients (if _reduced_per_ch >= 5)
            if self._reduced_per_ch >= 5:
                ch_result["price_x_media"] = _summarize(samples_np[:, base + 3])
                ch_result["distribution_x_media"] = _summarize(samples_np[:, base + 4])

            result[name] = ch_result

        return result

    def save(self, save_dir: str) -> None:
        """Save trained model to disk."""
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)

        torch.save({
            "embedding_state_dict": self.embedding_net.state_dict() if self.embedding_net is not None else None,
            "posterior": self.posterior,
            "max_T": self.max_T,
            "config": self.config,
            "is_trained": self._is_trained,
            "beta_scale": self._beta_scale,
            "roas_scale": self._roas_scale,
            "elasticity_scale": self._elasticity_scale,
            "n_reduced_params": self._n_reduced_params,
            "reduced_per_ch": self._reduced_per_ch,
            "global_reduced": self._global_reduced,
            "n_theta_channels": self._n_theta_channels,
            "use_simple": getattr(self, '_use_simple', False),
        }, str(path / "demantiq_neural.pt"))
        logger.info("Model saved to %s", path / "demantiq_neural.pt")

    def load(self, save_dir: str) -> None:
        """Load trained model from disk."""
        path = Path(save_dir) / "demantiq_neural.pt"
        checkpoint = torch.load(str(path), map_location=self.device, weights_only=False)

        self.config = checkpoint["config"]
        self.max_T = checkpoint["max_T"]

        self.embedding_net = EmbeddingNetwork(
            temporal_dim=self.config.temporal_dim,
            type_embed_dim=self.config.type_embed_dim,
            n_attn_heads=self.config.n_attn_heads,
            n_attn_layers=self.config.n_attn_layers,
            global_summary_dim=self.config.global_summary_dim,
            dropout=self.config.dropout,
        ).to(self.device)
        self.embedding_net.load_state_dict(checkpoint["embedding_state_dict"])

        self.posterior = checkpoint["posterior"]
        self._is_trained = checkpoint["is_trained"]
        self._beta_scale = checkpoint.get("beta_scale", 1.0)
        self._roas_scale = checkpoint.get("roas_scale", 1.0)
        self._elasticity_scale = checkpoint.get("elasticity_scale", 1.0)
        self._n_reduced_params = checkpoint.get("n_reduced_params", 62)
        self._reduced_per_ch = checkpoint.get("reduced_per_ch", 3)
        self._global_reduced = checkpoint.get("global_reduced", 2)
        self._n_theta_channels = checkpoint.get("n_theta_channels", MAX_CHANNELS)
        self._use_simple = checkpoint.get("use_simple", False)

        # Recreate sbi inference object for sequential rounds
        from sbi.inference import SNPE_C
        from sbi.neural_nets import posterior_nn
        from sbi.utils import BoxUniform

        prior_low = torch.full((self._n_reduced_params,), -0.1, device=self.device)
        prior_high = torch.full((self._n_reduced_params,), 1.1, device=self.device)
        self._sbi_prior = BoxUniform(low=prior_low, high=prior_high)

        embedding_wrapper = DemantiqEmbeddingWrapper(
            self.embedding_net, max_T=self.max_T
        ).to(self.device)

        density_builder = posterior_nn(
            "nsf",
            embedding_net=embedding_wrapper,
            hidden_features=128,
            num_transforms=self.config.n_flow_transforms,
            z_score_theta="independent",
        )
        self._sbi_inference = SNPE_C(
            prior=self._sbi_prior,
            density_estimator=density_builder,
            device=str(self.device),
        )

        logger.info("Model loaded from %s", path)


def _extract_summary_stats(
    y: Tensor, spend: Tensor, context: Tensor | None,
    n_channels: int, n_theta_channels: int,
) -> Tensor:
    """Extract hand-crafted summary statistics from a single observation.

    Instead of feeding raw time series to a learned embedding, compute
    interpretable statistics that capture the key patterns an MMM needs.

    Returns a fixed-size vector regardless of time series length.
    """
    stats = []
    T = int((y != 0).sum()) if y.dim() == 1 else y.shape[0]
    T = max(T, 1)
    y_valid = y[:T]

    # Y statistics (8 features)
    stats.extend([
        y_valid.mean(), y_valid.std(), y_valid.min(), y_valid.max(),
        y_valid[-1] - y_valid[0],  # trend
        float(T),  # series length
        float(n_channels),
    ])
    # Y autocorrelation (lag 1)
    if T > 1:
        y_centered = y_valid - y_valid.mean()
        autocorr = (y_centered[:-1] * y_centered[1:]).sum() / (y_centered**2).sum().clamp(min=1e-8)
        stats.append(float(autocorr))
    else:
        stats.append(0.0)

    # Per-channel spend statistics (5 features per channel)
    for ch_idx in range(n_theta_channels):
        if ch_idx < spend.shape[-1]:
            sp = spend[:T, ch_idx] if spend.dim() == 2 else spend[:T]
            stats.extend([
                float(sp.mean()), float(sp.std().clamp(min=1e-8)),
                float(sp.sum()),
            ])
            # Correlation with y
            if T > 1 and sp.std() > 1e-8:
                sp_centered = sp - sp.mean()
                y_centered = y_valid - y_valid.mean()
                corr = (sp_centered * y_centered).sum() / (
                    sp_centered.norm() * y_centered.norm() + 1e-8
                )
                stats.append(float(corr))
            else:
                stats.append(0.0)
            # Spend as fraction of y
            stats.append(float(sp.sum() / y_valid.sum().clamp(min=1e-8)))
        else:
            stats.extend([0.0, 0.0, 0.0, 0.0, 0.0])

    # Context statistics (3 features per context column)
    if context is not None and context.dim() == 2:
        n_ctx_cols = context.shape[1]
        for ctx_idx in range(min(n_ctx_cols, 10)):
            ctx_col = context[:T, ctx_idx]
            stats.extend([
                float(ctx_col.mean()),
                float(ctx_col.std().clamp(min=1e-8)),
            ])
            # Correlation with y
            if T > 1 and ctx_col.std() > 1e-8:
                ctx_c = ctx_col - ctx_col.mean()
                y_c = y_valid - y_valid.mean()
                corr = (ctx_c * y_c).sum() / (ctx_c.norm() * y_c.norm() + 1e-8)
                stats.append(float(corr))
            else:
                stats.append(0.0)
    else:
        # 10 context columns × 3 features
        stats.extend([0.0] * 30)

    # --- Interaction-detection features ---

    # Spend × price correlation per channel (captures price×media interaction)
    # Context column 0 = price
    if context is not None and context.dim() == 2 and context.shape[1] > 0:
        price = context[:T, 0]
        for ch_idx in range(n_theta_channels):
            if ch_idx < spend.shape[-1]:
                sp = spend[:T, ch_idx] if spend.dim() == 2 else spend[:T]
                if T > 1 and sp.std() > 1e-8 and price.std() > 1e-8:
                    sp_c = sp - sp.mean()
                    pr_c = price - price.mean()
                    corr = (sp_c * pr_c).sum() / (sp_c.norm() * pr_c.norm() + 1e-8)
                    stats.append(float(corr))
                else:
                    stats.append(0.0)
            else:
                stats.append(0.0)

        # Spend × distribution correlation per channel (captures dist×media interaction)
        # Context column 3 = distribution
        if context.shape[1] > 3:
            dist = context[:T, 3]
            for ch_idx in range(n_theta_channels):
                if ch_idx < spend.shape[-1]:
                    sp = spend[:T, ch_idx] if spend.dim() == 2 else spend[:T]
                    if T > 1 and sp.std() > 1e-8 and dist.std() > 1e-8:
                        sp_c = sp - sp.mean()
                        di_c = dist - dist.mean()
                        corr = (sp_c * di_c).sum() / (sp_c.norm() * di_c.norm() + 1e-8)
                        stats.append(float(corr))
                    else:
                        stats.append(0.0)
                else:
                    stats.append(0.0)
        else:
            stats.extend([0.0] * n_theta_channels)
    else:
        stats.extend([0.0] * n_theta_channels * 2)

    # Cross-channel spend correlations (upper triangle)
    for ch_a in range(n_theta_channels):
        for ch_b in range(ch_a + 1, n_theta_channels):
            if ch_a < spend.shape[-1] and ch_b < spend.shape[-1]:
                sp_a = spend[:T, ch_a] if spend.dim() == 2 else spend[:T]
                sp_b = spend[:T, ch_b] if spend.dim() == 2 else spend[:T]
                if T > 1 and sp_a.std() > 1e-8 and sp_b.std() > 1e-8:
                    a_c = sp_a - sp_a.mean()
                    b_c = sp_b - sp_b.mean()
                    corr = (a_c * b_c).sum() / (a_c.norm() * b_c.norm() + 1e-8)
                    stats.append(float(corr))
                else:
                    stats.append(0.0)
            else:
                stats.append(0.0)

    # Variance decomposition proxies: R² of y ~ sum(spend) and y ~ price
    if T > 2:
        # R² of y ~ total_spend
        total_sp = torch.zeros(T)
        for ch_idx in range(min(n_theta_channels, spend.shape[-1] if spend.dim() == 2 else 1)):
            total_sp += spend[:T, ch_idx] if spend.dim() == 2 else spend[:T]
        if total_sp.std() > 1e-8:
            corr_sp = ((total_sp - total_sp.mean()) * (y_valid - y_valid.mean())).sum()
            r2_spend = (corr_sp ** 2) / ((total_sp - total_sp.mean()).pow(2).sum() * (y_valid - y_valid.mean()).pow(2).sum() + 1e-8)
            stats.append(float(r2_spend))
        else:
            stats.append(0.0)

        # R² of y ~ price
        if context is not None and context.dim() == 2 and context.shape[1] > 0:
            price = context[:T, 0]
            if price.std() > 1e-8:
                corr_pr = ((price - price.mean()) * (y_valid - y_valid.mean())).sum()
                r2_price = (corr_pr ** 2) / ((price - price.mean()).pow(2).sum() * (y_valid - y_valid.mean()).pow(2).sum() + 1e-8)
                stats.append(float(r2_price))
            else:
                stats.append(0.0)
        else:
            stats.append(0.0)
    else:
        stats.extend([0.0, 0.0])

    return torch.tensor(stats, dtype=torch.float32)


def _extract_summary_stats_np(
    y: np.ndarray, spend: np.ndarray, context: np.ndarray | None,
    n_channels: int, n_theta_channels: int,
) -> torch.Tensor:
    """Numpy version of _extract_summary_stats for inference."""
    return _extract_summary_stats(
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(spend, dtype=torch.float32),
        torch.tensor(context, dtype=torch.float32) if context is not None else None,
        n_channels, n_theta_channels,
    )


def _summarize(samples: np.ndarray) -> dict[str, float]:
    """Compute summary statistics for posterior samples."""
    return {
        "mean": float(samples.mean()),
        "std": float(samples.std()),
        "q05": float(np.percentile(samples, 5)),
        "q95": float(np.percentile(samples, 95)),
    }


def _theta_to_config(
    theta: np.ndarray,
    channel_names: list[str],
    beta_scale: float,
    roas_scale: float,
    elasticity_scale: float,
    global_reduced: int,
    reduced_per_ch: int,
) -> "SimulationConfig":
    """Convert a reduced theta vector back to a SimulationConfig for re-simulation.

    Uses the sampled betas and a fixed set of reasonable defaults for
    saturation/adstock/noise/baseline. The goal is to generate plausible
    synthetic data around the posterior parameter region, not to exactly
    reconstruct the original config.

    Args:
        theta: (n_reduced_params,) reduced parameter vector.
        channel_names: List of channel names.
        beta_scale: Denormalization scale for betas.
        roas_scale: Denormalization scale for ROAS.
        elasticity_scale: Denormalization scale for price elasticity.
        global_reduced: Number of global params in reduced vector.
        reduced_per_ch: Number of per-channel params in reduced vector.

    Returns:
        SimulationConfig instance.
    """
    from demantiq.config.simulation_config import SimulationConfig
    from demantiq.config.channel_config import ChannelConfig
    from demantiq.config.baseline_config import BaselineConfig
    from demantiq.config.noise_config import NoiseConfig
    from demantiq.config.pricing_config import PricingConfig

    rng = np.random.default_rng()

    # Extract global params
    price_elasticity = float(theta[1] * elasticity_scale)

    # Build channels from sampled betas
    channels = []
    for i, name in enumerate(channel_names):
        base = global_reduced + i * reduced_per_ch
        beta = float(theta[base + 0] * beta_scale)
        beta = max(beta, 1.0)  # Clamp to positive

        channels.append(ChannelConfig(
            name=name,
            beta=beta,
            saturation_fn="logistic",
            saturation_params={"k": 3.0, "x0": 0.5},
            adstock_fn="geometric",
            adstock_params={"alpha": 0.5, "max_lag": 8},
            spend_mean=10000.0,
            spend_std=3000.0,
        ))

    # Add pricing if elasticity is nonzero
    pricing = None
    if abs(price_elasticity) > 0.1:
        pricing = PricingConfig(
            base_price=25.0,
            price_elasticity=float(np.clip(price_elasticity, -5.0, -0.1)),
            promo_frequency="monthly",
            promo_depth_mean=0.15,
        )

    return SimulationConfig(
        n_periods=104,
        channels=channels,
        noise=NoiseConfig(noise_scale=20.0),
        baseline=BaselineConfig(organic_level=1000.0),
        seed=int(rng.integers(0, 2**31)),
        pricing=pricing,
    )
