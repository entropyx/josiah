"""Quick overfit test for PFN-inspired full-sequence transformer.

Each week is a token. Self-attention across ALL weeks enables temporal
correlation computation — the signal needed for channel identification.

Usage:
    source venv/bin/activate
    python scripts/test_pfn_overfit.py
    python scripts/test_pfn_overfit.py --n-epochs 500 --d-model 128
"""

import argparse
import logging
import math
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class PFNDecompositionModel(nn.Module):
    """PFN-inspired full-sequence transformer for demand decomposition.

    Each week is a token with features [spend_ch1..chC, y, context, flags, time].
    Self-attention across all weeks computes temporal correlations.
    Output per week: per-channel contributions + baseline + non_media.
    """

    def __init__(self, n_input_features: int, n_outputs: int, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(n_input_features, d_model)

        # Learnable temporal encoding (added to token embeddings)
        self.max_len = 300
        self.temporal_encoding = nn.Parameter(torch.randn(1, self.max_len, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, n_input_features) — per-week features.

        Returns:
            (B, T, n_outputs) — per-week predictions (raw, before y-scaling).
        """
        B, T, _ = x.shape

        h = self.input_proj(x)  # (B, T, d_model)
        h = h + self.temporal_encoding[:, :T, :]  # add temporal encoding

        h = self.transformer(h)  # (B, T, d_model)

        return self.output_proj(h)  # (B, T, n_outputs)


def prepare_scenario_data(max_channels: int = 8, seed: int = 42):
    """Generate one scenario and prepare inputs/targets."""
    from demantiq.scenarios.scenario_sampler import ScenarioSampler
    from demantiq.core.demand_kernel import simulate
    from demantiq.orchestration.training_format import (
        extract_context_matrix, ground_truth_to_decomposition,
        DECOMP_IDX_BASELINE, DECOMP_IDX_CHANNELS_START,
        DECOMP_IDX_PRICE, DECOMP_IDX_COMPETITION, DECOMP_IDX_MACRO,
        MAX_CONTEXT_COLS,
    )

    sampler = ScenarioSampler(seed=seed, rich_context=True, channel_range=(3, 5))
    config = sampler.sample(1)[0]
    result = simulate(config)
    ch_names = [ch.name for ch in config.channels]
    n_ch = len(ch_names)
    T = config.n_periods

    y = result.observable_data["y"].values.astype(np.float32)
    spend = np.column_stack([
        result.observable_data[f"{ch}_spend"].values for ch in ch_names
    ]).astype(np.float32)
    context = extract_context_matrix(result.observable_data, T).astype(np.float32)

    # Pad spend to max_channels
    spend_padded = np.zeros((T, max_channels), dtype=np.float32)
    spend_padded[:, :n_ch] = spend

    # Presence flags for context
    context_present = (np.abs(context).sum(axis=0) > 0).astype(np.float32)
    context_flags = np.tile(context_present, (T, 1))  # (T, 10)

    # Temporal features: normalized time index + sin/cos week-of-year
    time_idx = np.linspace(0, 1, T).astype(np.float32).reshape(-1, 1)
    week_of_year = np.arange(T) % 52
    sin_week = np.sin(2 * np.pi * week_of_year / 52).astype(np.float32).reshape(-1, 1)
    cos_week = np.cos(2 * np.pi * week_of_year / 52).astype(np.float32).reshape(-1, 1)

    # Normalize inputs
    y_scale = np.abs(y).mean()
    y_normed = (y / max(y_scale, 1.0)).reshape(-1, 1)
    spend_scale = np.abs(spend_padded).max()
    spend_normed = spend_padded / max(spend_scale, 1.0)
    ctx_scale = np.abs(context).max(axis=0, keepdims=True)
    ctx_scale = np.maximum(ctx_scale, 1.0)
    context_normed = context / ctx_scale

    # Build input: [spend_ch1..8, y, context_1..10, flags_1..10, time, sin, cos]
    input_features = np.concatenate([
        spend_normed,       # (T, 8)
        y_normed,           # (T, 1)
        context_normed,     # (T, 10)
        context_flags,      # (T, 10)
        time_idx,           # (T, 1)
        sin_week,           # (T, 1)
        cos_week,           # (T, 1)
    ], axis=1)  # (T, 32)

    # Ground truth
    true_decomp = ground_truth_to_decomposition(result.ground_truth, ch_names, T)
    true_channels = np.zeros((T, max_channels), dtype=np.float32)
    for i in range(n_ch):
        true_channels[:, i] = true_decomp[:, DECOMP_IDX_CHANNELS_START + i]
    true_baseline = true_decomp[:, DECOMP_IDX_BASELINE].astype(np.float32)
    true_non_media = (
        true_decomp[:, DECOMP_IDX_PRICE]
        + true_decomp[:, DECOMP_IDX_COMPETITION]
        + true_decomp[:, DECOMP_IDX_MACRO]
    ).astype(np.float32)

    # Build target: [ch1..8, baseline, non_media] — normalized by y_scale
    target = np.concatenate([
        true_channels / max(y_scale, 1.0),     # (T, 8)
        (true_baseline / max(y_scale, 1.0)).reshape(-1, 1),   # (T, 1)
        (true_non_media / max(y_scale, 1.0)).reshape(-1, 1),  # (T, 1)
    ], axis=1)  # (T, 10)

    return {
        "input": torch.from_numpy(input_features).unsqueeze(0),  # (1, T, 32)
        "target": torch.from_numpy(target).unsqueeze(0),          # (1, T, 10)
        "y_scale": y_scale,
        "n_channels": n_ch,
        "max_channels": max_channels,
        "channel_names": ch_names,
        "T": T,
        "y": y,
        "true_channels": true_channels,
        "true_baseline": true_baseline,
        "true_non_media": true_non_media,
    }


def train_and_evaluate(n_epochs: int = 300, d_model: int = 128, n_layers: int = 6,
                       n_heads: int = 4, lr: float = 1e-3, seed: int = 42):
    """Train PFN model on single scenario and evaluate."""
    data = prepare_scenario_data(seed=seed)
    n_ch = data["n_channels"]
    C = data["max_channels"]
    T = data["T"]
    y_scale = data["y_scale"]
    ch_names = data["channel_names"]

    logger.info("Scenario: %d channels (%s), %d periods, y_scale=%.1f",
                n_ch, ", ".join(ch_names), T, y_scale)

    n_input = data["input"].shape[-1]
    n_output = C + 2  # channels + baseline + non_media

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PFNDecompositionModel(
        n_input_features=n_input, n_outputs=n_output,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: d=%d, layers=%d, heads=%d, params=%d", d_model, n_layers, n_heads, n_params)

    x = data["input"].to(device)   # (1, T, n_input)
    t = data["target"].to(device)  # (1, T, n_output)

    # Channel mask for loss: only compute loss on active channels
    ch_active = torch.zeros(C, device=device)
    ch_active[:n_ch] = 1.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    t_start = time.time()
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        pred = model(x)  # (1, T, n_output)

        # Split predictions
        pred_ch = pred[:, :, :C]        # (1, T, C)
        pred_base = pred[:, :, C]       # (1, T)
        pred_nm = pred[:, :, C + 1]     # (1, T)

        true_ch = t[:, :, :C]
        true_base = t[:, :, C]
        true_nm = t[:, :, C + 1]

        # Per-channel MSE (active channels only)
        ch_scale = true_ch.abs().mean(dim=1).clamp(min=0.01)  # (1, C)
        ch_diff = (pred_ch - true_ch) / ch_scale.unsqueeze(1)
        ch_diff = ch_diff * ch_active.unsqueeze(0).unsqueeze(0)
        l_ch = (ch_diff ** 2).sum() / (ch_active.sum() * T)

        # Baseline MSE
        base_scale = true_base.abs().mean().clamp(min=0.01)
        l_base = ((pred_base - true_base) / base_scale).pow(2).mean()

        # Non-media MSE
        nm_scale = true_nm.abs().mean().clamp(min=0.01)
        l_nm = ((pred_nm - true_nm) / nm_scale).pow(2).mean()

        # Reconstruction
        y_normed = x[:, :, C]  # y is at position C (after spend channels) in input
        pred_sum = pred_base + (pred_ch * ch_active).sum(dim=-1) + pred_nm
        l_recon = ((pred_sum - y_normed) ** 2).mean()

        loss = l_ch + l_base + l_nm + l_recon
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 50 == 0 or epoch == n_epochs - 1:
            logger.info("Epoch %3d/%d  loss=%.4f (ch=%.4f bl=%.4f nm=%.4f rc=%.4f)  lr=%.1e  [%.0fs]",
                        epoch + 1, n_epochs, loss.item(), l_ch.item(), l_base.item(),
                        l_nm.item(), l_recon.item(), optimizer.param_groups[0]["lr"],
                        time.time() - t_start)

    # --- Evaluate ---
    model.eval()
    with torch.no_grad():
        pred = model(x)

    pred_ch = pred[0, :, :C].cpu().numpy() * y_scale   # (T, C) denormalized
    pred_base = pred[0, :, C].cpu().numpy() * y_scale   # (T,)
    pred_nm = pred[0, :, C + 1].cpu().numpy() * y_scale  # (T,)

    true_ch = data["true_channels"][:, :n_ch]
    true_base = data["true_baseline"]
    y = data["y"]

    # Channel totals
    true_ch_totals = true_ch.sum(axis=0)
    pred_ch_totals = pred_ch[:, :n_ch].sum(axis=0)

    # Rank correlation
    rank_corr = spearmanr(true_ch_totals, pred_ch_totals)[0] if n_ch >= 3 else np.corrcoef(true_ch_totals, pred_ch_totals)[0, 1]

    # Per-channel time series correlation
    ch_time_corrs = []
    for i in range(n_ch):
        if np.std(true_ch[:, i]) > 1e-8 and np.std(pred_ch[:, i]) > 1e-8:
            ch_time_corrs.append(pearsonr(true_ch[:, i], pred_ch[:, i])[0])
    mean_time_corr = np.mean(ch_time_corrs) if ch_time_corrs else 0.0

    # Baseline
    total_y = y.sum()
    true_base_pct = true_base.sum() / total_y * 100
    pred_base_pct = pred_base.sum() / total_y * 100

    # Reconstruction R²
    y_hat = pred_base + pred_ch[:, :n_ch].sum(axis=1) + pred_nm
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    recon_r2 = 1 - ss_res / ss_tot if ss_tot > 1e-6 else 0.0

    print(f"\n{'='*70}")
    print(f"  PFN OVERFIT TEST RESULTS")
    print(f"  1 scenario, {n_epochs} epochs, {n_ch} channels, d_model={d_model}")
    print(f"  Model params: {n_params:,}")
    print(f"{'='*70}")

    print(f"\n  Baseline: true={true_base_pct:.1f}%  pred={pred_base_pct:.1f}%  err={abs(pred_base_pct - true_base_pct):.1f}pp")

    print(f"\n{'Channel':<16} {'True Total':>12} {'Pred Total':>12} {'Error %':>10} {'Time Corr':>10}")
    print("-" * 66)
    for i in range(n_ch):
        tc = true_ch_totals[i]
        pc = pred_ch_totals[i]
        err = abs(pc - tc) / max(abs(tc), 1e-6) * 100
        tcorr = ch_time_corrs[i] if i < len(ch_time_corrs) else 0.0
        print(f"{ch_names[i]:<16} {tc:>12.0f} {pc:>12.0f} {err:>9.1f}% {tcorr:>9.3f}")

    print(f"\n  Channel rank correlation:  {rank_corr:.3f}  (target: >0.9)")
    print(f"  Mean channel time corr:    {mean_time_corr:.3f}  (target: >0.8)")
    print(f"  Reconstruction R²:         {recon_r2:.4f}  (target: >0.9)")

    passed = rank_corr > 0.9 and mean_time_corr > 0.8 and recon_r2 > 0.9
    print(f"\n  {'PASSED' if passed else 'FAILED'}")


def main():
    parser = argparse.ArgumentParser(description="PFN overfit test")
    parser.add_argument("--n-epochs", type=int, default=300)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi", action="store_true",
                        help="Test on multiple hard scenarios (seeds 42,100,200,300,400,500)")
    args = parser.parse_args()

    if args.multi:
        seeds = [42, 100, 200, 300, 400, 500]
        for s in seeds:
            print(f"\n{'#'*70}")
            print(f"  SEED {s}")
            print(f"{'#'*70}")
            train_and_evaluate(
                n_epochs=args.n_epochs, d_model=args.d_model,
                n_layers=args.n_layers, n_heads=args.n_heads, lr=args.lr, seed=s,
            )
    else:
        train_and_evaluate(
            n_epochs=args.n_epochs, d_model=args.d_model,
            n_layers=args.n_layers, n_heads=args.n_heads, lr=args.lr, seed=args.seed,
        )


if __name__ == "__main__":
    main()
