"""Train and evaluate the Demantiq neural inference engine.

Uses ScenarioLibrary configs for evaluation with ground truth comparison plots.

Usage:
    source venv/bin/activate
    python scripts/train_neural.py                    # train + evaluate on clean_room
    python scripts/train_neural.py --scenario real_world
    python scripts/train_neural.py --skip-training     # load saved model, just evaluate
    python scripts/train_neural.py --n-train 5000      # more training data (better accuracy)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("neural_output")
TRAINING_DATA_DIR = OUTPUT_DIR / "training_data"
MODEL_DIR = OUTPUT_DIR / "trained_model"
PLOTS_DIR = OUTPUT_DIR / "plots"


def generate_training_data(n_total: int, n_workers: int = 4, n_fixed_channels: int | None = None):
    """Generate training data from random scenario sampling."""
    from demantiq.orchestration.training_pipeline import TrainingPipeline
    from demantiq.scenarios.scenario_sampler import ScenarioSampler

    TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)

    sampler = ScenarioSampler(seed=42, rich_context=True, n_fixed_channels=n_fixed_channels)
    pipeline = TrainingPipeline(
        sampler, output_dir=str(TRAINING_DATA_DIR), batch_size=100
    )

    t0 = time.time()
    pipeline.generate(n_total=n_total, n_workers=n_workers, seed=42)
    logger.info("Generated %d simulations in %.1f seconds", n_total, time.time() - t0)


def train_engine(n_epochs: int = 50, n_fixed_channels: int | None = None, simple_embedding: bool = False):
    """Train the neural inference engine."""
    from demantiq.neural.inference import DemantiqInferenceEngine, TrainingConfig

    if simple_embedding:
        # Track A: simple embedding — more flow capacity, larger batches
        config = TrainingConfig(
            n_epochs=n_epochs,
            batch_size=256,
            patience=30,
            learning_rate=5e-4,
            n_flow_transforms=8,
            simple_embedding=True,
            device="cpu",
        )
    else:
        # Track B: full embedding — lower LR for end-to-end stability
        config = TrainingConfig(
            n_epochs=n_epochs,
            batch_size=512,
            patience=40,
            learning_rate=1e-4,
            global_summary_dim=256,
            temporal_dim=64,
            type_embed_dim=16,
            n_attn_heads=4,
            n_attn_layers=2,
            n_flow_transforms=5,
            simple_embedding=False,
            device="cpu",
        )

    engine = DemantiqInferenceEngine(config)

    t0 = time.time()
    result = engine.train(
        data_dir=str(TRAINING_DATA_DIR),
        save_dir=str(MODEL_DIR),
        n_fixed_channels=n_fixed_channels,
    )
    logger.info("Training took %.1f seconds", time.time() - t0)
    logger.info("Validation metrics: %s", result.get("val_metrics", {}))
    return engine


def load_engine():
    """Load a previously trained engine."""
    from demantiq.neural.inference import DemantiqInferenceEngine

    engine = DemantiqInferenceEngine()
    engine.load(str(MODEL_DIR))
    return engine


# =========================================================================
#  DECOMPOSITION ENGINE (Phase A — per-period demand decomposition)
# =========================================================================

DECOMP_MODEL_DIR = OUTPUT_DIR / "decomp_model"


def train_decomposition_engine(
    n_epochs: int = 50,
    n_train: int = 500,
    n_fixed_channels: int | None = None,
    learning_rate: float = 1e-3,
):
    """Train the decomposition engine (encoder-decoder for per-period shares)."""
    from demantiq.neural.decomposition_engine import DecompositionEngine, DecompositionConfig

    config = DecompositionConfig(
        n_epochs=n_epochs,
        n_train=n_train,
        data_dir=str(TRAINING_DATA_DIR),
        learning_rate=learning_rate,
        batch_size=256,
        patience=15,
        n_fixed_channels=n_fixed_channels,
    )

    engine = DecompositionEngine(config)
    metrics = engine.train(data_dir=str(TRAINING_DATA_DIR), n_train=n_train)
    engine.save(str(DECOMP_MODEL_DIR))

    logger.info("Decomposition training metrics: %s", metrics)
    return engine


def evaluate_decomposition(engine, scenario_name: str):
    """Evaluate decomposition engine on a named scenario with time series plots."""
    from demantiq.scenarios.scenario_library import ScenarioLibrary
    from demantiq.core.demand_kernel import simulate
    from demantiq.orchestration.training_format import (
        extract_context_matrix,
        ground_truth_to_decomposition,
        DECOMP_IDX_BASELINE,
        DECOMP_IDX_CHANNELS_START,
        DECOMP_IDX_PRICE,
        DECOMP_IDX_DISTRIBUTION,
        DECOMP_IDX_COMPETITION,
        DECOMP_IDX_MACRO,
        DECOMP_IDX_NOISE,
        DECOMP_COLS,
    )
    from demantiq.neural.utils import CHANNEL_NAME_TO_IDX, MAX_CHANNELS

    logger.info("=== Evaluating decomposition: %s ===", scenario_name)

    config = ScenarioLibrary.get(scenario_name)
    result = simulate(config)

    channel_names = [ch.name for ch in config.channels]
    n_ch = len(channel_names)
    y = result.observable_data["y"].values
    T = config.n_periods

    # Build inputs
    spend_matrix = np.column_stack([
        result.observable_data[f"{ch}_spend"].values for ch in channel_names
    ])
    # Pad spend to MAX_CHANNELS
    if spend_matrix.shape[1] < MAX_CHANNELS:
        spend_matrix = np.pad(
            spend_matrix, ((0, 0), (0, MAX_CHANNELS - spend_matrix.shape[1]))
        )

    context_matrix = extract_context_matrix(result.observable_data, T)

    channel_type_ids = np.zeros(MAX_CHANNELS, dtype=np.int64)
    for i, ch_name in enumerate(channel_names):
        if i < MAX_CHANNELS:
            channel_type_ids[i] = CHANNEL_NAME_TO_IDX.get(ch_name, 0)

    # Run inference
    t0 = time.time()
    inf = engine.infer(y, spend_matrix, context_matrix, n_ch, channel_type_ids, T)
    logger.info("Inference took %.3f seconds", time.time() - t0)

    pred_shares = inf["shares"]       # (T, DECOMP_COLS)
    pred_contrib = inf["contributions"]  # (T, DECOMP_COLS)
    y_recon = inf["y_reconstructed"]  # (T,)

    # True decomposition
    true_decomp = ground_truth_to_decomposition(result.ground_truth, channel_names, T)

    # Compute true shares
    y_safe = np.maximum(y[:T], 1.0)
    true_shares = true_decomp / y_safe[:, np.newaxis]
    true_shares[:, DECOMP_IDX_DISTRIBUTION] = 0.0  # multiplicative, skip

    # ---- Metrics ----
    # Per-component contribution R² (meaningful metric — not trivially 1.0)
    # For each component, how well does predicted contribution match true contribution?
    component_names = ["baseline"] + channel_names + ["price", "distribution", "competition", "macro", "noise"]
    component_indices = [DECOMP_IDX_BASELINE]
    for i in range(n_ch):
        component_indices.append(DECOMP_IDX_CHANNELS_START + i)
    component_indices.extend([DECOMP_IDX_PRICE, DECOMP_IDX_DISTRIBUTION,
                              DECOMP_IDX_COMPETITION, DECOMP_IDX_MACRO, DECOMP_IDX_NOISE])

    # Overall share MSE across all components and timesteps
    active_mask = np.ones(DECOMP_COLS, dtype=bool)
    active_mask[DECOMP_IDX_DISTRIBUTION] = False
    overall_share_mse = np.mean((pred_shares[:, active_mask] - true_shares[:, active_mask]) ** 2)

    print(f"\n{'='*70}")
    print(f"  DECOMPOSITION EVALUATION — {scenario_name}")
    print(f"  Channels: {n_ch} | Periods: {T}")
    print(f"{'='*70}")
    print(f"\n  Overall Share MSE:       {overall_share_mse:.6f}")
    print(f"  Mean |y|:                {np.mean(np.abs(y[:T])):.1f}")
    print(f"  (Note: y-level R² is trivially 1.0 since softmax shares sum to 1)")

    print(f"\n{'Component':<16} {'True Share':>12} {'Pred Share':>12} {'Share MSE':>12} {'Comp R²':>10} {'True Total':>12} {'Pred Total':>12}")
    print("-" * 90)
    for name, idx in zip(component_names, component_indices):
        if idx == DECOMP_IDX_DISTRIBUTION:
            continue  # skip multiplicative
        ts = true_shares[:, idx]
        ps = pred_shares[:, idx]
        mse = np.mean((ts - ps) ** 2)
        true_total = np.sum(true_decomp[:, idx])
        pred_total = np.sum(pred_contrib[:, idx])
        # Per-component R²: how well does predicted contribution track true contribution over time?
        tc = true_decomp[:, idx]
        pc = pred_contrib[:, idx]
        ss_res = np.sum((tc - pc) ** 2)
        ss_tot = np.sum((tc - tc.mean()) ** 2)
        comp_r2 = 1 - ss_res / ss_tot if ss_tot > 1e-6 else 0.0
        print(f"{name:<16} {np.mean(ts):>12.4f} {np.mean(ps):>12.4f} {mse:>12.6f} {comp_r2:>10.4f} {true_total:>12.0f} {pred_total:>12.0f}")

    # High-level category breakdown
    total_y = np.sum(y[:T])
    true_baseline_total = np.sum(true_decomp[:, DECOMP_IDX_BASELINE])
    pred_baseline_total = np.sum(pred_contrib[:, DECOMP_IDX_BASELINE])
    true_media_total = sum(np.sum(true_decomp[:, DECOMP_IDX_CHANNELS_START + i]) for i in range(n_ch))
    pred_media_total = sum(np.sum(pred_contrib[:, DECOMP_IDX_CHANNELS_START + i]) for i in range(n_ch))
    true_other_total = (np.sum(true_decomp[:, DECOMP_IDX_PRICE])
                        + np.sum(true_decomp[:, DECOMP_IDX_COMPETITION])
                        + np.sum(true_decomp[:, DECOMP_IDX_MACRO]))
    pred_other_total = (np.sum(pred_contrib[:, DECOMP_IDX_PRICE])
                        + np.sum(pred_contrib[:, DECOMP_IDX_COMPETITION])
                        + np.sum(pred_contrib[:, DECOMP_IDX_MACRO]))
    true_noise_total = np.sum(true_decomp[:, DECOMP_IDX_NOISE])
    pred_noise_total = np.sum(pred_contrib[:, DECOMP_IDX_NOISE])

    print(f"\n--- Category Breakdown ---")
    print(f"{'Category':<16} {'True %':>10} {'Pred %':>10} {'Error':>10}")
    print("-" * 50)
    for cat, tv, pv in [
        ("Baseline",    true_baseline_total, pred_baseline_total),
        ("Media",       true_media_total,    pred_media_total),
        ("Other",       true_other_total,    pred_other_total),
        ("Noise",       true_noise_total,    pred_noise_total),
    ]:
        tp = tv / total_y * 100
        pp = pv / total_y * 100
        print(f"{cat:<16} {tp:>9.1f}% {pp:>9.1f}% {abs(pp - tp):>9.1f}pp")

    # Weekly accuracy: per-week correlation between true and predicted share vectors
    from scipy.stats import spearmanr
    active_cols = [DECOMP_IDX_BASELINE]
    for i in range(n_ch):
        active_cols.append(DECOMP_IDX_CHANNELS_START + i)
    active_cols.extend([DECOMP_IDX_PRICE, DECOMP_IDX_COMPETITION, DECOMP_IDX_MACRO, DECOMP_IDX_NOISE])

    weekly_corrs = []
    weekly_mapes = []
    for t in range(T):
        ts_week = true_shares[t, active_cols]
        ps_week = pred_shares[t, active_cols]
        # Spearman rank correlation — does the ranking hold?
        if np.std(ts_week) > 1e-8 and np.std(ps_week) > 1e-8:
            corr, _ = spearmanr(ts_week, ps_week)
            weekly_corrs.append(corr)
        # Weekly MAPE on shares
        denom = np.maximum(np.abs(ts_week), 0.01)
        weekly_mapes.append(np.mean(np.abs(ts_week - ps_week) / denom))

    weekly_corrs = np.array(weekly_corrs)
    weekly_mapes = np.array(weekly_mapes)

    # Weekly component accuracy: for each week, R² across components
    # "Given the true shares [0.62, 0.06, 0.15, 0.02, ...], how close are the predicted shares?"
    weekly_r2s = []
    for t in range(T):
        ts_week = true_shares[t, active_cols]
        ps_week = pred_shares[t, active_cols]
        ss_res = np.sum((ts_week - ps_week) ** 2)
        ss_tot = np.sum((ts_week - ts_week.mean()) ** 2)
        wr2 = 1 - ss_res / ss_tot if ss_tot > 1e-8 else 0.0
        weekly_r2s.append(wr2)
    weekly_r2s = np.array(weekly_r2s)

    print(f"\n--- Weekly Accuracy ---")
    print(f"  Mean weekly R² (across components): {np.mean(weekly_r2s):.4f}")
    print(f"  Median weekly R²:                   {np.median(weekly_r2s):.4f}")
    print(f"  Worst week R²:         {np.min(weekly_r2s):.4f}  (week {np.argmin(weekly_r2s) + 1})")
    print(f"  Best week R²:          {np.max(weekly_r2s):.4f}  (week {np.argmax(weekly_r2s) + 1})")
    print(f"  % weeks with R² > 0.8:              {np.mean(weekly_r2s > 0.8) * 100:.0f}%")
    print(f"  % weeks with R² > 0.5:              {np.mean(weekly_r2s > 0.5) * 100:.0f}%")
    print(f"  Mean weekly rank correlation:        {np.mean(weekly_corrs):.3f}")
    print(f"  Worst week correlation:              {np.min(weekly_corrs):.3f}  (week {np.argmin(weekly_corrs) + 1})")
    print(f"  % weeks with corr > 0.8:             {np.mean(weekly_corrs > 0.8) * 100:.0f}%")
    print(f"  Mean weekly share MAPE:              {np.mean(weekly_mapes) * 100:.1f}%")

    # Channel-only metrics (excludes baseline — catches channel collapse)
    ch_cols = [DECOMP_IDX_CHANNELS_START + i for i in range(n_ch)]
    ch_corrs = []
    for t in range(T):
        ts_ch = true_shares[t, ch_cols]
        ps_ch = pred_shares[t, ch_cols]
        if np.std(ts_ch) > 1e-8 and np.std(ps_ch) > 1e-8:
            corr, _ = spearmanr(ts_ch, ps_ch)
            ch_corrs.append(corr)
    ch_corrs = np.array(ch_corrs) if ch_corrs else np.array([0.0])

    # Channel-only total ranking
    true_ch_totals = np.array([np.sum(true_decomp[:, DECOMP_IDX_CHANNELS_START + i]) for i in range(n_ch)])
    pred_ch_totals = np.array([np.sum(pred_contrib[:, DECOMP_IDX_CHANNELS_START + i]) for i in range(n_ch)])
    true_rank = np.argsort(-true_ch_totals)
    pred_rank = np.argsort(-pred_ch_totals)
    true_rank_names = [channel_names[i] for i in true_rank]
    pred_rank_names = [channel_names[i] for i in pred_rank]
    if n_ch >= 2:
        overall_ch_corr, _ = spearmanr(true_ch_totals, pred_ch_totals)
    else:
        overall_ch_corr = 0.0

    print(f"\n--- Channel-Only Metrics (excludes baseline) ---")
    print(f"  Weekly channel rank correlation:     {np.mean(ch_corrs):.3f}")
    print(f"  % weeks channel corr > 0.5:          {np.mean(ch_corrs > 0.5) * 100:.0f}%")
    print(f"  Overall channel total correlation:   {overall_ch_corr:.3f}")
    print(f"  True ranking:  {' > '.join(true_rank_names)}")
    print(f"  Pred ranking:  {' > '.join(pred_rank_names)}")

    # Total contribution per channel
    print(f"\n{'Channel':<16} {'True Contrib':>14} {'Pred Contrib':>14} {'Error %':>10}")
    print("-" * 58)
    for i, ch in enumerate(channel_names):
        idx = DECOMP_IDX_CHANNELS_START + i
        true_total = np.sum(true_decomp[:, idx])
        pred_total = np.sum(pred_contrib[:, idx])
        err = abs(pred_total - true_total) / max(abs(true_total), 1.0) * 100
        print(f"{ch:<16} {true_total:>14.0f} {pred_total:>14.0f} {err:>9.1f}%")

    # Reconstruction R² (how well predicted contributions sum to actual y)
    ss_res = np.sum((y[:T] - y_recon) ** 2)
    ss_tot = np.sum((y[:T] - y[:T].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean((y[:T] - y_recon) ** 2))
    print(f"\n  Reconstruction R²:       {r2:.4f}  (sum of predicted contributions vs actual y)")
    print(f"  Reconstruction RMSE:     {rmse:.1f}")

    # ---- Plots ----
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    dates = result.observable_data["date"].values[:T]

    # 1. Actual vs Predicted y(t)
    _plot_decomp_actual_vs_predicted(dates, y[:T], y_recon, r2, rmse, scenario_name)

    # 2. Per-channel contribution time series
    _plot_decomp_channel_contributions(
        dates, true_decomp, pred_contrib, channel_names, scenario_name,
    )

    # 3. Monthly decomposition (stacked area)
    _plot_decomp_monthly_stacked(
        result.observable_data, true_decomp, pred_contrib,
        channel_names, scenario_name,
    )

    # 4. Training loss curve
    if hasattr(engine, "train_losses") and engine.train_losses:
        _plot_decomp_loss_curve(engine.train_losses, engine.val_losses, scenario_name)

    logger.info("Plots saved to %s/", PLOTS_DIR)

    # Save results CSV
    _save_decomp_results_csv(
        scenario_name, channel_names, T, r2, rmse,
        true_decomp, pred_contrib, pred_shares, true_shares,
    )


def _plot_decomp_actual_vs_predicted(dates, y_actual, y_pred, r2, rmse, scenario):
    """Plot actual vs predicted y with residuals."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1], sharex=True)

    ax1.plot(dates, y_actual, color="#2196F3", linewidth=1.5, label="Actual", alpha=0.9)
    ax1.plot(dates, y_pred, color="#FF9800", linewidth=1.5, label="Predicted (decomp sum)", alpha=0.9)
    ax1.fill_between(dates, y_actual, y_pred, alpha=0.1, color="gray")
    ax1.set_ylabel("Demand (y)")
    ax1.set_title(f"Actual vs Predicted — {scenario}  |  R²={r2:.4f}  RMSE={rmse:.1f}")
    ax1.legend()

    residuals = y_actual - y_pred
    colors = np.where(residuals >= 0, "#4CAF50", "#F44336")
    ax2.bar(dates, residuals, color=colors, alpha=0.6, width=5)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("Residual")
    ax2.set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{scenario}_decomp_actual_vs_predicted.png", dpi=150)
    plt.close(fig)
    logger.info("Actual vs predicted plot saved (R²=%.4f)", r2)


def _plot_decomp_channel_contributions(dates, true_decomp, pred_contrib, channel_names, scenario):
    """Per-channel contribution time series: true vs predicted."""
    from demantiq.orchestration.training_format import DECOMP_IDX_CHANNELS_START, DECOMP_IDX_BASELINE

    n_ch = len(channel_names)
    n_plots = n_ch + 1  # channels + baseline
    cols = min(n_plots, 3)
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), sharex=True)
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Baseline
    ax = axes[0]
    ax.plot(dates, true_decomp[:, DECOMP_IDX_BASELINE], color="#2196F3", label="True", linewidth=1.2)
    ax.plot(dates, pred_contrib[:, DECOMP_IDX_BASELINE], color="#FF9800", label="Predicted", linewidth=1.2, linestyle="--")
    ax.set_title("Baseline")
    ax.legend(fontsize=7)
    ax.set_ylabel("Contribution")

    # Per-channel
    for i, ch in enumerate(channel_names):
        ax = axes[i + 1]
        idx = DECOMP_IDX_CHANNELS_START + i
        ax.plot(dates, true_decomp[:, idx], color="#2196F3", label="True", linewidth=1.2)
        ax.plot(dates, pred_contrib[:, idx], color="#FF9800", label="Predicted", linewidth=1.2, linestyle="--")
        ax.set_title(ch)
        ax.legend(fontsize=7)

    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"Per-Component Contributions — {scenario}", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{scenario}_decomp_contributions.png", dpi=150)
    plt.close(fig)


def _plot_decomp_monthly_stacked(observable_data, true_decomp, pred_contrib, channel_names, scenario):
    """Stacked area chart of monthly decomposition: true vs predicted."""
    import pandas as pd
    from demantiq.orchestration.training_format import (
        DECOMP_IDX_BASELINE, DECOMP_IDX_CHANNELS_START,
        DECOMP_IDX_PRICE, DECOMP_IDX_MACRO, DECOMP_IDX_NOISE,
    )

    dates = observable_data["date"].values[:true_decomp.shape[0]]
    T = len(dates)

    # Build DataFrames for true and predicted
    def _build_df(matrix):
        df = pd.DataFrame({"date": dates})
        df["baseline"] = matrix[:T, DECOMP_IDX_BASELINE]
        for i, ch in enumerate(channel_names):
            df[ch] = matrix[:T, DECOMP_IDX_CHANNELS_START + i]
        if np.any(matrix[:T, DECOMP_IDX_PRICE] != 0):
            df["price_effect"] = matrix[:T, DECOMP_IDX_PRICE]
        if np.any(matrix[:T, DECOMP_IDX_MACRO] != 0):
            df["macro"] = matrix[:T, DECOMP_IDX_MACRO]
        return df

    true_df = _build_df(true_decomp)
    pred_df = _build_df(pred_contrib)

    true_monthly = true_df.set_index("date").resample("ME").sum()
    pred_monthly = pred_df.set_index("date").resample("ME").sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    cols = [c for c in true_monthly.columns]
    colors = plt.cm.Set2(np.linspace(0, 1, len(cols)))

    ax1.stackplot(true_monthly.index, [true_monthly[c].values for c in cols],
                  labels=cols, colors=colors, alpha=0.8)
    ax1.set_title("True Monthly Decomposition")
    ax1.set_ylabel("Demand")
    ax1.legend(loc="upper left", fontsize=7)

    cols_pred = [c for c in cols if c in pred_monthly.columns]
    ax2.stackplot(pred_monthly.index, [pred_monthly[c].values for c in cols_pred],
                  labels=cols_pred, colors=colors[:len(cols_pred)], alpha=0.8)
    ax2.set_title("Predicted Monthly Decomposition")
    ax2.set_ylabel("Demand")
    ax2.legend(loc="upper left", fontsize=7)

    fig.suptitle(f"Monthly Demand Decomposition — {scenario}", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{scenario}_decomp_monthly.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_decomp_loss_curve(train_losses, val_losses, scenario):
    """Training and validation loss over epochs."""
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss", color="#2196F3", linewidth=1.5)
    if val_losses:
        ax.plot(epochs, val_losses, label="Val Loss", color="#FF9800", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Training Loss — {scenario}")
    ax.legend()
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{scenario}_decomp_loss.png", dpi=150)
    plt.close(fig)


def _save_decomp_results_csv(scenario, channel_names, n_periods, r2, rmse,
                              true_decomp, pred_contrib, pred_shares, true_shares):
    """Save decomposition results to CSV."""
    import csv
    from demantiq.orchestration.training_format import DECOMP_IDX_CHANNELS_START

    RESULTS_DIR = OUTPUT_DIR / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Per-channel summary
    path = RESULTS_DIR / f"{scenario}_decomp_channels.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "channel", "true_total_contrib", "pred_total_contrib",
                     "error_pct", "mean_true_share", "mean_pred_share", "share_mse"])
        for i, ch in enumerate(channel_names):
            idx = DECOMP_IDX_CHANNELS_START + i
            true_total = np.sum(true_decomp[:, idx])
            pred_total = np.sum(pred_contrib[:, idx])
            err = abs(pred_total - true_total) / max(abs(true_total), 1.0) * 100
            ts_mean = np.mean(true_shares[:, idx])
            ps_mean = np.mean(pred_shares[:, idx])
            mse = np.mean((true_shares[:, idx] - pred_shares[:, idx]) ** 2)
            w.writerow([scenario, ch, f"{true_total:.1f}", f"{pred_total:.1f}",
                       f"{err:.2f}", f"{ts_mean:.6f}", f"{ps_mean:.6f}", f"{mse:.8f}"])

    # Summary
    summary_path = RESULTS_DIR / f"{scenario}_decomp_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "n_channels", "n_periods", "r2", "rmse"])
        w.writerow([scenario, len(channel_names), n_periods, f"{r2:.6f}", f"{rmse:.2f}"])

    logger.info("Decomposition results saved: %s, %s", path.name, summary_path.name)


def evaluate_scenario(engine, scenario_name: str):
    """Run inference on a named scenario and plot comparison with ground truth."""
    from demantiq.scenarios.scenario_library import ScenarioLibrary
    from demantiq.core.demand_kernel import simulate

    logger.info("=== Evaluating scenario: %s ===", scenario_name)

    # Get config and simulate
    config = ScenarioLibrary.get(scenario_name)
    result = simulate(config)

    channel_names = [ch.name for ch in config.channels]
    y = result.observable_data["y"].values
    spend_matrix = np.column_stack([
        result.observable_data[f"{ch}_spend"].values
        for ch in channel_names
    ])

    # Extract business context
    from demantiq.orchestration.training_format import extract_context_matrix
    context_matrix = extract_context_matrix(result.observable_data, config.n_periods)

    # Log what context is available
    ctx_present = []
    if config.pricing is not None:
        ctx_present.append("pricing")
    if config.distribution is not None:
        ctx_present.append("distribution")
    if config.competition is not None:
        ctx_present.append("competition")
    if config.macro is not None:
        ctx_present.append("macro")
    if config.endogeneity is not None:
        ctx_present.append("endogeneity")
    logger.info("Business context: %s", ", ".join(ctx_present) if ctx_present else "media-only")

    # Run inference
    t0 = time.time()
    inference = engine.infer(y, spend_matrix, channel_names, n_samples=5000,
                             context_matrix=context_matrix)
    logger.info("Inference took %.3f seconds", time.time() - t0)

    # Extract ground truth
    summary = result.summary_truth
    true_betas = {ch.name: ch.beta for ch in config.channels}
    true_roas = summary["true_roas"]
    true_contrib = summary["true_total_contribution"]
    total_contrib = sum(true_contrib.values())
    true_fracs = {k: v / total_contrib for k, v in true_contrib.items()} if total_contrib > 0 else {}

    # Print results table
    print(f"\n{'='*70}")
    print(f"  Scenario: {scenario_name}")
    print(f"  Channels: {len(channel_names)} | Periods: {config.n_periods}")
    print(f"{'='*70}")

    print(f"\n{'Channel':<14} {'True Beta':>10} {'Inferred':>10} {'95% CI':>20} {'Err%':>8}")
    print("-" * 66)
    for ch in channel_names:
        true = true_betas[ch]
        inf = inference[ch]["beta"]
        err = abs(inf["mean"] - true) / max(abs(true), 1e-6) * 100
        print(f"{ch:<14} {true:>10.1f} {inf['mean']:>10.1f} [{inf['q05']:>7.1f}, {inf['q95']:>7.1f}] {err:>7.1f}%")

    print(f"\n{'Channel':<14} {'True ROAS':>10} {'Inferred':>10} {'95% CI':>20} {'Err%':>8}")
    print("-" * 66)
    for ch in channel_names:
        true = true_roas[ch]
        inf = inference[ch]["roas"]
        err = abs(inf["mean"] - true) / max(abs(true), 1e-6) * 100
        print(f"{ch:<14} {true:>10.3f} {inf['mean']:>10.3f} [{inf['q05']:>7.3f}, {inf['q95']:>7.3f}] {err:>7.1f}%")

    true_media_pct = summary["true_total_media_contribution_pct"]
    inf_media_pct = inference["total_media_contribution_pct"]
    print(f"\nTotal media contribution: true={true_media_pct:.3f}, "
          f"inferred={inf_media_pct['mean']:.3f} "
          f"[{inf_media_pct['q05']:.3f}, {inf_media_pct['q95']:.3f}]")

    # Price elasticity (if pricing configured)
    true_elasticity = summary.get("true_price_elasticity", None)
    if true_elasticity is not None and true_elasticity != 0.0:
        inf_elast = inference["price_elasticity"]
        elast_err = abs(inf_elast["mean"] - true_elasticity) / max(abs(true_elasticity), 1e-6) * 100
        print(f"\nPrice elasticity: true={true_elasticity:.3f}, "
              f"inferred={inf_elast['mean']:.3f} "
              f"[{inf_elast['q05']:.3f}, {inf_elast['q95']:.3f}] "
              f"err={elast_err:.1f}%")
    else:
        true_elasticity = None

    # Channel ranking (Kendall's tau)
    from scipy.stats import kendalltau
    true_beta_vals = [true_betas[ch] for ch in channel_names]
    inf_beta_vals = [inference[ch]["beta"]["mean"] for ch in channel_names]
    if len(channel_names) >= 2:
        tau, tau_pval = kendalltau(true_beta_vals, inf_beta_vals)
        print(f"\nChannel ranking (Kendall's tau): {tau:.3f} (p={tau_pval:.3f})")
    else:
        tau = float("nan")

    # OLS baseline
    ols_results = _ols_baseline(result.observable_data, channel_names, config)
    if ols_results:
        print(f"\n--- OLS Baseline Comparison ---")
        print(f"{'Channel':<14} {'True Beta':>10} {'Neural':>10} {'OLS':>10} {'Neural Err%':>12} {'OLS Err%':>10}")
        print("-" * 70)
        for ch in channel_names:
            tb = true_betas[ch]
            nb = inference[ch]["beta"]["mean"]
            ob = ols_results["betas"].get(ch, 0.0)
            ne = abs(nb - tb) / max(abs(tb), 1e-6) * 100
            oe = abs(ob - tb) / max(abs(tb), 1e-6) * 100
            print(f"{ch:<14} {tb:>10.1f} {nb:>10.1f} {ob:>10.1f} {ne:>11.1f}% {oe:>9.1f}%")
        print(f"OLS R²: {ols_results['r2']:.4f}")

    # Interaction coefficients (if scenario has interactions)
    if config.interactions is not None:
        print(f"\n--- Interaction Coefficients ---")
        print(f"{'Channel':<14} {'True PxM':>10} {'Inferred':>10} {'95% CI':>20} "
              f"{'True DxM':>10} {'Inferred':>10} {'95% CI':>20}")
        print("-" * 100)
        for ch in channel_names:
            true_px = config.interactions.price_x_media.get(ch, 0.0)
            true_dx = config.interactions.distribution_x_media.get(ch, 0.0)
            inf_px = inference[ch].get("price_x_media", {"mean": 0, "q05": 0, "q95": 0})
            inf_dx = inference[ch].get("distribution_x_media", {"mean": 0, "q05": 0, "q95": 0})
            print(f"{ch:<14} {true_px:>10.3f} {inf_px['mean']:>10.3f} "
                  f"[{inf_px['q05']:>7.3f}, {inf_px['q95']:>7.3f}] "
                  f"{true_dx:>10.3f} {inf_dx['mean']:>10.3f} "
                  f"[{inf_dx['q05']:>7.3f}, {inf_dx['q95']:>7.3f}]")

    # Comprehensive diagnostics
    diagnostics = _compute_diagnostics(
        inference, channel_names, true_betas, true_roas, true_fracs,
        true_media_pct, true_elasticity, config,
    )
    _print_diagnostics(diagnostics)

    # Save results CSV
    RESULTS_DIR = OUTPUT_DIR / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _save_results_csv(channel_names, true_betas, true_roas, true_fracs,
                      true_media_pct, inference, scenario_name, config, ctx_present,
                      true_elasticity=true_elasticity, tau=tau,
                      ols_results=ols_results)
    _save_diagnostics_csv(diagnostics, scenario_name)

    # Generate plots
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    _plot_beta_comparison(channel_names, true_betas, inference, scenario_name)
    _plot_roas_comparison(channel_names, true_roas, inference, scenario_name)
    _plot_contribution_waterfall(channel_names, true_fracs, inference, scenario_name)
    _plot_posterior_distributions(channel_names, true_betas, inference, scenario_name)
    _plot_demand_decomposition(result, inference, channel_names, scenario_name)
    _plot_actual_vs_predicted(result, inference, channel_names, config, scenario_name)
    _plot_monthly_decomposition(result, inference, channel_names, config, scenario_name)

    logger.info("Plots saved to %s/", PLOTS_DIR)


def _compute_diagnostics(inference, channel_names, true_betas, true_roas, true_fracs,
                         true_media_pct, true_elasticity, config):
    """Compute comprehensive diagnostics for understanding inference quality."""
    diag = {"per_channel": {}, "per_type": {}, "global": {}}
    # Support both monolithic (raw_samples) and compositional (raw_channel_samples) output
    raw = inference.get("raw_samples")
    raw_channel = inference.get("raw_channel_samples", {})

    # Prior ranges for sharpness calculation
    # beta: [-0.1, 1.1] * beta_scale → effective range ~600
    # roas: [-0.1, 1.1] * roas_scale → effective range ~1.2
    # frac: [-0.1, 1.1] → range 1.2
    # interactions: [-0.15, 0.35] → range 0.5
    prior_ranges = {"beta": 1.2, "roas": 1.2, "contribution_fraction": 1.2,
                    "price_x_media": 0.5, "distribution_x_media": 0.5}

    beta_mapes, beta_biases, beta_sharpnesses = [], [], []
    roas_mapes, roas_biases, roas_sharpnesses = [], [], []
    px_mapes, px_biases, px_sharpnesses = [], [], []
    dx_mapes, dx_biases, dx_sharpnesses = [], [], []

    for ch in channel_names:
        inf = inference[ch]
        ch_diag = {}

        # Beta: signed bias + sharpness
        tb = true_betas[ch]
        ib = inf["beta"]["mean"]
        ch_diag["beta_true"] = tb
        ch_diag["beta_inferred"] = ib
        ch_diag["beta_signed_bias"] = ib - tb
        ch_diag["beta_signed_bias_pct"] = (ib - tb) / max(abs(tb), 1e-6) * 100
        ch_diag["beta_mape"] = abs(ib - tb) / max(abs(tb), 1e-6) * 100
        ci_width_norm = (inf["beta"]["q95"] - inf["beta"]["q05"])
        # Sharpness in normalized space (CI width / prior range)
        ch_diag["beta_sharpness"] = 1.0 - min(ci_width_norm / (prior_ranges["beta"] * 500), 1.0)  # scale by beta_max~500
        beta_mapes.append(ch_diag["beta_mape"])
        beta_biases.append(ch_diag["beta_signed_bias_pct"])
        beta_sharpnesses.append(ch_diag["beta_sharpness"])

        # ROAS
        tr = true_roas[ch]
        ir = inf["roas"]["mean"]
        ch_diag["roas_signed_bias_pct"] = (ir - tr) / max(abs(tr), 1e-6) * 100
        ch_diag["roas_mape"] = abs(ir - tr) / max(abs(tr), 1e-6) * 100
        roas_mapes.append(ch_diag["roas_mape"])
        roas_biases.append(ch_diag["roas_signed_bias_pct"])

        # Interactions
        if "price_x_media" in inf:
            true_px = config.interactions.price_x_media.get(ch, 0.0) if config.interactions else 0.0
            true_dx = config.interactions.distribution_x_media.get(ch, 0.0) if config.interactions else 0.0
            ipx = inf["price_x_media"]["mean"]
            idx_val = inf["distribution_x_media"]["mean"]
            ch_diag["px_true"] = true_px
            ch_diag["px_inferred"] = ipx
            ch_diag["px_mape"] = abs(ipx - true_px) / max(abs(true_px), 0.01) * 100
            ch_diag["px_signed_bias"] = ipx - true_px
            ch_diag["dx_true"] = true_dx
            ch_diag["dx_inferred"] = idx_val
            ch_diag["dx_mape"] = abs(idx_val - true_dx) / max(abs(true_dx), 0.01) * 100
            ch_diag["dx_signed_bias"] = idx_val - true_dx
            px_mapes.append(ch_diag["px_mape"])
            px_biases.append(ch_diag["px_signed_bias"])
            dx_mapes.append(ch_diag["dx_mape"])
            dx_biases.append(ch_diag["dx_signed_bias"])

            # Effective beta decomposition check
            eff_beta = ib * (1 + ipx) * (1 + idx_val)
            ch_diag["effective_beta"] = eff_beta

            # Posterior correlation: beta vs price_x_media
            # Support both monolithic (raw matrix) and compositional (per-channel dict)
            ch_raw = None
            if raw is not None:
                _GLOBAL_RED = 2
                _RED_PER_CH = 5
                ch_idx_pos = channel_names.index(ch)
                base = _GLOBAL_RED + ch_idx_pos * _RED_PER_CH
                if raw.shape[1] > base + 4:
                    ch_raw = raw[:, base:base + 5]
            elif ch in raw_channel:
                ch_raw = raw_channel[ch]

            if ch_raw is not None and ch_raw.shape[1] >= 5:
                beta_s = ch_raw[:, 0]
                px_s = ch_raw[:, 3]
                dx_s = ch_raw[:, 4]
                if beta_s.std() > 1e-8 and px_s.std() > 1e-8:
                    ch_diag["corr_beta_px"] = float(np.corrcoef(beta_s, px_s)[0, 1])
                else:
                    ch_diag["corr_beta_px"] = 0.0
                if beta_s.std() > 1e-8 and dx_s.std() > 1e-8:
                    ch_diag["corr_beta_dx"] = float(np.corrcoef(beta_s, dx_s)[0, 1])
                else:
                    ch_diag["corr_beta_dx"] = 0.0

        diag["per_channel"][ch] = ch_diag

    # Per-type summary
    diag["per_type"]["betas"] = {
        "n": len(beta_mapes),
        "mean_mape": float(np.mean(beta_mapes)) if beta_mapes else 0,
        "mean_signed_bias_pct": float(np.mean(beta_biases)) if beta_biases else 0,
        "mean_sharpness": float(np.mean(beta_sharpnesses)) if beta_sharpnesses else 0,
    }
    diag["per_type"]["roas"] = {
        "n": len(roas_mapes),
        "mean_mape": float(np.mean(roas_mapes)) if roas_mapes else 0,
        "mean_signed_bias_pct": float(np.mean(roas_biases)) if roas_biases else 0,
    }
    if px_mapes:
        diag["per_type"]["price_x_media"] = {
            "n": len(px_mapes),
            "mean_mape": float(np.mean(px_mapes)),
            "mean_signed_bias": float(np.mean(px_biases)),
        }
    if dx_mapes:
        diag["per_type"]["dist_x_media"] = {
            "n": len(dx_mapes),
            "mean_mape": float(np.mean(dx_mapes)),
            "mean_signed_bias": float(np.mean(dx_biases)),
        }

    # Global metrics
    diag["global"]["media_pct_true"] = true_media_pct
    diag["global"]["media_pct_inferred"] = inference["total_media_contribution_pct"]["mean"]
    diag["global"]["media_pct_bias"] = (
        inference["total_media_contribution_pct"]["mean"] - true_media_pct
    ) / max(abs(true_media_pct), 1e-6) * 100
    if true_elasticity:
        diag["global"]["elasticity_true"] = true_elasticity
        diag["global"]["elasticity_inferred"] = inference["price_elasticity"]["mean"]
        diag["global"]["elasticity_bias_pct"] = (
            inference["price_elasticity"]["mean"] - true_elasticity
        ) / max(abs(true_elasticity), 1e-6) * 100

    return diag


def _print_diagnostics(diag):
    """Print comprehensive diagnostics summary."""
    print(f"\n{'='*70}")
    print("  DIAGNOSTICS")
    print(f"{'='*70}")

    # Per-type summary
    print(f"\n{'Param Type':<20} {'N':>4} {'Mean MAPE':>12} {'Mean Bias':>12} {'Sharpness':>12}")
    print("-" * 64)
    for ptype, stats in diag["per_type"].items():
        sharpness = stats.get("mean_sharpness", "")
        if isinstance(sharpness, float):
            sharpness = f"{sharpness:.2f}"
        bias_key = "mean_signed_bias_pct" if "mean_signed_bias_pct" in stats else "mean_signed_bias"
        bias_val = stats.get(bias_key, 0)
        sign = "+" if bias_val >= 0 else ""
        print(f"{ptype:<20} {stats['n']:>4} {stats['mean_mape']:>11.1f}% {sign}{bias_val:>10.1f}% {sharpness:>12}")

    # Posterior correlations (identifiability check)
    has_corr = any("corr_beta_px" in d for d in diag["per_channel"].values())
    if has_corr:
        print(f"\n--- Posterior Correlations (identifiability) ---")
        print(f"{'Channel':<14} {'corr(β,PxM)':>14} {'corr(β,DxM)':>14} {'Effective β':>14}")
        print("-" * 60)
        for ch, d in diag["per_channel"].items():
            corr_px = d.get("corr_beta_px", 0)
            corr_dx = d.get("corr_beta_dx", 0)
            eff = d.get("effective_beta", 0)
            flag_px = " ⚠" if abs(corr_px) > 0.5 else ""
            flag_dx = " ⚠" if abs(corr_dx) > 0.5 else ""
            print(f"{ch:<14} {corr_px:>13.3f}{flag_px} {corr_dx:>13.3f}{flag_dx} {eff:>14.1f}")
        print("  (⚠ = strong anti-correlation, identifiability concern)")

    # Global
    g = diag["global"]
    print(f"\n--- Global ---")
    print(f"  Media contribution: bias = {g.get('media_pct_bias', 0):+.1f}%")
    if "elasticity_bias_pct" in g:
        print(f"  Price elasticity:   bias = {g['elasticity_bias_pct']:+.1f}%")


def _save_diagnostics_csv(diag, scenario):
    """Save diagnostics to CSV."""
    import csv
    RESULTS_DIR = OUTPUT_DIR / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    path = RESULTS_DIR / f"{scenario}_diagnostics.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "channel", "param", "true", "inferred", "signed_bias", "mape",
            "sharpness", "corr_beta_px", "corr_beta_dx", "effective_beta",
        ])
        for ch, d in diag["per_channel"].items():
            # Beta row
            w.writerow([ch, "beta", d.get("beta_true", ""), d.get("beta_inferred", ""),
                         f"{d.get('beta_signed_bias_pct', 0):.2f}%",
                         f"{d.get('beta_mape', 0):.2f}%",
                         f"{d.get('beta_sharpness', 0):.3f}",
                         f"{d.get('corr_beta_px', '')}", f"{d.get('corr_beta_dx', '')}",
                         f"{d.get('effective_beta', '')}"])
            # ROAS row
            w.writerow([ch, "roas", "", "", f"{d.get('roas_signed_bias_pct', 0):.2f}%",
                         f"{d.get('roas_mape', 0):.2f}%", "", "", "", ""])
            # Interaction rows
            if "px_true" in d:
                w.writerow([ch, "price_x_media", f"{d['px_true']:.3f}", f"{d['px_inferred']:.3f}",
                             f"{d['px_signed_bias']:.4f}", f"{d['px_mape']:.1f}%",
                             "", "", "", ""])
            if "dx_true" in d:
                w.writerow([ch, "dist_x_media", f"{d['dx_true']:.3f}", f"{d['dx_inferred']:.3f}",
                             f"{d['dx_signed_bias']:.4f}", f"{d['dx_mape']:.1f}%",
                             "", "", "", ""])

    logger.info("Diagnostics saved: %s", path.name)


def _ols_baseline(observable_data, channel_names, config):
    """Run a simple OLS regression as a baseline comparison."""
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        logger.warning("sklearn not installed, skipping OLS baseline")
        return None

    y = observable_data["y"].values
    features = {}
    feature_names = []

    # Add spend columns
    for ch in channel_names:
        col = f"{ch}_spend"
        if col in observable_data.columns:
            features[col] = observable_data[col].values
            feature_names.append(col)

    # Add pricing columns
    if "price" in observable_data.columns:
        features["price"] = observable_data["price"].values
        feature_names.append("price")
    if "is_promo" in observable_data.columns:
        features["is_promo"] = observable_data["is_promo"].values.astype(float)
        feature_names.append("is_promo")

    # Add distribution
    if "distribution" in observable_data.columns:
        features["distribution"] = observable_data["distribution"].values
        feature_names.append("distribution")

    # Add competition
    if "competitor_sov" in observable_data.columns:
        features["competitor_sov"] = observable_data["competitor_sov"].values
        feature_names.append("competitor_sov")

    X = np.column_stack([features[f] for f in feature_names])
    model = LinearRegression().fit(X, y)

    # Extract per-channel betas (coefficients for spend columns)
    betas = {}
    for i, fname in enumerate(feature_names):
        if fname.endswith("_spend"):
            ch = fname.replace("_spend", "")
            betas[ch] = float(model.coef_[i])

    # Price coefficient
    price_coef = None
    for i, fname in enumerate(feature_names):
        if fname == "price":
            price_coef = float(model.coef_[i])

    return {
        "betas": betas,
        "intercept": float(model.intercept_),
        "r2": float(model.score(X, y)),
        "price_coef": price_coef,
        "all_coefs": {fname: float(model.coef_[i]) for i, fname in enumerate(feature_names)},
    }


def _plot_demand_decomposition(result, inference, channel_names, scenario):
    """Stacked bar: true demand decomposition vs what the engine can see."""
    gt = result.ground_truth
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # True decomposition: compute variance contribution of each component
    components = {"baseline": gt["true_baseline"].values}
    for ch in channel_names:
        col = f"true_{ch}_contribution"
        if col in gt.columns:
            components[f"{ch} media"] = gt[col].values
    if "true_price_effect" in gt.columns:
        components["price effect"] = gt["true_price_effect"].values
    if "true_distribution_cap" in gt.columns:
        # Distribution cap is multiplicative, compute its additive effect
        components["distribution"] = (gt["true_distribution_cap"].values - 1.0) * gt["true_baseline"].values
    if "true_competition_effect" in gt.columns:
        components["competition"] = gt["true_competition_effect"].values
    if "true_macro_effect" in gt.columns:
        components["macro"] = gt["true_macro_effect"].values
    components["noise"] = gt["true_noise"].values

    # Compute mean absolute contribution for each component
    mean_abs = {k: float(np.mean(np.abs(v))) for k, v in components.items()}
    total_abs = sum(mean_abs.values())
    fracs = {k: v / total_abs for k, v in mean_abs.items()} if total_abs > 0 else mean_abs

    # Plot true decomposition
    ax = axes[0]
    labels = list(fracs.keys())
    values = [fracs[k] for k in labels]
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    bars = ax.barh(labels, values, color=colors)
    ax.set_xlabel("Share of Mean |Contribution|")
    ax.set_title("True Demand Decomposition")
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=8)

    # Inferred: what the neural engine reports
    ax2 = axes[1]
    inf_components = {}
    inf_media_pct = inference["total_media_contribution_pct"]["mean"]
    inf_components["total media"] = inf_media_pct
    inf_components["non-media (residual)"] = 1.0 - inf_media_pct

    # Per-channel breakdown within media
    for ch in channel_names:
        frac = inference[ch]["contribution_fraction"]["mean"]
        inf_components[f"  {ch}"] = frac * inf_media_pct

    labels2 = list(inf_components.keys())
    values2 = [inf_components[k] for k in labels2]
    colors2 = plt.cm.Set2(np.linspace(0, 1, len(labels2)))
    bars2 = ax2.barh(labels2, values2, color=colors2)
    ax2.set_xlabel("Share of Total Demand")
    ax2.set_title("Inferred Decomposition (Neural Engine)")
    for bar, val in zip(bars2, values2):
        if abs(val) > 0.005:
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1%}", va="center", fontsize=8)

    fig.suptitle(f"Demand Decomposition — {scenario}", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{scenario}_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_actual_vs_predicted(result, inference, channel_names, config, scenario):
    """Plot actual y(t) vs predicted y(t) from posterior mean parameters."""
    from demantiq.transforms.saturation import get_saturation_fn
    from demantiq.transforms.adstock import get_adstock_fn

    obs = result.observable_data
    y_actual = obs["y"].values
    T = len(y_actual)
    dates = obs["date"].values

    # Reconstruct y_predicted from posterior mean parameters
    y_predicted = np.full(T, inference["total_media_contribution_pct"]["mean"] * y_actual.mean())

    # Add media contributions
    for ch in config.channels:
        if ch.name not in inference:
            continue
        beta = inference[ch.name]["beta"]["mean"]
        spend = obs[f"{ch.name}_spend"].values

        # Apply adstock
        adstock_fn = get_adstock_fn(ch.adstock_fn)
        adstocked = adstock_fn(spend, **ch.adstock_params)

        # Normalize + saturate
        max_abs = np.max(np.abs(adstocked)) if np.max(np.abs(adstocked)) > 0 else 1.0
        normalized = adstocked / max_abs
        sat_fn = get_saturation_fn(ch.saturation_fn)
        saturated = sat_fn(normalized, **ch.saturation_params)

        # Apply interactions
        media_effect = beta * saturated
        px = inference[ch.name].get("price_x_media", {}).get("mean", 0)
        dx = inference[ch.name].get("distribution_x_media", {}).get("mean", 0)
        if "is_promo" in obs.columns:
            media_effect = media_effect * (1 + px * obs["is_promo"].values)
        if "distribution" in obs.columns:
            media_effect = media_effect * (1 + dx * obs["distribution"].values)

        y_predicted += media_effect

    # Add baseline (residual to match mean)
    baseline = y_actual.mean() - y_predicted.mean()
    y_predicted += baseline

    # Add price effect
    if "price" in obs.columns and "price_elasticity" in inference:
        elast = inference["price_elasticity"]["mean"]
        base_price = obs["price"].mean()
        price_effect = elast * np.log(obs["price"].values / base_price) * y_actual.mean()
        y_predicted += price_effect

    # Compute fit metrics
    ss_res = np.sum((y_actual - y_predicted) ** 2)
    ss_tot = np.sum((y_actual - y_actual.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean((y_actual - y_predicted) ** 2))

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1], sharex=True)

    ax1.plot(dates, y_actual, color="#2196F3", linewidth=1.5, label="Actual", alpha=0.9)
    ax1.plot(dates, y_predicted, color="#FF9800", linewidth=1.5, label="Predicted", alpha=0.9)
    ax1.fill_between(dates, y_actual, y_predicted, alpha=0.1, color="gray")
    ax1.set_ylabel("Demand (y)")
    ax1.set_title(f"Actual vs Predicted — {scenario}  |  R²={r2:.3f}  RMSE={rmse:.1f}")
    ax1.legend()

    # Residuals
    residuals = y_actual - y_predicted
    ax2.bar(dates, residuals, color=np.where(residuals >= 0, "#4CAF50", "#F44336"), alpha=0.6, width=5)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("Residual")
    ax2.set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{scenario}_actual_vs_predicted.png", dpi=150)
    plt.close(fig)
    logger.info("Saved actual vs predicted plot (R²=%.3f)", r2)


def _plot_monthly_decomposition(result, inference, channel_names, config, scenario):
    """Stacked area chart of monthly demand decomposition: true vs inferred."""
    import pandas as pd

    gt = result.ground_truth
    obs = result.observable_data

    # True decomposition
    true_components = pd.DataFrame({"date": gt["date"] if "date" in gt.columns else obs["date"]})
    true_components["baseline"] = gt["true_baseline"].values
    for ch in channel_names:
        col = f"true_{ch}_contribution"
        if col in gt.columns:
            true_components[ch] = gt[col].values
    if "true_price_effect" in gt.columns:
        true_components["price_effect"] = gt["true_price_effect"].values
    true_components["noise"] = gt["true_noise"].values

    # Aggregate to monthly
    true_monthly = true_components.set_index("date").resample("ME").sum()

    # Inferred decomposition (reconstruct from posterior means)
    from demantiq.transforms.saturation import get_saturation_fn
    from demantiq.transforms.adstock import get_adstock_fn

    inf_components = pd.DataFrame({"date": obs["date"]})
    y = obs["y"].values
    T = len(y)

    total_media = np.zeros(T)
    for ch_cfg in config.channels:
        if ch_cfg.name not in inference:
            continue
        beta = inference[ch_cfg.name]["beta"]["mean"]
        spend = obs[f"{ch_cfg.name}_spend"].values

        adstock_fn = get_adstock_fn(ch_cfg.adstock_fn)
        adstocked = adstock_fn(spend, **ch_cfg.adstock_params)
        max_abs = np.max(np.abs(adstocked)) if np.max(np.abs(adstocked)) > 0 else 1.0
        sat_fn = get_saturation_fn(ch_cfg.saturation_fn)
        saturated = sat_fn(adstocked / max_abs, **ch_cfg.saturation_params)

        media_effect = beta * saturated
        px = inference[ch_cfg.name].get("price_x_media", {}).get("mean", 0)
        dx = inference[ch_cfg.name].get("distribution_x_media", {}).get("mean", 0)
        if "is_promo" in obs.columns:
            media_effect = media_effect * (1 + px * obs["is_promo"].values)
        if "distribution" in obs.columns:
            media_effect = media_effect * (1 + dx * obs["distribution"].values)

        inf_components[ch_cfg.name] = media_effect
        total_media += media_effect

    # Price effect
    if "price" in obs.columns and "price_elasticity" in inference:
        elast = inference["price_elasticity"]["mean"]
        base_price = obs["price"].mean()
        inf_components["price_effect"] = elast * np.log(obs["price"].values / base_price) * y.mean()
    else:
        inf_components["price_effect"] = 0.0

    # Baseline = residual
    inf_components["baseline"] = y - total_media - inf_components["price_effect"].values
    inf_components["date"] = obs["date"]

    inf_monthly = inf_components.set_index("date").resample("ME").sum()

    # Plot side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # True decomposition
    cols_true = [c for c in true_monthly.columns if c != "noise"]
    colors = plt.cm.Set2(np.linspace(0, 1, len(cols_true)))
    ax1.stackplot(true_monthly.index, [true_monthly[c].values for c in cols_true],
                  labels=cols_true, colors=colors, alpha=0.8)
    ax1.set_title("True Monthly Decomposition")
    ax1.set_ylabel("Demand")
    ax1.legend(loc="upper left", fontsize=7)

    # Inferred decomposition
    cols_inf = [c for c in inf_monthly.columns if c in cols_true or c in channel_names]
    # Ensure same order
    cols_inf_ordered = [c for c in cols_true if c in inf_monthly.columns]
    colors_inf = colors[:len(cols_inf_ordered)]
    ax2.stackplot(inf_monthly.index, [inf_monthly[c].values for c in cols_inf_ordered],
                  labels=cols_inf_ordered, colors=colors_inf, alpha=0.8)
    ax2.set_title("Inferred Monthly Decomposition")
    ax2.set_ylabel("Demand")
    ax2.legend(loc="upper left", fontsize=7)

    fig.suptitle(f"Monthly Demand Decomposition — {scenario}", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{scenario}_monthly_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_results_csv(channels, true_betas, true_roas, true_fracs,
                      true_media_pct, inference, scenario, config, ctx_present,
                      true_elasticity=None, tau=float("nan"), ols_results=None):
    """Save per-channel results and scenario summary as CSVs."""
    import csv

    RESULTS_DIR = OUTPUT_DIR / "results"

    # --- Per-channel detail CSV ---
    detail_path = RESULTS_DIR / f"{scenario}_channels.csv"
    with open(detail_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "scenario", "channel",
            "true_beta", "inferred_beta", "beta_std", "beta_q05", "beta_q95", "beta_mape",
            "true_roas", "inferred_roas", "roas_std", "roas_q05", "roas_q95", "roas_mape",
            "true_contrib_frac", "inferred_contrib_frac", "frac_std", "frac_q05", "frac_q95",
        ])
        for ch in channels:
            tb = true_betas[ch]
            ib = inference[ch]["beta"]
            tr = true_roas[ch]
            ir = inference[ch]["roas"]
            tf = true_fracs.get(ch, 0.0)
            ifr = inference[ch]["contribution_fraction"]
            w.writerow([
                scenario, ch,
                f"{tb:.4f}", f"{ib['mean']:.4f}", f"{ib['std']:.4f}",
                f"{ib['q05']:.4f}", f"{ib['q95']:.4f}",
                f"{abs(ib['mean'] - tb) / max(abs(tb), 1e-6) * 100:.2f}",
                f"{tr:.6f}", f"{ir['mean']:.6f}", f"{ir['std']:.6f}",
                f"{ir['q05']:.6f}", f"{ir['q95']:.6f}",
                f"{abs(ir['mean'] - tr) / max(abs(tr), 1e-6) * 100:.2f}",
                f"{tf:.6f}", f"{ifr['mean']:.6f}", f"{ifr['std']:.6f}",
                f"{ifr['q05']:.6f}", f"{ifr['q95']:.6f}",
            ])

    # --- Scenario summary CSV (append mode — accumulates across scenarios) ---
    summary_path = RESULTS_DIR / "summary.csv"
    write_header = not summary_path.exists()
    inf_media = inference["total_media_contribution_pct"]

    # Compute aggregate metrics
    beta_mapes = [
        abs(inference[ch]["beta"]["mean"] - true_betas[ch]) / max(abs(true_betas[ch]), 1e-6)
        for ch in channels
    ]
    roas_mapes = [
        abs(inference[ch]["roas"]["mean"] - true_roas[ch]) / max(abs(true_roas[ch]), 1e-6)
        for ch in channels
    ]

    # Check if true values fall within 90% CI
    beta_covered = sum(
        1 for ch in channels
        if inference[ch]["beta"]["q05"] <= true_betas[ch] <= inference[ch]["beta"]["q95"]
    )
    roas_covered = sum(
        1 for ch in channels
        if inference[ch]["roas"]["q05"] <= true_roas[ch] <= inference[ch]["roas"]["q95"]
    )

    # Elasticity MAPE
    elast_mape = ""
    if true_elasticity is not None and true_elasticity != 0.0:
        inf_elast = inference["price_elasticity"]["mean"]
        elast_mape = f"{abs(inf_elast - true_elasticity) / abs(true_elasticity) * 100:.2f}"

    # OLS comparison
    ols_mean_beta_mape = ""
    if ols_results:
        ols_mapes = [
            abs(ols_results["betas"].get(ch, 0) - true_betas[ch]) / max(abs(true_betas[ch]), 1e-6)
            for ch in channels
        ]
        ols_mean_beta_mape = f"{np.mean(ols_mapes) * 100:.2f}"

    with open(summary_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "scenario", "n_channels", "n_periods", "business_context",
                "true_media_pct", "inferred_media_pct", "media_pct_err",
                "mean_beta_mape", "median_beta_mape",
                "mean_roas_mape", "median_roas_mape",
                "beta_90ci_coverage", "roas_90ci_coverage",
                "kendall_tau", "elasticity_mape",
                "ols_mean_beta_mape",
            ])
        n_ch = len(channels)
        w.writerow([
            scenario, n_ch, config.n_periods,
            "|".join(ctx_present) if ctx_present else "media_only",
            f"{true_media_pct:.4f}", f"{inf_media['mean']:.4f}",
            f"{abs(inf_media['mean'] - true_media_pct) / max(abs(true_media_pct), 1e-6) * 100:.2f}",
            f"{np.mean(beta_mapes) * 100:.2f}", f"{np.median(beta_mapes) * 100:.2f}",
            f"{np.mean(roas_mapes) * 100:.2f}", f"{np.median(roas_mapes) * 100:.2f}",
            f"{beta_covered / n_ch:.2f}", f"{roas_covered / n_ch:.2f}",
            f"{tau:.3f}", elast_mape,
            ols_mean_beta_mape,
        ])

    # --- Raw posterior samples CSV (for deep analysis) ---
    samples_path = RESULTS_DIR / f"{scenario}_posterior_samples.csv"
    raw = inference.get("raw_samples")
    raw_global = inference.get("raw_global_samples")
    raw_channel = inference.get("raw_channel_samples", {})

    _REDUCED_PER_CH = 5
    _GLOBAL_RED = 2
    header = ["sample_id", "media_contribution_pct", "price_elasticity_norm"]
    for ch in channels:
        header.extend([f"{ch}_beta_norm", f"{ch}_roas_norm", f"{ch}_contrib_frac",
                       f"{ch}_price_x_media", f"{ch}_dist_x_media"])

    with open(samples_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        n_samples = 2000

        if raw is not None:
            # Monolithic format
            for i in range(min(n_samples, raw.shape[0])):
                row = [i, f"{raw[i, 0]:.6f}", f"{raw[i, 1]:.6f}"]
                for j in range(len(channels)):
                    base = _GLOBAL_RED + j * _REDUCED_PER_CH
                    for k in range(_REDUCED_PER_CH):
                        if base + k < raw.shape[1]:
                            row.append(f"{raw[i, base + k]:.6f}")
                        else:
                            row.append("0.0")
                w.writerow(row)
        elif raw_global is not None:
            # Compositional format
            n_out = min(n_samples, raw_global.shape[0])
            for i in range(n_out):
                row = [i, f"{raw_global[i, 0]:.6f}", f"{raw_global[i, 1]:.6f}"]
                for ch in channels:
                    if ch in raw_channel and i < raw_channel[ch].shape[0]:
                        for k in range(_REDUCED_PER_CH):
                            row.append(f"{raw_channel[ch][i, k]:.6f}")
                    else:
                        row.extend(["0.0"] * _REDUCED_PER_CH)
                w.writerow(row)

    logger.info("Results saved: %s, %s, %s",
                detail_path.name, summary_path.name, samples_path.name)


def _plot_beta_comparison(channels, true_betas, inference, scenario):
    """Bar chart: true vs inferred betas with 90% credible intervals."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(channels))
    width = 0.35

    true_vals = [true_betas[ch] for ch in channels]
    inf_means = [inference[ch]["beta"]["mean"] for ch in channels]
    inf_lo = [inference[ch]["beta"]["q05"] for ch in channels]
    inf_hi = [inference[ch]["beta"]["q95"] for ch in channels]
    errors = [[m - lo for m, lo in zip(inf_means, inf_lo)],
              [hi - m for m, hi in zip(inf_means, inf_hi)]]

    ax.bar(x - width / 2, true_vals, width, label="True", color="#2196F3", alpha=0.8)
    ax.bar(x + width / 2, inf_means, width, label="Inferred", color="#FF9800", alpha=0.8,
           yerr=errors, capsize=4, error_kw={"linewidth": 1.5})

    ax.set_xlabel("Channel")
    ax.set_ylabel("Beta")
    ax.set_title(f"Beta Recovery — {scenario}")
    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{scenario}_betas.png", dpi=150)
    plt.close(fig)


def _plot_roas_comparison(channels, true_roas, inference, scenario):
    """Bar chart: true vs inferred ROAS."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(channels))
    width = 0.35

    true_vals = [true_roas[ch] for ch in channels]
    inf_means = [inference[ch]["roas"]["mean"] for ch in channels]
    inf_lo = [inference[ch]["roas"]["q05"] for ch in channels]
    inf_hi = [inference[ch]["roas"]["q95"] for ch in channels]
    errors = [[m - lo for m, lo in zip(inf_means, inf_lo)],
              [hi - m for m, hi in zip(inf_means, inf_hi)]]

    ax.bar(x - width / 2, true_vals, width, label="True", color="#2196F3", alpha=0.8)
    ax.bar(x + width / 2, inf_means, width, label="Inferred", color="#FF9800", alpha=0.8,
           yerr=errors, capsize=4, error_kw={"linewidth": 1.5})

    ax.set_xlabel("Channel")
    ax.set_ylabel("ROAS")
    ax.set_title(f"ROAS Recovery — {scenario}")
    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{scenario}_roas.png", dpi=150)
    plt.close(fig)


def _plot_contribution_waterfall(channels, true_fracs, inference, scenario):
    """Side-by-side contribution fraction comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    true_vals = [true_fracs.get(ch, 0) for ch in channels]
    inf_vals = [inference[ch]["contribution_fraction"]["mean"] for ch in channels]

    colors = plt.cm.Set2(np.linspace(0, 1, len(channels)))

    ax1.pie(true_vals, labels=channels, autopct="%1.1f%%", colors=colors, startangle=90)
    ax1.set_title("True Contribution Split")

    ax2.pie(inf_vals, labels=channels, autopct="%1.1f%%", colors=colors, startangle=90)
    ax2.set_title("Inferred Contribution Split")

    fig.suptitle(f"Channel Contribution — {scenario}", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{scenario}_contributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_posterior_distributions(channels, true_betas, inference, scenario):
    """Posterior distributions for each channel's beta with true value marked."""
    n_ch = len(channels)
    cols = min(n_ch, 3)
    rows = (n_ch + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_ch == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, ch in enumerate(channels):
        ax = axes[i]
        # Use the inferred beta summary to reconstruct samples
        inf_beta = inference[ch]["beta"]
        # Draw from a normal approximation for visualization
        samples = np.random.normal(inf_beta["mean"], max(inf_beta["std"], 1e-6), size=5000)

        ax.hist(samples, bins=50, density=True, alpha=0.7, color="#FF9800", label="Posterior")
        ax.axvline(true_betas[ch], color="#2196F3", linewidth=2, linestyle="--", label=f"True ({true_betas[ch]:.0f})")
        ax.set_title(ch)
        ax.set_xlabel("Beta")
        ax.legend(fontsize=8)

    # Hide unused axes
    for i in range(n_ch, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"Posterior Distributions — {scenario}", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{scenario}_posteriors.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Demantiq neural engine")
    parser.add_argument("--scenario", default="clean_room",
                        help="Scenario to evaluate (default: clean_room)")
    parser.add_argument("--n-train", type=int, default=500,
                        help="Number of training simulations (default: 500)")
    parser.add_argument("--n-epochs", type=int, default=50,
                        help="Training epochs (default: 50)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Load saved model instead of training")
    parser.add_argument("--all-scenarios", action="store_true",
                        help="Evaluate on all 15 scenarios")
    parser.add_argument("--fixed-channels", type=int, default=None,
                        help="Fix channel count for training (matches theta dims to scenario)")
    parser.add_argument("--simple-embedding", action="store_true",
                        help="Use summary statistics + MLP instead of full embedding network")
    parser.add_argument("--compositional", action="store_true",
                        help="Use compositional architecture (separate global + per-channel NSFs)")
    parser.add_argument("--sequential", action="store_true",
                        help="Run 2 sequential SNPE rounds focused on target scenario")
    parser.add_argument("--seq-rounds", type=int, default=2,
                        help="Number of sequential rounds (default: 2)")
    parser.add_argument("--seq-sims", type=int, default=2000,
                        help="Simulations per sequential round (default: 2000)")
    parser.add_argument("--decomposition", action="store_true",
                        help="Use per-period decomposition engine (Phase A — supervised regression)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for decomposition engine (default: 1e-3)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Decomposition path (Phase A) ----
    if args.decomposition:
        if args.skip_training:
            from demantiq.neural.decomposition_engine import DecompositionEngine, DecompositionConfig
            config = DecompositionConfig(n_fixed_channels=args.fixed_channels)
            engine = DecompositionEngine(config)
            engine.load(str(DECOMP_MODEL_DIR))
        else:
            # Step 1: Generate training data (with per-period decomposition)
            logger.info("=== Generating %d training simulations ===", args.n_train)
            generate_training_data(args.n_train, n_fixed_channels=args.fixed_channels)

            # Step 2: Train decomposition engine
            logger.info("=== Training DECOMPOSITION engine (%d epochs) ===", args.n_epochs)
            engine = train_decomposition_engine(
                n_epochs=args.n_epochs,
                n_train=args.n_train,
                n_fixed_channels=args.fixed_channels,
                learning_rate=args.lr,
            )

        # Step 3: Evaluate
        if args.all_scenarios:
            from demantiq.scenarios.scenario_library import ScenarioLibrary
            for name in ScenarioLibrary.list_scenarios():
                try:
                    evaluate_decomposition(engine, name)
                except Exception as e:
                    logger.error("Failed on %s: %s", name, e)
        else:
            evaluate_decomposition(engine, args.scenario)

        print(f"\nPlots saved to {PLOTS_DIR}/")
        print(f"Results saved to {OUTPUT_DIR / 'results'}/")
        print(f"Model saved to {DECOMP_MODEL_DIR}/")
        return

    # ---- SBI paths (monolithic / compositional) ----
    if args.skip_training:
        logger.info("Loading saved model from %s", MODEL_DIR)
        if args.compositional:
            from demantiq.neural.compositional import CompositionalInferenceEngine
            engine = CompositionalInferenceEngine()
            engine.load(str(MODEL_DIR))
        else:
            engine = load_engine()
    elif args.compositional:
        # Compositional architecture: separate global + per-channel NSFs
        from demantiq.neural.compositional import CompositionalInferenceEngine, CompositionalConfig

        # Step 1: Generate training data
        logger.info("=== Generating %d training simulations ===", args.n_train)
        generate_training_data(args.n_train, n_fixed_channels=args.fixed_channels)

        # Step 2: Train compositional engine
        logger.info("=== Training COMPOSITIONAL engine ===")
        comp_config = CompositionalConfig(
            embed_epochs=10,
            embed_lr=1e-3,
            embed_batch_size=512,
            nsf_epochs=args.n_epochs,
            nsf_patience=30,
            nsf_batch_size=512,
        )
        engine = CompositionalInferenceEngine(comp_config)
        engine.train(
            data_dir=str(TRAINING_DATA_DIR),
            save_dir=str(MODEL_DIR),
            n_fixed_channels=args.fixed_channels,
        )
    else:
        # Step 1: Generate training data
        logger.info("=== Generating %d training simulations ===", args.n_train)
        generate_training_data(args.n_train, n_fixed_channels=args.fixed_channels)

        # Step 2: Train monolithic engine
        logger.info("=== Training neural engine (%d epochs) ===", args.n_epochs)
        engine = train_engine(args.n_epochs, n_fixed_channels=args.fixed_channels,
                              simple_embedding=args.simple_embedding)

    # Step 2.5: Sequential refinement (if requested, runs after training)
    if args.sequential and not args.all_scenarios:
        from demantiq.scenarios.scenario_library import ScenarioLibrary
        from demantiq.core.demand_kernel import simulate as sim_fn
        from demantiq.orchestration.training_format import extract_context_matrix

        logger.info("=== Running %d sequential rounds on %s ===",
                     args.seq_rounds, args.scenario)
        config = ScenarioLibrary.get(args.scenario)
        result = sim_fn(config)

        ch_names = [ch.name for ch in config.channels]
        y_obs = result.observable_data["y"].values
        spend_obs = np.column_stack([
            result.observable_data[f"{ch}_spend"].values for ch in ch_names
        ])
        ctx_obs = extract_context_matrix(result.observable_data, config.n_periods)

        engine.train_sequential(
            x_obs=y_obs,
            spend_obs=spend_obs,
            channel_names=ch_names,
            context_obs=ctx_obs,
            n_rounds=args.seq_rounds,
            n_sims_per_round=args.seq_sims,
        )
        # Save updated model
        engine.save(str(MODEL_DIR))

    # Step 3: Evaluate
    if args.all_scenarios:
        from demantiq.scenarios.scenario_library import ScenarioLibrary
        for name in ScenarioLibrary.list_scenarios():
            try:
                evaluate_scenario(engine, name)
            except Exception as e:
                logger.error("Failed on %s: %s", name, e)
    else:
        evaluate_scenario(engine, args.scenario)

    print(f"\nPlots saved to {PLOTS_DIR}/")
    print(f"Results saved to {OUTPUT_DIR / 'results'}/")
    print(f"Model saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
