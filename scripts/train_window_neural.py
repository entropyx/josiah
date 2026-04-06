"""Train and evaluate the window-based per-channel contribution model.

Usage:
    source venv/bin/activate

    # Generate data + train + evaluate
    python scripts/train_window_neural.py --n-train 1000 --n-epochs 20

    # Skip data generation (reuse existing)
    python scripts/train_window_neural.py --n-epochs 50 --skip-datagen

    # Evaluate only (load saved model)
    python scripts/train_window_neural.py --skip-training --random-eval 10

    # Smoke test
    python scripts/train_window_neural.py --n-train 100 --n-epochs 5 --random-eval 3
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("neural_output")
TRAINING_DATA_DIR = OUTPUT_DIR / "window_training_data"
MODEL_DIR = OUTPUT_DIR / "window_model"


def generate_training_data(n_total: int, n_workers: int = 4, channel_range: tuple[int, int] = (2, 8)):
    """Generate training data with variable channel counts."""
    from demantiq.orchestration.training_pipeline import TrainingPipeline
    from demantiq.scenarios.scenario_sampler import ScenarioSampler

    TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)

    sampler = ScenarioSampler(seed=42, rich_context=True, channel_range=channel_range)
    pipeline = TrainingPipeline(sampler, output_dir=str(TRAINING_DATA_DIR), batch_size=100)

    t0 = time.time()
    pipeline.generate(n_total=n_total, n_workers=n_workers, seed=42)
    logger.info("Generated %d simulations in %.1f seconds", n_total, time.time() - t0)


def train_engine(
    n_epochs: int = 100,
    n_train: int = 100000,
    learning_rate: float = 3e-4,
    batch_size: int = 512,
    window_size: int = 16,
    patience: int = 20,
):
    """Train the window contribution engine."""
    from demantiq.neural.window_engine import WindowContributionEngine, WindowContributionConfig

    config = WindowContributionConfig(
        n_epochs=n_epochs,
        n_train=n_train,
        learning_rate=learning_rate,
        batch_size=batch_size,
        window_size=window_size,
        patience=patience,
        data_dir=str(TRAINING_DATA_DIR),
    )

    engine = WindowContributionEngine(config)
    metrics = engine.train()
    engine.save(str(MODEL_DIR))

    logger.info("Training metrics: %s", metrics)
    return engine


def load_engine():
    """Load a previously trained engine."""
    from demantiq.neural.window_engine import WindowContributionEngine, WindowContributionConfig
    import json

    config_path = MODEL_DIR / "window_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = WindowContributionConfig(**json.load(f))
    else:
        config = WindowContributionConfig()

    engine = WindowContributionEngine(config)
    engine.load(str(MODEL_DIR))
    return engine


def evaluate_scenario(engine, scenario_name: str | None = None, scenario_seed: int | None = None):
    """Evaluate on a named scenario or random scenario.

    Returns dict with evaluation metrics.
    """
    from demantiq.scenarios.scenario_sampler import ScenarioSampler
    from demantiq.core.demand_kernel import simulate
    from demantiq.orchestration.training_format import (
        extract_context_matrix,
        ground_truth_to_decomposition,
        DECOMP_IDX_BASELINE,
        DECOMP_IDX_CHANNELS_START,
        DECOMP_IDX_PRICE,
        DECOMP_IDX_COMPETITION,
        DECOMP_IDX_MACRO,
        DECOMP_IDX_NOISE,
    )

    if scenario_name is not None:
        from demantiq.scenarios.scenario_library import ScenarioLibrary
        config = ScenarioLibrary.get(scenario_name)
    else:
        sampler = ScenarioSampler(seed=scenario_seed, rich_context=True, channel_range=(2, 8))
        config = sampler.sample(1)[0]

    result = simulate(config)
    channel_names = [ch.name for ch in config.channels]
    n_ch = len(channel_names)
    y = result.observable_data["y"].values.astype(np.float32)
    T = config.n_periods

    # Build spend matrix (T, n_ch)
    spend_matrix = np.column_stack([
        result.observable_data[f"{ch}_spend"].values for ch in channel_names
    ]).astype(np.float32)

    context_matrix = extract_context_matrix(result.observable_data, T).astype(np.float32)

    # Run inference
    t0 = time.time()
    inf = engine.infer(y, spend_matrix, context_matrix, n_ch)
    inf_time = time.time() - t0

    pred_channels = inf["channel_contributions"]  # (T, n_ch)
    pred_baseline = inf["baseline"]               # (T,)
    pred_non_media = inf["non_media"]             # (T,)
    y_hat = inf["y_hat"]                          # (T,)

    # Ground truth
    true_decomp = ground_truth_to_decomposition(result.ground_truth, channel_names, T)

    true_baseline = true_decomp[:, DECOMP_IDX_BASELINE]
    true_channels = np.column_stack([
        true_decomp[:, DECOMP_IDX_CHANNELS_START + i] for i in range(n_ch)
    ])
    true_non_media = (
        true_decomp[:, DECOMP_IDX_PRICE]
        + true_decomp[:, DECOMP_IDX_COMPETITION]
        + true_decomp[:, DECOMP_IDX_MACRO]
    )

    total_y = y[:T].sum()

    # Category metrics
    true_baseline_pct = true_baseline.sum() / total_y * 100
    pred_baseline_pct = pred_baseline.sum() / total_y * 100
    true_media_pct = true_channels.sum() / total_y * 100
    pred_media_pct = pred_channels.sum() / total_y * 100
    category_error = abs(pred_baseline_pct - true_baseline_pct)

    # Channel metrics
    from scipy.stats import spearmanr, pearsonr

    # Per-channel total contributions
    true_ch_totals = true_channels.sum(axis=0)
    pred_ch_totals = pred_channels.sum(axis=0)

    if n_ch >= 3:
        rank_corr, _ = spearmanr(true_ch_totals, pred_ch_totals)
    else:
        rank_corr = float(np.corrcoef(true_ch_totals, pred_ch_totals)[0, 1])

    # Per-channel time series correlation (mean across channels)
    ch_time_corrs = []
    for i in range(n_ch):
        tc = true_channels[:, i]
        pc = pred_channels[:, i]
        if np.std(tc) > 1e-8 and np.std(pc) > 1e-8:
            corr, _ = pearsonr(tc, pc)
            ch_time_corrs.append(corr)
    mean_ch_time_corr = np.mean(ch_time_corrs) if ch_time_corrs else 0.0

    # Per-channel contribution error
    ch_errors = []
    for i in range(n_ch):
        true_total = abs(true_ch_totals[i])
        if true_total > 1e-6:
            err = abs(pred_ch_totals[i] - true_ch_totals[i]) / true_total * 100
            ch_errors.append(err)
    mean_ch_error = np.mean(ch_errors) if ch_errors else 0.0

    # Reconstruction R²
    ss_res = np.sum((y[:T] - y_hat[:T]) ** 2)
    ss_tot = np.sum((y[:T] - y[:T].mean()) ** 2)
    recon_r2 = 1 - ss_res / ss_tot if ss_tot > 1e-6 else 0.0

    # Channel ranking
    true_ranking = sorted(range(n_ch), key=lambda i: -true_ch_totals[i])
    pred_ranking = sorted(range(n_ch), key=lambda i: -pred_ch_totals[i])
    true_rank_names = [channel_names[i] for i in true_ranking]
    pred_rank_names = [channel_names[i] for i in pred_ranking]

    # Print results
    label = scenario_name or f"random(seed={scenario_seed})"
    print(f"\n{'='*70}")
    print(f"  WINDOW MODEL EVALUATION — {label}")
    print(f"  Channels: {n_ch} ({', '.join(channel_names)}) | Periods: {T}")
    print(f"  Inference time: {inf_time:.3f}s")
    print(f"{'='*70}")

    print(f"\n--- Category Breakdown ---")
    print(f"{'Category':<16} {'True %':>10} {'Pred %':>10} {'Error':>10}")
    print("-" * 50)
    for cat, tp, pp in [
        ("Baseline", true_baseline_pct, pred_baseline_pct),
        ("Media", true_media_pct, pred_media_pct),
    ]:
        print(f"{cat:<16} {tp:>9.1f}% {pp:>9.1f}% {abs(pp - tp):>9.1f}pp")

    print(f"\n--- Channel Contributions ---")
    print(f"{'Channel':<16} {'True Total':>12} {'Pred Total':>12} {'Error %':>10} {'Time Corr':>10}")
    print("-" * 66)
    for i in range(n_ch):
        tc = true_ch_totals[i]
        pc = pred_ch_totals[i]
        err = abs(pc - tc) / max(abs(tc), 1e-6) * 100
        tcorr = ch_time_corrs[i] if i < len(ch_time_corrs) else 0.0
        print(f"{channel_names[i]:<16} {tc:>12.0f} {pc:>12.0f} {err:>9.1f}% {tcorr:>9.3f}")

    print(f"\n--- Summary Metrics ---")
    print(f"  Category error:          {category_error:.1f}pp")
    print(f"  Channel rank correlation: {rank_corr:.3f}")
    print(f"  Mean channel time corr:   {mean_ch_time_corr:.3f}")
    print(f"  Mean channel error:       {mean_ch_error:.1f}%")
    print(f"  Reconstruction R²:        {recon_r2:.4f}")
    print(f"  True ranking:  {' > '.join(true_rank_names)}")
    print(f"  Pred ranking:  {' > '.join(pred_rank_names)}")

    return {
        "scenario": label,
        "n_channels": n_ch,
        "n_periods": T,
        "category_error": category_error,
        "rank_corr": rank_corr,
        "mean_ch_time_corr": mean_ch_time_corr,
        "mean_ch_error": mean_ch_error,
        "recon_r2": recon_r2,
        "true_baseline_pct": true_baseline_pct,
        "pred_baseline_pct": pred_baseline_pct,
        "true_media_pct": true_media_pct,
        "pred_media_pct": pred_media_pct,
    }


def evaluate_random(engine, n_scenarios: int = 10, seed: int = 999):
    """Evaluate on diverse random scenarios."""
    print(f"\n{'#'*70}")
    print(f"  RANDOM EVALUATION — {n_scenarios} scenarios (seed={seed})")
    print(f"{'#'*70}")

    all_metrics = []
    for i in range(n_scenarios):
        scenario_seed = seed + i
        metrics = evaluate_scenario(engine, scenario_seed=scenario_seed)
        all_metrics.append(metrics)

    # Summary
    cat_errors = [m["category_error"] for m in all_metrics]
    rank_corrs = [m["rank_corr"] for m in all_metrics]
    time_corrs = [m["mean_ch_time_corr"] for m in all_metrics]
    ch_errors = [m["mean_ch_error"] for m in all_metrics]
    recon_r2s = [m["recon_r2"] for m in all_metrics]

    print(f"\n{'='*70}")
    print(f"  RANDOM EVAL SUMMARY ({n_scenarios} scenarios)")
    print(f"{'='*70}")

    print(f"\n{'Scenario':<30} {'Cat Err':>8} {'Rank r':>8} {'Time r':>8} {'Ch Err':>8} {'R²':>8}")
    print("-" * 74)
    for m in all_metrics:
        print(f"{m['scenario']:<30} {m['category_error']:>7.1f}pp {m['rank_corr']:>7.3f} "
              f"{m['mean_ch_time_corr']:>7.3f} {m['mean_ch_error']:>7.1f}% {m['recon_r2']:>7.4f}")

    print(f"\n--- Aggregated ---")
    print(f"  Mean category error:      {np.mean(cat_errors):.1f}pp  (target: <5pp)")
    print(f"  Mean rank correlation:    {np.mean(rank_corrs):.3f}  (target: >0.7)")
    print(f"  Mean time correlation:    {np.mean(time_corrs):.3f}")
    print(f"  Mean channel error:       {np.mean(ch_errors):.1f}%  (target: <25%)")
    print(f"  Mean reconstruction R²:   {np.mean(recon_r2s):.4f}  (target: >0.9)")
    print(f"  Scenarios with rank>0.5:  {sum(1 for r in rank_corrs if r > 0.5)}/{n_scenarios}")

    return all_metrics


def run_overfit_test(n_epochs: int = 200, lr: float = 1e-3, batch_size: int = 32, window_size: int = 16):
    """Train on a single scenario and evaluate on the SAME scenario.

    If the model can't overfit one scenario, the architecture is broken.
    Target: channel rank correlation > 0.9, channel time correlation > 0.8.
    """
    from demantiq.scenarios.scenario_sampler import ScenarioSampler
    from demantiq.core.demand_kernel import simulate
    from demantiq.orchestration.training_pipeline import TrainingPipeline
    from demantiq.neural.window_engine import WindowContributionEngine, WindowContributionConfig
    from demantiq.orchestration.training_format import extract_context_matrix

    OVERFIT_DATA_DIR = OUTPUT_DIR / "overfit_data"
    OVERFIT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Generate exactly 1 scenario with seed=42
    sampler = ScenarioSampler(seed=42, rich_context=True, channel_range=(3, 5))
    pipeline = TrainingPipeline(sampler, output_dir=str(OVERFIT_DATA_DIR), batch_size=100)
    pipeline.generate(n_total=1, n_workers=1, seed=42)

    # Also simulate the same scenario for evaluation
    sampler2 = ScenarioSampler(seed=42, rich_context=True, channel_range=(3, 5))
    eval_config = sampler2.sample(1)[0]
    eval_result = simulate(eval_config)
    channel_names = [ch.name for ch in eval_config.channels]
    n_ch = len(channel_names)
    logger.info("Overfit test: %d channels (%s), %d periods",
                n_ch, ", ".join(channel_names), eval_config.n_periods)

    # Train with no early stopping
    config = WindowContributionConfig(
        n_epochs=n_epochs,
        n_train=1,
        learning_rate=lr,
        batch_size=batch_size,
        window_size=window_size,
        patience=n_epochs,  # no early stopping
        val_fraction=0.01,  # minimal validation
        data_dir=str(OVERFIT_DATA_DIR),
    )

    engine = WindowContributionEngine(config)
    metrics = engine.train()

    # Evaluate on the SAME scenario
    y = eval_result.observable_data["y"].values.astype(np.float32)
    spend = np.column_stack([
        eval_result.observable_data[f"{ch}_spend"].values for ch in channel_names
    ]).astype(np.float32)
    context = extract_context_matrix(eval_result.observable_data, eval_config.n_periods).astype(np.float32)

    result = engine.infer(y, spend, context, n_ch)

    # Compare to ground truth
    from demantiq.orchestration.training_format import (
        ground_truth_to_decomposition,
        DECOMP_IDX_BASELINE, DECOMP_IDX_CHANNELS_START,
        DECOMP_IDX_PRICE, DECOMP_IDX_COMPETITION, DECOMP_IDX_MACRO,
    )
    from scipy.stats import spearmanr, pearsonr

    T = eval_config.n_periods
    true_decomp = ground_truth_to_decomposition(eval_result.ground_truth, channel_names, T)

    true_baseline = true_decomp[:, DECOMP_IDX_BASELINE]
    true_channels = np.column_stack([
        true_decomp[:, DECOMP_IDX_CHANNELS_START + i] for i in range(n_ch)
    ])

    pred_channels = result["channel_contributions"]
    pred_baseline = result["baseline"]

    total_y = y[:T].sum()
    true_base_pct = true_baseline.sum() / total_y * 100
    pred_base_pct = pred_baseline.sum() / total_y * 100

    true_ch_totals = true_channels.sum(axis=0)
    pred_ch_totals = pred_channels.sum(axis=0)
    rank_corr = spearmanr(true_ch_totals, pred_ch_totals)[0] if n_ch >= 3 else np.corrcoef(true_ch_totals, pred_ch_totals)[0, 1]

    ch_time_corrs = []
    for i in range(n_ch):
        if np.std(true_channels[:, i]) > 1e-8 and np.std(pred_channels[:, i]) > 1e-8:
            ch_time_corrs.append(pearsonr(true_channels[:, i], pred_channels[:, i])[0])
    mean_time_corr = np.mean(ch_time_corrs) if ch_time_corrs else 0.0

    print(f"\n{'='*70}")
    print(f"  OVERFIT TEST RESULTS")
    print(f"  1 scenario, {n_epochs} epochs, {n_ch} channels")
    print(f"{'='*70}")

    print(f"\n  Baseline: true={true_base_pct:.1f}%  pred={pred_base_pct:.1f}%  err={abs(pred_base_pct - true_base_pct):.1f}pp")

    print(f"\n{'Channel':<16} {'True Total':>12} {'Pred Total':>12} {'Error %':>10} {'Time Corr':>10}")
    print("-" * 66)
    for i in range(n_ch):
        tc = true_ch_totals[i]
        pc = pred_ch_totals[i]
        err = abs(pc - tc) / max(abs(tc), 1e-6) * 100
        tcorr = ch_time_corrs[i] if i < len(ch_time_corrs) else 0.0
        print(f"{channel_names[i]:<16} {tc:>12.0f} {pc:>12.0f} {err:>9.1f}% {tcorr:>9.3f}")

    # Reconstruction R²
    ss_res = np.sum((y[:T] - result["y_hat"][:T]) ** 2)
    ss_tot = np.sum((y[:T] - y[:T].mean()) ** 2)
    recon_r2 = 1 - ss_res / ss_tot if ss_tot > 1e-6 else 0.0

    print(f"\n  Channel rank correlation:  {rank_corr:.3f}  (target: >0.9)")
    print(f"  Mean channel time corr:    {mean_time_corr:.3f}  (target: >0.8)")
    print(f"  Reconstruction R²:         {recon_r2:.4f}  (target: >0.9)")

    passed = rank_corr > 0.9 and mean_time_corr > 0.8 and recon_r2 > 0.9
    print(f"\n  {'PASSED' if passed else 'FAILED'}")
    if not passed:
        print("  The architecture cannot overfit a single scenario.")
        print("  This suggests a fundamental issue that more data/epochs won't fix.")


def main():
    parser = argparse.ArgumentParser(description="Train window-based contribution model")
    parser.add_argument("--n-train", type=int, default=1000, help="Number of training scenarios")
    parser.add_argument("--n-epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--window-size", type=int, default=16, help="Window size in weeks")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--skip-datagen", action="store_true", help="Skip data generation")
    parser.add_argument("--skip-training", action="store_true", help="Load saved model")
    parser.add_argument("--scenario", type=str, default=None, help="Named scenario to evaluate")
    parser.add_argument("--random-eval", type=int, default=0, help="Evaluate on N random scenarios")
    parser.add_argument("--n-workers", type=int, default=4, help="Data generation workers")
    parser.add_argument("--channel-min", type=int, default=2, help="Min channels")
    parser.add_argument("--channel-max", type=int, default=8, help="Max channels")
    parser.add_argument("--overfit-test", action="store_true",
                        help="Train on 1 scenario and evaluate on the SAME scenario")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.overfit_test:
        run_overfit_test(
            n_epochs=args.n_epochs, lr=args.lr,
            batch_size=args.batch_size, window_size=args.window_size,
        )
        return

    if not args.skip_training:
        # Generate training data
        if not args.skip_datagen:
            generate_training_data(
                args.n_train, n_workers=args.n_workers,
                channel_range=(args.channel_min, args.channel_max),
            )

        # Train
        engine = train_engine(
            n_epochs=args.n_epochs,
            n_train=args.n_train,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            window_size=args.window_size,
            patience=args.patience,
        )
    else:
        engine = load_engine()

    # Evaluate
    if args.scenario:
        evaluate_scenario(engine, scenario_name=args.scenario)

    if args.random_eval > 0:
        evaluate_random(engine, n_scenarios=args.random_eval)

    if not args.scenario and args.random_eval == 0:
        # Default: quick eval on 3 random scenarios
        evaluate_random(engine, n_scenarios=3)


if __name__ == "__main__":
    main()
