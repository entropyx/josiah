"""Train and evaluate the PFN-based demand decomposition model.

Usage:
    source venv/bin/activate

    # Generate data + train + evaluate
    python scripts/train_pfn.py --n-train 1000 --n-epochs 50

    # Skip data generation (reuse existing)
    python scripts/train_pfn.py --n-epochs 100 --skip-datagen

    # Evaluate only (load saved model)
    python scripts/train_pfn.py --skip-training --random-eval 10

    # Smoke test
    python scripts/train_pfn.py --n-train 100 --n-epochs 10 --random-eval 3
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("neural_output")
TRAINING_DATA_DIR = OUTPUT_DIR / "pfn_training_data"
MODEL_DIR = OUTPUT_DIR / "pfn_model"


def generate_training_data(n_total: int, n_workers: int = 4, channel_range: tuple[int, int] = (2, 8)):
    from demantiq.orchestration.training_pipeline import TrainingPipeline
    from demantiq.scenarios.scenario_sampler import ScenarioSampler

    TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    sampler = ScenarioSampler(seed=42, rich_context=True, channel_range=channel_range)
    pipeline = TrainingPipeline(sampler, output_dir=str(TRAINING_DATA_DIR), batch_size=100)

    t0 = time.time()
    pipeline.generate(n_total=n_total, n_workers=n_workers, seed=42)
    logger.info("Generated %d simulations in %.1f seconds", n_total, time.time() - t0)


def train_engine(n_epochs: int = 100, n_train: int = 100000, learning_rate: float = 1e-3,
                 batch_size: int = 32, patience: int = 20, d_model: int = 128,
                 n_layers: int = 6):
    from demantiq.neural.pfn_engine import PFNEngine, PFNConfig

    config = PFNConfig(
        n_epochs=n_epochs, n_train=n_train, learning_rate=learning_rate,
        batch_size=batch_size, patience=patience, d_model=d_model,
        n_layers=n_layers, data_dir=str(TRAINING_DATA_DIR),
    )

    engine = PFNEngine(config)
    metrics = engine.train()
    engine.save(str(MODEL_DIR))
    logger.info("Training metrics: %s", metrics)
    return engine


def load_engine():
    from demantiq.neural.pfn_engine import PFNEngine, PFNConfig
    import json

    config_path = MODEL_DIR / "pfn_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = PFNConfig(**json.load(f))
    else:
        config = PFNConfig()

    engine = PFNEngine(config)
    engine.load(str(MODEL_DIR))
    return engine


def evaluate_scenario(engine, scenario_name: str | None = None, scenario_seed: int | None = None):
    from demantiq.scenarios.scenario_sampler import ScenarioSampler
    from demantiq.core.demand_kernel import simulate
    from demantiq.orchestration.training_format import (
        extract_context_matrix, ground_truth_to_decomposition,
        DECOMP_IDX_BASELINE, DECOMP_IDX_CHANNELS_START,
        DECOMP_IDX_PRICE, DECOMP_IDX_COMPETITION, DECOMP_IDX_MACRO,
    )
    from scipy.stats import spearmanr, pearsonr

    if scenario_name is not None:
        from demantiq.scenarios.scenario_library import ScenarioLibrary
        config = ScenarioLibrary.get(scenario_name)
    else:
        sampler = ScenarioSampler(seed=scenario_seed, rich_context=True, channel_range=(2, 8))
        config = sampler.sample(1)[0]

    result = simulate(config)
    ch_names = [ch.name for ch in config.channels]
    n_ch = len(ch_names)
    T = config.n_periods
    y = result.observable_data["y"].values.astype(np.float32)

    spend = np.column_stack([
        result.observable_data[f"{ch}_spend"].values for ch in ch_names
    ]).astype(np.float32)
    imp = np.column_stack([
        result.observable_data[f"{ch}_impressions"].values for ch in ch_names
    ]).astype(np.float32)
    clk = np.column_stack([
        result.observable_data[f"{ch}_clicks"].values for ch in ch_names
    ]).astype(np.float32)
    context = extract_context_matrix(result.observable_data, T).astype(np.float32)

    t0 = time.time()
    inf = engine.infer(y, spend, context, n_ch, impressions=imp, clicks=clk)
    inf_time = time.time() - t0

    pred_ch = inf["channel_contributions"]
    pred_base = inf["baseline"]
    pred_nm = inf["non_media"]
    y_hat = inf["y_hat"]

    true_decomp = ground_truth_to_decomposition(result.ground_truth, ch_names, T)
    true_base = true_decomp[:, DECOMP_IDX_BASELINE]
    true_ch = np.column_stack([
        true_decomp[:, DECOMP_IDX_CHANNELS_START + i] for i in range(n_ch)
    ])

    total_y = y[:T].sum()
    true_base_pct = true_base.sum() / total_y * 100
    pred_base_pct = pred_base.sum() / total_y * 100
    cat_error = abs(pred_base_pct - true_base_pct)

    true_ch_totals = true_ch.sum(axis=0)
    pred_ch_totals = pred_ch.sum(axis=0)
    rank_corr = spearmanr(true_ch_totals, pred_ch_totals)[0] if n_ch >= 3 else float(np.corrcoef(true_ch_totals, pred_ch_totals)[0, 1])

    ch_time_corrs = []
    for i in range(n_ch):
        if np.std(true_ch[:, i]) > 1e-8 and np.std(pred_ch[:, i]) > 1e-8:
            ch_time_corrs.append(pearsonr(true_ch[:, i], pred_ch[:, i])[0])
    mean_time_corr = np.mean(ch_time_corrs) if ch_time_corrs else 0.0

    ch_errors = []
    for i in range(n_ch):
        if abs(true_ch_totals[i]) > 1e-6:
            ch_errors.append(abs(pred_ch_totals[i] - true_ch_totals[i]) / abs(true_ch_totals[i]) * 100)
    mean_ch_error = np.mean(ch_errors) if ch_errors else 0.0

    ss_res = np.sum((y[:T] - y_hat[:T]) ** 2)
    ss_tot = np.sum((y[:T] - y[:T].mean()) ** 2)
    recon_r2 = 1 - ss_res / ss_tot if ss_tot > 1e-6 else 0.0

    true_ranking = sorted(range(n_ch), key=lambda i: -true_ch_totals[i])
    pred_ranking = sorted(range(n_ch), key=lambda i: -pred_ch_totals[i])

    label = scenario_name or f"random(seed={scenario_seed})"
    print(f"\n{'='*70}")
    print(f"  PFN EVALUATION — {label}")
    print(f"  Channels: {n_ch} ({', '.join(ch_names)}) | Periods: {T} | Inference: {inf_time:.3f}s")
    print(f"{'='*70}")

    print(f"\n  Baseline: true={true_base_pct:.1f}%  pred={pred_base_pct:.1f}%  err={cat_error:.1f}pp")

    print(f"\n{'Channel':<16} {'True Total':>12} {'Pred Total':>12} {'Error %':>10} {'Time Corr':>10}")
    print("-" * 66)
    for i in range(n_ch):
        tc = true_ch_totals[i]
        pc = pred_ch_totals[i]
        err = abs(pc - tc) / max(abs(tc), 1e-6) * 100
        tcorr = ch_time_corrs[i] if i < len(ch_time_corrs) else 0.0
        print(f"{ch_names[i]:<16} {tc:>12.0f} {pc:>12.0f} {err:>9.1f}% {tcorr:>9.3f}")

    print(f"\n  Rank correlation:   {rank_corr:.3f}")
    print(f"  Mean time corr:     {mean_time_corr:.3f}")
    print(f"  Mean channel err:   {mean_ch_error:.1f}%")
    print(f"  Reconstruction R²:  {recon_r2:.4f}")
    print(f"  True ranking:  {' > '.join(ch_names[i] for i in true_ranking)}")
    print(f"  Pred ranking:  {' > '.join(ch_names[i] for i in pred_ranking)}")

    return {
        "scenario": label, "n_channels": n_ch, "n_periods": T,
        "category_error": cat_error, "rank_corr": rank_corr,
        "mean_ch_time_corr": mean_time_corr, "mean_ch_error": mean_ch_error,
        "recon_r2": recon_r2,
    }


def evaluate_random(engine, n_scenarios: int = 10, seed: int = 999):
    print(f"\n{'#'*70}")
    print(f"  RANDOM EVALUATION — {n_scenarios} scenarios (seed={seed})")
    print(f"{'#'*70}")

    all_metrics = []
    for i in range(n_scenarios):
        metrics = evaluate_scenario(engine, scenario_seed=seed + i)
        all_metrics.append(metrics)

    cat_errs = [m["category_error"] for m in all_metrics]
    rank_corrs = [m["rank_corr"] for m in all_metrics]
    time_corrs = [m["mean_ch_time_corr"] for m in all_metrics]
    ch_errs = [m["mean_ch_error"] for m in all_metrics]
    recon_r2s = [m["recon_r2"] for m in all_metrics]

    print(f"\n{'='*70}")
    print(f"  SUMMARY ({n_scenarios} scenarios)")
    print(f"{'='*70}")

    print(f"\n{'Scenario':<30} {'Cat Err':>8} {'Rank':>8} {'Time r':>8} {'Ch Err':>8} {'R²':>8}")
    print("-" * 74)
    for m in all_metrics:
        print(f"{m['scenario']:<30} {m['category_error']:>7.1f}pp {m['rank_corr']:>7.3f} "
              f"{m['mean_ch_time_corr']:>7.3f} {m['mean_ch_error']:>7.1f}% {m['recon_r2']:>7.4f}")

    print(f"\n  Mean category error:    {np.mean(cat_errs):.1f}pp")
    print(f"  Mean rank correlation:  {np.mean(rank_corrs):.3f}")
    print(f"  Mean time correlation:  {np.mean(time_corrs):.3f}")
    print(f"  Mean channel error:     {np.mean(ch_errs):.1f}%")
    print(f"  Mean reconstruction R²: {np.mean(recon_r2s):.4f}")
    print(f"  Scenarios rank > 0.7:   {sum(1 for r in rank_corrs if r > 0.7)}/{n_scenarios}")

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Train PFN decomposition model")
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--skip-datagen", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--random-eval", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--channel-min", type=int, default=2)
    parser.add_argument("--channel-max", type=int, default=8)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_training:
        if not args.skip_datagen:
            generate_training_data(
                args.n_train, n_workers=args.n_workers,
                channel_range=(args.channel_min, args.channel_max),
            )
        engine = train_engine(
            n_epochs=args.n_epochs, n_train=args.n_train, learning_rate=args.lr,
            batch_size=args.batch_size, patience=args.patience,
            d_model=args.d_model, n_layers=args.n_layers,
        )
    else:
        engine = load_engine()

    if args.scenario:
        evaluate_scenario(engine, scenario_name=args.scenario)

    if args.random_eval > 0:
        evaluate_random(engine, n_scenarios=args.random_eval)

    if not args.scenario and args.random_eval == 0:
        evaluate_random(engine, n_scenarios=3)


if __name__ == "__main__":
    main()
