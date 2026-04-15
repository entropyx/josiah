"""Train and evaluate the proper PFN model.

Usage:
  source venv/bin/activate
  python scripts/train_proper_pfn.py --n-train 50000 --n-epochs 100 --random-eval 10
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from demantiq.scenarios.scenario_sampler import ScenarioSampler
from demantiq.core.demand_kernel import simulate
from demantiq.orchestration.training_pipeline import TrainingPipeline
from demantiq.orchestration.training_format import (
    extract_context_matrix, ground_truth_to_decomposition,
    DECOMP_IDX_BASELINE, DECOMP_IDX_CHANNELS_START,
)
from demantiq.neural.proper_pfn.engine import ProperPFNEngine, ProperPFNConfig
from demantiq.neural.proper_pfn.metrics import scenario_level_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT = Path("neural_output")
DATA = OUT / "proper_pfn_data"
MODEL = OUT / "proper_pfn_model"


def generate_data(n_total: int, n_workers: int = 4):
    DATA.mkdir(parents=True, exist_ok=True)
    sampler = ScenarioSampler(seed=42, rich_context=True, channel_range=(2, 8))
    pipeline = TrainingPipeline(sampler, output_dir=str(DATA), batch_size=100)
    t0 = time.time()
    pipeline.generate(n_total=n_total, n_workers=n_workers, seed=42)
    logger.info("Generated %d scenarios in %.0fs", n_total, time.time() - t0)


def train(args):
    config = ProperPFNConfig(
        n_epochs=args.n_epochs,
        n_train=args.n_train,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        d_model=args.d_model,
        n_layers=args.n_layers,
        week_mask_fraction=args.mask_frac,
        patience=args.patience,
        data_dir=str(DATA),
    )
    engine = ProperPFNEngine(config)
    metrics = engine.train()
    engine.save(str(MODEL))
    logger.info("Training metrics: %s", metrics)
    return engine


def load_engine():
    import json
    with open(MODEL / "proper_pfn_config.json") as f:
        config = ProperPFNConfig(**json.load(f))
    engine = ProperPFNEngine(config)
    engine.load(str(MODEL))
    return engine


def evaluate_scenario(engine, seed: int):
    sampler = ScenarioSampler(seed=seed, rich_context=True, channel_range=(2, 8))
    cfg = sampler.sample(1)[0]
    result = simulate(cfg)
    ch_names = [c.name for c in cfg.channels]
    n_ch = len(ch_names)
    T = cfg.n_periods
    y = result.observable_data["y"].values.astype(np.float32)
    spend = np.column_stack([result.observable_data[f"{c}_spend"].values for c in ch_names]).astype(np.float32)
    imp = np.column_stack([result.observable_data[f"{c}_impressions"].values for c in ch_names]).astype(np.float32)
    clk = np.column_stack([result.observable_data[f"{c}_clicks"].values for c in ch_names]).astype(np.float32)
    ctx = extract_context_matrix(result.observable_data, T).astype(np.float32)

    out = engine.infer(y, spend, imp, clk, ctx, n_ch)

    true_decomp = ground_truth_to_decomposition(result.ground_truth, ch_names, T)
    true_ch = np.column_stack([true_decomp[:, DECOMP_IDX_CHANNELS_START + i] for i in range(n_ch)])
    true_base = true_decomp[:, DECOMP_IDX_BASELINE]

    metrics = scenario_level_metrics(
        out["channel_contributions"], out["baseline"],
        true_ch, true_base, y,
    )
    metrics["scenario"] = f"seed={seed}"
    metrics["n_channels"] = n_ch
    metrics["n_periods"] = T
    return metrics


def evaluate_random(engine, n: int, seed: int = 999):
    rows = []
    for i in range(n):
        m = evaluate_scenario(engine, seed + i)
        rows.append(m)
        print(f"seed={seed+i}  ch_r2={m['channel_r2_mean']:.3f}  "
              f"base_r2={m['baseline_r2']:.3f}  rank={m['rank_corr']:.3f}  "
              f"cat_err={m['category_error_pp']:.1f}pp  ch_err={m['per_channel_error_pct']:.1f}%")

    print(f"\n{'='*70}\n  SUMMARY ({n} scenarios)\n{'='*70}")
    print(f"  Mean channel R²:    {np.mean([m['channel_r2_mean'] for m in rows]):.3f}")
    print(f"  Mean baseline R²:   {np.mean([m['baseline_r2'] for m in rows]):.3f}")
    print(f"  Mean rank corr:     {np.mean([m['rank_corr'] for m in rows]):.3f}")
    print(f"  Mean category err:  {np.mean([m['category_error_pp'] for m in rows]):.1f}pp")
    print(f"  Mean channel err:   {np.mean([m['per_channel_error_pct'] for m in rows]):.1f}%")
    print(f"  Scenarios R² > 0.5: {sum(1 for m in rows if m['channel_r2_mean'] > 0.5)}/{n}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--mask-frac", type=float, default=0.3)
    parser.add_argument("--random-eval", type=int, default=10)
    parser.add_argument("--skip-datagen", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--n-workers", type=int, default=4)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if not args.skip_training:
        if not args.skip_datagen:
            generate_data(args.n_train, n_workers=args.n_workers)
        engine = train(args)
    else:
        engine = load_engine()

    if args.random_eval > 0:
        evaluate_random(engine, args.random_eval)


if __name__ == "__main__":
    main()
