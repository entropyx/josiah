"""Overfit test for pfn v2 (channels-as-features + impressions/clicks).

Trains and evaluates on THE SAME scenario by loading from DemantiqDataset
after pipeline.generate() writes the .npz. Avoids the bug where
pipeline-generated scenarios differ from sampler.sample() output.

Supports --multi to test across several seeds.

Usage:
  python scripts/test_pfn_overfit_v2.py --n-epochs 2000
  python scripts/test_pfn_overfit_v2.py --n-epochs 2000 --multi
"""

import argparse
import logging
import tempfile
import numpy as np

from demantiq.scenarios.scenario_sampler import ScenarioSampler
from demantiq.orchestration.training_pipeline import TrainingPipeline
from demantiq.orchestration.training_format import (
    DECOMP_IDX_BASELINE, DECOMP_IDX_CHANNELS_START,
)
from demantiq.neural.data_loader import DemantiqDataset
from demantiq.neural.pfn_engine import PFNEngine, PFNConfig
from demantiq.neural.metrics import scenario_level_metrics

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)


def run_one(seed: int, args) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        sampler = ScenarioSampler(seed=seed, rich_context=True, channel_range=(3, 5))
        pipeline = TrainingPipeline(sampler, output_dir=tmpdir, batch_size=1)
        pipeline.generate(n_total=1, n_workers=1, seed=seed)

        config = PFNConfig(
            n_epochs=args.n_epochs, n_train=1, batch_size=args.batch_size,
            learning_rate=args.lr, d_model=args.d_model, n_layers=args.n_layers,
            week_mask_fraction=args.mask_frac, patience=args.n_epochs,
            val_fraction=0.0, data_dir=tmpdir,
        )
        engine = PFNEngine(config)
        engine.train()

        backing = DemantiqDataset(tmpdir)
        ch_names = backing.channel_names[0]
        n_ch = len(ch_names)
        T = int(backing.n_periods[0])
        y = backing.y[0, :T].astype(np.float32)
        spend = backing.spend[0, :T, :n_ch].astype(np.float32)
        imp = backing.impressions[0, :T, :n_ch].astype(np.float32)
        clk = backing.clicks[0, :T, :n_ch].astype(np.float32)
        ctx = backing.context[0, :T, :].astype(np.float32)
        decomp = backing.decomposition[0, :T, :]

        out = engine.infer(y, spend, ctx, n_ch, impressions=imp, clicks=clk)

        true_ch = np.column_stack(
            [decomp[:, DECOMP_IDX_CHANNELS_START + i] for i in range(n_ch)]
        )
        true_base = decomp[:, DECOMP_IDX_BASELINE]

        metrics = scenario_level_metrics(
            out["channel_contributions"], out["baseline"],
            true_ch, true_base, y,
        )

        # Characterize scenario diversity
        true_totals = true_ch.sum(axis=0)
        pred_totals = out["channel_contributions"].sum(axis=0)
        metrics["seed"] = seed
        metrics["n_channels"] = n_ch
        metrics["n_periods"] = T
        metrics["channel_names"] = ch_names
        metrics["true_base_pct"] = float(true_base.sum() / y.sum() * 100)
        metrics["channel_spread"] = float(true_totals.max() / max(true_totals.min(), 1e-6))
        metrics["baseline_cv"] = float(true_base.std() / max(true_base.mean(), 1e-6))
        metrics["true_channel_totals"] = true_totals.tolist()
        metrics["pred_channel_totals"] = pred_totals.tolist()
        metrics["true_baseline_total"] = float(true_base.sum())
        metrics["pred_baseline_total"] = float(out["baseline"].sum())
        return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-epochs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--mask-frac", type=float, default=0.0)
    parser.add_argument("--multi", action="store_true",
                        help="Test on multiple seeds (42, 100, 200, 300, 400, 500)")
    args = parser.parse_args()

    seeds = [42, 100, 200, 300, 400, 500] if args.multi else [args.seed]

    rows = []
    for seed in seeds:
        print(f"\n>>> Running overfit on seed={seed}...")
        m = run_one(seed, args)
        rows.append(m)
        print(
            f"  seed={seed}  "
            f"ch={m['n_channels']}  T={m['n_periods']}  "
            f"base={m['true_base_pct']:.0f}%  "
            f"spread={m['channel_spread']:.1f}x  "
            f"bl_cv={m['baseline_cv']:.3f}"
        )
        print(
            f"  ➜ rank={m['rank_corr']:.3f}  "
            f"ch_r2={m['channel_r2_mean']:.3f}  "
            f"cat_err={m['category_error_pp']:.1f}pp  "
            f"ch_err={m['per_channel_error_pct']:.1f}%"
        )
        # Per-channel breakdown
        print(f"  {'channel':<14} {'true_total':>11} {'pred_total':>11} {'err%':>7} {'true_avg':>9} {'pred_avg':>9}")
        for name, true_tot, pred_tot in zip(
            m['channel_names'], m['true_channel_totals'], m['pred_channel_totals']
        ):
            err = abs(pred_tot - true_tot) / max(abs(true_tot), 1e-6) * 100
            true_avg = true_tot / m['n_periods']
            pred_avg = pred_tot / m['n_periods']
            print(f"  {name:<14} {true_tot:>11.0f} {pred_tot:>11.0f} {err:>6.1f}% {true_avg:>9.1f} {pred_avg:>9.1f}")
        tb = m['true_baseline_total']
        pb = m['pred_baseline_total']
        be = abs(pb - tb) / max(abs(tb), 1e-6) * 100
        print(f"  {'BASELINE':<14} {tb:>11.0f} {pb:>11.0f} {be:>6.1f}%")

    if args.multi:
        print(f"\n{'='*80}")
        print(f"  OVERFIT SUMMARY ({len(rows)} scenarios)")
        print(f"{'='*80}")
        print(f"  Mean rank corr:      {np.mean([m['rank_corr'] for m in rows]):.3f}")
        print(f"  Mean channel R²:     {np.mean([m['channel_r2_mean'] for m in rows]):.3f}")
        print(f"  Mean category err:   {np.mean([m['category_error_pp'] for m in rows]):.1f}pp")
        print(f"  Mean per-channel:    {np.mean([m['per_channel_error_pct'] for m in rows]):.1f}%")
        passed = sum(
            1 for m in rows
            if m['rank_corr'] > 0.9 and m['channel_r2_mean'] > 0.8
            and m['category_error_pp'] < 5.0 and m['per_channel_error_pct'] < 20.0
        )
        print(f"  Overfit passed:      {passed}/{len(rows)}")


if __name__ == "__main__":
    main()
