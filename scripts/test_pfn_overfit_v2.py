"""Overfit test for pfn v2 (channels-as-features + impressions/clicks).

Trains and evaluates on THE SAME scenario by loading from DemantiqDataset
after pipeline.generate() writes the .npz. Avoids the bug where
pipeline-generated scenarios differ from sampler.sample() output.

Usage:
  python scripts/test_pfn_overfit_v2.py --n-epochs 200
"""

import argparse
import logging
import tempfile
import numpy as np

from demantiq.scenarios.scenario_sampler import ScenarioSampler
from demantiq.orchestration.training_pipeline import TrainingPipeline
from demantiq.orchestration.training_format import (
    DECOMP_IDX_BASELINE, DECOMP_IDX_CHANNELS_START,
    DECOMP_IDX_PRICE, DECOMP_IDX_COMPETITION, DECOMP_IDX_MACRO,
)
from demantiq.neural.data_loader import DemantiqDataset
from demantiq.neural.pfn_engine import PFNEngine, PFNConfig
from demantiq.neural.proper_pfn.metrics import scenario_level_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--mask-frac", type=float, default=0.3)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate one scenario via the pipeline (writes to npz)
        sampler = ScenarioSampler(seed=args.seed, rich_context=True, channel_range=(3, 5))
        pipeline = TrainingPipeline(sampler, output_dir=tmpdir, batch_size=1)
        pipeline.generate(n_total=1, n_workers=1, seed=args.seed)

        # Train
        config = PFNConfig(
            n_epochs=args.n_epochs, n_train=1, batch_size=args.batch_size,
            learning_rate=args.lr, d_model=args.d_model, n_layers=args.n_layers,
            week_mask_fraction=args.mask_frac, patience=args.n_epochs,
            val_fraction=0.0, data_dir=tmpdir,
        )
        engine = PFNEngine(config)
        engine.train()

        # Evaluate on the SAME scenario by loading from the dataset
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

        print(f"\n{'='*70}")
        print(f"  PFN V2 OVERFIT TEST — seed={args.seed}, {args.n_epochs} epochs")
        print(f"  {n_ch} channels ({', '.join(ch_names)}), {T} periods")
        print(f"{'='*70}")
        print(f"\n  Channel rank corr:    {metrics['rank_corr']:.3f}  (target: >0.9)")
        print(f"  Channel R² mean:      {metrics['channel_r2_mean']:.3f}  (target: >0.8)")
        print(f"  Baseline R²:          {metrics['baseline_r2']:.3f}  (target: >0.8)")
        print(f"  Category error:       {metrics['category_error_pp']:.1f}pp  (target: <5pp)")
        print(f"  Per-channel err:      {metrics['per_channel_error_pct']:.1f}%  (target: <20%)")
        print(f"  True base %: {metrics['true_baseline_pct']:.1f}  Pred: {metrics['pred_baseline_pct']:.1f}")

        passed = (
            metrics['rank_corr'] > 0.9
            and metrics['channel_r2_mean'] > 0.8
            and metrics['baseline_r2'] > 0.8
        )
        print(f"\n  {'PASSED' if passed else 'FAILED'}")


if __name__ == "__main__":
    main()
