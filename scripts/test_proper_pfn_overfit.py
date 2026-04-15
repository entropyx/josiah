"""Overfit test: train proper PFN on 1 scenario and evaluate on the SAME.

Target: channel rank corr > 0.9, channel r2 mean > 0.8, baseline r2 > 0.8.
If this fails, the architecture is broken.

Usage:
  source venv/bin/activate
  python scripts/test_proper_pfn_overfit.py --n-epochs 300 --seed 42
"""

import argparse
import logging
import tempfile
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        sampler = ScenarioSampler(seed=args.seed, rich_context=True, channel_range=(3, 5))
        pipeline = TrainingPipeline(sampler, output_dir=tmpdir, batch_size=1)
        pipeline.generate(n_total=1, n_workers=1, seed=args.seed)

        config = ProperPFNConfig(
            n_epochs=args.n_epochs,
            n_train=1,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            d_model=args.d_model,
            n_layers=args.n_layers,
            patience=args.n_epochs,  # no early stopping
            val_fraction=0.0,
            data_dir=tmpdir,
        )
        engine = ProperPFNEngine(config)
        engine.train()

        sampler2 = ScenarioSampler(seed=args.seed, rich_context=True, channel_range=(3, 5))
        eval_config = sampler2.sample(1)[0]
        result = simulate(eval_config)
        ch_names = [c.name for c in eval_config.channels]
        n_ch = len(ch_names)
        T = eval_config.n_periods
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

        print(f"\n{'='*70}")
        print(f"  OVERFIT TEST RESULTS — seed={args.seed}, {args.n_epochs} epochs")
        print(f"  {n_ch} channels ({', '.join(ch_names)}), {T} periods")
        print(f"{'='*70}")
        print(f"\n  Channel rank corr:    {metrics['rank_corr']:.3f}  (target: >0.9)")
        print(f"  Channel R² mean:      {metrics['channel_r2_mean']:.3f}  (target: >0.8)")
        print(f"  Baseline R²:          {metrics['baseline_r2']:.3f}  (target: >0.8)")
        print(f"  Category error:       {metrics['category_error_pp']:.1f}pp  (target: <5pp)")
        print(f"  Per-channel err:      {metrics['per_channel_error_pct']:.1f}%  (target: <20%)")

        passed = (
            metrics['rank_corr'] > 0.9
            and metrics['channel_r2_mean'] > 0.8
            and metrics['baseline_r2'] > 0.8
        )
        print(f"\n  {'PASSED' if passed else 'FAILED'}")


if __name__ == "__main__":
    main()
