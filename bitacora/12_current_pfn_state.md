# Current PFN State (Active Architecture)

## Summary

The active neural decomposition architecture is `pfn_model` + `pfn_dataset` +
`pfn_engine` (channels-as-features in a single per-week token). The
`proper_pfn` channels-as-tokens experiment was attempted and abandoned after
verifying it could not even overfit a single scenario — the permutation
invariance over channels caused the model to collapse to predicting equal
contributions for all channels.

## Active modules

| File | Purpose |
|------|---------|
| `demantiq/neural/pfn_model.py` | Per-week-token transformer + `build_pfn_input` |
| `demantiq/neural/pfn_dataset.py` | Full-scenario dataset with share targets and BERT masking |
| `demantiq/neural/pfn_engine.py` | Training loop, weighted held-out loss, inference |
| `demantiq/neural/metrics.py` | Per-component R², rank correlation, per-channel error |
| `scripts/test_pfn_overfit_v2.py` | Overfit verification (single scenario or `--multi`) |
| `scripts/train_pfn.py` | Multi-scenario training + evaluation |
| `scripts/vast_setup.sh` | One-step vast.ai instance setup |

## Key design decisions (current state)

- **Channels as features** (positions in input vector), not as set tokens.
  Permutation-invariant tokens were tried (`proper_pfn`) and could not
  overfit — they collapse to averaging.
- **Targets are shares** (component / y[t]) rather than normalized
  contributions. Hard reconstruction constraint that shares sum to ~1.
- **Per-channel input includes spend, impressions, and clicks.**
  Impressions/clicks are synthesized by the simulator from per-channel
  CPM and CTR values (added to `ChannelConfig`).
- **BERT-style masking with weighted loss** (5x on masked weeks, 1x on
  visible). Pure held-out loss caused train/inference distribution
  shift; weighted loss preserves both.
- **Global normalization for per-channel features** (max across all
  channels). Per-channel normalization loses inter-channel scale info
  and degrades performance.

## Overfit verification (current architecture, 2000 epochs, 6 seeds)

| Seed | Baseline % | Rank | Cat err | Per-channel err |
|------|-----------|------|---------|-----------------|
| 500 | 49% | 1.000 | 0.1pp | 2.4% |
| 300 | 72% | 1.000 | 0.3pp | 2.1% |
| 200 | 93% | 1.000 | 0.2pp | 10% |
| 100 | 90% | 1.000 | 0.1pp | 35% |
| 400 | 87% | 1.000 | 0.3pp | 56% |
| 42  | 105% | 1.000 | 0.7pp | 266% |

**Pattern:** lower baseline → better channel magnitudes. Rank correlation
and category error are perfect across all 6. Per-channel magnitude error
correlates with baseline dominance — when baseline >85%, channels are tiny
and small absolute baseline errors blow up channel relative errors.

## What this does and does not prove

**Proves:**
- Architecture can learn per-scenario decomposition.
- Channel ranking is reliable across diverse scenarios.
- Baseline accuracy is excellent regardless of scenario type.

**Does NOT prove:**
- Multi-scenario generalization. The cross-scenario test (training on
  50K scenarios and evaluating on unseen) is the next experiment.
- Behavior on real client data outside the simulator's distribution.

## Next experiment (Task 14, vast.ai)

```bash
git push origin feature/implement-v1
# On vast.ai:
cd /workspace/josiah && bash scripts/vast_setup.sh
python scripts/train_pfn.py --n-train 50000 --n-epochs 100 --random-eval 10 --batch-size 32 --patience 30
```

Target metrics on 10 unseen scenarios:
- Mean rank corr > 0.5 = progress; > 0.7 = breakthrough
- Mean category error < 15pp = progress; < 5pp = breakthrough
- Mean per-channel error < 40% = progress; < 20% = breakthrough

Previous runs on this benchmark (older architecture) topped out at
mean rank corr ≈ 0.34 with mean per-channel error ≈ 87%. The added
impressions/clicks signal and proven overfit capacity should push past
this if cross-scenario generalization is feasible.
