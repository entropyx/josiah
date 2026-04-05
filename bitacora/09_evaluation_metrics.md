# Evaluation Metrics Reference

## Category Breakdown

```
--- Category Breakdown ---
Category             True %     Pred %      Error
--------------------------------------------------
Baseline              62.1%      66.3%       4.2pp
Media                 37.7%      30.1%       7.5pp
Other                  0.3%      -0.4%       0.7pp
Noise                 -0.0%       0.1%       0.1pp
```

- **What it measures**: Overall decomposition accuracy — is the model getting the big picture right?
- **Good result**: <5pp error on baseline and media
- **Current best**: 2-9pp across diverse scenarios (Approach 3)

## Channel-Only Metrics

```
--- Channel-Only Metrics (excludes baseline) ---
  Weekly channel rank correlation:     0.281
  % weeks channel corr > 0.5:          12%
  Overall channel total correlation:   0.200
  True ranking:  google > email > facebook > youtube > tiktok
  Pred ranking:  facebook > youtube > google > email > tiktok
```

- **Why "channel-only"**: The overall weekly R² and rank correlation are inflated by baseline dominance. A model that gets baseline right but channels wrong still shows high overall R² (0.96). Channel-only metrics exclude baseline to catch channel collapse.
- **Good result**: Channel correlation > 0.5, correct ranking order
- **Current best**: 0.08 (random) — no approach has solved this with neural networks

## Weekly Accuracy

```
--- Weekly Accuracy ---
  Mean weekly R² (across components): 0.9540
  Median weekly R²:                   0.9610
  Worst week R²:         0.8265  (week 73)
  Best week R²:          0.9867  (week 35)
  % weeks with R² > 0.8:              100%
```

- **What it measures**: Per-week, how close is the predicted share vector to the true share vector across all components?
- **Caveat**: Inflated by baseline dominance (baseline is ~62%, so getting baseline right = high R²)
- **Good result**: 100% weeks > 0.8 (achieved)

## Per-Channel Contribution Errors

```
Channel            True Contrib   Pred Contrib    Error %
----------------------------------------------------------
facebook                  12330          12316       0.1%
google                    29430          11890      59.6%
```

- **What it measures**: Absolute contribution accuracy per channel (total across all weeks)
- **Good result**: <20% per channel
- **Current**: All channels predict ~12K (the average)

## Random Eval Summary

```
python scripts/train_neural.py --decomposition --skip-training --fixed-channels 5 --random-eval 10
```

Evaluates on 10 diverse random scenarios (seed=999, reproducible). Shows:
- Per-scenario: True Base%, Pred Base%, True Media%, Pred Media%, Cat Err, Ch Corr, Best/Worst Ch%
- Summary: Mean category error, mean channel corr, % scenarios with corr > 0.5

**This is the most important evaluation** — it tests generalization across diverse business types.

## Metric Interpretations

| Metric | Good | OK | Bad |
|--------|------|------|-----|
| Category error | <5pp | 5-15pp | >15pp |
| Channel rank corr | >0.8 | 0.3-0.8 | <0.3 |
| Per-channel error | <20% | 20-50% | >50% |
| Weekly R² | >0.9 | 0.7-0.9 | <0.7 |
| Reconstruction R² | >0.9 | 0.5-0.9 | <0.5 |

## Known Metric Pitfalls

1. **Weekly R² inflated by baseline**: A model predicting constant 65% baseline + equal channels gets R² ~0.83 because baseline variance dominates.
2. **Reconstruction R² trivially 1.0 with shares**: If shares sum to ~1.0, then sum(shares × y) ≈ y regardless of decomposition quality.
3. **Facebook 0.1% error is coincidence**: When all channels predict ~12K and Facebook's true value happens to be ~12K, the error looks good but it's just luck.
