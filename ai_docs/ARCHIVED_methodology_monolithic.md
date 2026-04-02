# Demantiq: Monolithic Inference Approach

**End-to-end embedding + single NSF density estimator**

*Entropy — March 2026*

---

## Why This Approach

The monolithic approach trains the embedding network and NSF density estimator **end-to-end** — gradients flow from the NSF loss through the embedding, teaching both components simultaneously what features matter.

This produced our best results on aggregate metrics:
- ROAS: 0.5-22% error (production quality)
- Price elasticity: 17.6% error
- Media contribution: +10% bias

The compositional approach (separate global + per-channel NSFs with pre-trained frozen embedding) achieved channel differentiation but lost the end-to-end gradient that made ROAS accurate.

## Architecture

```
Observable Data (y, spend, price, distribution, competition, macro)
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              EMBEDDING NETWORK (trained end-to-end)          │
│                                                              │
│  Per-channel Dilated CNN → Channel-type embeddings           │
│       → Set Transformer (cross-channel attention)            │
│  Outcome Dilated CNN → Context Dilated CNN                   │
│       → Global MLP → 256-dim summary                         │
│                                                              │
│  ALL trained jointly with the NSF — the embedding learns     │
│  what features make the NSF's job easier                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│            SINGLE NSF (27-dim parameter space)               │
│                                                              │
│  Global (2):                                                 │
│    media_contribution_pct, price_elasticity                  │
│                                                              │
│  Per channel × 5 (25):                                       │
│    beta, ROAS, contribution_fraction,                        │
│    price×media interaction, distribution×media interaction   │
│                                                              │
│  5 spline transforms, 128 hidden features                    │
│  All 27 parameters inferred jointly                          │
└─────────────────────────────────────────────────────────────┘
```

## Key Difference vs Compositional

| Aspect | Monolithic | Compositional |
|--------|-----------|---------------|
| Embedding training | End-to-end with NSF | Pre-trained, then frozen |
| NSF count | 1 (27-dim) | 2 (2-dim global + 5-dim per-channel) |
| Data per NSF dim | ~1850 (50K / 27) | ~50,000 (250K / 5) |
| Training time | ~5-10 hours | ~2 hours |
| ROAS accuracy | **0.5-22%** (best) | 56-87% (regression) |
| Channel differentiation | Via end-to-end gradient | Via pre-trained per-channel objective |
| Interaction recovery | Unknown (first test with this approach) | 38-82% (detected but noisy) |

## What's Different From Earlier Monolithic Runs

This run includes fixes that earlier monolithic runs didn't have:

1. **Interaction decomposition**: 27-dim theta includes price×media and dist×media coefficients per channel. Earlier runs had 17-dim theta without interactions, causing the 1.7x beta bias.

2. **ROAS normalization fix**: `roas_scale` now computed from actual data max (~0.45) instead of defaulting to 1.0. Earlier runs compressed all ROAS values into bottom 5% of the prior range.

3. **Prior bounds for elasticity**: `[-1.1, 0.1]` for normalized elasticity. Earlier runs clipped negative values.

4. **Differentiated evaluation scenario** (`realistic_brand`): 5 channels with different betas (120-350), different spend levels (3K-30K), different saturation (hill vs logistic), different adstock (geometric vs weibull). Earlier runs used `interaction_heavy` where all channels were identical.

5. **Training data with rich context**: 100% pricing, ~70% distribution, ~39% full interactions. Sampler bug fixed — per-batch samplers now inherit `rich_context` and `n_fixed_channels`.

## Expected Results

Based on what each fix addresses:

| Metric | Previous monolithic (no interaction decomp) | Expected with all fixes |
|--------|----------------------------------------------|------------------------|
| ROAS | 0.5-22% | Similar — the end-to-end gradient that produced this is preserved |
| Elasticity | 17.6% | Similar ~15-20% |
| Betas | 65-81% (1.7x interaction bias) | **20-40%** — interaction coefficients absorb the multiplicative effect |
| Interactions | Not inferred | **Unknown** — first test, no baseline to compare against |
| Media contribution | +10% | Similar |

## What Could Go Wrong

1. **27 dims may be too many**: The NSF handled 17 dims well. Adding 10 interaction dims increases the parameter space 60%. At 50K samples, that's ~1850 samples per dim — borderline for NSF.

2. **Interaction identifiability**: Even with the embedding seeing promo and distribution time series, separating `beta × (1 + interaction × modifier)` into `beta` and `interaction` requires the NSF to learn a multiplicative decomposition. This is harder than additive separation.

3. **Training time**: End-to-end training with 50K samples through the full embedding is slow (~5-10 hours on CPU). No way to speed this up without GPU.

## Training Command

```bash
rm -rf neural_output/trained_model/
python scripts/train_neural.py --n-train 50000 --n-epochs 100 --fixed-channels 5 --scenario realistic_brand
```

## Relationship to Compositional Approach

These are two strategies for the same underlying architecture:

- **Monolithic**: simpler, end-to-end gradient, proven ROAS accuracy, but slower training and higher-dimensional NSF
- **Compositional**: faster training, lower-dimensional per-channel NSF, but requires the embedding to be good WITHOUT end-to-end gradient from the NSF

If the monolithic approach works with the interaction decomposition, it becomes the production path. If betas are still bad, the compositional approach needs a better embedding strategy (e.g., end-to-end per-channel training, or a contrastive learning objective that forces channel differentiation).

Both approaches are documented and the code supports both via the `--compositional` flag.
