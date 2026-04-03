# Demantiq Neural Engine — Experiment Log

*Last updated: 2026-04-03*

---

## Summary of All Approaches

| # | Approach | Category Accuracy | Channel Correlation | Status | Why Abandoned |
|---|---------|-------------------|-------------------|--------|---------------|
| 1 | Monolithic SBI (SNPE-C + NSF) | N/A | N/A | Abandoned | ROAS 0.5-22% (good) but betas 65-81% error, constant params only, NSF can't scale past ~30 dims |
| 2 | Compositional SBI (2 NSFs) | N/A | N/A | Abandoned | ROAS regressed to 56-87%, frozen embedding killed end-to-end gradients |
| 3 | Encoder-decoder (per-timestep MLP, simple MSE) | **4-8pp** | -0.50 (random) | Best category result | Channels all predict ~14.5K (average). CNN features identical (cosine 0.98). Only tested on ~60% baseline scenarios |
| 4 | + Composite loss (learnable sigmas) | 34pp (collapsed) | -0.50 | Abandoned | Self-balancing let model collapse — pushed everything to baseline, media at 3.7% vs true 38% |
| 5 | + Composite loss (running-mean normalization) | 8.8pp | -0.50 | Abandoned | Channels still ~14.5K. Running-mean didn't collapse but didn't improve channels |
| 6 | + Per-channel decoder (separate from projection) | 4-6pp | -0.50 | Abandoned | Projection layer (529→256) mixed channel positions. Channels still identical |
| 7 | + Raw spend feature (1 dim added) | 5-7pp | -0.14 | Abandoned | 1 dim in 210 total — decoder barely used it (zeroing changed output by 0.001) |
| 8 | NAM with y input (channel CNN sees spend+y+context) | 25pp | -0.15 | Abandoned | y is polluted by all channels. Spend↔y correlation doesn't predict share (twitter -0.42 corr = 2% share, direct_mail +0.42 corr = 2% share) |
| 9 | NAM without y (additive, predict absolute contributions) | 39pp | -0.06 | Abandoned | Without y, CNN can't infer beta. Same spend → different contributions across scenarios because beta varies 20-800 |

---

## Key Diagnostics & Findings

### CNN features are identical across channels (Approach 3-7)
- **Test**: Cosine similarity of CNN temporal features across 5 channels
- **Result**: 0.98+ for all pairs
- **Cause**: Shared CNN learns temporal SHAPE (pulsed vs steady) but not MAGNITUDE. 10x spend difference → only 1.5x feature norm difference
- **Impact**: All channels produce same decoder input → same output

### Dying ReLU (Approach 3-5)
- **Test**: Layer-by-layer CNN output on realistic_brand spend data
- **Result**: Last ReLU killed ALL outputs to zero. Conv layer output was entirely negative (mean=-3947)
- **Fix**: LeakyReLU + spend normalization (divide by global max)
- **Status**: Fixed, CNN produces nonzero output after fix

### Projection layer destroys channel identity (Approach 6)
- **Test**: Channel features before and after 529→256 projection
- **Result**: Before projection: channels well-differentiated (cosine -0.46 to 0.60). After projection + decoder: all predict ~0.06 share
- **Cause**: Different channels appear at different positions across training scenarios. Projection learns "position 0 = average channel"

### Learnable loss weights are gameable (Approach 4)
- **Test**: Trained with Kendall et al. uncertainty weighting (4 learnable sigma params)
- **Result**: Model grew σ_category to dampen category loss, collapsed media to 3.7%
- **Fix**: Running-mean normalization (non-learnable)
- **Lesson**: Learnable loss parameters can find degenerate solutions

### Spend↔y correlation doesn't predict channel share (Approach 8)
- **Test**: Computed Pearson correlation between each channel's spend and y across scenarios
- **Result**: twitter corr=-0.42 → share=0.02. direct_mail corr=+0.42 → share=0.02. youtube corr=+0.69 → share=0.04. podcast corr=+0.61 → share=0.15
- **Cause**: y contains ALL channels + baseline + noise. Correlation is dominated by confounders
- **Impact**: Per-channel NAM that sees [spend_i, y] gets contradictory training signals

### Toy test: CNN CAN learn channel differentiation (Approach 8)
- **Test 1**: 1 channel, simple data → CNN learns correlation 0.87 in 50 epochs
- **Test 2**: 5 channels, simple data → CNN learns correlation 0.55 in 80 epochs, still climbing
- **Test 3**: 5 channels, real (nonlinear) data → CNN fails (correlation 0.0)
- **Cause**: Saturation + adstock make the spend→contribution relationship nonlinear. Combined with multi-channel confounding, the signal is too weak for the CNN

### Training data diversity matters
- **Before**: organic_level 200-3000 with uniform sampling → 78% of scenarios have baseline 60-80%
- **After**: Regime system (media_dominant / balanced / baseline_dominant) → baseline ranges 7-102%
- **Impact**: Models trained on diverse data don't just learn "predict 65% baseline"

### Early stopping kills learning (Approach 8-9)
- **Observation**: Model early-stops at epoch 16 (patience=15, best at epoch 1)
- **Cause**: Val loss spikes after epoch 1 — model overfits training patterns quickly
- **Impact**: Never reaches the point where channel differentiation would emerge (toy test took 40+ epochs)

---

## The Fundamental Problem

**Across 50K training scenarios, the same channel type with the same spend pattern can have wildly different contributions because beta varies 20-800.** The neural network can only see spend (and optionally y), but contribution = beta × saturation(adstock(spend)). Without knowing beta, it predicts the average contribution — useless.

Analogy: "Someone drove for 2 hours. How far did they go?" Without knowing speed (beta), you can only guess the average distance.

**OLS solves this per-scenario** by fitting regression coefficients to one business's data. But OLS can't capture saturation, adstock, or nonlinear interactions — it's too simple for production MMM.

---

## What Works (proven)

1. **Category-level decomposition**: Encoder-decoder with simple MSE gets baseline vs media at 4-8pp error on ~60% baseline scenarios
2. **Weekly stability**: 100% of weeks above R² 0.8 (dominated by baseline)
3. **Training infrastructure**: Simulator generates clean ground truth, data pipeline works, evaluation framework catches problems
4. **Vast.ai GPU training**: 5x speedup at $0.04/hr, workflow proven

## What Doesn't Work (proven)

1. **Cross-scenario channel differentiation**: No architecture has achieved channel correlation > 0.3 on diverse evaluation scenarios
2. **Shared CNN for channels**: Produces identical features regardless of channel identity
3. **Single-channel NAM**: Can't isolate one channel's effect from confounded y
4. **Predicting shares vs absolute**: Neither format solves the beta-unknown problem

---

## Untested Ideas

### Two-stage decomposition
1. Stage 1: Neural network predicts category split (baseline / media / other) — proven to work
2. Stage 2: Neural network splits total_media across channels — easier target (channels are 100% of signal, no baseline noise)
- **Key question**: Does Stage 1 generalize across diverse baseline levels (12-102%)? Only tested at ~60%.

### Single multivariate CNN (all inputs → all outputs)
- One CNN takes ALL spends + context + y simultaneously
- Outputs ALL contributions at once
- Each CNN filter naturally captures cross-variable patterns
- Never tried — all previous approaches used separate per-channel encoding

### Meta-learning / per-scenario fine-tuning
- Pre-train on 50K scenarios (learns general patterns)
- At inference, fine-tune on the specific scenario's data (learns beta)
- Based on MAML (Finn et al. 2017)
- Most principled solution but most complex to implement

### Google NNN-style architecture
- Per-channel sub-networks with SEPARATE learned weights
- Attribution via integrated gradients (post-hoc, not structural)
- Requires enough data per channel type to learn type-specific transformations
