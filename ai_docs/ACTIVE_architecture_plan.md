# Demantiq: Quality Architecture for Time-Varying Demand Inference

## Context

After 10+ iterations, we've identified the fundamental limitation: the neural engine infers **constant parameters** (one beta per channel for the entire period) from **time-varying data** (104 weeks of changing spend, promotions, distribution). This causes:

1. **Beta bias**: The engine averages over time-varying effectiveness, getting the wrong answer for every sub-period
2. **Interaction weakness**: Temporal co-occurrence (ads during promos) is compressed away by AdaptiveAvgPool
3. **Scale mismatch**: The actual-vs-predicted R² is -17 because constant parameters can't reproduce time-varying demand patterns

Meanwhile, the simulator ALREADY generates per-period ground truth (true_facebook_contribution for every week, true_price_effect for every week, etc.) but we throw it all away — the data loader never loads it, the training pipeline never saves it.

**The real question**: Do we want a model that says "Facebook beta = 180 for the whole year" or one that says "Facebook was effective in Q1-Q2 (beta ~220) but declined in Q3-Q4 (beta ~140) after the agency switch, and ads worked 25% better during the monthly promos"?

---

## What We're Building

### The shift: from constant parameters to per-period decomposition

```
BEFORE (SBI + Neural Spline Flow):
  Data → Embedding → NSF → "beta=180, ROAS=0.012" (constants for entire period)

AFTER (Supervised encoder-decoder):
  Data → Embedding → Temporal Decoder → Week-by-week decomposition:
    Week 1: baseline=62%, facebook=12%, google=15%, price=8%, noise=3%
    Week 2: baseline=58%, facebook=14%, google=16%, price=9%, noise=3%
    ...
    Week 104: baseline=55%, facebook=18%, google=13%, price=11%, noise=3%
```

**Still a neural network. Still trained on 50K simulator datasets. Still instant inference (<1 sec).** Just predicting the right thing now — weekly demand decomposition instead of constant parameters.

**Output: fractional shares per period.** The decoder outputs what fraction of demand each driver is responsible for, per week. Shares sum to ~1.0 (with noise as the residual). This is scale-invariant — the network learns proportions that transfer across businesses regardless of their revenue scale.

To get absolute numbers for client delivery: `facebook_contribution(week_3) = share(0.18) × observed_sales(1,243) = 223.7 units`. This is standard practice in all MMM tools (PyMC, Meridian, Robyn).

---

## What Needs to Change

### Problem 1: We discard per-period ground truth

The simulator generates `ground_truth` DataFrame with per-period:
- `true_baseline` (per week)
- `true_{channel}_contribution` (per week, per channel)
- `true_price_effect` (per week)
- `true_distribution_cap` (per week)
- `true_competition_effect`, `true_macro_effect`, `true_regime_effects`
- `true_interaction_{type}` (per week)
- `true_noise`

But `training_pipeline.py` only saves `summary_to_ext_vector()` — constant aggregates. The per-period decomposition is generated and thrown away.

**Fix**: Save the per-period ground truth decomposition in the .npz training batches. This is the true signal for time-varying inference.

### Problem 2: AdaptiveAvgPool kills temporal structure

The DilatedConv1D encoder produces rich temporal features (104 timesteps × 64 filters), then AdaptiveAvgPool collapses them into 64 numbers — losing WHEN things happened. The NSF only sees "Facebook had pulsed campaigns" not "Facebook pulsed in Q1 and Q4 but went dark in Q2-Q3."

**Fix**: Replace AdaptiveAvgPool with a temporal output that preserves per-period information — keep the full (T × 64) feature map for the decoder.

### Problem 3: The simulator assumes constant betas

The simulator generates one beta per channel, constant across all periods. Real betas change (agency switches, creative refreshes, platform algorithm changes, competitive shifts).

**Fix (Phase B)**: Extend the simulator to support time-varying parameters:
- Random walk: `beta(t) = beta(t-1) + drift + noise`
- Regime-switching: `beta = beta_1` for weeks 1-52, `beta = beta_2` for weeks 53-104
- Seasonal variation: `beta(t) = beta_base × (1 + seasonal_modifier(t))`

---

## Architecture

```
Observable Data (y, spend, context) — 104 weeks × ~30 columns
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│              EMBEDDING NETWORK (modified)                     │
│                                                              │
│  Per-channel Dilated CNN → Channel-type embeddings           │
│       → Set Transformer (cross-channel attention)            │
│  Outcome Dilated CNN → Context Dilated CNN                   │
│                                                              │
│  KEY CHANGE: No AdaptiveAvgPool                              │
│  Output: per-period embeddings (104 × 256)                   │
│  instead of single summary (256)                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              TEMPORAL DECODER (NEW)                           │
│                                                              │
│  Input: per-period embeddings (104 × 256)                    │
│                                                              │
│  For each week t:                                            │
│    embedding(t) → MLP → softmax → component shares:          │
│      [baseline_share, fb_share, google_share, tiktok_share,  │
│       email_share, youtube_share, price_share,               │
│       distribution_share, noise_share]                       │
│                                                              │
│  Output: (104 × N_components) fractional shares              │
│  Shares sum to ~1.0 per week                                 │
│                                                              │
│  Loss: MSE(predicted_shares, true_shares)                    │
│  Training: standard PyTorch, no sbi dependency               │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   OUTPUT                                     │
│                                                              │
│  Per week: fractional shares of each demand driver           │
│  Multiply by observed y(t) → absolute contributions          │
│                                                              │
│  Week 1: fb=223, google=186, price=99, baseline=770, ...     │
│  Week 2: fb=193, google=222, price=125, baseline=805, ...    │
│  ...                                                         │
│  Week 104: fb=262, google=189, price=145, baseline=798, ...  │
│                                                              │
│  Sum across weeks → total channel contributions, ROAS        │
│  Average shares → approximate "betas" (time-varying!)        │
└─────────────────────────────────────────────────────────────┘
```

### Why regression instead of SBI (Neural Spline Flows)

| Aspect | SBI + NSF (what we had) | Supervised regression (what we're building) |
|--------|------------------------|---------------------------------------------|
| Output dimensionality | 27 params (struggled above 30) | 104 × 10 = 1040 values (no problem for regression) |
| What it predicts | Constant parameters (beta=180 forever) | Per-period decomposition (shares per week) |
| Ground truth | Summary aggregates | Per-period DataFrame (already generated, was thrown away) |
| Training | sbi library, SNPE-C algorithm | Standard PyTorch training loop |
| Uncertainty | Full posterior distribution | Ensemble or MC dropout (future) |
| Training time | 5-10 hours | ~1-2 hours |
| Inference time | < 1 second | < 1 second |

---

## Three-Phase Plan

### Phase A: Per-Period Decomposition (the right MVP)

**Step 1: Save per-period ground truth** (1 day)

Files to modify:
- `demantiq/orchestration/training_pipeline.py` — save `ground_truth` DataFrame to .npz
- `demantiq/orchestration/training_format.py` — add format for per-period decomposition
- `demantiq/neural/data_loader.py` — load per-period ground truth

**Step 2: Build temporal decoder** (1-2 weeks)

Files to create/modify:
- `demantiq/neural/temporal_decoder.py` — NEW: sequence decoder outputting per-period shares
- `demantiq/neural/encoders.py` — modify to preserve temporal dimension
- `demantiq/neural/decomposition_engine.py` — NEW: training loop (standard PyTorch, no sbi)
- `scripts/train_neural.py` — add `--decomposition` flag

**Step 3: Time series evaluation** (1 day)

- Actual vs predicted y(t) — overlay plot with R²
- Monthly decomposition — stacked area chart (true vs inferred side by side)
- Per-channel contribution over time — line plots

### Phase B: Time-Varying Simulator (1 week)

- Extend ChannelConfig with drift/volatility/regime parameters
- Extend ScenarioSampler to generate time-varying configs
- Retrain decoder on time-varying data

### Phase C: Neural Superstatistics (1-3 months, research-grade)

Based on Radev et al. (2023):
- Per-timestep amortized inference
- Transition model learned from data
- Would make Demantiq the only fast MMM engine with time-varying parameters

---

## What We Keep vs What Changes

### Keep (everything we've built):
- The simulator (demand kernel, 15-step pipeline)
- The embedding network architecture (DilatedConv1D + Set Transformer + context encoder)
- The training pipeline (50K synthetic datasets, .npz batches)
- The scenario library (including `realistic_brand`)
- The evaluation framework (diagnostics, CSV export, plots)
- All the training data on disk (50K batches)

### Change:
- **Drop NSF / sbi** — wrong tool for high-dimensional per-period output
- **Add temporal decoder** — predicts weekly decomposition shares
- **Modify embedding** — remove AdaptiveAvgPool, keep temporal features
- **Save per-period ground truth** — use what the simulator already computes
- **Training becomes standard PyTorch** — simpler, faster, more debuggable

---

## Expected Impact

| Metric | Current (constant params) | Phase A (per-period decomposition) |
|--------|--------------------------|-----------------------------------|
| Actual vs predicted R² | -17 (terrible) | > 0.8 (good fit) |
| Channel contribution accuracy | ~40-70% aggregate MAPE | Per-period MSE, visible in time plots |
| Interaction detection | 38-82% MAPE | Implicit — interactions show up as higher contributions during promo weeks |
| Time-varying effects | Not captured | Naturally captured (different contribution each week) |
| Business deliverable | Parameter tables | Weekly decomposition charts (CFO-ready) |
| Training time | 5-10 hours (SBI) | ~1-2 hours (supervised regression) |
| Inference time | < 1 second | < 1 second |

---

## Verification

```bash
# Step 1: After saving per-period ground truth
python -c "
from demantiq.neural.data_loader import DemantiqDataset
ds = DemantiqDataset('training_data')
item = ds[0]
print('Has decomposition:', 'decomposition' in item)
print('Shape:', item['decomposition'].shape)  # (104, N_components)
"

# Step 2: After building decomposition decoder
python scripts/train_neural.py --n-train 50000 --decomposition --scenario realistic_brand

# Step 3: Check time series plots
ls neural_output/plots/realistic_brand_actual_vs_predicted.png
ls neural_output/plots/realistic_brand_monthly_decomposition.png

# Expect R² > 0.8 on actual vs predicted
```

---

## References

1. Radev et al. (2023). Neural Superstatistics. *Scientific Reports*. — Per-timestep amortized SBI
2. Dew, Padilla, Shchetkina (2024). Your MMM is Broken. *arXiv:2408.07678*. — Identifiability of time-varying + nonlinear effects
3. Gloeckler et al. (2025). Compositional SBI for Time Series. *ICLR 2025*. — Markov factorization
4. PyMC Marketing — Time-varying media parameters via Hilbert Space GP
