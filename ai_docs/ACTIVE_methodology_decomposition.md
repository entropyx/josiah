# Demantiq: Per-Period Demand Decomposition

**Supervised encoder-decoder for weekly demand attribution**

*Entropy — April 2026*

---

## Plain English: What This Does and How

### The goal

You run ads on Facebook, Google, TikTok. You change prices. You expand distribution to new stores. Sales go up and down every week. The question: **how much of each week's sales came from each driver?**

That's what Demantiq does. Give it your weekly data (sales, ad spend, prices, etc.) and it tells you: "In week 23, Facebook drove 14% of your sales, Google drove 18%, your price cut drove 8%, and 55% was organic baseline." For every single week.

### The 4-step pipeline

**Step 1 — We build a fake business simulator.** We created a synthetic data generator that produces realistic business data where we know *exactly* how much each driver contributed each week — because we set the rules. Think of it as a video game economy: we control the physics. We control how effective Facebook is, how price-sensitive customers are, how distribution affects sales. And because we control it, we know the true answer.

**Step 2 — We generate 50,000 fake businesses.** Each one is different. Some have 3 channels, some have 10. Some are heavily driven by advertising, others are mostly organic. Some have aggressive pricing strategies, others don't. Some have strong competitors, others don't. This diversity is critical — the network needs to see every possible business shape to generalize.

**Step 3 — We train a neural network on all 50,000.** The network sees what a real analyst would see (sales, ad spend, prices, distribution) and learns to predict what's hidden (the true decomposition — how much of each week's sales came from each driver). After seeing 50,000 examples with their answers, it learns the pattern. Think of it like training a doctor: show them 50,000 X-rays with diagnoses, and they learn to diagnose new patients.

**Step 4 — We apply the trained network to real data.** Give it a client's real data (never seen before). In under 1 second, it outputs the weekly decomposition. No MCMC sampling (the slow Bayesian approach), no hours of fitting. Instant.

### Why a neural network? Why not statistics?

Traditional Marketing Mix Models (PyMC, Google Meridian, Meta Robyn) use Bayesian statistics. They take one business's data and try to figure out the parameters by running millions of simulations (MCMC sampling). This works, but:

- **It's slow** — takes 2-8 hours per client
- **It gives you constants** — "Facebook effectiveness = 180 for the whole year." But in reality, Facebook's effectiveness changes week to week (creative fatigue, algorithm changes, seasonality)
- **It doesn't scale** — every new client needs a full re-fit from scratch

Our approach is fundamentally different. We don't fit one client at a time. We train a neural network ONCE on 50,000 simulated businesses, and then it can instantly analyze any new business. It's like the difference between a calculator (solves one problem at a time) and a trained brain (recognizes patterns instantly).

### The neural network: two parts

The network has two connected pieces — an **encoder** that reads the data, and a **decoder** that makes the prediction.

#### Part 1: The Encoder ("the reader")

The encoder's job is to read 2 years of weekly data and build an understanding of what's happening each week. It needs to answer questions like: "Was there a spike in week 15? Was that spike because Facebook spend went up, or because there was a promotion, or because of seasonality?"

We use a specific type of neural network called a **Dilated Convolutional Network (Dilated CNN)**. Here's why:

Imagine you're reading a book. A regular CNN reads 3 words at a time — it sees local patterns ("the cat sat") but misses the big picture. A dilated CNN reads with expanding gaps:
- First pass: reads every word → catches daily/weekly patterns (this week's spike)
- Second pass: reads every 2nd word → catches monthly patterns (this month's trend)
- Third pass: reads every 4th word → catches quarterly patterns (this quarter's seasonality)

This gives the network **multi-scale context** in just 3 layers. When it sees week 23's data, it understands it in the context of what happened in the surrounding weeks, months, and quarters. This is critical for demand attribution — was this week's sales spike from a Facebook campaign, or from the seasonal ramp-up that happens every Q4?

The encoder runs this dilated CNN separately on each input signal:
- Facebook spend over time → "Facebook had a pulsed campaign with spikes in Q1 and Q3"
- Google spend over time → "Google had steady spend with a ramp-up in Q4"
- Sales (y) over time → "Sales peaked in December and dipped in February"
- Price over time → "There was a 20% promo in week 15 and week 40"
- Distribution over time → "Store count grew from 500 to 800 over the year"

#### Handling variable numbers of channels

One client has 3 channels (Facebook, Google, email). Another has 8. The encoder needs to handle any number without redesigning the network.

We solve this with a **Set Transformer** — a neural network component designed for variable-size inputs. Think of it like a meeting: you can have a meeting with 3 people or 8 people. The Set Transformer "listens" to all channels at once and figures out how they relate to each other. If Facebook and Google are both spending heavily in the same weeks, it notices the overlap. If email is doing something completely different from the paid channels, it notices that too.

Each channel also gets a **channel-type embedding** — a small learned fingerprint that tells the network "this is Facebook data" vs "this is Google data." The network learns that Facebook spend patterns behave differently from email patterns, even if the raw numbers are similar.

After processing everything, the encoder outputs a **256-number summary for each week**. These 256 numbers encode everything the network understands about that week — the spend patterns, the business context, the cross-channel dynamics. This is the encoder's "understanding" passed to the decoder.

#### Part 2: The Decoder ("the predictor")

The decoder takes the encoder's per-week understanding (256 numbers per week) and predicts what fraction of demand each component is responsible for.

It's a simple **feed-forward network (MLP)** — basically a stack of matrix multiplications with nonlinear activation functions. Nothing fancy. For each week independently:

```
256 numbers → multiply by learned weights → 128 numbers → apply nonlinearity
→ multiply by learned weights → 128 numbers → apply nonlinearity
→ multiply by learned weights → 26 numbers (one per demand component)
```

The 26 output numbers are the predicted shares for that week:
- Share going to baseline (organic demand)
- Share going to Facebook
- Share going to Google
- Share going to TikTok
- ... (up to 20 channels)
- Share going to price effect
- Share going to distribution
- Share going to competition
- Share going to macro environment
- Share going to noise (random variation)

To get actual dollar/unit contributions: multiply each share by that week's observed sales. If the model says Facebook's share is 0.14 and sales that week were 1,200 units, then Facebook drove 168 units that week.

### How training works

The training is straightforward supervised learning — the most battle-tested approach in all of machine learning:

1. Show the network a simulated business's observable data (sales, spend, prices)
2. The network predicts weekly shares
3. Compare predictions to the true shares (which we know because we simulated the data)
4. Calculate the error (mean squared error — how far off was each prediction?)
5. Use backpropagation to adjust all the weights slightly to reduce the error
6. Repeat 50,000 × 50 times (50,000 businesses, 50 passes through the data)

The encoder and decoder are trained **together end-to-end** — the encoder learns what features to extract BECAUSE the decoder needs them. This is powerful: the encoder doesn't just extract generic features, it learns to extract exactly the features that make decomposition prediction accurate.

### What this replaces and why

Our previous approach used **Simulation-Based Inference (SBI) with Neural Spline Flows (NSF)**. In plain terms: instead of predicting the decomposition directly, it tried to learn a probability distribution over constant parameters. The output was things like "Facebook beta = 180 for the whole year (with 90% chance between 150-210)."

We moved away from this because:

1. **Constants are wrong for business** — channel effectiveness changes over time. Creative gets stale, algorithms change, competitors enter/exit. Saying "Facebook beta = 180 for 2 years" is like saying "the weather was 72F for all of 2024."

2. **The math couldn't scale** — Neural Spline Flows work well for 5-30 parameters, but per-week decomposition needs 1,000+ output values. The NSF would need to learn a probability distribution in 1,000-dimensional space, which is intractable.

3. **Business users want charts, not parameters** — A CMO doesn't care that "beta = 180." They want to see a stacked area chart showing what drove sales each month, how channel contributions shifted over time, and when the ROI started declining.

The new approach gives them exactly that: week-by-week decomposition, ready for monthly charts and quarterly reviews.

---

## Technical Architecture

```
Observable Data (y, spend, context) — T weeks × ~30 columns
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│              ENCODER (EmbeddingNetwork.forward_temporal)      │
│                                                              │
│  For each channel:                                           │
│    spend(t) → Dilated CNN (3 layers, dilations 1,2,4)        │
│            → temporal features (T × 64)                      │
│    + learnable channel-type embedding (16-dim)               │
│                                                              │
│  Aggregate across channels: masked mean per timestep         │
│  y(t) → Dilated CNN → temporal features (T × 64)            │
│  context(t) → Dilated CNN → temporal features (T × 64)      │
│                                                              │
│  Concatenate + project: (T × 209) → (T × 256)               │
│                                                              │
│  KEY: No pooling. Every timestep preserved.                  │
└──────────────────────────┬──────────────────────────────────┘
                           │  (batch, T, 256)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              TEMPORAL DECODER (TemporalDecoder)               │
│                                                              │
│  Shared MLP applied to each timestep independently:          │
│    embedding(t) → Linear(256,128) → ReLU → Dropout           │
│                 → Linear(128,128) → ReLU → Dropout           │
│                 → Linear(128, 26) → raw share predictions     │
│                                                              │
│  26 output columns per timestep:                             │
│    [0]     baseline                                          │
│    [1-20]  channel contributions (up to 20 channels)         │
│    [21]    price effect                                       │
│    [22]    distribution cap                                   │
│    [23]    competition effect                                 │
│    [24]    macro + regime effects                             │
│    [25]    noise                                             │
│                                                              │
│  Output is UNCONSTRAINED (no softmax):                       │
│    - Noise and price can be negative                         │
│    - Inactive channels learn to output zero                  │
│    - Shares don't need to sum to exactly 1.0                 │
│                                                              │
│  Loss: MSE(predicted_shares, true_shares)                    │
│  true_shares = ground_truth_component(t) / y(t)             │
└──────────────────────────┬──────────────────────────────────┘
                           │  (batch, T, 26)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   INFERENCE OUTPUT                            │
│                                                              │
│  shares(t, c) = predicted fractional share of component c    │
│  contribution(t, c) = shares(t, c) × y(t)                   │
│                                                              │
│  Derived metrics:                                            │
│    Total channel contribution = Σ_t contribution(t, ch)      │
│    ROAS = total_contribution / total_spend                   │
│    Time-varying effectiveness = contribution(t) / spend(t)   │
│    Monthly decomposition = aggregate shares by month         │
└─────────────────────────────────────────────────────────────┘
```

---

## The Encoder: Dilated CNN + Set Transformer

### Why Dilated CNN?

Standard CNNs look at a fixed window (e.g., 3 weeks). Dilated CNNs use exponentially spaced gaps:
- Layer 1 (dilation=1): sees weeks [t-3, t+3] — immediate neighbors
- Layer 2 (dilation=2): sees weeks [t-6, t+6] — monthly patterns
- Layer 3 (dilation=4): sees weeks [t-12, t+12] — quarterly patterns

This gives the network a **multi-scale receptive field** — it understands what's happening this week in the context of what happened this month and this quarter, all in 3 layers instead of 12.

The conv layers are:
```
Conv1d(1, 32, kernel=7, dilation=1)  → 32 filters, local patterns
Conv1d(32, 64, kernel=5, dilation=2) → 64 filters, medium-range
Conv1d(64, 64, kernel=3, dilation=4) → 64 filters, long-range
```

Each channel's spend goes through the same shared encoder (weight sharing — the network doesn't need separate weights per channel since the channel-type embedding handles identity).

### Why Set Transformer?

The number of media channels varies (1 to 20). Standard neural networks need fixed input sizes. The Set Transformer handles variable-size sets by design — it uses self-attention across channels, learning which channels interact with each other.

In the temporal encoder, we use a simpler aggregation: **masked mean across channels per timestep**. The Set Transformer is still available for the pooled (summary) path used by the SBI engines.

### Temporal vs Pooled

The same encoder supports two modes:
- `forward()` — pools temporal features into a single 256-dim summary (used by SBI engines)
- `forward_temporal()` — preserves per-timestep features as (T × 256) output (used by decomposition decoder)

The difference: `forward()` applies AdaptiveAvgPool1d which collapses time → 1. `forward_temporal()` skips pooling entirely, keeping the full temporal resolution.

---

## The Decoder: Per-Timestep MLP

The decoder is deliberately simple: a 2-layer MLP (256 → 128 → 128 → 26) applied independently to each timestep. No recurrence, no attention across time — temporal structure is handled by the encoder's dilated receptive field.

### Why no softmax?

Early versions used softmax to force shares to sum to 1.0. This caused three problems:

1. **Negative components**: Noise and price effects can be negative (a price increase reduces demand). Softmax forces all outputs positive.
2. **Inactive channels**: With 26 output slots but only ~7 active components, the model wastes capacity pushing 19 slots to near-zero through extreme negative logits.
3. **Training difficulty**: The softmax creates competition between components — increasing one share decreases all others, making gradient flow indirect.

Without softmax, the model directly outputs share values. Inactive channels learn to output exactly 0.0, negative components are naturally represented, and each output is trained independently via MSE.

---

## Training Pipeline

### Data generation

```python
# ScenarioSampler generates random business configurations
# Each has: 1-20 channels, 52-260 weeks, pricing/distribution/competition/macro
sampler = ScenarioSampler(seed=42, rich_context=True, n_fixed_channels=5)

# TrainingPipeline runs the simulator and saves .npz batches
# NEW: now saves per-period ground_truth decomposition matrix (T × 26)
pipeline = TrainingPipeline(sampler, output_dir="training_data", batch_size=100)
pipeline.generate(n_total=50000, n_workers=4)
```

### Ground truth format

Each .npz batch contains:
- `y`: (N, T) observed demand
- `spend`: (N, T, max_channels) spend matrix
- `context`: (N, T, 10) business context (price, promo, distribution, competition, macro)
- `decomposition`: (N, T, 26) **per-period ground truth** — the true contribution of each component at each timestep

The decomposition matrix columns match the decoder output layout:
| Index | Component | Type |
|-------|-----------|------|
| 0 | baseline | always present |
| 1-20 | per-channel contribution | post-interaction, zero-padded for unused slots |
| 21 | price effect | additive (can be negative) |
| 22 | distribution cap | multiplicative (raw value ~0.8-1.0) |
| 23 | competition effect | additive |
| 24 | macro + regime effects | additive |
| 25 | noise | additive (can be negative) |

### Training loop

Standard PyTorch:
```
For each epoch:
    For each batch:
        1. encoder.forward_temporal(y, spend, context, ...) → embeddings (B, T, 256)
        2. decoder(embeddings) → predicted_shares (B, T, 26)
        3. true_shares = ground_truth_decomposition / y
        4. loss = MSE(predicted_shares, true_shares) over valid timesteps only
        5. loss.backward() → update encoder + decoder jointly

    Validate on held-out 10%
    ReduceLROnPlateau scheduler
    Early stopping (patience=15)
```

---

## Evaluation Metrics

### Primary: Per-Component Share MSE

How well does each predicted component share match the true share, averaged across all timesteps?

```
Share MSE(component) = (1/T) Σ_t (predicted_share(t) - true_share(t))²
```

Lower is better. A model that assigns everything to baseline will have high share MSE for channels and low for baseline.

### Secondary: Per-Component R²

For each component, how well does the predicted absolute contribution track the true contribution over time?

```
Component R²(c) = 1 - Σ_t(true(t,c) - pred(t,c))² / Σ_t(true(t,c) - mean(true(t,c)))²
```

A positive R² means the model captures the temporal variation of that component. Negative means it's worse than predicting the mean.

### Derived: Total Contribution Error

Sum predicted contributions across all weeks, compare to true totals:
```
Error%(channel) = |Σ_t pred_contrib(t) - Σ_t true_contrib(t)| / |Σ_t true_contrib(t)|
```

This is what matters for business decisions — "did the model get the total Facebook contribution right?"

### Not meaningful: y-level R²

Because the decoder outputs shares that (approximately) sum to 1.0, and `y_reconstructed = Σ_c shares(c) × y ≈ y`, the y-level R² is trivially close to 1.0. It does not indicate model quality.

---

## Comparison to Previous Approaches

| Aspect | Monolithic SBI | Compositional SBI | Decomposition (current) |
|--------|---------------|-------------------|------------------------|
| What it predicts | 27 constant params | 2 global + 5/channel constant | T × 26 per-period shares |
| Architecture | Embedding + NSF | Frozen embedding + 2 NSFs | Encoder-decoder (MLP) |
| Library | sbi (SNPE-C) | sbi (SNPE-C) | Pure PyTorch |
| Output dim | 27 | 2 + 5 | T × 26 (~2700) |
| Time-varying | No | No | Yes |
| Negative values | Via prior bounds | Via prior bounds | Natural |
| Uncertainty | Full posterior | Full posterior | Future (ensemble/MC dropout) |
| Training time | 5-10 hrs | ~2 hrs | ~1-2 hrs |
| Inference time | < 1 sec | < 1 sec | < 1 sec |

---

## Running It

```bash
# Generate training data + train + evaluate
python scripts/train_neural.py \
    --decomposition \
    --n-train 50000 \
    --n-epochs 50 \
    --fixed-channels 5 \
    --scenario realistic_brand

# Load saved model and evaluate only
python scripts/train_neural.py \
    --decomposition \
    --skip-training \
    --scenario realistic_brand
```

### Output

- `neural_output/decomp_model/` — saved encoder + decoder weights
- `neural_output/plots/` — actual vs predicted, per-channel contributions, monthly decomposition, loss curve
- `neural_output/results/` — per-channel CSV, summary CSV

---

## Future Work

### Phase B: Time-Varying Simulator

The simulator currently generates constant betas. Real businesses have time-varying effectiveness. Extending the simulator with random walks, regime switching, and seasonal modulation will produce training data that better matches reality.

### Phase C: Uncertainty Quantification

The current architecture gives point estimates. Options for uncertainty:
- **MC Dropout**: Enable dropout at inference, run N forward passes → approximate posterior
- **Deep Ensemble**: Train 5 models with different seeds → ensemble spread = uncertainty
- **Neural Superstatistics** (Radev et al., 2023): Per-timestep amortized SBI with learned transition models

---

## References

1. Radev et al. (2023). Neural Superstatistics. *Scientific Reports*.
2. Dew, Padilla, Shchetkina (2024). Your MMM is Broken. *arXiv:2408.07678*.
3. Bai, Kolter, Koltun (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling — dilated causal convolutions.
4. Lee, Lee, Kim, Shin (2019). Set Transformer: A Framework for Attention-based Permutation-Invariant Input.
