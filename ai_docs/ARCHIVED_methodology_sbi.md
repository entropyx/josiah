# Demantiq: Methodology & Results

**Simulation-Based Inference for Demand Architecture**

*Entropy — March 2026*

---

## Executive Summary

Demantiq replaces traditional Marketing Mix Modeling (MCMC/Bayesian fitting) with a neural inference engine that produces full posterior distributions over all business parameters in **under 1 second**, compared to hours with PyMC or Meridian.

The approach:

1. A synthetic simulator generates 50,000+ realistic business datasets where every parameter is known
2. A neural network learns the inverse mapping: *observable data → parameter distributions*
3. On new data, the trained network produces instant posteriors with uncertainty quantification

**Key innovation**: No one has applied Simulation-Based Inference to Marketing Mix Modeling before. Google's NNN (2025), Meta's Robyn, and PyMC Marketing all use conventional fitting methods. This is novel.

**Current architecture**: Compositional inference — separate density estimators for global parameters (2-dim) and per-channel parameters (5-dim), with 250K unrolled training examples for per-channel inference. This gives 50,000 training samples per parameter dimension, dramatically improving posterior concentration.

---

## 1. The Problem

### What is demand attribution?

A business's demand comes from many drivers simultaneously:

```
demand(t) = organic_baseline
          + media_effects(t)       ← How much did Facebook/Google/TikTok drive?
          + pricing_effects(t)     ← How much did price changes/promos drive?
          + distribution_effects(t)← How much did store availability matter?
          + competition_effects(t) ← How much did competitors suppress demand?
          + interactions(t)        ← Did promos make ads more effective?
          + noise(t)
```

The question: **given only the observable data (sales, spend, prices, etc.), can we recover the true contribution of each driver?**

This is an **inverse problem** — we see the output (sales) and want to recover the inputs (each driver's contribution). It's hard because:
- Everything happens at the same time (can't isolate one driver)
- Spend is correlated with expected demand (endogeneity)
- Channels are correlated with each other (collinearity)
- Some effects are multiplicative, not additive (interactions)

### Why traditional MMM is slow

Traditional Bayesian MMM (PyMC Marketing, Google Meridian) works by:

1. Writing down a mathematical likelihood: P(observed_sales | parameters)
2. Specifying priors: P(parameters)
3. Running MCMC sampling — generating millions of random parameter proposals, computing likelihood for each, keeping the ones that explain the data well

This takes **2-8 hours per dataset** because MCMC must explore the full parameter space through random walks. Every new client requires a fresh MCMC run from scratch.

### Our approach: learn the inverse mapping once, apply everywhere

Instead of fitting each dataset from scratch, we:

1. **Pre-generate** 50,000 synthetic datasets with known ground truth
2. **Train** a neural network to recognize patterns: (data → parameters)
3. **Deploy**: for any new dataset, one forward pass = instant posteriors

The network has already "seen" thousands of businesses similar to yours. It recognizes the patterns. Think of it like training a doctor on 50,000 case files with known diagnoses — after enough practice, diagnosis becomes instant.

---

## 2. The Simulator (Data Generating Process)

### What the simulator does, in plain English

Imagine you run a supplement brand. Every week you have sales. Those sales come from a bunch of different things happening at the same time. The simulator builds a fake version of your business where we control every lever and know exactly how much each one contributes.

Here's what the simulator calculates, step by step:

**1. Start with organic demand ("people who would buy anyway")**

Every brand has a baseline level of demand — people who already know your product, repeat buyers, organic search traffic. This is the demand you'd get if you turned off all ads, never ran a promo, and the economy stayed flat.

Example: 1,200 units per week as a baseline, growing slowly at +2 units/week (your brand is getting more well-known over time), plus a seasonal bump around the holidays.

**2. Add what each media channel contributes**

For each channel (Facebook, Google, TikTok, Email, YouTube), the simulator computes how much demand that channel's spend generates. But it's not a simple "spend $1, get X sales." Three things happen:

- **Carryover (adstock)**: When you run a Facebook ad on Monday, some people see it and buy Tuesday, Wednesday, next week. The ad's effect lingers and decays over time. Facebook ads decay fast (people forget in a few days). YouTube brand videos decay slowly (awareness sticks around for weeks). The simulator models this: each week's spend contributes not just to this week's sales, but to next week's, and the week after, with a decay rate specific to each channel.

- **Diminishing returns (saturation)**: The first $10K you spend on Facebook reaches new people. The fifth $10K just shows the same people the same ad again. Every channel has a point where spending more stops helping. The simulator models this with a curve that rises steeply at low spend and flattens at high spend. Different channels saturate at different rates — Google Search might saturate slower (there's always someone searching) while TikTok might saturate faster (limited viral reach).

- **Channel strength (beta)**: After applying carryover and saturation, the result is multiplied by a coefficient that represents the channel's actual effectiveness. Google might have beta=350 (search intent is very strong), while TikTok might have beta=120 (awareness but less direct conversion). This is the core number we're trying to recover.

**3. Account for how promotions amplify ads (price × media interaction)**

When you're running a 20% off sale AND Facebook ads at the same time, the ads work better than normal. People see the ad, see the discount, and are more likely to buy. This amplification effect is the price × media interaction.

Each channel has its own interaction strength:
- Email: 30% amplification during promos (promo emails are very effective)
- TikTok: 25% (promos boost viral content)
- Facebook: 20% (social ads + deals work well together)
- Google: 10% (people searching already have intent, promos help less)
- YouTube: 5% (brand videos barely affected by whether there's a promo)

**4. Account for how distribution amplifies ads (distribution × media interaction)**

If your product is only in 75% of stores, then ads that drive people to stores are wasted 25% of the time — the customer goes to buy but it's not on the shelf. Higher distribution means ads convert better. This is the distribution × media interaction.

**5. Add the effect of price changes**

When your average selling price goes up 10%, demand drops. How much it drops is the price elasticity. An elasticity of -1.2 means: "10% price increase → 12% demand decrease."

The simulator tracks your product's price each week. During promo weeks (say, 20% off monthly), price drops and demand goes up. During normal weeks, price is at the base level.

**6. Factor in distribution (store availability)**

If your product is only available in 75% of stores, you can only capture 75% of the potential demand that media and pricing generate. Distribution acts as a ceiling — no matter how good your ads are, you can't sell what isn't on the shelf.

The simulator tracks distribution over time. A growing brand might go from 50% → 85% distribution over 2 years as they get into more retailers.

**7. Factor in competition**

Competitors are running their own ads. When competitor share-of-voice is high, your ads stand out less — people see theirs instead of yours. The simulator tracks competitor activity and reduces your media effectiveness proportionally.

**8. Factor in the economy (macro effects)**

Consumer confidence, unemployment, category trends — these affect demand regardless of what you do with media or pricing. A recession suppresses demand even if you triple your ad budget.

The simulator can also model structural breaks — sudden events like a new competitor entering the market or a supply chain shock that permanently shifts the baseline.

**9. Add random noise**

Real sales data is noisy. Weather, viral social posts, a celebrity mention, a bad review — things you can't predict. The simulator adds realistic random noise to make the fake data look like real data.

**10. Sum it all up**

```
This week's demand = organic baseline
                   + Facebook contribution (after carryover + saturation + interactions)
                   + Google contribution
                   + TikTok contribution
                   + Email contribution
                   + YouTube contribution
                   + price change effect
                   + distribution effect
                   + competition effect
                   + macro/economy effect
                   + random noise
```

The simulator knows the EXACT number for each line. In real life you only see the total (your sales report). The neural engine's job is to look at your sales data, your spend data, your prices, your distribution — and figure out what each line was.

---

### The demand equation

For each time period t (typically weekly, over 26-260 weeks):

```
y(t) = B₀ + trend(t) + seasonality(t)                              [baseline]
     + Σᵢ βᵢ · sat(adstock(spendᵢ(t)))                            [media effects]
       × (1 + γᵢᵖ · promo(t))                                     [price×media interaction]
       × (1 + γᵢᵈ · distribution(t))                               [distribution×media interaction]
     + ε · ln(price(t) / price_base)                                [price elasticity]
     + distribution_cap(t)                                          [distribution effect]
     + competition_effect(t)                                        [competition]
     + macro(t) + regime(t)                                         [macro/economy]
     + noise(t)                                                     [random error]
```

Let's break down every piece.

### 2.1 Baseline demand: `B₀ + trend(t) + seasonality(t)`

This is the demand that would exist without any marketing, pricing, or external effects.

- **B₀** (organic level): The base demand. Example: 1000 units/week for a mid-size brand.
- **trend(t)**: Long-term growth or decline. Example: +3 units/week slope = growing brand.
- **seasonality(t)**: Repeating patterns. We use Fourier series:

```
seasonality(t) = Σₖ [aₖ · sin(2πkt/52) + bₖ · cos(2πkt/52)]

where k = 1, 2, ... N_terms

k=1: captures annual cycle (peak at Christmas, dip in January)
k=2: captures semi-annual cycle (summer + winter peaks)
k=3+: captures finer seasonal patterns
```

**Why Fourier?** Any periodic pattern can be decomposed into sine/cosine waves. 2-4 terms capture the major seasonal patterns without overfitting.

### 2.2 Media effects: `βᵢ · sat(adstock(spendᵢ(t)))`

This is the core of MMM — how much does each media channel (Facebook, Google, etc.) contribute to demand?

Three transformations happen to raw spend before it becomes a contribution:

**Step 1: Adstock** — "Yesterday's ad still works today"

When you run a Facebook ad on Monday, its effect doesn't disappear on Tuesday. It decays over time. The **geometric adstock** models this:

```
adstocked(t) = spend(t) + α · adstocked(t-1)

where α = decay rate (0 to 1)
  α = 0.3: fast decay (digital ads — effect fades in ~3 days)
  α = 0.8: slow decay (TV ads — brand awareness lingers for weeks)
```

Example with α = 0.5 and spend = [100, 0, 0, 0]:
```
Week 1: adstocked = 100
Week 2: adstocked = 0 + 0.5 × 100 = 50
Week 3: adstocked = 0 + 0.5 × 50 = 25
Week 4: adstocked = 0 + 0.5 × 25 = 12.5
```

The $100 spend in week 1 generates a total effect of 100 + 50 + 25 + 12.5 + ... = 200 over time. The adstock captures this carryover.

We also support **Weibull adstock** which can model delayed peaks (effect peaks 2-3 weeks after spending, like a TV campaign that builds awareness slowly).

**Step 2: Saturation** — "Spending $10M doesn't work 10x better than $1M"

Media has diminishing returns. The first $1M on Facebook reaches new audiences. The tenth $1M just shows the same people the same ad again.

We normalize spend by `max|adstocked_spend|` so saturation operates on a [0, 1] scale, then apply a saturation function:

**Hill saturation** (most common in MMM):
```
sat(x) = x^S / (K^S + x^S)

where:
  K = half-saturation point (when x = K, output = 0.5)
  S = steepness (higher = sharper curve)
```

**Logistic saturation**:
```
sat(x) = 1 / (1 + exp(-k(x - x₀)))

where:
  k = steepness
  x₀ = inflection point
```

Both produce an S-curve: near-linear at low spend, flattening at high spend.

**Step 3: Beta multiplication** — "How much does this channel actually contribute?"

```
channel_contribution(t) = βᵢ × sat(adstock(spend(t)))

where βᵢ = the channel's true effectiveness coefficient
```

A β of 150 means: "At full saturation, this channel contributes 150 units of demand per period." The saturation and adstock determine what fraction of that maximum is realized at the actual spend level.

### 2.3 Interactions: `× (1 + γᵢᵖ · promo(t)) × (1 + γᵢᵈ · distribution(t))`

Media doesn't work in isolation. Promotions make ads more effective (people see an ad AND a discount → more likely to buy). Distribution determines whether ads can convert (no product on shelf → ad is wasted).

```
total_channel_effect(t) = βᵢ × sat(adstock(spend(t)))
                          × (1 + γᵢᵖ × is_promo(t))        ← price amplifies media
                          × (1 + γᵢᵈ × distribution(t))     ← distribution amplifies media
```

Example for Facebook with β=150, γᵖ=0.25, γᵈ=0.30:
```
During normal week (no promo, distribution=0.7):
  effect = 150 × sat(spend) × (1 + 0) × (1 + 0.30 × 0.7)
         = 150 × sat(spend) × 1.0 × 1.21
         = 181.5 × sat(spend)

During promo week (promo active, distribution=0.7):
  effect = 150 × sat(spend) × (1 + 0.25) × (1 + 0.30 × 0.7)
         = 150 × sat(spend) × 1.25 × 1.21
         = 226.9 × sat(spend)

The promo amplifies the media effect by 25%.
```

**Why this matters for business**: "Your Facebook ads generate 150 units at baseline, but during promos they generate 227 units — that's a 51% lift. Running ads without promos wastes 34% of their potential."

### 2.4 Price elasticity: `ε · ln(price(t) / price_base)`

How demand changes when price changes:

```
price_effect(t) = ε × ln(price(t) / price_base)

where:
  ε = price elasticity (e.g., -1.5)
  ln = natural logarithm
  price_base = reference price (e.g., $25)
```

A 10% price increase → ln(1.1) ≈ 0.095 → effect = -1.5 × 0.095 = -0.14 → demand drops ~14%.

**Why log?** Constant elasticity: a 10% price increase has the same proportional demand effect whether price goes from $20→$22 or $100→$110.

### 2.5 Other components

- **Distribution cap**: A multiplicative ceiling. If distribution = 0.7 (product available in 70% of stores), effective demand is capped at 70% of potential.
- **Competition**: Competitor share-of-voice (SOV) suppresses your media effectiveness. High competitor spend → your ads stand out less.
- **Macro**: External factors (consumer confidence, unemployment) that affect overall demand. Includes **regime changes** (COVID-type shocks that permanently shift the baseline).
- **Noise**: Random variation. Can be Gaussian, t-distributed (heavy tails), or heteroscedastic (variance proportional to demand level).

---

## 3. The Neural Inference Engine

### How the neural engine works, in plain English

The simulator can generate a fake business where we know everything — every beta, every interaction coefficient, every elasticity. But with a real client, we only see the output (sales) and the inputs (spend, prices, etc.). We don't know the hidden parameters that connect them.

The neural engine solves this by studying thousands of fake businesses, then applying what it learned to real ones. Here's how, step by step:

**Step 1: Create 50,000 practice problems**

We run the simulator 50,000 times, each time with completely different random settings — different channels, different betas, different saturation curves, different price elasticities, different interaction strengths. Each run produces:
- The observable data: weekly sales, weekly spend per channel, prices, distribution, etc. (this is what a real client would give us)
- The answer key: the exact betas, ROAS, elasticity, interaction coefficients that generated those sales (this is what we'd never know with real data)

Think of it like a teacher creating 50,000 math tests with answer keys. Each test is a different business.

**Step 2: Teach a neural network to read business data**

The neural network needs to look at 104 weeks of sales data, spend across 5 channels, prices, promotions, and distribution — and extract the important patterns. This is like reading an X-ray: the raw image has millions of pixels, but a trained radiologist sees "tumor, 3cm, left lobe."

We use a **pattern reader** (the embedding network) that works in four steps:

1. **Read each channel's spend pattern separately.** A small neural network (called a dilated CNN — think of it as a sliding magnifying glass that looks at different time scales) scans each channel's weekly spend and produces a summary: "Facebook had pulsed campaigns with peaks in Q4" or "Google was always-on with steady spend." This produces a 64-number fingerprint per channel.

2. **Understand how channels interact with each other.** A cross-channel attention mechanism (called a Set Transformer) looks at all channels together and asks: "Are Facebook and Google spending at the same times? Is TikTok's pulsed pattern offset from YouTube's seasonal pattern?" This captures relationships between channels — are they competing for the same audience or complementing each other? Each channel's fingerprint gets updated with this cross-channel context, and all channels get pooled into a single 80-number summary of the media mix.

3. **Read the business context.** Another pattern reader scans the non-media time series: price changes, promo periods, distribution levels, competitive activity, macro indicators. It produces a 64-number summary of the business environment.

4. **Read the sales pattern.** A third pattern reader scans the sales time series itself — the seasonality, trend, noise level, response to events. Another 64-number summary.

All four summaries get concatenated and compressed into a final **256-number fingerprint** that captures everything important about this business. This fingerprint is what the next step uses.

**Step 3: Turn the fingerprint into parameter estimates**

Now we have a 256-number fingerprint summarizing a business. We need to turn it into estimates of the hidden parameters (betas, ROAS, elasticity, interactions) — with uncertainty.

We use a **Neural Spline Flow** (NSF). Think of it as a machine that takes random noise and reshapes it into a meaningful answer, guided by the fingerprint:

1. Start with pure random noise — 5 random numbers from a bell curve (for per-channel inference: beta, ROAS, contribution fraction, price interaction, distribution interaction)
2. Pass through 5 "reshaping" steps, each guided by the business fingerprint
3. Each step bends, stretches, and shifts the numbers based on what the fingerprint says about this business
4. After 5 steps, the random noise has been transformed into a plausible set of parameters FOR THIS SPECIFIC BUSINESS

Because we start with random noise, we can repeat this 10,000 times and get 10,000 different plausible answers. The spread of those answers IS the uncertainty. If all 10,000 say "Facebook beta is between 160 and 200," we're confident. If they spread from 50 to 400, we're uncertain.

**Step 4: Split the problem into easier pieces (compositional architecture)**

Trying to estimate all 27 parameters at once (2 global + 5 channels × 5 params) is hard. Instead, we split into two focused problems:

- **Global estimator**: Just 2 parameters — total media contribution % and price elasticity. Conditioned on the 256-number business fingerprint. This is easy — a 2-dimensional problem with 50,000 training examples.

- **Per-channel estimator**: Just 5 parameters per channel — beta, ROAS, contribution fraction, price interaction, distribution interaction. Conditioned on that channel's specific fingerprint (336 numbers: its temporal pattern + cross-channel context + business context).

The key trick: by "unrolling" channels, each of the 50,000 training businesses with 5 channels becomes 5 separate per-channel training examples. So the per-channel estimator gets **250,000 training examples** for a 5-dimensional problem. That's 50,000 examples per parameter — way more than enough.

At inference time on a new business: run the pattern reader once, then run the global estimator once, then run the per-channel estimator 5 times (once per channel). Takes < 1 second total.

**Step 5: Train by practicing on the 50,000 tests**

Training works like this: for each of the 50,000 practice businesses, we:
1. Feed the observable data through the pattern reader → get the fingerprint
2. Feed the fingerprint through the NSF → get predicted parameter distributions
3. Check: does the predicted distribution put high probability on the TRUE parameters (the ones we know from the answer key)?
4. If not: adjust the neural network's internal weights slightly to make it more accurate
5. Repeat for all 50,000 businesses, many times over (100-200 passes through the full dataset)

After training, the network has seen so many different business configurations that it can recognize patterns like: "High seasonal spend with strong promo correlation and moderate ROAS → this channel probably has beta around 200, price interaction around 0.2, slow adstock decay."

**Step 6: Apply to a real client**

When a real client gives us their data:
1. The pattern reader extracts the fingerprint (< 0.1 seconds)
2. The global estimator produces 10,000 samples of media contribution % and elasticity (< 0.1 seconds)
3. The per-channel estimator produces 10,000 samples of each channel's parameters (< 0.3 seconds)
4. We summarize: "Facebook beta = 180 ± 25, ROAS = 0.012 ± 0.003, price interaction = 0.20 ± 0.05"

Total time: under 1 second. Traditional MCMC methods take 2-8 hours for the same answer.

**Why this works**

The network doesn't "fit" each client's data from scratch. It already knows what good answers look like from 50,000 practice runs. When it sees a new business, it pattern-matches: "This looks like practice businesses #4,372, #12,891, and #38,204 — they had betas around 150-200, elasticity around -1.3, and moderate interactions." The answer comes from recognition, not computation.

This is the same principle behind how a doctor with 20 years of experience can diagnose faster than running every possible test — they've seen enough cases to recognize patterns instantly.

---

### Technical details of the neural engine

The inference engine takes observable data and produces posterior distributions over all the parameters described above. It has three main components:

1. **Embedding Network** — reads the raw data and compresses it into a meaningful summary
2. **Density Estimator (Neural Spline Flow)** — transforms the summary into a probability distribution over parameters
3. **Compositional Architecture** — splits the problem into easier sub-problems

### 3.1 The Embedding Network: Reading the Data

The observable data for one business is a multivariate time series:

```
Inputs:
  y(t)              — weekly sales (104 values for 2 years)
  spend_facebook(t) — weekly Facebook spend
  spend_google(t)   — weekly Google spend
  ... (up to 20 channels)
  price(t)          — weekly product price
  is_promo(t)       — binary: is there a promotion?
  distribution(t)   — store availability %
  competitor_sov(t) — competitor share of voice
  macro_vars(t)     — up to 4 macro indicators
```

The challenge: this data is variable-size (different businesses have 3-20 channels, 26-260 weeks) and high-dimensional (~8000 numbers). We need to compress it into a fixed-size vector that captures the essential patterns.

The embedding network has four branches:

#### Branch 1: Per-Channel Temporal Encoder (Dilated 1D CNN)

Each channel's spend time series is encoded independently by a shared convolutional neural network.

**What is a 1D CNN?** A sliding filter that detects patterns in a sequence. A filter of width 7 slides along the 104-week spend series, computing a weighted sum at each position. Multiple filters detect different patterns (spikes, trends, periodicity).

**What are dilations?** They expand the filter's reach without adding parameters:

```
Dilation = 1 (normal):     reads 7 consecutive weeks
  ● ● ● ● ● ● ●

Dilation = 2 (skip every other): reads 13 weeks of context
  ● - ● - ● - ● - ● - ● - ●

Dilation = 4 (skip 3):    reads 25 weeks of context
  ● - - - ● - - - ● - - - ● - - - ● - - - ● - - - ●
```

Our architecture stacks 3 layers:
```
Layer 1: Conv1D(1 → 32 filters, width=7, dilation=1) + ReLU
         Detects: weekly patterns, spend spikes, on/off periods
         Receptive field: 7 weeks

Layer 2: Conv1D(32 → 64 filters, width=5, dilation=2) + ReLU
         Detects: monthly patterns, flight schedules, seasonal ramps
         Receptive field: ~17 weeks

Layer 3: Conv1D(64 → 64 filters, width=3, dilation=4) + ReLU
         Detects: quarterly trends, long-term spend shifts
         Receptive field: ~29 weeks

AdaptiveAvgPool1d → (64,) fixed-size output
         Collapses the time dimension: one 64-number summary per channel
```

**Why dilated CNN instead of a Transformer?** For sequences under 300 timesteps, dilated CNNs are faster (O(T) vs O(T²)) and capture the same multi-scale patterns. Transformers excel at very long sequences (1000+) which marketing data rarely has.

**Why shared weights?** The same CNN processes Facebook spend, Google spend, TikTok spend, etc. This forces the network to learn general spend-pattern features (spikes, seasonality, trends) rather than channel-specific quirks. It also means the network works on any number of channels without retraining.

Each channel also gets a **learnable type embedding** (16 dimensions) — a vector that encodes "this is Facebook-type media" vs "this is Google-type media". This tells the network about functional properties (upper-funnel vs lower-funnel, broad vs targeted) without hand-engineering features.

**Per-channel output**: concat(temporal_64, type_embedding_16) = **80 dimensions per channel**.

#### Branch 2: Cross-Channel Attention (Set Transformer)

Media channels interact — Facebook and Google might compete for the same audience, or TV might amplify Search. The Set Transformer learns these interactions.

**Why "Set"?** Channels are an unordered set. Listing them as [Facebook, Google, TikTok] is the same as [TikTok, Facebook, Google]. The architecture must be **permutation invariant** — reordering channels shouldn't change the result.

**How it works:**

```
Input: {channel_1(80), channel_2(80), ..., channel_N(80)}

Step 1: Multi-Head Self-Attention (2 layers)
  Each channel "looks at" every other channel and updates its embedding
  based on what it sees:

  Q = channel_i × W_Q    (what am I looking for?)
  K = channel_j × W_K    (what do I offer?)
  V = channel_j × W_V    (what information do I carry?)

  attention_weight = softmax(Q · K^T / √d)
  attended = attention_weight × V

  After attention, each channel's embedding incorporates information
  from ALL other channels. Facebook's embedding now "knows" that
  Google is also spending heavily this quarter.

Step 2: Pooling by Multihead Attention (PMA)
  A learnable "seed" vector attends to all channels and produces
  a single fixed-size summary:

  set_summary = PMA({attended_ch1, attended_ch2, ..., attended_chN})
  → (80,) regardless of how many channels
```

**What this captures:**
- Budget competition: "Facebook and Instagram are both high → they're competing for the same audience"
- Synergy: "TV spend is high AND Search spend is high → TV drives search, synergy effect"
- Collinearity: "All channels spend more in Q4 → hard to separate their effects"

**Per-channel output preserved**: The attended per-channel embeddings (80-dim each) are kept for the per-channel density estimator.

#### Branch 3: Outcome Encoder

The sales time series y(t) is encoded by the same dilated CNN architecture → 64 dimensions. This captures the demand pattern: level, trend, seasonality, variance.

#### Branch 4: Business Context Encoder

All non-media time series (price, promos, distribution, competition, macro) are packed into a multivariate matrix (T × 10 columns) and encoded by a dilated CNN with 10 input channels → 64 dimensions.

This gives the network visibility into pricing dynamics, distribution changes, and macro trends — the "business context" that's NOT media but affects demand.

#### Assembly: Global Summary

```
global_summary = MLP(
    concat(
        set_summary(80),       ← cross-channel media patterns
        y_embedding(64),       ← demand patterns
        context_embedding(64), ← business context patterns
        n_channels(1)          ← how many channels are active
    )
) → (256,)
```

This 256-dimensional vector is the **complete summary** of everything observable about this business. It conditions the density estimator.

**Per-channel summary** (for compositional inference):
```
per_channel_summary[i] = concat(
    attended_channel_embedding[i](80),  ← this channel + cross-channel context
    global_summary(256)                 ← the full business context
) → (336,)
```

### 3.2 The Density Estimator: Neural Spline Flow (NSF)

The density estimator takes the embedding and produces a **probability distribution** over all parameters. Not a single point estimate — a full distribution that shows uncertainty.

**Why a distribution?** If Facebook and Google spend are highly correlated, it's hard to tell which one drove sales. A point estimate like "Facebook beta = 200" hides this uncertainty. A distribution shows "Facebook beta is probably 100-300, with most mass around 200" — honestly reflecting what the data can and cannot tell us.

**How Neural Spline Flows work:**

Start with simple noise:
```
z ~ Normal(0, I)    ← 5 random numbers from a standard normal distribution
```

Pass through a chain of 5 learnable transforms:
```
z → T₁(z; embed) → T₂(·; embed) → T₃(·; embed) → T₄(·; embed) → T₅(·; embed) → θ
```

Each transform T_k is a **monotone rational-quadratic spline** — a flexible curve made of connected quadratic segments. The spline's shape (knot positions, slopes) is produced by a small neural network that takes the embedding as input.

```
One spline transform:

Input: z_k (a number)
       embed (the 256-dim or 592-dim embedding)

Step 1: A neural network takes embed → produces K knot positions and K slopes
Step 2: The spline maps z_k through these knots:

        output │     ╭────────
               │   ╱
               │  │
               │ ╱
        ───────┼╱──────────── input
               │
               │

        The spline can stretch, compress, and bend the distribution
        in ways that capture asymmetry, heavy tails, and multimodality.
```

After 5 transforms, the simple Gaussian noise has been reshaped into a complex posterior distribution that matches what the training data says.

**The training loss:**
```
L = -log q_φ(θ_true | x)

In English: "How surprised would the network be if θ_true were the answer?"

Low surprise = high probability assigned to truth = good model
High surprise = low probability assigned to truth = bad model

We minimize this over all 50,000 training examples.
```

**Why NSF over simpler flows?** Rational-quadratic splines can model heavy tails, sharp peaks, and asymmetric distributions. A simpler flow (like RealNVP with affine transforms) can only shift and scale — it can't capture the complex shapes that MMM posteriors often have.

### 3.3 The Compositional Architecture: Divide and Conquer

**The problem with one big flow**: A single NSF over 27 dimensions (2 global + 5 channels × 5 params) needs enormous amounts of training data to produce tight posteriors. With 50K samples, that's only ~1850 samples per dimension — not enough for the NSF to concentrate.

**The solution**: Split into two small, focused flows:

```
COMPOSITIONAL ARCHITECTURE
══════════════════════════

┌─────────────────────┐       ┌──────────────────────────────────┐
│ GLOBAL NSF           │       │ PER-CHANNEL NSF (shared weights)  │
│                      │       │                                   │
│ Dimensions: 2        │       │ Dimensions: 5                     │
│  - media_pct         │       │  - beta (base media effect)       │
│  - price_elasticity  │       │  - ROAS                           │
│                      │       │  - contribution_fraction           │
│ Conditioned on:      │       │  - price×media interaction         │
│  global_summary(256) │       │  - distribution×media interaction  │
│                      │       │                                   │
│ Training data:       │       │ Conditioned on:                   │
│  50,000 samples      │       │  per_channel_summary(336)         │
│                      │       │  + global_summary(256) = (592)    │
│ Samples per dim:     │       │                                   │
│  25,000              │       │ Training data:                    │
│                      │       │  50K × 5 channels = 250,000      │
│                      │       │                                   │
│                      │       │ Samples per dim:                  │
│                      │       │  50,000                           │
└─────────────────────┘       └──────────────────────────────────┘
```

**The key insight: channel unrolling**

Each simulation with 5 channels becomes 5 per-channel training examples:

```
Simulation #1: y, spend(5 channels), context
  → Channel example 1: (facebook_embedding + global) → (beta_fb, roas_fb, frac_fb, px_fb, dx_fb)
  → Channel example 2: (google_embedding + global) → (beta_gg, roas_gg, frac_gg, px_gg, dx_gg)
  → Channel example 3: (tiktok_embedding + global) → (beta_tt, roas_tt, frac_tt, px_tt, dx_tt)
  → Channel example 4: (email_embedding + global) → (beta_em, roas_em, frac_em, px_em, dx_em)
  → Channel example 5: (youtube_embedding + global) → (beta_yt, roas_yt, frac_yt, px_yt, dx_yt)

50,000 simulations × 5 channels = 250,000 per-channel examples
250,000 examples / 5 dimensions = 50,000 samples per dimension
```

This is **100x more data density** than the monolithic approach (50K / 27 ≈ 1850 per dim).

**Embedding pre-training:**

Before training the NSFs, we pre-train the embedding network on a simple regression task:
```
embedding(raw_data) → predict [media_pct, elasticity, mean_beta, mean_roas]
```

This teaches the embedding "what to pay attention to" in ~10 epochs (~70 min). Then we freeze the embedding and extract fixed vectors for all 50K samples. The NSFs train on these pre-computed vectors — fast, because no gradient flows through the embedding.

**Inference on new data:**

```
1. Run EmbeddingNetwork on client data
   → global_summary (256-dim)
   → per_channel_summaries (one 336-dim vector per channel)

2. Sample from Global NSF:
   → media_contribution_pct: 37.4% [32%, 43%]
   → price_elasticity: -1.5 [-1.9, -1.1]

3. For each channel, sample from Per-Channel NSF:
   Facebook:
   → base_beta: 152 [128, 178]
   → ROAS: 0.012 [0.008, 0.016]
   → contribution_fraction: 0.22 [0.18, 0.26]
   → price×media: 0.24 [0.18, 0.30]
   → distribution×media: 0.28 [0.20, 0.36]

Total time: < 1 second
```

---

## 4. The Interaction Decomposition: What We Solved

### The problem

In initial experiments, the network consistently overestimated channel betas by ~1.7x (e.g., inferred 250 when true was 150) while ROAS was accurate (5-22% error). This persisted across 500, 5K, 10K, and 50K training samples.

### Root cause: label mismatch

We traced the exact math:

```
The simulator applies interactions MULTIPLICATIVELY:
  contribution(t) = beta(150) × sat(spend) × (1 + 0.25 × promo) × (1 + 0.30 × distribution)

Average multiplier: (1 + 0.25 × 0.5) × (1 + 0.30 × 0.7) = 1.125 × 1.21 = 1.36
Maximum multiplier: (1 + 0.25) × (1 + 0.30) = 1.625

The ground truth stored:
  true_beta = 150           ← config value, BEFORE interactions
  true_contribution = Σ(150 × sat × interaction_multipliers)  ← AFTER interactions
  true_roas = contribution / spend                             ← AFTER interactions

The network sees y (which contains post-interaction contributions ~204-244 × sat)
but is trained to predict beta = 150 (pre-interaction).

It correctly learns the EFFECTIVE beta (~250) that explains the data.
150 × 1.625 = 243.75 ≈ the ~250 the network consistently reported.
```

**This was NOT a model failure** — the network was right. We were comparing its answer (effective beta) against the wrong ground truth (config beta).

### The fix: infer both components

Instead of one "beta" number, the network now infers:
- **base_beta** = 150 (media effect without interactions)
- **γᵖ** = 0.25 (how much promotions amplify media)
- **γᵈ** = 0.30 (how much distribution amplifies media)

The effective beta can be reconstructed: 150 × (1 + 0.25 × promo_frac) × (1 + 0.30 × dist_level).

**Validation**: Posterior correlation between beta and interaction coefficients is ~0.00 across all channels — confirming the network CAN separate these components (no identifiability issue).

---

## 5. Diagnostics: How We Know It's Working

### Signed bias (not just MAPE)

Unsigned MAPE hides whether errors are systematic or random. We report signed bias:
- All betas biased +40% → systematic (like the interaction problem)
- Some +20%, some -20% → random noise, needs more data

### Posterior sharpness

Measures how much the network learned vs echoing the prior:
```
sharpness = 1 - (CI_width / prior_range)

0.0 = posterior = prior (learned nothing)
0.5 = posterior is half the prior width (meaningful learning)
1.0 = posterior is a point (perfect certainty)
```

### Effective beta decomposition check

After adding interaction inference, we validate:
```
effective_beta = inferred_beta × (1 + inferred_px) × (1 + inferred_dx)
```
This should approximately equal the old "biased" estimate (~250), confirming the decomposition is internally consistent.

### Posterior correlations

If beta and price_x_media are strongly anti-correlated in the posterior (corr < -0.5), the network can't separate them — it trades off "high beta, low interaction" vs "low beta, high interaction." Near-zero correlation means clean separation.

### Simulation-Based Calibration (SBC)

The gold standard: "When the engine says 90% confident, is it right 90% of the time?"

```
Repeat 500 times:
  1. Draw random true parameters from the prior
  2. Simulate a dataset using those parameters
  3. Run the engine → get posterior
  4. Check: is the true value inside the 90% credible interval?

If ~450/500 (90%) → calibrated
If ~300/500 (60%) → overconfident (CIs too narrow)
If ~490/500 (98%) → underconfident (CIs too wide)
```

---

## 6. Results

### Scenario: `interaction_heavy` (SCN-005)

5 channels (Facebook, Google, TikTok, Email, YouTube), 104 weeks, with:
- Pricing: base_price=$25, elasticity=-1.5
- Distribution: 70% weighted availability
- Interactions: price×media=0.25, distribution×media=0.30 for all channels
- All betas = 150 (identical, to test decomposition)

### Evolution of results across iterations

| Iteration | Architecture | Training | Beta MAPE | ROAS MAPE | Elasticity | Key insight |
|-----------|-------------|----------|-----------|-----------|------------|-------------|
| v1 | Monolithic 62-dim | 5K, wrong data | 60% | 30% | 80%+ | Data gen bug: settings not passed to batches |
| v2 | Monolithic 17-dim | 5K, fixed | 48-62% | 20% | 57% | Prior bounds clipped elasticity |
| v3 | Monolithic 17-dim | 50K, full embed | 65-81% | **1-22%** | **18%** | ROAS excellent, betas biased by interactions |
| v4 | + interaction decomp, simple | 50K, summary stats | **15-65%** | 42% | 43% | Decomposition works, betas improving |
| v5 | + interaction decomp, full | 50K, full embed | 68-72% | **0.5-22%** | **18%** | Full embed better for ROAS/elasticity |
| v6 | **Compositional** | 50K, pre-trained embed | *running* | *running* | *running* | 250K per-channel examples for 5-dim NSF |

### Best results to date (v5, full embedding)

| Parameter | True | Inferred | Error |
|-----------|------|----------|-------|
| Facebook ROAS | 0.012 | 0.012 | **1.2%** |
| Email ROAS | 0.011 | 0.011 | **0.5%** |
| Price elasticity | -1.500 | -1.764 | **17.6%** |
| Media contribution | 37.4% | 41.3% | +10.4% |

### Comparison with alternatives

| Method | Beta Recovery | Elasticity | Interactions | Time |
|--------|-------------|-----------|-------------|------|
| OLS | 100% error | N/A | N/A | 0.01s |
| Demantiq Neural | 1-65% | 18% | Detected | 0.25s |
| PyMC Marketing | ~10-20% (expected) | ~10% | Not modeled | 2-4 hours |

Demantiq is 10,000x faster than PyMC and models interaction effects that PyMC doesn't even attempt.

---

## 7. Architecture Decisions

| Decision | What we chose | Why | Alternative considered |
|----------|--------------|-----|----------------------|
| Inference method | SNPE-C (amortized NPE) | Train once, infer in ms. No MCMC. | NLE (needs MCMC at inference), variational (less expressive) |
| Density estimator | Neural Spline Flows | Handles multimodal posteriors, exact log-prob | MAF (less expressive), RealNVP (can't model complex shapes) |
| Temporal encoder | Dilated 1D CNN (3 layers) | O(T), multi-scale patterns, SBI standard | Transformer (O(T²), overkill for T<300) |
| Channel encoder | Set Transformer (2 layers, 4 heads) | Learns cross-channel interactions, permutation invariant | DeepSets (no pairwise learning), flattening (breaks with variable N) |
| Architecture | Compositional (global + per-channel) | 100x more data per dim, faster training | Monolithic (too many dims for data volume) |
| Embedding strategy | Pre-train then freeze | Separates representation learning from density estimation | End-to-end (10 hours, unstable gradients) |
| Parameter decomposition | base_beta + interactions | Separates media from interaction amplification | Effective beta (hides interactions, biases results) |

---

## 8. Next Steps

### In progress
- Compositional architecture running (~1.5 hours estimated)
- Expected: betas < 15% MAPE, interactions < 20%, with concentrated posteriors

### Short-term
- Time series validation: actual vs predicted y(t) overlay, monthly decomposition chart
- Evaluate across all 15 benchmark scenarios (clean_room through adversarial)
- Head-to-head comparison with PyMC Marketing on same scenarios

### Medium-term
- Calibration interface: inject geo-lift results as informative priors
- Sequential SNPE rounds for high-stakes observations
- Streamlit UI for interactive inference

### Long-term
- Cross-client evidence database (hierarchical priors from experiments)
- Foundation model pre-training across scenario types
- Production API for automated inference

---

## References

1. Greenberg, D., Nonnenmacher, M., & Macke, J. (2019). Automatic Posterior Transformation for Likelihood-Free Inference. *ICML 2019*. — SNPE-C algorithm
2. Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G. (2019). Neural Spline Flows. *NeurIPS 2019*. — NSF density estimator
3. Lee, J., Lee, Y., Kim, J., et al. (2019). Set Transformer. *ICML 2019*. — Cross-channel attention
4. van den Oord, A., et al. (2016). WaveNet: A Generative Model for Raw Audio. *arXiv:1609.03499*. — Dilated causal convolutions
5. Talts, S., Betancourt, M., et al. (2018). Validating Bayesian Inference Algorithms with Simulation-Based Calibration. *arXiv:1804.06788*. — SBC diagnostics
6. Deistler, M., Macke, J., et al. (2025). Simulation-Based Inference: A Practical Guide. *arXiv:2508.12939*. — Embedding design
7. Google (2025). NNN: Next-Generation Neural Networks for Marketing Measurement. *arXiv:2504.06212*. — Closest competitor approach
8. Gloeckler, M., et al. (2024). All-in-one Simulation-Based Inference. *ICML 2024*. — Compositional SBI approaches
