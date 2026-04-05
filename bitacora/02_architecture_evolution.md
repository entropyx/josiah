# Architecture Evolution

## Approach 1: Monolithic SBI (SNPE-C + Neural Spline Flow)

**What**: Single NSF density estimator over 27 constant parameters (betas, ROAS, elasticity).
**Result**: ROAS 0.5-22% error (good!), betas 65-81% error (bad).
**Why abandoned**: NSF can't scale past ~30 dims. Constant parameters can't capture time-varying effects. Actual-vs-predicted R² = -17.

**Key learning**: The embedding network CAN extract useful signal from observable data (ROAS worked). End-to-end gradient (encoder→NSF→loss) is critical — this is why monolithic beat compositional.

## Approach 2: Compositional SBI (2 separate NSFs)

**What**: Global NSF (2 params: media_pct, elasticity) + per-channel NSF (5 params per channel) with "channel unrolling" (50K × 5 channels = 250K per-channel examples).
**Result**: ROAS regressed to 56-87%.
**Why abandoned**: Frozen embedding (pre-trained, then fixed) killed end-to-end gradients. The per-channel NSF got embeddings that weren't optimized for its task.

**Key learning**: Never freeze the encoder. End-to-end training is essential.

## Approach 3: Encoder-Decoder with Simple MSE ← BEST CATEGORY RESULT

**What**: EmbeddingNetwork.forward_temporal() → (B,T,256) embeddings → TemporalDecoder MLP → 26 shares per timestep. Simple MSE loss.
**Architecture**:
- 3 Dilated CNNs (channels, y, context) with LeakyReLU + input normalization
- Set Transformer for cross-channel attention (pooled path)
- Per-timestep MLP decoder (256→128→128→26)
**Result**: Category accuracy 2-9pp across diverse baselines. Channel correlation 0.08 (random).
**Status**: Current best for category. Still the active encoder-decoder.

**Key learning**: Simple MSE works better than composite losses for category. The model genuinely learns scenario characteristics from observable data.

## Approach 4: Composite Loss (Learnable Sigmas)

**What**: 4-term loss (share MSE, channel totals, category, ranking) with Kendall et al. uncertainty weighting.
**Result**: Collapsed — media predicted at 3.7% vs true 38%.
**Why abandoned**: Learnable sigma parameters found degenerate solution. Model dampened category loss and pushed everything to baseline.

**Key learning**: Learnable loss weights can be gamed. Never use them.

## Approach 5: Composite Loss (Running-Mean Normalization)

**What**: Same 4 losses but with non-learnable EMA normalization.
**Result**: Category 8.8pp, channels still ~14.5K each.
**Why abandoned**: Didn't collapse but didn't improve channels either.

## Approach 6: Per-Channel Decoder

**What**: Each channel processed individually through shared MLP (not mixed in projection).
**Result**: Channels still ~14.5K.
**Why abandoned**: Diagnostic revealed the 529→256 projection mixed channel positions. Different channels at same position across training scenarios = ambiguity.

## Approach 7: Raw Spend Feature

**What**: Added normalized spend (1 dim) to per-channel features to preserve magnitude.
**Result**: Decoder barely used it (zeroing changed output by 0.001).
**Why abandoned**: 1 dim in 210 total = 0.5% signal. Insufficient.

## Approach 8: NAM with y Input

**What**: Per-channel CNN seeing [spend_i, y, context, type_emb] for full time series.
**Result**: Channel correlation -0.15.
**Why abandoned**: y is polluted by all channels. Spend↔y correlation doesn't predict per-channel share (proven by diagnostic: twitter corr=-0.42 = 2% share, direct_mail corr=+0.42 = 2% share).

**Key learning**: The spend↔y relationship is confounded by other channels. Per-channel networks that see y get contradictory training signals.

## Approach 9: NAM without y (Additive Decomposition)

**What**: Channel CNN sees ONLY spend + type_emb + global_stats. Predicts absolute contributions (not shares). Components sum to reconstruct y.
**Result**: Category 39pp (worst), channels -0.06.
**Why abandoned**: Without y, CNN can't infer beta. Beta varies 20-800 across scenarios for same channel type. Network can only predict average contribution.

**Key learning**: Removing y removes too much information. The model needs SOME signal about the scenario's scale.

## Current: Encoder-Decoder (Approach 3 restored)

The original encoder-decoder with simple MSE remains the best performer for category-level decomposition. It's the foundation for the two-stage hybrid approach.
