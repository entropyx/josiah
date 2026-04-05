# Key Diagnostics & Findings

## 1. CNN Features Are Identical Across Channels

**Test**: Cosine similarity of per-channel CNN temporal features on realistic_brand.
**Result**: 0.98+ cosine similarity between ALL channel pairs.
**Root cause**: The shared Dilated CNN learns temporal SHAPE (pulsed, steady, seasonal) but not MAGNITUDE. A 10x spend difference produces only 1.5x feature norm difference with cosine 0.987.
**Impact**: The decoder receives essentially identical inputs for all channels → produces identical outputs.
**Attempted fixes**: Raw spend feature (1 dim, drowned), per-channel decoder (didn't help), NAM (didn't help).

## 2. Dying ReLU in Channel CNN

**Test**: Layer-by-layer CNN output on realistic_brand spend (10K-80K range).
**Result**: Final ReLU killed ALL outputs to zero. Conv output was entirely negative (mean=-3947).
**Fix applied**: LeakyReLU(0.01) + global spend normalization (divide by max across all channels).
**Status**: FIXED. CNN produces nonzero output after fix.

## 3. Projection Layer Destroys Channel Identity

**Test**: Channel features before vs after 529→256 projection.
**Result**: Before projection — channels well-differentiated (cosine -0.46 to 0.60). After projection — all channels produce ~0.06 share.
**Root cause**: Different channels appear at different positions across training scenarios. Position 0 is sometimes facebook, sometimes pinterest. Projection learns "position 0 = average channel."
**Status**: Fundamental limitation of concatenate+project with variable channel positions.

## 4. Spend↔y Correlation Does NOT Predict Channel Share

**Test**: Pearson correlation between each channel's spend and y across multiple scenarios.
**Result**: 
- twitter: corr=-0.42, true_share=0.020
- direct_mail: corr=+0.42, true_share=0.022
- youtube: corr=+0.69, true_share=0.044
- podcast: corr=+0.61, true_share=0.150
**Root cause**: y contains ALL channels + baseline + noise. Correlation is dominated by confounders, not by that channel's actual effectiveness.
**Impact**: Any model that uses spend↔y correlation to predict per-channel share will get contradictory training signals.

## 5. Toy Test: CNN CAN Learn Channel Differentiation

**Test 1**: Single channel, simple data (y = baseline + beta*spend + noise).
**Result**: Correlation 0.87 in 50 epochs. CNN learns perfectly.

**Test 2**: 5 channels sharing y, simple data.
**Result**: Correlation 0.55 in 80 epochs, still climbing.

**Test 3**: 5 channels, real nonlinear data (saturation + adstock).
**Result**: Correlation 0.0 — fails completely.

**Conclusion**: The CNN architecture CAN learn channel differentiation in principle. It fails on real data because saturation + adstock + multi-channel confounding makes the signal too weak for cross-scenario generalization.

## 6. OLS Gets Perfect Channel Ranking on Single Scenario

**Test**: Standard OLS regression y ~ b0 + b1*fb_spend + b2*google_spend + ... on realistic_brand.
**Result**: Perfect ranking (Spearman 1.0). google > email > facebook > youtube > tiktok — exactly correct.
**Key insight**: The information IS in the observable data. Per-scenario fitting works; cross-scenario generalization doesn't.
**Limitation**: OLS can't capture saturation, adstock, or nonlinear effects — too simple for production.

## 7. Training Data Diversity Matters

**Before**: organic_level 200-3000 uniform → 78% of scenarios have baseline 60-80%.
**After**: Regime system (media_dominant / balanced / baseline_dominant) → baseline ranges 7-102%.
**Impact**: Models trained on diverse data are directionally correct (high baseline scenarios get high baseline predictions). Without diversity, model just predicts ~65% for everything.

## 8. The Fundamental Problem

**Across 50K training scenarios, the same channel type with the same spend pattern can have wildly different contributions because beta varies 20-800.** The neural network can only see spend (and optionally y), but contribution = beta × saturation(adstock(spend)). Without knowing beta, it predicts the average contribution.

This is NOT solvable by any cross-scenario neural network architecture. It requires per-scenario fitting (like OLS does, but with nonlinear functions).
