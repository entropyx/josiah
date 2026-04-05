# What Works (Proven)

## 1. Category-Level Decomposition (Baseline vs Media vs Other)

**Approach**: Encoder-decoder with simple MSE on shares.
**Accuracy**: 2-9pp on diverse scenarios (12-102% baseline range).
**Details**:
- High baseline (>80%): 2-6pp error — excellent
- Medium baseline (50-70%): 8-9pp — good
- Low baseline (<20%): 19pp — needs improvement (more epochs should help)
**Model is NOT predicting constant 65%** — it responds to different inputs directionally.

**Key config that produced this result**:
```
--decomposition --n-train 50000 --n-epochs 50 --fixed-channels 5
```
With diverse training data (ScenarioSampler with regime system).

## 2. Weekly Stability

100% of weeks have R² > 0.8 (across components per week).
The model produces consistent predictions across all timesteps, not concentrated in a few weeks.

## 3. Training Infrastructure

- Simulator generates clean, separable ground truth for all components
- .npz training data pipeline works (50K scenarios, batched)
- DemantiqDataset loads all data correctly (y, spend, context, decomposition, type_ids)
- Evaluation framework catches problems (category breakdown, channel-only metrics, weekly accuracy)

## 4. Vast.ai GPU Workflow

- Tested on RTX A2000 ($0.044/hr)
- 5x speedup over CPU (50K/50 epochs in ~60 min vs ~5 hrs)
- SSH workflow: push code → git pull on server → train → pull results → destroy
- Scripts: `scripts/vast_train.py` for search, launch, logs, pull, destroy
- GitHub token needed for private repo cloning

## 5. Diverse Training Data

Regime system in ScenarioSampler produces:
- media_dominant: organic 100-800, betas 200-800, spend 5K-80K
- balanced: organic 500-2000, betas 50-500, spend 1K-50K  
- baseline_dominant: organic 1500-5000, betas 10-150, spend 500-15K

Resulting baseline% range: 7-102% (vs old 22-102% clustered at 68%).

## 6. Encoder Fixes (LeakyReLU + Normalization)

- LeakyReLU(0.01) replaces ReLU in all Dilated CNNs — prevents dying neurons
- Spend normalization: divide by global max across all channels per batch
- Y normalization: divide by max(abs(y)) per batch
- Context normalization: divide by max(abs) per column per batch
- Raw spend feature: 1 additional dim concatenated to channel features (preserves magnitude)
