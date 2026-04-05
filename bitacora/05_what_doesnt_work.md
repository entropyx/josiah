# What Doesn't Work (Proven)

## 1. Cross-Scenario Neural Network for Channel Differentiation

**Tried 9 different approaches.** Channel correlation never exceeded 0.08 on diverse evaluation. All channels predict approximately the same value (~average contribution).

**Root cause**: Beta (channel effectiveness) varies 20-800 across training scenarios for the same channel type. The network sees the same spend pattern but different true contributions depending on beta. It can only learn the average.

**Analogy**: "Someone drove for 2 hours. How far did they go?" Without knowing speed (beta), you predict the average distance. Wrong for every individual case.

## 2. Shared CNN for Channel Encoding

Produces features with cosine similarity 0.98+ across all channels. The CNN learns temporal patterns (shape) but not per-channel magnitude or effectiveness. No architecture change downstream can fix this if the input features are identical.

## 3. Per-Channel NAM (with or without y)

- **With y**: y is polluted by all channels. Contradictory training signals (same correlation → different shares).
- **Without y**: Can't infer beta without any demand signal. Predicts average.

## 4. Composite Loss Functions

- **Learnable sigmas (Kendall et al.)**: Model found degenerate solution — dampened category loss, collapsed everything to baseline.
- **Running-mean normalization**: Prevented collapse but didn't improve channels. 
- **Multiple loss terms don't help if the model can't differentiate channels in the first place.** The loss function optimizes whatever signal the architecture provides. If the architecture produces identical features per channel, no loss function can create differentiation.

## 5. Predicting Shares (component/y) for Channel Accuracy

Shares lose scale information. Two channels with very different absolute contributions can have similar shares if they track y proportionally. Absolute contributions are more informative for channel-level accuracy.

## 6. Position-Based Channel Encoding

With variable channel combinations across training scenarios (position 0 = facebook in scenario A, pinterest in scenario B), any position-based encoding learns the AVERAGE channel behavior at that position. Type embeddings (16 dims) are too small to overcome this.

## 7. Early Stopping with Complex Losses

Complex losses (composite, NAM) cause val_loss to spike early (epoch 1-6) leading to early stopping before the model can learn channel differentiation. The toy test showed differentiation emerges at epoch 40+, but real training stops at epoch 16.
