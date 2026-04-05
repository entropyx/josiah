# Hybrid Approach: Neural Network + Differential Evolution

## The Core Insight

After 9 neural network approaches, we learned:
- **Category decomposition** (baseline vs media) works with cross-scenario neural networks
- **Channel decomposition** (which channel contributed how much) CANNOT work with cross-scenario training because beta varies per scenario

The solution: use each method for what it's good at.

## Two-Stage Pipeline

### Stage 1: Neural Network (instant, cross-scenario)

**Already proven** — the encoder-decoder with simple MSE achieves 2-9pp category accuracy.

```
Observable data → Neural Network → baseline(t), total_media(t), price(t), competition(t), macro(t)
```

From this: `y_media(t) = y(t) - baseline(t) - price(t) - competition(t) - macro(t)`

This gives us the total media contribution per week — the "budget" that needs to be split across channels.

### Stage 2: Differential Evolution (seconds, per-scenario)

For each new scenario/client, optimize per-channel parameters:

```python
For each channel i, find: beta_i, saturation_params_i, adstock_params_i

Such that:
  y_media(t) ≈ Σ beta_i × saturation(adstock(spend_i(t), alpha_i, lag_i), K_i, S_i)

Minimize: MSE(sum_of_channel_contributions, y_media)
```

**Parameters per channel**: ~4 (beta, alpha/shape, K/k, S/x0)
**Total for 5 channels**: ~20 parameters
**Optimizer**: scipy.optimize.differential_evolution (or CMA-ES)
**Time**: ~1-5 seconds per scenario

### Why This Works

1. **Neural network handles temporal complexity**: trend, seasonality, baseline level, price effects, competition — all the nonlinear temporal patterns it's good at.

2. **Optimizer handles per-scenario beta**: It fits THIS business's specific channel effectiveness. No cross-scenario generalization needed.

3. **Known functional forms**: The simulator uses specific saturation (hill, logistic) and adstock (geometric, weibull) functions. We can use the same functions in the optimizer, or let the optimizer discover which function fits best.

4. **We already have the building blocks**:
   - `demantiq/components/saturation.py` — hill_saturation(), logistic_saturation()
   - `demantiq/components/adstock.py` — geometric_adstock(), exponential_adstock()
   - `demantiq/scenarios/scenario_sampler.py` — knows parameter ranges for bounds

### Computational Cost

| Step | Time | Cost |
|------|------|------|
| Stage 1 (neural) | <1 second | Free (model already trained) |
| Stage 2 (optimizer) | 1-5 seconds | Free (runs on CPU) |
| **Total** | **<6 seconds** | **Free at inference** |

Compare: PyMC MCMC = 2-8 hours. Google Meridian = 1-4 hours. Meta Robyn = 10-30 minutes.

### Uncertainty Quantification

- **Stage 1**: MC Dropout or ensemble for category-level uncertainty
- **Stage 2**: Multiple random starts of differential evolution → spread = uncertainty. Or bootstrap (resample weeks, re-optimize, get distribution of betas).

### Validation Strategy

We can validate end-to-end on simulator data:
1. Generate scenario with known ground truth
2. Run Stage 1 → get y_media prediction
3. Run Stage 2 → get per-channel params and contributions
4. Compare to true per-channel contributions

Target metrics:
- Category accuracy: <5pp (Stage 1)
- Channel ranking: Spearman > 0.8 (Stage 2)
- Channel contribution error: <20% per channel (Stage 2)
- Total reconstruction R² > 0.9

## Comparison to Meta Robyn

Meta Robyn does something similar:
- Ridge regression + Nevergrad (evolutionary optimization) for channel params
- Prophet for trend/seasonality decomposition

Our approach is better because:
- Neural network for trend/seasonality/baseline is more flexible than Prophet
- Simulator provides ground truth for validation (Robyn can't verify accuracy)
- Same functional forms (hill, adstock) but with neural pre-decomposition

## Implementation Plan

1. **Improve Stage 1** (more epochs, better extreme-baseline accuracy)
2. **Build Stage 2**: define objective function, parameter bounds, run scipy DE
3. **End-to-end evaluation**: simulate → Stage 1 → Stage 2 → compare to ground truth
4. **Uncertainty**: bootstrap/multi-start for confidence intervals
5. **Client-ready output**: weekly decomposition charts, ROAS, contribution tables
