# Next Steps

## Priority 1: Implement the Hybrid Approach (Neural + Differential Evolution)

### Stage 1: Improve Category Accuracy

The encoder-decoder already achieves 2-9pp. To improve:
- **More epochs**: Training stopped at epoch 27 (early stopping). Try patience=30 or 50.
- **More data**: 50K is good but 100K might help for extreme baselines.
- **Extreme baseline accuracy**: Currently 19pp error when baseline is <20%. The model struggles with media-dominant businesses. May need oversampling of media_dominant regime.

Run on vast.ai:
```bash
python scripts/train_neural.py --decomposition --n-train 50000 --n-epochs 100 \
    --fixed-channels 5 --random-eval 10
```

### Stage 2: Build Differential Evolution Channel Optimizer

**New file needed**: `demantiq/neural/channel_optimizer.py`

```python
def optimize_channels(
    y_media: np.ndarray,          # (T,) media contribution from Stage 1
    spend: dict[str, np.ndarray], # per-channel spend time series
    channel_configs: list,         # saturation/adstock function types
) -> dict:
    """Find per-channel betas and curve params via differential evolution.
    
    Returns per-channel contributions and parameters.
    """
```

Implementation:
1. Define objective: `MSE(sum(beta_i * sat(adstock(spend_i))), y_media)`
2. Parameter bounds from ScenarioSampler ranges (beta: 10-800, alpha: 0.1-0.9, etc.)
3. Use `scipy.optimize.differential_evolution` (or try multiple: DE, CMA-ES, L-BFGS-B)
4. Return per-channel contributions + fitted parameters

**Key decisions needed**:
- Which saturation function? Hill or logistic? Or try both and pick best fit?
- Which adstock function? Geometric or weibull? Or try both?
- How to handle multiple local minima? Multiple restarts?
- How to regularize? (prevent one channel from absorbing all credit)

### Stage 3: End-to-End Pipeline

```python
def decompose(observable_data: pd.DataFrame, channel_names: list[str]) -> dict:
    """Full demand decomposition: neural network + optimization.
    
    1. Neural network → category split (baseline, media total, other)
    2. Differential evolution → per-channel contributions within media
    3. Return weekly decomposition with uncertainty
    """
```

### Stage 4: Validation on Simulator Data

Run end-to-end on 100+ simulated scenarios with known ground truth:
- Category accuracy (should be <5pp)
- Channel ranking accuracy (target: Spearman > 0.8)
- Channel contribution error (target: <20% per channel)
- Reconstruction R² (target: >0.9)
- Runtime (target: <10 seconds per scenario)

### Stage 5: Uncertainty Quantification

- **Stage 1**: MC Dropout (enable dropout at inference, 100 forward passes)
- **Stage 2**: Multiple random starts of DE → distribution of parameters

## Priority 2: Things to Investigate

### Why does category accuracy degrade at low baselines?
The model predicts 33% when true is 12%. Is this because:
- Not enough media-dominant training scenarios?
- The encoder can't distinguish "lots of media" from "moderate media"?
- The MSE loss weights baseline errors and media errors equally?

### Should we switch from shares to absolute contributions for Stage 1?
Currently predicting shares (component/y). The share clipping to [-2, 2] might lose information. Absolute contributions in demand units might be more natural.

### Can Stage 2 work with unknown saturation/adstock functions?
In production, we don't know if a client's channels follow hill or logistic saturation. The optimizer could try multiple function families and pick the best fit (lowest MSE).

## Priority 3: Production Readiness

### Streamlit UI
Wire the two-stage pipeline into the Streamlit app (`app.py`):
- Upload client data (CSV)
- Auto-detect channels
- Run Stage 1 + Stage 2
- Display weekly decomposition charts
- Download results

### Documentation
- Update `ai_docs/ACTIVE_methodology_decomposition.md` with hybrid approach
- Plain English explanation for client presentations
- Technical docs for team

## Key Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `demantiq/neural/channel_optimizer.py` | CREATE | Differential evolution for per-channel params |
| `demantiq/neural/decomposition_engine.py` | MODIFY | Add `decompose()` method that runs Stage 1 + Stage 2 |
| `scripts/train_neural.py` | MODIFY | Add evaluation for hybrid approach |
| `ai_docs/ACTIVE_methodology_decomposition.md` | UPDATE | Document hybrid approach |

## User Preferences (from memory)

- Quality over speed — no shortcuts, no "working" versions that need 10 more iterations
- Must use neural network approaches (never suggest OLS as a solution)
- OK with long training times, no due date
- Wants to understand the math and be able to explain to boss in plain English
- Activate venv before running tests: `source venv/bin/activate`
