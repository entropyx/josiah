# Current Codebase State

## Active Neural Engine Files

### `demantiq/neural/temporal_decoder.py`
Simple per-timestep MLP decoder: (B, T, 256) → (B, T, 26) shares.
Architecture: Linear(256,128) → LeakyReLU → Dropout → Linear(128,128) → LeakyReLU → Dropout → Linear(128,26).
This is the decoder from Approach 3 (best category result).

### `demantiq/neural/encoders.py`
`EmbeddingNetwork` with `forward_temporal()` method:
- 3 Dilated CNNs: channel_temporal_encoder, y_encoder, context_encoder
- Each CNN: Conv1d(in→32→64→64) with LeakyReLU, dilations 1,2,4
- Channel features: CNN(64) + type_emb(16) + raw_spend(1) = 81 per channel
- `channel_dim` = 80 (for Set Transformer), `channel_dim_with_spend` = 81 (for temporal path)
- With n_fixed_channels: concatenates all channel features (5×81=405)
- Without: masked mean across channels
- All inputs normalized (spend by global max, y by max, context per column)
- Projection: concatenated features → Linear(D, 256) → embedding

### `demantiq/neural/decomposition_engine.py`
`DecompositionEngine` with simple MSE loss:
- `_compute_true_shares()`: decomposition / y, clipped to [-2, 2], distribution_cap zeroed
- `_train_step()`: encoder.forward_temporal → decoder → MSE(pred_shares, true_shares)
- `infer()`: returns shares, contributions (shares × y), y_reconstructed
- Save/load: encoder.pt + decoder.pt + decomp_config.json

### `demantiq/neural/losses.py`
Currently contains the composite `DecompositionLoss` with running-mean normalization. 
NOT used by current engine (which uses inline simple MSE). Kept for reference.

### `demantiq/neural/data_loader.py`
`DemantiqDataset`: loads .npz batches with y, spend, context, decomposition, type_ids, config_vectors.
Returns dict per item with all tensors. Handles variable-length padding.

### `demantiq/orchestration/training_format.py`
Decomposition matrix layout (DECOMP_COLS = 26):
- Index 0: baseline
- Index 1-20: per-channel contributions (MAX_CHANNELS=20, typically 5 active)
- Index 21: price effect
- Index 22: distribution cap (MULTIPLICATIVE — zeroed in training)
- Index 23: competition effect
- Index 24: macro + regime effects
- Index 25: noise

`ground_truth_to_decomposition()`: converts simulator's ground_truth DataFrame to (T, 26) matrix.

### `demantiq/scenarios/scenario_sampler.py`
`ScenarioSampler` with regime system:
- media_dominant: low organic (100-800), high betas (200-800), high spend (5K-80K)
- balanced: mid organic (500-2000), mid betas (50-500), mid spend (1K-50K)
- baseline_dominant: high organic (1500-5000), low betas (10-150), low spend (500-15K)
- `rich_context=True`: pricing always, distribution 70%, interactions 70%
- `n_fixed_channels`: fixes channel count (used with --fixed-channels 5)

### `scripts/train_neural.py`
Main training + evaluation script. Key flags:
- `--decomposition`: uses decomposition engine
- `--n-train N`: number of training simulations
- `--n-epochs N`: training epochs
- `--fixed-channels N`: fix channel count
- `--skip-training`: load saved model
- `--random-eval N`: evaluate on N diverse random scenarios
- `--all-scenarios`: evaluate on all compatible ScenarioLibrary scenarios
- `--scenario NAME`: evaluate on specific scenario
- `--lr`: learning rate

Evaluation outputs: category breakdown, channel-only metrics (rank correlation, ranking), weekly accuracy (R², rank corr, MAPE), per-channel contribution errors, reconstruction R².

### `scripts/vast_train.py`
Vast.ai GPU management:
- `search`: browse GPUs
- `test`: quick verification run
- `launch`: start training instance
- `logs`: view instance logs
- `pull`: copy results locally
- `destroy`: stop billing
- Reads API key from .env via python-dotenv

## Git Branch

`feature/implement-v1` — latest commit has the restored encoder-decoder (Approach 3).

## Config for Best Category Result

```python
DecompositionConfig(
    n_epochs=50, n_train=50000, batch_size=256, learning_rate=1e-3,
    n_fixed_channels=5, patience=15,
    temporal_dim=64, type_embed_dim=16, embedding_dim=256,
    decoder_hidden_dim=128, decoder_n_layers=2,
)
```
