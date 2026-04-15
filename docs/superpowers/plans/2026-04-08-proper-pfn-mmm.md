# Proper PFN-based MMM Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an amortized neural inference engine for Marketing Mix Modeling that does per-scenario decomposition in a single forward pass, using proper PFN held-out training, channels-as-permutation-invariant-tokens, and richer per-channel signals (impressions/clicks).

**Architecture:** (1) Extend simulator to generate per-channel impressions and clicks alongside spend. (2) Rebuild the transformer so each (channel, week) pair is a separate token — permutation-invariant over channels. (3) Proper PFN training: mask random weeks and compute loss ONLY on masked weeks (eliminates averaging shortcut). (4) Replace the misleading reconstruction R² metric with proper per-component R² and per-channel accuracy metrics.

**Tech Stack:** PyTorch, NumPy, existing Demantiq simulator and training pipeline. All new code in `demantiq/neural/proper_pfn/` to keep it isolated from the current (deprecated) pfn_* modules.

---

## Context for the engineer

If you're picking this up cold, read these first in order:

1. `bitacora/01_project_overview.md` through `bitacora/11_pfn_breakthrough.md` — full context of why we're here
2. `demantiq/core/demand_kernel.py` — the current simulator
3. `demantiq/neural/pfn_model.py`, `pfn_engine.py`, `pfn_dataset.py` — the current (failing) PFN implementation that this plan replaces
4. The PFN paper: https://arxiv.org/abs/2112.10510

**Why the current implementation fails:**
- Channels are features at fixed positions in input vectors → model learns position-specific shortcuts that don't generalize
- Loss is computed on all weeks → averaging strategy works locally → model never learns to use context
- Reconstruction R² only measures whether shares sum to 1 (trivial), not whether decomposition is correct

**What this plan fixes:**
- Channel as set-token (permutation invariant, no positional shortcut possible)
- Held-out loss (only on masked weeks — averaging gets zero gradient signal)
- Per-component R² and proper metrics (can't be gamed by trivially satisfying constraints)
- Richer per-channel input (spend + impressions + clicks breaks multicollinearity)

## File structure

**New files (all under `demantiq/neural/proper_pfn/`):**
- `__init__.py` — package exports
- `model.py` — the new channel-token transformer architecture
- `dataset.py` — full-scenario dataset with proper held-out masking
- `engine.py` — training loop with loss computed on held-out tokens only
- `metrics.py` — per-component R², channel accuracy, reconstruction-free metrics
- `inference.py` — inference API (no masking, predicts all tokens)

**Modified files:**
- `demantiq/core/demand_kernel.py` — add impressions/clicks generation
- `demantiq/config/channel_config.py` — add CPM, CTR fields
- `demantiq/scenarios/scenario_sampler.py` — sample CPM and CTR parameters
- `demantiq/orchestration/training_format.py` — add impressions/clicks to batch format
- `demantiq/orchestration/training_pipeline.py` — write impressions/clicks to .npz
- `scripts/train_proper_pfn.py` — new training script

**Files NOT modified (keep current state for reference):**
- `demantiq/neural/pfn_*.py` — leave as-is; this plan doesn't touch them

---

## Task 1: Add CPM and CTR fields to ChannelConfig

**Files:**
- Modify: `demantiq/config/channel_config.py`
- Test: `tests/demantiq/config/test_channel_config.py`

- [ ] **Step 1: Read the existing ChannelConfig**

```bash
cat demantiq/config/channel_config.py
```

Familiarize yourself with the current fields (name, spend_pattern, beta, saturation_fn, adstock_fn, etc.).

- [ ] **Step 2: Write the failing test**

Create or modify `tests/demantiq/config/test_channel_config.py`:

```python
from demantiq.config.channel_config import ChannelConfig

def test_channel_config_has_cpm_and_ctr():
    """CPM and CTR should be configurable per channel with sensible defaults."""
    ch = ChannelConfig(name="facebook")
    assert hasattr(ch, "cpm")
    assert hasattr(ch, "ctr")
    assert ch.cpm > 0
    assert 0 < ch.ctr < 1

def test_channel_config_accepts_custom_cpm_ctr():
    ch = ChannelConfig(name="facebook", cpm=12.0, ctr=0.015)
    assert ch.cpm == 12.0
    assert ch.ctr == 0.015
```

- [ ] **Step 3: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/demantiq/config/test_channel_config.py -v`
Expected: FAIL with AttributeError (cpm/ctr don't exist yet).

- [ ] **Step 4: Add CPM and CTR fields**

Modify `demantiq/config/channel_config.py`. Find the `@dataclass` definition and add these fields with defaults:

```python
    # Media efficiency parameters (used for synthetic impressions/clicks generation)
    cpm: float = 10.0  # Cost per 1000 impressions in dollars
    ctr: float = 0.01  # Click-through rate (fraction of impressions that become clicks)
```

Place these after existing fields, before any `__post_init__` method.

- [ ] **Step 5: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/demantiq/config/test_channel_config.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add demantiq/config/channel_config.py tests/demantiq/config/test_channel_config.py
git commit -m "feat(config): add CPM and CTR fields to ChannelConfig"
```

---

## Task 2: Generate CPM and CTR in ScenarioSampler

**Files:**
- Modify: `demantiq/scenarios/scenario_sampler.py`
- Test: `tests/demantiq/scenarios/test_scenario_sampler_cpm_ctr.py`

- [ ] **Step 1: Write the failing test**

Create `tests/demantiq/scenarios/test_scenario_sampler_cpm_ctr.py`:

```python
from demantiq.scenarios.scenario_sampler import ScenarioSampler

def test_sampler_generates_varying_cpm():
    sampler = ScenarioSampler(seed=42, channel_range=(3, 5))
    configs = sampler.sample(10)
    all_cpms = []
    for config in configs:
        for ch in config.channels:
            all_cpms.append(ch.cpm)
    assert min(all_cpms) < max(all_cpms), "CPMs should vary across channels"
    assert all(0.5 < cpm < 100.0 for cpm in all_cpms), "CPMs should be in realistic range"

def test_sampler_generates_varying_ctr():
    sampler = ScenarioSampler(seed=42, channel_range=(3, 5))
    configs = sampler.sample(10)
    all_ctrs = []
    for config in configs:
        for ch in config.channels:
            all_ctrs.append(ch.ctr)
    assert min(all_ctrs) < max(all_ctrs), "CTRs should vary across channels"
    assert all(0.0001 < ctr < 0.2 for ctr in all_ctrs), "CTRs should be in realistic range"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/demantiq/scenarios/test_scenario_sampler_cpm_ctr.py -v`
Expected: FAIL (CPMs/CTRs are all default 10.0/0.01).

- [ ] **Step 3: Add CPM/CTR sampling in `_sample_one`**

Modify `demantiq/scenarios/scenario_sampler.py`. Find where `ChannelConfig` is constructed inside `_sample_one`. Add CPM and CTR sampling before creating the config:

```python
            # Sample CPM (cost per 1000 impressions) — varies by channel type
            # Range chosen to span realistic values: $1 CPM (email) to $50 CPM (CTV)
            cpm = float(rng.uniform(1.0, 50.0))
            # Sample CTR (click-through rate) — varies widely by channel
            # Range: 0.1% (display) to 10% (search/direct)
            ctr = float(10 ** rng.uniform(-3.0, -1.0))  # log-uniform from 0.001 to 0.1
```

Then pass these to `ChannelConfig(..., cpm=cpm, ctr=ctr)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/demantiq/scenarios/test_scenario_sampler_cpm_ctr.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add demantiq/scenarios/scenario_sampler.py tests/demantiq/scenarios/test_scenario_sampler_cpm_ctr.py
git commit -m "feat(scenarios): sample CPM and CTR per channel"
```

---

## Task 3: Generate impressions and clicks in demand_kernel

**Files:**
- Modify: `demantiq/core/demand_kernel.py`
- Test: `tests/demantiq/core/test_demand_kernel_impressions.py`

- [ ] **Step 1: Write the failing test**

Create `tests/demantiq/core/test_demand_kernel_impressions.py`:

```python
import numpy as np
from demantiq.scenarios.scenario_sampler import ScenarioSampler
from demantiq.core.demand_kernel import simulate

def test_simulate_returns_impressions_and_clicks():
    sampler = ScenarioSampler(seed=42, channel_range=(3, 5))
    config = sampler.sample(1)[0]
    result = simulate(config)
    for ch in config.channels:
        impressions_col = f"{ch.name}_impressions"
        clicks_col = f"{ch.name}_clicks"
        assert impressions_col in result.observable_data.columns, f"Missing {impressions_col}"
        assert clicks_col in result.observable_data.columns, f"Missing {clicks_col}"

def test_impressions_follow_spend_times_cpm():
    sampler = ScenarioSampler(seed=42, channel_range=(3, 5))
    config = sampler.sample(1)[0]
    result = simulate(config)
    for ch in config.channels:
        spend = result.observable_data[f"{ch.name}_spend"].values
        impressions = result.observable_data[f"{ch.name}_impressions"].values
        # impressions ≈ (spend / cpm) * 1000, within noise band
        expected_ratio = 1000.0 / ch.cpm
        nonzero = spend > 0
        if nonzero.sum() > 10:
            empirical_ratio = (impressions[nonzero] / spend[nonzero]).mean()
            # Allow 30% slack due to noise in generation
            assert 0.7 * expected_ratio < empirical_ratio < 1.3 * expected_ratio

def test_clicks_are_fraction_of_impressions():
    sampler = ScenarioSampler(seed=42, channel_range=(3, 5))
    config = sampler.sample(1)[0]
    result = simulate(config)
    for ch in config.channels:
        impressions = result.observable_data[f"{ch.name}_impressions"].values
        clicks = result.observable_data[f"{ch.name}_clicks"].values
        nonzero = impressions > 0
        if nonzero.sum() > 10:
            empirical_ctr = (clicks[nonzero] / impressions[nonzero]).mean()
            assert 0.5 * ch.ctr < empirical_ctr < 1.5 * ch.ctr
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/demantiq/core/test_demand_kernel_impressions.py -v`
Expected: FAIL (columns don't exist).

- [ ] **Step 3: Generate impressions and clicks in simulate()**

Modify `demantiq/core/demand_kernel.py`. Find the loop where per-channel contributions are computed (around line 92-115 in the current file). After channel contributions are computed and before the aggregate step, add:

```python
    # Generate synthetic impressions and clicks per channel
    # impressions_t = (spend_t / cpm) * 1000 * (1 + log_normal_noise)
    # clicks_t = impressions_t * ctr * (1 + small_noise)
    impressions_dict = {}
    clicks_dict = {}
    noise_rng = np.random.default_rng(int(config.seed) + 9999)
    for ch in config.channels:
        ch_spend = spend[ch.name]
        expected_impressions = (ch_spend / ch.cpm) * 1000.0
        imp_noise = np.exp(noise_rng.normal(0.0, 0.1, size=len(ch_spend)))  # ~10% multiplicative
        impressions_ch = expected_impressions * imp_noise
        impressions_ch = np.maximum(impressions_ch, 0.0)
        impressions_dict[ch.name] = impressions_ch

        expected_clicks = impressions_ch * ch.ctr
        clk_noise = np.exp(noise_rng.normal(0.0, 0.1, size=len(ch_spend)))
        clicks_ch = expected_clicks * clk_noise
        clicks_dict[ch.name] = np.maximum(clicks_ch, 0.0)
```

Then find `_build_observable` function and pass these dicts to it. Inside `_build_observable`, add columns:

```python
    for ch in config.channels:
        observable[f"{ch.name}_spend"] = spend[ch.name]
        observable[f"{ch.name}_impressions"] = impressions.get(ch.name, np.zeros(n))
        observable[f"{ch.name}_clicks"] = clicks.get(ch.name, np.zeros(n))
```

Update the `simulate` function's call to `_build_observable` to pass impressions and clicks.

- [ ] **Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/demantiq/core/test_demand_kernel_impressions.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add demantiq/core/demand_kernel.py tests/demantiq/core/test_demand_kernel_impressions.py
git commit -m "feat(core): generate synthetic impressions and clicks per channel"
```

---

## Task 4: Persist impressions and clicks in training pipeline

**Files:**
- Modify: `demantiq/orchestration/training_format.py`
- Modify: `demantiq/orchestration/training_pipeline.py`
- Test: `tests/demantiq/orchestration/test_training_pipeline_impressions.py`

- [ ] **Step 1: Write the failing test**

Create `tests/demantiq/orchestration/test_training_pipeline_impressions.py`:

```python
import tempfile
import numpy as np
from pathlib import Path
from demantiq.scenarios.scenario_sampler import ScenarioSampler
from demantiq.orchestration.training_pipeline import TrainingPipeline

def test_pipeline_writes_impressions_and_clicks():
    with tempfile.TemporaryDirectory() as tmpdir:
        sampler = ScenarioSampler(seed=42, rich_context=True, channel_range=(3, 5))
        pipeline = TrainingPipeline(sampler, output_dir=tmpdir, batch_size=2)
        pipeline.generate(n_total=2, n_workers=1, seed=42)

        npz_files = list(Path(tmpdir).glob("batch_*.npz"))
        assert len(npz_files) == 1
        data = np.load(str(npz_files[0]), allow_pickle=False)
        assert "impressions" in data.files
        assert "clicks" in data.files
        # Shape: (n_samples, n_periods, max_channels)
        assert data["impressions"].ndim == 3
        assert data["clicks"].ndim == 3
        assert data["impressions"].shape == data["spend"].shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/demantiq/orchestration/test_training_pipeline_impressions.py -v`
Expected: FAIL ("impressions" not in npz files).

- [ ] **Step 3: Modify training_format.py to expose impressions/clicks**

Open `demantiq/orchestration/training_format.py`. Find the function that converts a `SimulationResult` to training tensors (search for `observable_data` and `spend`).

Add helper functions:

```python
def extract_impressions_matrix(observable_data, n_periods: int, channel_names: list[str]) -> np.ndarray:
    """Extract per-channel impressions into a matrix (n_periods, n_channels)."""
    n_ch = len(channel_names)
    matrix = np.zeros((n_periods, n_ch), dtype=np.float32)
    for i, name in enumerate(channel_names):
        col = f"{name}_impressions"
        if col in observable_data.columns:
            matrix[:, i] = observable_data[col].values[:n_periods]
    return matrix


def extract_clicks_matrix(observable_data, n_periods: int, channel_names: list[str]) -> np.ndarray:
    """Extract per-channel clicks into a matrix (n_periods, n_channels)."""
    n_ch = len(channel_names)
    matrix = np.zeros((n_periods, n_ch), dtype=np.float32)
    for i, name in enumerate(channel_names):
        col = f"{name}_clicks"
        if col in observable_data.columns:
            matrix[:, i] = observable_data[col].values[:n_periods]
    return matrix
```

- [ ] **Step 4: Modify training_pipeline.py to write impressions and clicks**

Open `demantiq/orchestration/training_pipeline.py`. Find the `generate` method and the part that builds arrays from each simulation result.

Find the loop that collects `spend_matrix` and `context_matrix`. Alongside those, collect:

```python
            impressions_matrix = extract_impressions_matrix(
                result.observable_data, n_periods, channel_names
            )
            clicks_matrix = extract_clicks_matrix(
                result.observable_data, n_periods, channel_names
            )
```

Import these helpers at the top:

```python
from demantiq.orchestration.training_format import (
    extract_impressions_matrix, extract_clicks_matrix, ...
)
```

Collect them into padded batch arrays (same shape as spend_matrix), then include in the `np.savez_compressed(...)` call:

```python
np.savez_compressed(
    out_path,
    y=y_batch,
    spend=spend_batch,
    impressions=impressions_batch,
    clicks=clicks_batch,
    context=context_batch,
    decomposition=decomp_batch,
    ...
)
```

Padding logic: follow the exact same pattern used for spend_batch (pad to max_channels and max_periods).

- [ ] **Step 5: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/demantiq/orchestration/test_training_pipeline_impressions.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add demantiq/orchestration/training_format.py demantiq/orchestration/training_pipeline.py tests/demantiq/orchestration/test_training_pipeline_impressions.py
git commit -m "feat(orchestration): persist impressions and clicks in .npz batches"
```

---

## Task 5: Load impressions and clicks in DemantiqDataset

**Files:**
- Modify: `demantiq/neural/data_loader.py`
- Test: `tests/demantiq/neural/test_data_loader_impressions.py`

- [ ] **Step 1: Write the failing test**

Create `tests/demantiq/neural/test_data_loader_impressions.py`:

```python
import tempfile
import numpy as np
import torch
from demantiq.scenarios.scenario_sampler import ScenarioSampler
from demantiq.orchestration.training_pipeline import TrainingPipeline
from demantiq.neural.data_loader import DemantiqDataset

def test_dataset_exposes_impressions_and_clicks():
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        sampler = ScenarioSampler(seed=42, rich_context=True, channel_range=(3, 5))
        pipeline = TrainingPipeline(sampler, output_dir=tmpdir, batch_size=3)
        pipeline.generate(n_total=3, n_workers=1, seed=42)

        dataset = DemantiqDataset(tmpdir)
        item = dataset[0]
        assert "impressions" in item
        assert "clicks" in item
        assert item["impressions"].shape == item["spend"].shape
        assert item["clicks"].shape == item["spend"].shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/demantiq/neural/test_data_loader_impressions.py -v`
Expected: FAIL (impressions not in item dict).

- [ ] **Step 3: Update DemantiqDataset to load impressions and clicks**

Open `demantiq/neural/data_loader.py`. Find `_load_all_batches` and the `__getitem__` method.

In `_load_all_batches`, add lists:

```python
        all_impressions = []
        all_clicks = []
```

In the per-batch loop, add:

```python
            if "impressions" in data:
                all_impressions.append(data["impressions"][:n_in_batch])
            else:
                all_impressions.append(np.zeros_like(data["spend"][:n_in_batch]))
            if "clicks" in data:
                all_clicks.append(data["clicks"][:n_in_batch])
            else:
                all_clicks.append(np.zeros_like(data["spend"][:n_in_batch]))
```

In the padding section, pad impressions and clicks the same way as spend:

```python
        imp_arrays = []
        clk_arrays = []
        for imp_arr, clk_arr, sp_arr in zip(all_impressions, all_clicks, all_spend):
            pad_t = max_t - imp_arr.shape[1]
            pad_c = max_c - imp_arr.shape[2]
            if pad_t > 0 or pad_c > 0:
                imp_arr = np.pad(imp_arr, ((0, 0), (0, pad_t), (0, pad_c)))
                clk_arr = np.pad(clk_arr, ((0, 0), (0, pad_t), (0, pad_c)))
            imp_arrays.append(imp_arr)
            clk_arrays.append(clk_arr)

        self.impressions = np.concatenate(imp_arrays, axis=0).astype(np.float32)
        self.clicks = np.concatenate(clk_arrays, axis=0).astype(np.float32)
```

In `__getitem__`, add:

```python
            "impressions": torch.from_numpy(self.impressions[idx]),
            "clicks": torch.from_numpy(self.clicks[idx]),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/demantiq/neural/test_data_loader_impressions.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add demantiq/neural/data_loader.py tests/demantiq/neural/test_data_loader_impressions.py
git commit -m "feat(neural): expose impressions and clicks in DemantiqDataset"
```

---

## Task 6: Create proper_pfn package structure

**Files:**
- Create: `demantiq/neural/proper_pfn/__init__.py`

- [ ] **Step 1: Create the package file**

```bash
mkdir -p demantiq/neural/proper_pfn
cat > demantiq/neural/proper_pfn/__init__.py << 'EOF'
"""Proper PFN-based MMM decomposition.

Replaces the earlier pfn_* modules with an architecture that:
- Treats each (channel, week) pair as a permutation-invariant token
- Uses held-out loss (only masked tokens contribute to gradient)
- Predicts per-component shares of y per timestep
- Includes per-channel impressions and clicks as input features
"""

from demantiq.neural.proper_pfn.model import ProperPFNModel
from demantiq.neural.proper_pfn.dataset import ProperPFNDataset, proper_pfn_collate_fn
from demantiq.neural.proper_pfn.engine import ProperPFNEngine, ProperPFNConfig

__all__ = [
    "ProperPFNModel",
    "ProperPFNDataset",
    "proper_pfn_collate_fn",
    "ProperPFNEngine",
    "ProperPFNConfig",
]
EOF
```

- [ ] **Step 2: Commit**

```bash
git add demantiq/neural/proper_pfn/__init__.py
git commit -m "feat(neural): create proper_pfn package"
```

---

## Task 7: Build ProperPFNDataset with channel-token structure and held-out masking

**Files:**
- Create: `demantiq/neural/proper_pfn/dataset.py`
- Test: `tests/demantiq/neural/proper_pfn/test_dataset.py`

- [ ] **Step 1: Create the test file directory**

```bash
mkdir -p tests/demantiq/neural/proper_pfn
touch tests/demantiq/neural/proper_pfn/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/demantiq/neural/proper_pfn/test_dataset.py`:

```python
import tempfile
import torch
from demantiq.scenarios.scenario_sampler import ScenarioSampler
from demantiq.orchestration.training_pipeline import TrainingPipeline
from demantiq.neural.data_loader import DemantiqDataset
from demantiq.neural.proper_pfn.dataset import ProperPFNDataset, proper_pfn_collate_fn


def _make_dataset(n_scenarios: int = 3, mask_fraction: float = 0.3, training: bool = True):
    tmpdir = tempfile.mkdtemp()
    sampler = ScenarioSampler(seed=42, rich_context=True, channel_range=(3, 5))
    pipeline = TrainingPipeline(sampler, output_dir=tmpdir, batch_size=n_scenarios)
    pipeline.generate(n_total=n_scenarios, n_workers=1, seed=42)
    backing = DemantiqDataset(tmpdir)
    return ProperPFNDataset(
        backing, max_channels=8, week_mask_fraction=mask_fraction, training=training
    )


def test_dataset_returns_channel_tokens():
    ds = _make_dataset()
    item = ds[0]
    # Channel tokens: (T, C_max, channel_feature_dim)
    assert "channel_tokens" in item
    assert item["channel_tokens"].ndim == 3
    # Global tokens (y, context per week): (T, global_feature_dim)
    assert "global_tokens" in item
    assert item["global_tokens"].ndim == 2
    # Number of active channels
    assert "n_channels" in item


def test_dataset_produces_masked_flags():
    ds = _make_dataset(mask_fraction=0.3, training=True)
    item = ds[0]
    # Per-week mask indicator
    assert "week_is_masked" in item
    assert item["week_is_masked"].ndim == 1
    n_masked = item["week_is_masked"].sum().item()
    T = item["week_is_masked"].shape[0]
    # Roughly 30% masked (allow slack for rounding on short sequences)
    assert n_masked >= 1
    assert n_masked <= T


def test_dataset_target_is_shares():
    ds = _make_dataset()
    item = ds[0]
    # Targets: per-week shares for each component
    # Shape: (T, max_channels + 2) — channels + baseline + non_media
    assert "target" in item
    assert item["target"].shape[1] == ds.max_channels + 2


def test_collate_pads_variable_lengths():
    ds1 = _make_dataset(n_scenarios=3)
    items = [ds1[0], ds1[1], ds1[2]]
    batch = proper_pfn_collate_fn(items)
    # All items should have the same T dimension after padding
    assert batch["channel_tokens"].shape[0] == 3
    assert batch["global_tokens"].shape[0] == 3
    assert batch["week_is_masked"].shape[0] == 3
    assert batch["target"].shape[0] == 3
    # time_pad_mask marks padded positions
    assert "time_pad_mask" in batch
    assert batch["time_pad_mask"].dtype == torch.bool
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `source venv/bin/activate && pytest tests/demantiq/neural/proper_pfn/test_dataset.py -v`
Expected: FAIL (dataset module doesn't exist).

- [ ] **Step 4: Implement ProperPFNDataset**

Create `demantiq/neural/proper_pfn/dataset.py`:

```python
"""Full-scenario dataset for the proper PFN architecture.

Produces:
  - channel_tokens: (T, max_channels, channel_feature_dim)
      Per-channel per-week features: [spend, impressions, clicks] normalized.
  - global_tokens: (T, global_feature_dim)
      Per-week features shared across channels: [y, context, presence_flags, time].
  - week_is_masked: (T,) float
      1.0 for held-out weeks during training, 0.0 otherwise.
  - channel_pad_mask: (max_channels,) bool
      True for padded (inactive) channels.
  - target: (T, max_channels + 2)
      Per-week shares: [ch1_share, ..., chC_share, baseline_share, non_media_share].

The masking zeros input observations for masked weeks (model can only use
time features and other weeks' observations to infer decomposition).
"""

from __future__ import annotations

import math
import numpy as np
import torch
from torch.utils.data import Dataset

from demantiq.neural.data_loader import DemantiqDataset
from demantiq.orchestration.training_format import (
    DECOMP_IDX_BASELINE,
    DECOMP_IDX_CHANNELS_START,
    DECOMP_IDX_PRICE,
    DECOMP_IDX_COMPETITION,
    DECOMP_IDX_MACRO,
    MAX_CONTEXT_COLS,
)


CHANNEL_FEATURE_DIM = 3  # spend, impressions, clicks (all normalized)


class ProperPFNDataset(Dataset):
    """Per-scenario dataset with channel-token structure and held-out masking."""

    def __init__(
        self,
        backing_dataset: DemantiqDataset,
        max_channels: int = 8,
        n_context_dims: int = MAX_CONTEXT_COLS,
        week_mask_fraction: float = 0.3,
        training: bool = True,
    ) -> None:
        self.backing = backing_dataset
        self.max_channels = max_channels
        self.n_context_dims = n_context_dims
        self.week_mask_fraction = week_mask_fraction
        self.training = training

        self._valid_indices = self._filter()

    def _filter(self) -> list[int]:
        valid: list[int] = []
        for i in range(len(self.backing.n_periods)):
            n_ch = len(self.backing.channel_names[i])
            if 2 <= n_ch <= self.max_channels:
                valid.append(i)
        return valid

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> dict:
        scenario_idx = self._valid_indices[idx]
        T = int(self.backing.n_periods[scenario_idx])
        n_ch = len(self.backing.channel_names[scenario_idx])

        # --- y and y_denom for share computation ---
        y_raw = self.backing.y[scenario_idx, :T]  # numpy
        y_scale = float(max(np.mean(np.abs(y_raw)), 1.0))
        y_denom = np.maximum(np.abs(y_raw), max(y_scale * 0.1, 1.0))

        # --- Channel tokens: (T, max_channels, 3) with [spend, impressions, clicks] ---
        spend_raw = self.backing.spend[scenario_idx, :T, :n_ch]
        imp_raw = self.backing.impressions[scenario_idx, :T, :n_ch]
        clk_raw = self.backing.clicks[scenario_idx, :T, :n_ch]

        # Per-channel normalization: divide each by its own max-abs over time.
        # This keeps small-spend channels visible while log-magnitude features
        # handle relative scale (not needed here — scale info is redundant with
        # the target being a share of y).
        def per_channel_normalize(arr: np.ndarray) -> np.ndarray:
            # arr shape: (T, n_ch)
            scales = np.maximum(np.abs(arr).max(axis=0, keepdims=True), 1.0)
            return arr / scales

        spend_norm = per_channel_normalize(spend_raw)
        imp_norm = per_channel_normalize(imp_raw)
        clk_norm = per_channel_normalize(clk_raw)

        channel_tokens = np.zeros((T, self.max_channels, CHANNEL_FEATURE_DIM), dtype=np.float32)
        channel_tokens[:, :n_ch, 0] = spend_norm
        channel_tokens[:, :n_ch, 1] = imp_norm
        channel_tokens[:, :n_ch, 2] = clk_norm

        # --- Context ---
        ctx_raw = self.backing.context[scenario_idx, :T, :].copy()  # (T, 10)
        context_present = (np.abs(ctx_raw).sum(axis=0) > 0).astype(np.float32)  # (10,)
        # Normalize each context column by its own max-abs
        col_scales = np.maximum(np.abs(ctx_raw).max(axis=0, keepdims=True), 1.0)
        ctx_norm = ctx_raw / col_scales

        # Normalize y by y_scale for input (but target uses y_denom per timestep)
        y_norm = (y_raw / y_scale).astype(np.float32)

        # --- Temporal features ---
        time_index = np.linspace(0.0, 1.0, T, dtype=np.float32)
        week_of_year = np.arange(T, dtype=np.float32) % 52.0
        sin_week = np.sin(2.0 * np.pi * week_of_year / 52.0)
        cos_week = np.cos(2.0 * np.pi * week_of_year / 52.0)

        # --- Week masking (training only) ---
        week_is_masked = np.zeros(T, dtype=np.float32)
        if self.training and self.week_mask_fraction > 0.0:
            n_mask = max(1, int(math.ceil(T * self.week_mask_fraction)))
            idxs = np.random.permutation(T)[:n_mask]
            week_is_masked[idxs] = 1.0

            # Zero out input observations for masked weeks (spend, impressions,
            # clicks, y, context). Time features remain visible.
            visible = 1.0 - week_is_masked
            channel_tokens = channel_tokens * visible[:, None, None]
            y_norm = y_norm * visible
            ctx_norm = ctx_norm * visible[:, None]
            # Keep context_present unchanged — it's a scenario-level flag

        # --- Assemble global tokens: (T, global_feature_dim) ---
        # [y, ctx (10), presence_flags (10), time_index, sin_week, cos_week, is_masked]
        presence_flags = np.tile(context_present, (T, 1))  # (T, 10)
        global_tokens = np.concatenate(
            [
                y_norm.reshape(T, 1),
                ctx_norm.astype(np.float32),
                presence_flags,
                time_index.reshape(T, 1),
                sin_week.reshape(T, 1),
                cos_week.reshape(T, 1),
                week_is_masked.reshape(T, 1),
            ],
            axis=1,
        )  # (T, 1 + 10 + 10 + 3 + 1 = 25)

        # --- Target: shares per timestep ---
        ch_start = DECOMP_IDX_CHANNELS_START
        ch_shares = np.zeros((T, self.max_channels), dtype=np.float32)
        ch_shares[:, :n_ch] = (
            self.backing.decomposition[scenario_idx, :T, ch_start : ch_start + n_ch]
            / y_denom[:, None]
        )

        base_share = (
            self.backing.decomposition[scenario_idx, :T, DECOMP_IDX_BASELINE] / y_denom
        ).astype(np.float32)

        nm_share = (
            (
                self.backing.decomposition[scenario_idx, :T, DECOMP_IDX_PRICE]
                + self.backing.decomposition[scenario_idx, :T, DECOMP_IDX_COMPETITION]
                + self.backing.decomposition[scenario_idx, :T, DECOMP_IDX_MACRO]
            )
            / y_denom
        ).astype(np.float32)

        target = np.concatenate(
            [ch_shares, base_share.reshape(T, 1), nm_share.reshape(T, 1)],
            axis=1,
        )  # (T, max_channels + 2)

        # --- Channel pad mask ---
        channel_pad_mask = np.ones(self.max_channels, dtype=bool)
        channel_pad_mask[:n_ch] = False

        return {
            "channel_tokens": torch.from_numpy(channel_tokens),
            "global_tokens": torch.from_numpy(global_tokens),
            "week_is_masked": torch.from_numpy(week_is_masked),
            "target": torch.from_numpy(target),
            "y_raw": torch.from_numpy(y_raw.astype(np.float32)),
            "y_denom": torch.from_numpy(y_denom.astype(np.float32)),
            "y_scale": torch.tensor(y_scale, dtype=torch.float32),
            "n_channels": torch.tensor(n_ch, dtype=torch.int32),
            "n_periods": torch.tensor(T, dtype=torch.int32),
            "channel_pad_mask": torch.from_numpy(channel_pad_mask),
        }


def proper_pfn_collate_fn(batch: list[dict]) -> dict:
    """Pad variable-length scenarios to batch max."""
    max_t = max(item["channel_tokens"].shape[0] for item in batch)
    b = len(batch)
    max_channels = batch[0]["channel_tokens"].shape[1]
    channel_feat = batch[0]["channel_tokens"].shape[2]
    global_feat = batch[0]["global_tokens"].shape[1]
    target_feat = batch[0]["target"].shape[1]

    channel_tokens = torch.zeros(b, max_t, max_channels, channel_feat, dtype=torch.float32)
    global_tokens = torch.zeros(b, max_t, global_feat, dtype=torch.float32)
    target = torch.zeros(b, max_t, target_feat, dtype=torch.float32)
    week_is_masked = torch.zeros(b, max_t, dtype=torch.float32)
    time_pad_mask = torch.ones(b, max_t, dtype=torch.bool)
    y_raw = torch.zeros(b, max_t, dtype=torch.float32)
    y_denom = torch.ones(b, max_t, dtype=torch.float32)
    y_scale = torch.zeros(b, dtype=torch.float32)
    n_channels = torch.zeros(b, dtype=torch.int32)
    n_periods = torch.zeros(b, dtype=torch.int32)
    channel_pad_mask = torch.zeros(b, max_channels, dtype=torch.bool)

    for i, item in enumerate(batch):
        t = item["channel_tokens"].shape[0]
        channel_tokens[i, :t] = item["channel_tokens"]
        global_tokens[i, :t] = item["global_tokens"]
        target[i, :t] = item["target"]
        week_is_masked[i, :t] = item["week_is_masked"]
        time_pad_mask[i, :t] = False  # valid
        y_raw[i, :t] = item["y_raw"]
        y_denom[i, :t] = item["y_denom"]
        y_scale[i] = item["y_scale"]
        n_channels[i] = item["n_channels"]
        n_periods[i] = item["n_periods"]
        channel_pad_mask[i] = item["channel_pad_mask"]

    return {
        "channel_tokens": channel_tokens,
        "global_tokens": global_tokens,
        "target": target,
        "week_is_masked": week_is_masked,
        "time_pad_mask": time_pad_mask,
        "y_raw": y_raw,
        "y_denom": y_denom,
        "y_scale": y_scale,
        "n_channels": n_channels,
        "n_periods": n_periods,
        "channel_pad_mask": channel_pad_mask,
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `source venv/bin/activate && pytest tests/demantiq/neural/proper_pfn/test_dataset.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add demantiq/neural/proper_pfn/dataset.py tests/demantiq/neural/proper_pfn/
git commit -m "feat(proper_pfn): channel-token dataset with held-out masking"
```

---

## Task 8: Build ProperPFNModel with channel-token attention

**Files:**
- Create: `demantiq/neural/proper_pfn/model.py`
- Test: `tests/demantiq/neural/proper_pfn/test_model.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/demantiq/neural/proper_pfn/test_model.py`:

```python
import torch
from demantiq.neural.proper_pfn.model import ProperPFNModel


def _dummy_batch(B=2, T=32, C=8, channel_feat=3, global_feat=25):
    channel_tokens = torch.randn(B, T, C, channel_feat)
    global_tokens = torch.randn(B, T, global_feat)
    channel_pad_mask = torch.zeros(B, C, dtype=torch.bool)
    channel_pad_mask[:, 5:] = True  # 5 active channels
    time_pad_mask = torch.zeros(B, T, dtype=torch.bool)
    return channel_tokens, global_tokens, channel_pad_mask, time_pad_mask


def test_model_forward_output_shapes():
    model = ProperPFNModel(
        channel_feat_dim=3, global_feat_dim=25, max_channels=8,
        d_model=64, n_heads=4, n_layers=2,
    )
    ch_tok, glob_tok, ch_mask, t_mask = _dummy_batch()
    out = model(ch_tok, glob_tok, ch_mask, t_mask)
    assert out["channel_shares"].shape == (2, 32, 8)
    assert out["baseline_share"].shape == (2, 32)
    assert out["non_media_share"].shape == (2, 32)


def test_model_is_permutation_invariant_over_channels():
    """Swapping active channel positions should swap outputs identically."""
    torch.manual_seed(0)
    model = ProperPFNModel(
        channel_feat_dim=3, global_feat_dim=25, max_channels=8,
        d_model=64, n_heads=4, n_layers=2,
    )
    model.eval()
    ch_tok, glob_tok, ch_mask, t_mask = _dummy_batch(B=1)
    # Keep 5 active channels (idx 0..4), pad 5..7
    with torch.no_grad():
        out_original = model(ch_tok, glob_tok, ch_mask, t_mask)

    # Swap channel 0 and channel 2 in input
    perm = torch.arange(8)
    perm[0], perm[2] = 2, 0
    ch_tok_swapped = ch_tok[:, :, perm, :]
    with torch.no_grad():
        out_swapped = model(ch_tok_swapped, glob_tok, ch_mask, t_mask)

    # Channel 0 output in original should equal channel 2 output in swapped
    assert torch.allclose(
        out_original["channel_shares"][:, :, 0],
        out_swapped["channel_shares"][:, :, 2],
        atol=1e-5,
    )
    assert torch.allclose(
        out_original["channel_shares"][:, :, 2],
        out_swapped["channel_shares"][:, :, 0],
        atol=1e-5,
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source venv/bin/activate && pytest tests/demantiq/neural/proper_pfn/test_model.py -v`
Expected: FAIL (module doesn't exist).

- [ ] **Step 3: Implement ProperPFNModel**

Create `demantiq/neural/proper_pfn/model.py`:

```python
"""Channel-token transformer for proper PFN MMM decomposition.

Architecture:
  1. Project channel and global features to d_model.
  2. Build a sequence of tokens: [global_week_1, ch_1_week_1, ch_2_week_1, ..., global_week_2, ...]
     Actually: process as two streams — per-week global attends to all channels
     at that week (and to other weeks' globals/channels via cross-attention).
  3. Channel stream is permutation-invariant (no channel-positional encoding).
  4. Temporal encoding added to both streams for time ordering.
  5. Output per channel per week (shares) + per week (baseline, non_media).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ProperPFNModel(nn.Module):
    """Transformer with channel-token permutation invariance.

    Sequence layout for attention: tokens = concat of
      - Global tokens (one per week): encode y, context, time — index [0 .. T-1]
      - Channel tokens (one per (channel, week)): index [T .. T + T*max_channels - 1]

    Total sequence length: T + T * max_channels
    For T=200, max_channels=8: 200 + 1600 = 1800 tokens.

    Channel tokens have NO positional encoding across the channel dimension
    (only the time dimension), ensuring permutation invariance over channels.
    """

    def __init__(
        self,
        channel_feat_dim: int,
        global_feat_dim: int,
        max_channels: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 300,  # max weeks per scenario
    ) -> None:
        super().__init__()
        self.max_channels = max_channels
        self.d_model = d_model

        self.channel_proj = nn.Linear(channel_feat_dim, d_model)
        self.global_proj = nn.Linear(global_feat_dim, d_model)

        # Temporal encoding — shared between global and channel streams at same week.
        self.temporal_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # Learned type embeddings: one for global-week tokens, one for channel tokens.
        # These are added to token embeddings so the model can distinguish their roles.
        self.global_type_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.channel_type_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Separate output heads
        self.channel_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.baseline_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.non_media_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        channel_tokens: Tensor,   # (B, T, C, channel_feat)
        global_tokens: Tensor,    # (B, T, global_feat)
        channel_pad_mask: Tensor, # (B, C) True=padded
        time_pad_mask: Tensor,    # (B, T) True=padded
    ) -> dict[str, Tensor]:
        B, T, C, _ = channel_tokens.shape

        # --- Project to d_model ---
        ch_emb = self.channel_proj(channel_tokens)  # (B, T, C, d)
        gl_emb = self.global_proj(global_tokens)    # (B, T, d)

        # --- Add temporal encoding (shared across channels at same week) ---
        temp = self.temporal_encoding[:, :T, :]  # (1, T, d)
        ch_emb = ch_emb + temp.unsqueeze(2)      # (B, T, C, d)
        gl_emb = gl_emb + temp                   # (B, T, d)

        # --- Add type embeddings (distinguishes global from channel tokens) ---
        ch_emb = ch_emb + self.channel_type_emb  # broadcast
        gl_emb = gl_emb + self.global_type_emb

        # --- Flatten to a single sequence ---
        # Order: [global_1, ..., global_T, ch_1_1, ch_2_1, ..., ch_C_1, ch_1_2, ...]
        # Reshape channel tokens to (B, T*C, d)
        ch_seq = ch_emb.reshape(B, T * C, self.d_model)
        seq = torch.cat([gl_emb, ch_seq], dim=1)  # (B, T + T*C, d)

        # --- Build attention padding mask ---
        # Global tokens: mask if week is padded.
        # Channel tokens: mask if week is padded OR channel is padded.
        global_mask_flat = time_pad_mask  # (B, T)

        # Channel mask: (B, T, C) True=padded
        # A channel-week is padded if its week is padded OR its channel is padded.
        channel_mask_2d = time_pad_mask.unsqueeze(-1) | channel_pad_mask.unsqueeze(1)  # (B, T, C)
        channel_mask_flat = channel_mask_2d.reshape(B, T * C)

        attn_mask = torch.cat([global_mask_flat, channel_mask_flat], dim=1)  # (B, T + T*C)

        # --- Transformer ---
        out = self.transformer(seq, src_key_padding_mask=attn_mask)  # (B, T + T*C, d)

        # --- Split back ---
        out_global = out[:, :T, :]                        # (B, T, d)
        out_channel = out[:, T:, :].reshape(B, T, C, self.d_model)  # (B, T, C, d)

        # --- Heads ---
        channel_shares = self.channel_head(out_channel).squeeze(-1)  # (B, T, C)
        baseline_share = self.baseline_head(out_global).squeeze(-1)  # (B, T)
        non_media_share = self.non_media_head(out_global).squeeze(-1)  # (B, T)

        # Zero out padded channels in output
        active_mask = (~channel_pad_mask).float()  # (B, C)
        channel_shares = channel_shares * active_mask.unsqueeze(1)

        return {
            "channel_shares": channel_shares,
            "baseline_share": baseline_share,
            "non_media_share": non_media_share,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source venv/bin/activate && pytest tests/demantiq/neural/proper_pfn/test_model.py -v`
Expected: PASS. If permutation invariance test fails, the issue is almost certainly that you added a channel-positional encoding — remove it.

- [ ] **Step 5: Commit**

```bash
git add demantiq/neural/proper_pfn/model.py tests/demantiq/neural/proper_pfn/test_model.py
git commit -m "feat(proper_pfn): channel-token transformer with permutation invariance"
```

---

## Task 9: Build proper held-out loss function

**Files:**
- Create: `demantiq/neural/proper_pfn/loss.py`
- Test: `tests/demantiq/neural/proper_pfn/test_loss.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/demantiq/neural/proper_pfn/test_loss.py`:

```python
import torch
from demantiq.neural.proper_pfn.loss import held_out_loss


def test_loss_ignores_visible_weeks_completely():
    """Loss on visible weeks should have ZERO gradient contribution."""
    B, T, C = 2, 10, 4
    pred = {
        "channel_shares": torch.randn(B, T, C, requires_grad=True),
        "baseline_share": torch.randn(B, T, requires_grad=True),
        "non_media_share": torch.randn(B, T, requires_grad=True),
    }
    target = torch.randn(B, T, C + 2)
    time_pad = torch.zeros(B, T, dtype=torch.bool)
    ch_pad = torch.zeros(B, C, dtype=torch.bool)

    # All visible (no held-out): loss should be zero
    week_masked = torch.zeros(B, T)
    total, _ = held_out_loss(pred, target, time_pad, ch_pad, week_masked, C)
    # Loss is computed only on held-out weeks. With zero held-out, loss is zero.
    assert total.item() == 0.0


def test_loss_nonzero_when_masked_weeks_exist():
    B, T, C = 2, 10, 4
    pred = {
        "channel_shares": torch.zeros(B, T, C),
        "baseline_share": torch.zeros(B, T),
        "non_media_share": torch.zeros(B, T),
    }
    target = torch.ones(B, T, C + 2)
    time_pad = torch.zeros(B, T, dtype=torch.bool)
    ch_pad = torch.zeros(B, C, dtype=torch.bool)
    week_masked = torch.zeros(B, T)
    week_masked[:, :3] = 1.0  # mask first 3 weeks

    total, comp = held_out_loss(pred, target, time_pad, ch_pad, week_masked, C)
    # pred=0, target=1, squared error = 1, averaged = 1
    assert total.item() > 0.0
    assert "channels_masked" in comp
    assert "baseline_masked" in comp


def test_loss_computes_only_on_held_out():
    B, T, C = 1, 10, 4
    # perfect predictions on visible weeks, zero on masked weeks
    week_masked = torch.zeros(B, T)
    week_masked[0, :3] = 1.0

    pred = {
        "channel_shares": torch.ones(B, T, C),
        "baseline_share": torch.ones(B, T),
        "non_media_share": torch.ones(B, T),
    }
    target = torch.ones(B, T, C + 2)
    # Now set predictions on held-out weeks to wrong value
    pred["channel_shares"][0, :3, :] = 0.0
    pred["baseline_share"][0, :3] = 0.0
    pred["non_media_share"][0, :3] = 0.0

    time_pad = torch.zeros(B, T, dtype=torch.bool)
    ch_pad = torch.zeros(B, C, dtype=torch.bool)

    total, _ = held_out_loss(pred, target, time_pad, ch_pad, week_masked, C)
    # Only held-out weeks contribute. Error = (1-0)^2 = 1 on masked.
    assert total.item() > 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source venv/bin/activate && pytest tests/demantiq/neural/proper_pfn/test_loss.py -v`
Expected: FAIL (module doesn't exist).

- [ ] **Step 3: Implement the loss**

Create `demantiq/neural/proper_pfn/loss.py`:

```python
"""Held-out loss for proper PFN training.

Critical design: loss is computed ONLY on held-out (masked) weeks. Visible
weeks contribute zero gradient. This eliminates the averaging shortcut
because the model must use visible-week context to predict held-out weeks.
"""

from __future__ import annotations

import torch


def held_out_loss(
    pred: dict,
    target: torch.Tensor,
    time_pad_mask: torch.Tensor,
    channel_pad_mask: torch.Tensor,
    week_is_masked: torch.Tensor,
    max_channels: int,
) -> tuple[torch.Tensor, dict]:
    """Compute MSE loss ONLY on held-out (masked) weeks.

    Args:
        pred: dict with channel_shares (B,T,C), baseline_share (B,T), non_media_share (B,T).
        target: (B, T, max_channels + 2). Last two are [baseline, non_media].
        time_pad_mask: (B, T) bool, True=padded week.
        channel_pad_mask: (B, C) bool, True=padded channel.
        week_is_masked: (B, T) float, 1.0=held-out, 0.0=visible.
        max_channels: int, number of channel slots.

    Returns:
        total_loss (scalar), components (dict of floats).
    """
    ch_active = (~channel_pad_mask).float()  # (B, C)
    time_valid = (~time_pad_mask).float()    # (B, T)

    # Held-out weight: 1.0 on held-out valid weeks, 0.0 elsewhere
    weight = time_valid * week_is_masked  # (B, T)
    n_valid = weight.sum().clamp(min=1.0)

    pred_ch = pred["channel_shares"]
    pred_base = pred["baseline_share"]
    pred_nm = pred["non_media_share"]

    true_ch = target[:, :, :max_channels]
    true_base = target[:, :, max_channels]
    true_nm = target[:, :, max_channels + 1]

    # Channel loss: MSE over active channels on held-out weeks
    ch_err_sq = (pred_ch - true_ch) ** 2  # (B, T, C)
    ch_err_sq = ch_err_sq * ch_active.unsqueeze(1) * weight.unsqueeze(-1)
    denom_ch = (ch_active.sum(dim=1).unsqueeze(1) * weight).sum().clamp(min=1.0)
    l_ch = ch_err_sq.sum() / denom_ch

    # Baseline loss (held-out only)
    l_base = ((pred_base - true_base) ** 2 * weight).sum() / n_valid

    # Non-media loss (held-out only)
    l_nm = ((pred_nm - true_nm) ** 2 * weight).sum() / n_valid

    # Reconstruction: shares should sum to ~1 on held-out weeks
    pred_sum = pred_base + (pred_ch * ch_active.unsqueeze(1)).sum(dim=-1) + pred_nm
    target_sum = torch.ones_like(pred_sum)
    l_recon = ((pred_sum - target_sum) ** 2 * weight).sum() / n_valid

    total = l_ch + l_base + l_nm + l_recon

    return total, {
        "channels_masked": l_ch.item(),
        "baseline_masked": l_base.item(),
        "non_media_masked": l_nm.item(),
        "reconstruction_masked": l_recon.item(),
        "total": total.item(),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source venv/bin/activate && pytest tests/demantiq/neural/proper_pfn/test_loss.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add demantiq/neural/proper_pfn/loss.py tests/demantiq/neural/proper_pfn/test_loss.py
git commit -m "feat(proper_pfn): held-out loss computes only on masked weeks"
```

---

## Task 10: Build proper metrics (replace misleading reconstruction R²)

**Files:**
- Create: `demantiq/neural/proper_pfn/metrics.py`
- Test: `tests/demantiq/neural/proper_pfn/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/demantiq/neural/proper_pfn/test_metrics.py`:

```python
import numpy as np
from demantiq.neural.proper_pfn.metrics import (
    per_component_r2,
    channel_rank_correlation,
    scenario_level_metrics,
)


def test_per_component_r2_perfect_prediction():
    true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(per_component_r2(true, pred) - 1.0) < 1e-6


def test_per_component_r2_worse_than_mean():
    # Predicting constant mean gives R² = 0
    true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = np.full_like(true, true.mean())
    assert abs(per_component_r2(true, pred)) < 1e-6


def test_per_component_r2_negative_for_bad_prediction():
    true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # reversed
    assert per_component_r2(true, pred) < 0


def test_channel_rank_correlation_perfect():
    true_totals = np.array([100.0, 50.0, 25.0, 10.0])
    pred_totals = np.array([99.0, 49.0, 24.0, 9.0])  # same order
    corr = channel_rank_correlation(true_totals, pred_totals)
    assert corr > 0.99


def test_channel_rank_correlation_reversed():
    true_totals = np.array([100.0, 50.0, 25.0, 10.0])
    pred_totals = np.array([10.0, 25.0, 50.0, 100.0])  # reversed
    corr = channel_rank_correlation(true_totals, pred_totals)
    assert corr < -0.99


def test_scenario_level_metrics_structure():
    T = 50
    true_channels = np.random.rand(T, 3) * 10
    pred_channels = true_channels + np.random.randn(T, 3) * 0.1
    true_baseline = np.random.rand(T) * 100
    pred_baseline = true_baseline + np.random.randn(T) * 0.5
    y = true_baseline + true_channels.sum(axis=1)

    metrics = scenario_level_metrics(
        pred_channels, pred_baseline,
        true_channels, true_baseline, y,
    )
    assert "channel_r2_mean" in metrics
    assert "baseline_r2" in metrics
    assert "rank_corr" in metrics
    assert "per_channel_error_pct" in metrics
    # Near-perfect predictions should give high R²
    assert metrics["channel_r2_mean"] > 0.95
    assert metrics["baseline_r2"] > 0.95
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source venv/bin/activate && pytest tests/demantiq/neural/proper_pfn/test_metrics.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement metrics**

Create `demantiq/neural/proper_pfn/metrics.py`:

```python
"""Metrics for proper PFN evaluation.

Replaces the misleading "reconstruction R²" (which only measured whether
shares sum to 1) with metrics that actually measure decomposition quality.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def per_component_r2(true: np.ndarray, pred: np.ndarray) -> float:
    """R² of predicted time series against true time series.

    This is the key metric for decomposition quality: does the predicted
    component track the true component over time? Cannot be gamed by
    satisfying sum-to-1 constraints.
    """
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    if ss_tot < 1e-10:
        return 0.0
    return 1.0 - ss_res / ss_tot


def channel_rank_correlation(true_totals: np.ndarray, pred_totals: np.ndarray) -> float:
    """Spearman rank correlation of per-channel total contributions."""
    if len(true_totals) < 2:
        return 0.0
    if len(true_totals) == 2:
        # Spearman on 2 items is ±1 — use Pearson instead
        c = float(np.corrcoef(true_totals, pred_totals)[0, 1])
        if np.isnan(c):
            return 0.0
        return c
    corr, _ = spearmanr(true_totals, pred_totals)
    if np.isnan(corr):
        return 0.0
    return float(corr)


def scenario_level_metrics(
    pred_channels: np.ndarray,   # (T, n_ch) absolute contributions
    pred_baseline: np.ndarray,   # (T,) absolute baseline
    true_channels: np.ndarray,   # (T, n_ch)
    true_baseline: np.ndarray,   # (T,)
    y: np.ndarray,               # (T,)
) -> dict:
    """Compute all scenario-level evaluation metrics."""
    n_ch = pred_channels.shape[1]

    # Per-channel R² (time series accuracy)
    ch_r2s = []
    for i in range(n_ch):
        if np.std(true_channels[:, i]) > 1e-10:
            ch_r2s.append(per_component_r2(true_channels[:, i], pred_channels[:, i]))
    channel_r2_mean = float(np.mean(ch_r2s)) if ch_r2s else 0.0

    # Baseline R²
    baseline_r2 = per_component_r2(true_baseline, pred_baseline)

    # Rank correlation of per-channel totals
    true_totals = true_channels.sum(axis=0)
    pred_totals = pred_channels.sum(axis=0)
    rank_corr = channel_rank_correlation(true_totals, pred_totals)

    # Per-channel total contribution error (percent)
    per_ch_err = []
    for i in range(n_ch):
        if abs(true_totals[i]) > 1e-6:
            err = abs(pred_totals[i] - true_totals[i]) / abs(true_totals[i]) * 100.0
            per_ch_err.append(err)
    per_channel_error_pct = float(np.mean(per_ch_err)) if per_ch_err else 0.0

    # Category error (baseline share)
    total_y = float(y.sum())
    true_base_pct = float(true_baseline.sum() / total_y * 100) if total_y > 0 else 0.0
    pred_base_pct = float(pred_baseline.sum() / total_y * 100) if total_y > 0 else 0.0
    category_error_pp = abs(pred_base_pct - true_base_pct)

    return {
        "channel_r2_mean": channel_r2_mean,
        "baseline_r2": baseline_r2,
        "rank_corr": rank_corr,
        "per_channel_error_pct": per_channel_error_pct,
        "category_error_pp": category_error_pp,
        "true_baseline_pct": true_base_pct,
        "pred_baseline_pct": pred_base_pct,
        "per_channel_r2": ch_r2s,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source venv/bin/activate && pytest tests/demantiq/neural/proper_pfn/test_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add demantiq/neural/proper_pfn/metrics.py tests/demantiq/neural/proper_pfn/test_metrics.py
git commit -m "feat(proper_pfn): per-component R² and rank metrics (replace misleading reconstruction R²)"
```

---

## Task 11: Build ProperPFNEngine (training + inference)

**Files:**
- Create: `demantiq/neural/proper_pfn/engine.py`
- Test: `tests/demantiq/neural/proper_pfn/test_engine.py`

- [ ] **Step 1: Write a minimal end-to-end smoke test**

Create `tests/demantiq/neural/proper_pfn/test_engine.py`:

```python
import tempfile
from demantiq.scenarios.scenario_sampler import ScenarioSampler
from demantiq.orchestration.training_pipeline import TrainingPipeline
from demantiq.neural.proper_pfn.engine import ProperPFNEngine, ProperPFNConfig


def test_engine_trains_without_errors():
    import numpy as np
    with tempfile.TemporaryDirectory() as tmpdir:
        sampler = ScenarioSampler(seed=42, rich_context=True, channel_range=(3, 5))
        pipeline = TrainingPipeline(sampler, output_dir=tmpdir, batch_size=10)
        pipeline.generate(n_total=10, n_workers=1, seed=42)

        config = ProperPFNConfig(
            n_epochs=2,
            n_train=10,
            batch_size=4,
            d_model=32,
            n_layers=1,
            n_heads=2,
            data_dir=tmpdir,
        )
        engine = ProperPFNEngine(config)
        metrics = engine.train()
        assert "best_val_loss" in metrics

        # Test inference
        from demantiq.core.demand_kernel import simulate
        config_scenario = sampler.sample(1)[0]
        result = simulate(config_scenario)
        ch_names = [c.name for c in config_scenario.channels]
        n_ch = len(ch_names)
        T = config_scenario.n_periods
        y = result.observable_data["y"].values.astype(np.float32)
        spend = np.column_stack([
            result.observable_data[f"{c}_spend"].values for c in ch_names
        ]).astype(np.float32)
        imp = np.column_stack([
            result.observable_data[f"{c}_impressions"].values for c in ch_names
        ]).astype(np.float32)
        clk = np.column_stack([
            result.observable_data[f"{c}_clicks"].values for c in ch_names
        ]).astype(np.float32)
        from demantiq.orchestration.training_format import extract_context_matrix
        ctx = extract_context_matrix(result.observable_data, T).astype(np.float32)

        out = engine.infer(y, spend, imp, clk, ctx, n_ch)
        assert out["channel_contributions"].shape == (T, n_ch)
        assert out["baseline"].shape == (T,)
        assert out["y_hat"].shape == (T,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/demantiq/neural/proper_pfn/test_engine.py -v`
Expected: FAIL (module doesn't exist).

- [ ] **Step 3: Implement ProperPFNEngine**

Create `demantiq/neural/proper_pfn/engine.py`:

```python
"""Training loop and inference for proper PFN."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from demantiq.neural.data_loader import DemantiqDataset
from demantiq.neural.proper_pfn.dataset import (
    ProperPFNDataset, proper_pfn_collate_fn, CHANNEL_FEATURE_DIM,
)
from demantiq.neural.proper_pfn.model import ProperPFNModel
from demantiq.neural.proper_pfn.loss import held_out_loss

logger = logging.getLogger(__name__)


@dataclass
class ProperPFNConfig:
    # Architecture
    max_channels: int = 8
    n_context_dims: int = 10
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1

    # Training
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    n_epochs: int = 100
    patience: int = 20
    grad_clip: float = 1.0
    val_fraction: float = 0.1
    week_mask_fraction: float = 0.3

    # Data
    n_train: int = 50000
    data_dir: str = "training_data"


# Global feature dim = y(1) + ctx(10) + flags(10) + time_idx(1) + sin(1) + cos(1) + is_masked(1) = 25
GLOBAL_FEATURE_DIM = 25


class ProperPFNEngine:
    def __init__(self, config: ProperPFNConfig | None = None):
        self.config = config or ProperPFNConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ProperPFNModel(
            channel_feat_dim=CHANNEL_FEATURE_DIM,
            global_feat_dim=GLOBAL_FEATURE_DIM,
            max_channels=self.config.max_channels,
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            dropout=self.config.dropout,
        ).to(self.device)

        self.optimizer = None
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def train(self, data_dir: str | None = None, n_train: int | None = None) -> dict:
        data_dir = data_dir or self.config.data_dir
        n_train = n_train or self.config.n_train

        logger.info("Loading training data from %s (max %d)", data_dir, n_train)
        backing = DemantiqDataset(data_dir, max_samples=n_train)
        logger.info("Loaded %d scenarios", len(backing))

        # Split by scenario
        n = len(backing)
        if n == 1:
            train_idx, val_idx = [0], []
        else:
            n_val = max(1, int(n * self.config.val_fraction))
            g = torch.Generator().manual_seed(42)
            perm = torch.randperm(n, generator=g).tolist()
            train_idx, val_idx = perm[:n - n_val], perm[n - n_val:]

        train_ds = _SubsetDataset(
            backing, train_idx,
            max_channels=self.config.max_channels,
            week_mask_fraction=self.config.week_mask_fraction,
            training=True,
        )
        val_ds = None
        if val_idx:
            val_ds = _SubsetDataset(
                backing, val_idx,
                max_channels=self.config.max_channels,
                week_mask_fraction=self.config.week_mask_fraction,  # mask at val too
                training=True,  # to apply mask — otherwise mask=0 and loss is 0
            )

        train_loader = DataLoader(
            train_ds, batch_size=self.config.batch_size, shuffle=True,
            num_workers=0, pin_memory=self.device.type == "cuda",
            collate_fn=proper_pfn_collate_fn,
        )
        val_loader = None
        if val_ds and len(val_ds) > 0:
            val_loader = DataLoader(
                val_ds, batch_size=self.config.batch_size, shuffle=False,
                num_workers=0, pin_memory=self.device.type == "cuda",
                collate_fn=proper_pfn_collate_fn,
            )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.n_epochs, eta_min=1e-6,
        )

        best_val = float("inf")
        best_state = None
        patience = 0

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            "Training: %d scenarios, %d epochs, batch=%d, lr=%.1e, params=%d",
            len(train_ds), self.config.n_epochs, self.config.batch_size,
            self.config.learning_rate, n_params,
        )

        t0 = time.time()
        for epoch in range(self.config.n_epochs):
            train_loss, train_comp = self._train_epoch(train_loader)
            val_loss = train_loss
            if val_loader:
                val_loss, _ = self._validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            scheduler.step()

            if epoch % 5 == 0 or epoch == self.config.n_epochs - 1:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch %3d/%d  train=%.4f (ch=%.4f bl=%.4f nm=%.4f rc=%.4f) "
                    "val=%.4f  lr=%.1e  [%.0fs]",
                    epoch + 1, self.config.n_epochs, train_loss,
                    train_comp["channels_masked"], train_comp["baseline_masked"],
                    train_comp["non_media_masked"], train_comp["reconstruction_masked"],
                    val_loss, lr, time.time() - t0,
                )

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= self.config.patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        if best_state:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        return {
            "best_val_loss": best_val,
            "final_train_loss": self.train_losses[-1],
            "n_epochs_trained": len(self.train_losses),
            "training_time_minutes": (time.time() - t0) / 60,
        }

    def _train_epoch(self, loader: DataLoader) -> tuple[float, dict]:
        self.model.train()
        total = 0.0
        total_comp = {}
        n = 0
        for batch in loader:
            self.optimizer.zero_grad()
            loss, comp = self._compute_loss(batch)
            loss.backward()
            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            total += loss.item()
            for k, v in comp.items():
                total_comp[k] = total_comp.get(k, 0.0) + v
            n += 1
        n = max(n, 1)
        return total / n, {k: v / n for k, v in total_comp.items()}

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> tuple[float, dict]:
        self.model.eval()
        total = 0.0
        total_comp = {}
        n = 0
        for batch in loader:
            loss, comp = self._compute_loss(batch)
            total += loss.item()
            for k, v in comp.items():
                total_comp[k] = total_comp.get(k, 0.0) + v
            n += 1
        n = max(n, 1)
        return total / n, {k: v / n for k, v in total_comp.items()}

    def _compute_loss(self, batch: dict):
        channel_tokens = batch["channel_tokens"].to(self.device)
        global_tokens = batch["global_tokens"].to(self.device)
        target = batch["target"].to(self.device)
        time_pad = batch["time_pad_mask"].to(self.device)
        ch_pad = batch["channel_pad_mask"].to(self.device)
        week_masked = batch["week_is_masked"].to(self.device)

        pred = self.model(channel_tokens, global_tokens, ch_pad, time_pad)
        return held_out_loss(
            pred, target, time_pad, ch_pad, week_masked, self.config.max_channels,
        )

    @torch.no_grad()
    def infer(
        self,
        y: np.ndarray,
        spend: np.ndarray,
        impressions: np.ndarray,
        clicks: np.ndarray,
        context: np.ndarray,
        n_channels: int,
    ) -> dict:
        """Inference on a single scenario (no masking — all weeks visible)."""
        self.model.eval()
        T = len(y)
        C = self.config.max_channels

        # Pad channel data to max_channels
        def pad_channel_arr(arr):
            out = np.zeros((T, C), dtype=np.float32)
            out[:, :n_channels] = arr[:, :n_channels]
            return out

        spend_p = pad_channel_arr(spend)
        imp_p = pad_channel_arr(impressions)
        clk_p = pad_channel_arr(clicks)

        # Per-channel normalization
        def per_channel_normalize(arr):
            scales = np.maximum(np.abs(arr).max(axis=0, keepdims=True), 1.0)
            return arr / scales

        spend_n = per_channel_normalize(spend_p)
        imp_n = per_channel_normalize(imp_p)
        clk_n = per_channel_normalize(clk_p)

        channel_tokens = np.stack([spend_n, imp_n, clk_n], axis=-1)  # (T, C, 3)
        channel_tokens = channel_tokens.astype(np.float32)

        # Global features
        y_scale = max(float(np.abs(y).mean()), 1.0)
        y_norm = (y / y_scale).astype(np.float32)

        ctx_present = (np.abs(context).sum(axis=0) > 0).astype(np.float32)
        col_scales = np.maximum(np.abs(context).max(axis=0, keepdims=True), 1.0)
        ctx_norm = (context / col_scales).astype(np.float32)
        presence = np.tile(ctx_present, (T, 1))

        time_idx = np.linspace(0.0, 1.0, T, dtype=np.float32)
        woy = np.arange(T, dtype=np.float32) % 52.0
        sin_w = np.sin(2.0 * np.pi * woy / 52.0)
        cos_w = np.cos(2.0 * np.pi * woy / 52.0)
        is_masked = np.zeros(T, dtype=np.float32)  # inference: no masking

        global_tokens = np.concatenate(
            [
                y_norm.reshape(T, 1),
                ctx_norm,
                presence,
                time_idx.reshape(T, 1),
                sin_w.reshape(T, 1),
                cos_w.reshape(T, 1),
                is_masked.reshape(T, 1),
            ],
            axis=1,
        ).astype(np.float32)

        # Batch dim
        ch_tok = torch.from_numpy(channel_tokens).unsqueeze(0).to(self.device)
        gl_tok = torch.from_numpy(global_tokens).unsqueeze(0).to(self.device)
        ch_pad = torch.ones(1, C, dtype=torch.bool, device=self.device)
        ch_pad[0, :n_channels] = False
        t_pad = torch.zeros(1, T, dtype=torch.bool, device=self.device)

        pred = self.model(ch_tok, gl_tok, ch_pad, t_pad)

        ch_shares = pred["channel_shares"][0].cpu().numpy()  # (T, C)
        base_share = pred["baseline_share"][0].cpu().numpy()  # (T,)
        nm_share = pred["non_media_share"][0].cpu().numpy()  # (T,)

        y_f = y.astype(np.float32)
        ch_contrib = ch_shares[:, :n_channels] * y_f[:, None]
        base_contrib = base_share * y_f
        nm_contrib = nm_share * y_f
        y_hat = base_contrib + ch_contrib.sum(axis=1) + nm_contrib

        return {
            "channel_contributions": ch_contrib.astype(np.float32),
            "baseline": base_contrib.astype(np.float32),
            "non_media": nm_contrib.astype(np.float32),
            "y_hat": y_hat.astype(np.float32),
        }

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out / "proper_pfn.pt")
        (out / "proper_pfn_config.json").write_text(json.dumps(asdict(self.config), indent=2))

    def load(self, path: str | Path) -> None:
        state = torch.load(Path(path) / "proper_pfn.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)


class _SubsetDataset(ProperPFNDataset):
    """ProperPFNDataset limited to a list of scenario indices."""

    def __init__(self, backing, scenario_indices, **kwargs):
        super().__init__(backing, **kwargs)
        self._valid_indices = [i for i in scenario_indices if 2 <= len(backing.channel_names[i]) <= self.max_channels]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/demantiq/neural/proper_pfn/test_engine.py -v -s`
Expected: PASS. Training is slow because it generates data — be patient.

- [ ] **Step 5: Commit**

```bash
git add demantiq/neural/proper_pfn/engine.py tests/demantiq/neural/proper_pfn/test_engine.py
git commit -m "feat(proper_pfn): training engine with held-out loss"
```

---

## Task 12: Overfit test (critical verification)

**Files:**
- Create: `scripts/test_proper_pfn_overfit.py`

This is the critical test. The overfit test must pass to show the architecture works on a single scenario. If it fails, nothing else is worth doing.

- [ ] **Step 1: Create the overfit script**

Create `scripts/test_proper_pfn_overfit.py`:

```python
"""Overfit test: train proper PFN on 1 scenario and evaluate on the SAME.

Target: channel rank corr > 0.9, channel r2 mean > 0.8, baseline r2 > 0.8.
If this fails, the architecture is broken.

Usage:
  source venv/bin/activate
  python scripts/test_proper_pfn_overfit.py --n-epochs 500 --seed 42
"""

import argparse
import logging
import tempfile
import numpy as np

from demantiq.scenarios.scenario_sampler import ScenarioSampler
from demantiq.core.demand_kernel import simulate
from demantiq.orchestration.training_pipeline import TrainingPipeline
from demantiq.orchestration.training_format import (
    extract_context_matrix, ground_truth_to_decomposition,
    DECOMP_IDX_BASELINE, DECOMP_IDX_CHANNELS_START,
    DECOMP_IDX_PRICE, DECOMP_IDX_COMPETITION, DECOMP_IDX_MACRO,
)
from demantiq.neural.proper_pfn.engine import ProperPFNEngine, ProperPFNConfig
from demantiq.neural.proper_pfn.metrics import scenario_level_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate 1 scenario
        sampler = ScenarioSampler(seed=args.seed, rich_context=True, channel_range=(3, 5))
        pipeline = TrainingPipeline(sampler, output_dir=tmpdir, batch_size=1)
        pipeline.generate(n_total=1, n_workers=1, seed=args.seed)

        # Train
        config = ProperPFNConfig(
            n_epochs=args.n_epochs,
            n_train=1,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            d_model=args.d_model,
            n_layers=args.n_layers,
            patience=args.n_epochs,  # no early stopping
            val_fraction=0.0,
            data_dir=tmpdir,
        )
        engine = ProperPFNEngine(config)
        engine.train()

        # Evaluate on the same scenario
        sampler2 = ScenarioSampler(seed=args.seed, rich_context=True, channel_range=(3, 5))
        eval_config = sampler2.sample(1)[0]
        result = simulate(eval_config)
        ch_names = [c.name for c in eval_config.channels]
        n_ch = len(ch_names)
        T = eval_config.n_periods
        y = result.observable_data["y"].values.astype(np.float32)
        spend = np.column_stack([result.observable_data[f"{c}_spend"].values for c in ch_names]).astype(np.float32)
        imp = np.column_stack([result.observable_data[f"{c}_impressions"].values for c in ch_names]).astype(np.float32)
        clk = np.column_stack([result.observable_data[f"{c}_clicks"].values for c in ch_names]).astype(np.float32)
        ctx = extract_context_matrix(result.observable_data, T).astype(np.float32)

        out = engine.infer(y, spend, imp, clk, ctx, n_ch)

        # Ground truth
        true_decomp = ground_truth_to_decomposition(result.ground_truth, ch_names, T)
        true_ch = np.column_stack([true_decomp[:, DECOMP_IDX_CHANNELS_START + i] for i in range(n_ch)])
        true_base = true_decomp[:, DECOMP_IDX_BASELINE]

        metrics = scenario_level_metrics(
            out["channel_contributions"], out["baseline"],
            true_ch, true_base, y,
        )

        print(f"\n{'='*70}")
        print(f"  OVERFIT TEST RESULTS — seed={args.seed}, {args.n_epochs} epochs")
        print(f"  {n_ch} channels ({', '.join(ch_names)}), {T} periods")
        print(f"{'='*70}")
        print(f"\n  Channel rank corr:    {metrics['rank_corr']:.3f}  (target: >0.9)")
        print(f"  Channel R² mean:      {metrics['channel_r2_mean']:.3f}  (target: >0.8)")
        print(f"  Baseline R²:          {metrics['baseline_r2']:.3f}  (target: >0.8)")
        print(f"  Category error:       {metrics['category_error_pp']:.1f}pp  (target: <5pp)")
        print(f"  Per-channel err:      {metrics['per_channel_error_pct']:.1f}%  (target: <20%)")

        passed = (
            metrics['rank_corr'] > 0.9
            and metrics['channel_r2_mean'] > 0.8
            and metrics['baseline_r2'] > 0.8
        )
        print(f"\n  {'PASSED' if passed else 'FAILED'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the overfit test**

Run: `source venv/bin/activate && python scripts/test_proper_pfn_overfit.py --n-epochs 300 --seed 42`
Expected: overfit test should PASS. If it fails, STOP and investigate — the architecture has a bug.

- [ ] **Step 3: Commit**

```bash
git add scripts/test_proper_pfn_overfit.py
git commit -m "feat(proper_pfn): overfit verification script"
```

---

## Task 13: Multi-scenario training script with proper evaluation

**Files:**
- Create: `scripts/train_proper_pfn.py`

- [ ] **Step 1: Create the training script**

Create `scripts/train_proper_pfn.py`:

```python
"""Train and evaluate the proper PFN model.

Usage:
  source venv/bin/activate
  python scripts/train_proper_pfn.py --n-train 50000 --n-epochs 100 --random-eval 10
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from demantiq.scenarios.scenario_sampler import ScenarioSampler
from demantiq.core.demand_kernel import simulate
from demantiq.orchestration.training_pipeline import TrainingPipeline
from demantiq.orchestration.training_format import (
    extract_context_matrix, ground_truth_to_decomposition,
    DECOMP_IDX_BASELINE, DECOMP_IDX_CHANNELS_START,
    DECOMP_IDX_PRICE, DECOMP_IDX_COMPETITION, DECOMP_IDX_MACRO,
)
from demantiq.neural.proper_pfn.engine import ProperPFNEngine, ProperPFNConfig
from demantiq.neural.proper_pfn.metrics import scenario_level_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT = Path("neural_output")
DATA = OUT / "proper_pfn_data"
MODEL = OUT / "proper_pfn_model"


def generate_data(n_total: int, n_workers: int = 4):
    DATA.mkdir(parents=True, exist_ok=True)
    sampler = ScenarioSampler(seed=42, rich_context=True, channel_range=(2, 8))
    pipeline = TrainingPipeline(sampler, output_dir=str(DATA), batch_size=100)
    t0 = time.time()
    pipeline.generate(n_total=n_total, n_workers=n_workers, seed=42)
    logger.info("Generated %d scenarios in %.0fs", n_total, time.time() - t0)


def train(args):
    config = ProperPFNConfig(
        n_epochs=args.n_epochs,
        n_train=args.n_train,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        d_model=args.d_model,
        n_layers=args.n_layers,
        week_mask_fraction=args.mask_frac,
        patience=args.patience,
        data_dir=str(DATA),
    )
    engine = ProperPFNEngine(config)
    metrics = engine.train()
    engine.save(str(MODEL))
    logger.info("Training metrics: %s", metrics)
    return engine


def load_engine():
    import json
    with open(MODEL / "proper_pfn_config.json") as f:
        config = ProperPFNConfig(**json.load(f))
    engine = ProperPFNEngine(config)
    engine.load(str(MODEL))
    return engine


def evaluate_scenario(engine, seed: int):
    sampler = ScenarioSampler(seed=seed, rich_context=True, channel_range=(2, 8))
    cfg = sampler.sample(1)[0]
    result = simulate(cfg)
    ch_names = [c.name for c in cfg.channels]
    n_ch = len(ch_names)
    T = cfg.n_periods
    y = result.observable_data["y"].values.astype(np.float32)
    spend = np.column_stack([result.observable_data[f"{c}_spend"].values for c in ch_names]).astype(np.float32)
    imp = np.column_stack([result.observable_data[f"{c}_impressions"].values for c in ch_names]).astype(np.float32)
    clk = np.column_stack([result.observable_data[f"{c}_clicks"].values for c in ch_names]).astype(np.float32)
    ctx = extract_context_matrix(result.observable_data, T).astype(np.float32)

    out = engine.infer(y, spend, imp, clk, ctx, n_ch)

    true_decomp = ground_truth_to_decomposition(result.ground_truth, ch_names, T)
    true_ch = np.column_stack([true_decomp[:, DECOMP_IDX_CHANNELS_START + i] for i in range(n_ch)])
    true_base = true_decomp[:, DECOMP_IDX_BASELINE]

    metrics = scenario_level_metrics(
        out["channel_contributions"], out["baseline"],
        true_ch, true_base, y,
    )
    metrics["scenario"] = f"seed={seed}"
    metrics["n_channels"] = n_ch
    metrics["n_periods"] = T
    return metrics


def evaluate_random(engine, n: int, seed: int = 999):
    rows = []
    for i in range(n):
        m = evaluate_scenario(engine, seed + i)
        rows.append(m)
        print(f"seed={seed+i}  ch_r2={m['channel_r2_mean']:.3f}  "
              f"base_r2={m['baseline_r2']:.3f}  rank={m['rank_corr']:.3f}  "
              f"cat_err={m['category_error_pp']:.1f}pp  ch_err={m['per_channel_error_pct']:.1f}%")

    print(f"\n{'='*70}\n  SUMMARY ({n} scenarios)\n{'='*70}")
    print(f"  Mean channel R²:    {np.mean([m['channel_r2_mean'] for m in rows]):.3f}")
    print(f"  Mean baseline R²:   {np.mean([m['baseline_r2'] for m in rows]):.3f}")
    print(f"  Mean rank corr:     {np.mean([m['rank_corr'] for m in rows]):.3f}")
    print(f"  Mean category err:  {np.mean([m['category_error_pp'] for m in rows]):.1f}pp")
    print(f"  Mean channel err:   {np.mean([m['per_channel_error_pct'] for m in rows]):.1f}%")
    print(f"  Scenarios R² > 0.5: {sum(1 for m in rows if m['channel_r2_mean'] > 0.5)}/{n}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--mask-frac", type=float, default=0.3)
    parser.add_argument("--random-eval", type=int, default=10)
    parser.add_argument("--skip-datagen", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--n-workers", type=int, default=4)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if not args.skip_training:
        if not args.skip_datagen:
            generate_data(args.n_train, n_workers=args.n_workers)
        engine = train(args)
    else:
        engine = load_engine()

    if args.random_eval > 0:
        evaluate_random(engine, args.random_eval)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test locally**

Run: `source venv/bin/activate && python scripts/train_proper_pfn.py --n-train 100 --n-epochs 5 --random-eval 2 --n-workers 2 --batch-size 8`
Expected: runs to completion without errors.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_proper_pfn.py
git commit -m "feat(proper_pfn): training + evaluation script with proper metrics"
```

---

## Task 14: Push and train at scale on vast.ai

- [ ] **Step 1: Push all commits**

```bash
git push origin feature/implement-v1
```

- [ ] **Step 2: On vast.ai instance, pull and run**

On the running vast.ai instance:
```bash
cd /workspace/josiah
bash scripts/vast_setup.sh

# First: overfit test (verify architecture works)
python scripts/test_proper_pfn_overfit.py --n-epochs 300

# If overfit test PASSES, run full training
python scripts/train_proper_pfn.py --n-train 50000 --n-epochs 100 --random-eval 10 --batch-size 16 --patience 30
```

- [ ] **Step 3: Interpret results**

The critical metrics on unseen scenarios:
- **Channel R² > 0.5** on majority of scenarios → architecture is doing in-context learning
- **Rank corr > 0.7** on majority → channel identification works across scenarios
- **Category error < 10pp** → baseline differentiation works

If rank corr stays around 0.3-0.4 (like previous runs): the held-out loss approach ALSO didn't break through — this is the definitive signal that cross-scenario amortized MMM decomposition may not be solvable with neural networks. Document this and pivot.

If rank corr > 0.7: we have the breakthrough. Document, write up, move to production.

---

## Self-review checklist

- ✅ Every task has exact file paths
- ✅ Every code step has actual code (no "TBD" or "implement similar")
- ✅ Type/method names are consistent across tasks (CHANNEL_FEATURE_DIM, GLOBAL_FEATURE_DIM, proper_pfn_collate_fn, held_out_loss, ProperPFNModel all defined exactly once and used consistently)
- ✅ Tests are written FIRST and shown to fail before implementation
- ✅ Each task ends with a commit
- ✅ Integration task (Task 14) verifies the full system end-to-end
- ✅ Overfit task (Task 12) is the critical verification before scaling — if it fails, stop and debug
