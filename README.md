# Josiah

Synthetic Marketing Mix Model (MMM) data generator with known ground truth parameters. Built for testing and validating MMM implementations like [PyMC Marketing](https://www.pymc-marketing.io/).

## Why

When building or evaluating an MMM, you need data where you already know the true effect of each channel. Josiah generates realistic datasets with configurable adstock, saturation, trend, seasonality, controls, and promotions — then exports the ground truth parameters alongside the data so you can measure how well your model recovers them.

## Install

```bash
pip install -e .
```

Requires Python 3.9+.

## Quick Start

### Streamlit App

```bash
streamlit run app.py
```

The app has three pages:

1. **Scenario Builder** — configure single or batch scenarios, pick an engine, set scale presets, and tune channel/control/promo parameters.
2. **Generate & Preview** — run generation, inspect the output DataFrame and decomposition charts.
3. **Export** — download CSVs, Parquet files, or a ZIP bundle with ground truth JSON sidecars.

### Python API

```python
from josiah import ScenarioConfig, ChannelConfig, ControlConfig, PromoConfig, generate_single

config = ScenarioConfig(
    name="my_test",
    engine="pymc",
    start_date="2022-01-01",
    end_date="2024-12-31",
    frequency="W",
    intercept=5000.0,
    noise_std=100.0,
    trend_type="linear",
    trend_params={"slope": 0.5},
    seasonality_n_terms=2,
    channels=[
        ChannelConfig(name="facebook", alpha=0.7, l_max=8, lam=2.0, beta=800.0, spend_mean=3000.0, spend_std=500.0),
        ChannelConfig(name="google", alpha=0.5, l_max=4, lam=3.0, beta=1200.0, spend_mean=5000.0, spend_std=1000.0),
    ],
    controls=[
        ControlConfig(name="z1", gamma_shape=2.0, gamma_scale=1.0, coefficient=150.0),
    ],
    promos=[
        PromoConfig(name="black_friday", coefficient=500.0, n_occurrences=1, duration_days=3),
    ],
    seed=42,
)

df, ground_truth, decomp_df = generate_single(config)
```

### Batch Generation

Generate multiple scenarios with randomized parameters:

```python
from josiah import BatchConfig, generate_batch, run_batch

# Create randomized scenario configs
batch = BatchConfig(
    n_scenarios=10,
    engine="pymc",
    n_channels_range=(2, 5),
    beta_range=(200.0, 1500.0),
    intercept_range=(500.0, 2000.0),
    master_seed=42,
)
configs = generate_batch(batch)

# Generate all datasets
results = run_batch(configs)  # list of (df, ground_truth, decomp_df|None)
```

### Export

```python
from josiah import export_scenario, export_batch_to_zip

# Single scenario to files
export_scenario(df, ground_truth, path="output/", fmt="csv", decomp_df=decomp_df)

# Batch to ZIP (returns BytesIO)
zip_bytes = export_batch_to_zip(results, fmt="csv")
```

## Engines

### PyMC Engine (recommended)

Matches PyMC Marketing's formulas:

```
y = intercept + trend + seasonality + controls + channels + promos + noise
```

Where each channel contribution is:

```
beta * logistic_saturation(geometric_adstock(spend / max|spend|, alpha, l_max), lam)
```

Spend is normalized by `max(abs(spend))` per channel before saturation, matching PyMC Marketing's `MaxAbsScaler`. The ground truth JSON includes `channel_scales` so you can denormalize.

Output columns: `date`, `{channel}_spend`, `{control}`, `{promo}`, `y`

### Legacy Engine

Hill CPM curves + exponential adstock. Uses daily frequency.

Output columns: `date`, `{channel}_spend`, `{channel}_impressions`, `{channel}_cpm`, `{channel}_revenue`, `seasonality_revenue`, `total_revenue`, `revenue`, `y`, `is_preflight`

## Ground Truth

Every generated dataset includes a JSON sidecar with the true parameters used for generation. This lets you validate model recovery — compare your fitted parameters against the known truth.

The PyMC ground truth includes: intercept, trend, seasonality coefficients, per-channel adstock/saturation/beta params, channel scales, control coefficients, promo coefficients, total ROAS, and the full formula string.

## Available Channels

facebook, google, tiktok, pinterest, email, youtube, snapchat, linkedin, twitter, display

## Available Promos

black_friday, cyber_monday, prime_day, summer_sale, holiday_sale, flash_sale, new_year_sale, back_to_school, valentines, spring_sale, labor_day, memorial_day

## License

MIT
