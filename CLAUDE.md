# CLAUDE.md

## Project Overview

Josiah is a Streamlit app + Python package for generating synthetic Marketing Mix Model (MMM) data with known ground truth parameters. Used for testing and validating MMM implementations (e.g. PyMC Marketing).

## Structure

```
josiah/
├── app.py                        # Streamlit entry point
├── josiah/
│   ├── __init__.py               # Public API exports
│   ├── engines/
│   │   ├── pymc_engine.py        # geometric adstock + logistic saturation → returns (df, ground_truth, decomp_df)
│   │   └── legacy_engine.py      # Hill curves + exponential adstock → returns (df, ground_truth)
│   ├── components/
│   │   ├── adstock.py            # geometric_adstock(), exponential_adstock()
│   │   ├── saturation.py         # logistic_saturation(), hill_saturation()
│   │   ├── trend.py              # linear_trend(), cube_root_trend()
│   │   ├── seasonality.py        # fourier_seasonality(), sine_seasonality()
│   │   ├── channels.py           # generate_spend(), channel_effect()
│   │   ├── controls.py           # generate_controls()
│   │   └── promos.py             # generate_promo_indicators(), add_promos_legacy()
│   ├── scenario.py               # ScenarioConfig, BatchConfig, ChannelConfig, ControlConfig, PromoConfig, generate_batch()
│   ├── generator.py              # generate_single() → (df, ground_truth, decomp_df|None), generate_batch()
│   ├── export.py                 # export_scenario(), export_batch_to_zip(), export_single_to_bytes()
│   └── visualization.py          # plot_revenue_decomposition(), plot_channel_spend()
├── pages/
│   ├── 1_Scenario_Builder.py     # Batch or single scenario config
│   ├── 2_Generate_Preview.py     # Run generation, inspect + download results
│   └── 3_Export.py               # Download single or batch ZIP
├── pyproject.toml
└── requirements.txt
```

## Install & Run

```bash
pip install -e .
streamlit run app.py
```

## Architecture

**PyMC Engine** (recommended): `y = intercept + trend + seasonality + controls + channels + promos + noise`
- Channel: `beta * logistic_saturation(geometric_adstock(spend / max|spend|, alpha, l_max), lam)`
- Spend normalized by max(abs) before saturation so lambda operates on [0,1] scale (matches PyMC Marketing's MaxAbsScaler)
- Ground truth JSON includes `channel_scales` (per-channel max abs spend) — essential for denormalizing spend predictions
- Controls are continuous gamma-distributed values with configurable coefficient
- Promos are 0/1 indicators with configurable coefficient, duration, and occurrences
- Returns 3-tuple: `(DataFrame, ground_truth_dict, decomposition_DataFrame)`

**Legacy Engine**: Hill CPM curves + exponential adstock (refactored from sim.ipynb)
- Uses daily frequency (not weekly)
- Returns 2-tuple: `(DataFrame, ground_truth_dict)` — no decomposition DataFrame

**Data Flow**: BatchConfig → generate_batch() (uses master_seed for per-scenario seeds) → list[ScenarioConfig] → generate_single() → (DataFrame, ground_truth, decomp_df|None) → export

## Conventions

- PyMC columns: `date`, `{channel}_spend`, `{control}`, `{promo}`, `y`
- PyMC decomposition columns: `date`, `intercept`, `trend`, `seasonality`, `{channel}_contribution`, `{control}_contribution`, `{promo}_contribution`, `noise`, `y`
- Legacy columns: `date`, `{channel}_spend`, `{channel}_impressions`, `{channel}_cpm`, `{channel}_revenue`, `seasonality_revenue`, `total_revenue`, `revenue`, `y`, `is_preflight`
- Ground truth JSON sidecar has all true parameters for model recovery
- Channels: facebook, google, tiktok, pinterest, email, youtube, snapchat, linkedin, twitter, display
- Promos: black_friday, cyber_monday, prime_day, summer_sale, holiday_sale, flash_sale, new_year_sale, back_to_school, valentines, spring_sale, labor_day, memorial_day
- Scale presets: Thousands (K), Tens of Thousands, Hundreds of Thousands, Millions (M), Billions (B), Custom
  - Ranges derived from scale factor S (e.g. intercept: 0.5S–2.0S, beta: 0.2S–1.5S, spend_mean: 0.1S–2.0S, noise_std: 0.01S–0.1S)
