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
│   │   ├── pymc_engine.py        # geometric adstock + logistic saturation
│   │   └── legacy_engine.py      # Hill curves + exponential adstock
│   ├── components/
│   │   ├── adstock.py            # geometric_adstock(), exponential_adstock()
│   │   ├── saturation.py         # logistic_saturation(), hill_saturation()
│   │   ├── trend.py              # linear_trend(), cube_root_trend()
│   │   ├── seasonality.py        # fourier_seasonality(), sine_seasonality()
│   │   ├── channels.py           # generate_spend(), channel_effect()
│   │   ├── controls.py           # generate_controls()
│   │   └── promos.py             # generate_promo_indicators(), add_promos_legacy()
│   ├── scenario.py               # ScenarioConfig, BatchConfig, generate_batch()
│   ├── generator.py              # generate_single() orchestrator
│   ├── export.py                 # CSV/Parquet/ZIP export + ground truth JSON
│   └── visualization.py          # Plotly charts
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
- Ground truth JSON includes `channel_scales` (per-channel max abs spend) for reference
- Promos are 0/1 indicators with configurable coefficient, duration, and occurrences

**Legacy Engine**: Hill CPM curves + exponential adstock (refactored from sim.ipynb)

**Data Flow**: BatchConfig → generate_batch() → list[ScenarioConfig] → generate_single() → (DataFrame, ground_truth) → export

## Conventions

- PyMC columns: `date`, `{channel}_spend`, `{control}`, `{promo}`, `y`
- Legacy columns: `date`, `{channel}_spend`, `{channel}_impressions`, `{channel}_cpm`, `{channel}_revenue`, `revenue`, `y`
- Ground truth JSON sidecar has all true parameters for model recovery
- Channels: facebook, google, tiktok, pinterest, email, youtube, snapchat, linkedin, twitter, display
- Scale presets: Thousands (K), Tens of Thousands, Hundreds of Thousands, Millions (M), Billions (B), Custom
