# Demantiq Simulator — Design Specification

**Date:** 2026-03-13
**Status:** Approved
**PRD Reference:** [docs/prd.md](../prd.md)

---

## 1. Context

The Josiah project currently provides a simple MMM synthetic data generator (~2,500 lines) with geometric adstock, logistic/Hill saturation, trend, seasonality, controls, and promos. The Demantiq PRD describes a far more comprehensive simulator adding pricing, distribution, competition, macro variables, endogeneity, cross-variable interactions, a 15-step demand kernel, 15 named scenarios, an evaluation framework, calibration system, and Monte Carlo orchestration.

This spec defines how to build the full Demantiq simulator as a sibling package alongside Josiah, delivered across 8 milestones with autonomous agent-driven development.

## 2. Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Package structure | Sibling packages: `josiah/` (frozen) + `demantiq/` (new) | Clean separation, no coupling, Josiah stays simple |
| Compute backend | NumPy first, JAX migration later | Faster development, easier debugging, JAX port is mechanical |
| Milestone execution | Parallel where dependencies allow (Approach B) | M2 and M3 can parallelize after M1; maximizes throughput |
| UI framework | Streamlit (existing) | Already in use, adequate for configuration/visualization |
| Evaluation/Calibration | API-only, no Streamlit pages | Consumed by MMM training service at scale |
| Agent orchestration | Claude Flow per-milestone workflows | One workflow per milestone, coordinator dispatches |
| Execution mode | Fully autonomous (ralph approach) | Coordinator runs end-to-end, validators gate each milestone |

## 3. Repository Structure

```
semarang/
├── josiah/                          # FROZEN — simple MMM generator
│
├── demantiq/                        # NEW — complex demand simulator
│   ├── __init__.py                  # Public API: Simulator, SimulationConfig, ScenarioLibrary
│   ├── config/
│   │   ├── __init__.py
│   │   ├── simulation_config.py     # Master config dataclass
│   │   ├── channel_config.py        # Per-channel parameters
│   │   ├── pricing_config.py        # Pricing mechanics
│   │   ├── distribution_config.py   # Distribution/availability
│   │   ├── competition_config.py    # Competitive dynamics
│   │   ├── macro_config.py          # External variables + regime changes
│   │   ├── endogeneity_config.py    # Feedback/confounding
│   │   ├── interaction_config.py    # Cross-variable interactions
│   │   ├── noise_config.py          # Error term specification
│   │   └── baseline_config.py       # Trend, seasonality, organic level
│   │
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── spend_generator.py       # Correlated media spend (log-normal, gamma, flighting)
│   │   ├── pricing_engine.py        # Price/promo simulation
│   │   ├── distribution_generator.py # Weighted distribution over time
│   │   ├── competition_generator.py # Competitor SOV and activity
│   │   ├── macro_generator.py       # External variable time series
│   │   └── baseline_generator.py    # Trend + seasonality + organic
│   │
│   ├── transforms/
│   │   ├── __init__.py
│   │   ├── saturation.py            # Hill, logistic, power, piecewise linear, custom
│   │   ├── adstock.py               # Geometric, weibull CDF/PDF, delayed geometric, custom
│   │   └── interactions.py          # Cross-variable interaction functions
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── demand_kernel.py         # 15-step DGP pipeline
│   │   ├── endogeneity_layer.py     # Feedback loops and confounding injection
│   │   └── noise_model.py           # Heteroscedastic and correlated noise
│   │
│   ├── ground_truth/
│   │   ├── __init__.py
│   │   ├── contributions.py         # True channel/lever contributions
│   │   ├── roas_calculator.py       # True ROAS per channel
│   │   ├── margin_attribution.py    # True margin decomposition
│   │   ├── interaction_effects.py   # True interaction magnitudes
│   │   ├── elasticities.py          # True price/media elasticities
│   │   └── counterfactuals.py       # True zero-spend counterfactuals
│   │
│   ├── scenarios/
│   │   ├── __init__.py
│   │   ├── scenario_library.py      # 15 named pre-built scenarios (SCN-001 to SCN-015)
│   │   ├── scenario_sampler.py      # Random scenario generation for Monte Carlo
│   │   └── difficulty_scorer.py     # Quantify scenario estimation difficulty
│   │
│   ├── orchestration/
│   │   ├── __init__.py
│   │   ├── monte_carlo.py           # Batch simulation runner
│   │   ├── parallel_runner.py       # Multi-core parallelization
│   │   └── serializer.py            # Config JSON/YAML save/load
│   │
│   ├── evaluation/                  # API-only (no Streamlit UI)
│   │   ├── __init__.py
│   │   ├── parameter_recovery.py    # Bias, MAPE, coverage, interval width, rank correlation
│   │   ├── contribution_accuracy.py # Per-channel MAPE, ranking, zero-detection
│   │   ├── roas_accuracy.py         # ROAS estimation accuracy
│   │   ├── interaction_detection.py # Interaction identification accuracy
│   │   ├── optimization_quality.py  # Recommended vs. true optimal allocation
│   │   ├── capability_surface.py    # Multi-dimensional accuracy mapping
│   │   └── model_comparison.py      # Head-to-head engine benchmarking
│   │
│   ├── calibration/                 # API-only (no Streamlit UI)
│   │   ├── __init__.py
│   │   ├── empirical_distributions.py # Parameter distributions from real data
│   │   ├── public_data_adapter.py   # Public financial data ingestion
│   │   └── realism_validator.py     # Synthetic vs. real data statistical tests
│   │
│   └── utils/
│       ├── __init__.py
│       ├── distributions.py         # Custom probability distributions
│       ├── time_series.py           # Fourier, trend, structural break utilities
│       ├── correlation.py           # Correlation matrix generation (copulas)
│       └── random.py                # Seeded RNG management
│
├── app.py                           # Updated: route to Josiah or Demantiq
├── pages/
│   ├── 1_Scenario_Builder.py        # Existing Josiah page (unchanged)
│   ├── 2_Generate_Preview.py        # Existing Josiah page (unchanged)
│   ├── 3_Export.py                   # Existing Josiah page (unchanged)
│   ├── 4_Demantiq_Config.py         # NEW: Full DGP configuration
│   ├── 5_Demantiq_Scenarios.py      # NEW: Browse/select/customize scenarios
│   ├── 6_Demantiq_Generate.py       # NEW: Run simulation, preview results
│   ├── 7_Monte_Carlo.py             # NEW: Batch runs, progress monitoring
│   └── 8_Model_Comparison.py        # NEW: Head-to-head results visualization
│
├── tests/
│   └── demantiq/                    # Mirrors demantiq/ structure
│       ├── test_config/
│       ├── test_generators/
│       ├── test_transforms/
│       ├── test_core/
│       ├── test_ground_truth/
│       ├── test_scenarios/
│       ├── test_orchestration/
│       ├── test_evaluation/
│       ├── test_calibration/
│       └── test_integration/        # End-to-end: all 15 scenarios produce valid output
│
├── docs/
│   ├── prd.md
│   ├── README.md
│   ├── specs/
│   │   └── 2026-03-13-demantiq-simulator-design.md  # This file
│   ├── project/
│   │   ├── milestones.md
│   │   └── stage-status.md
│   └── workflows/
│       ├── coordinator.yml
│       ├── m1-core-dgp.yml
│       ├── m2-pricing-dist.yml
│       ├── m3-endog-comp.yml
│       ├── m4-interactions.yml
│       ├── m5-scenarios-mc.yml
│       ├── m6-evaluation.yml
│       ├── m7-calibration.yml
│       └── m8-training-data.yml
│
└── pyproject.toml                   # Updated: add demantiq package + test deps
```

## 4. Milestone Plan

### M1: Core DGP (Foundation)

**Goal:** Demand kernel with media channels, saturation, adstock, baseline, noise.

**Delivers:**
- `demantiq/config/`: `SimulationConfig`, `ChannelConfig`, `BaselineConfig`, `NoiseConfig`
- `demantiq/transforms/saturation.py`: Hill, logistic, power, piecewise linear
- `demantiq/transforms/adstock.py`: Geometric, weibull CDF, weibull PDF, delayed geometric
- `demantiq/generators/spend_generator.py`: Correlated spend with flighting patterns, log-normal/gamma marginals, correlation groups
- `demantiq/generators/baseline_generator.py`: Trend (linear, cube root) + seasonality (Fourier) + organic level
- `demantiq/core/demand_kernel.py`: Steps 1-6, 12-13 of the 15-step pipeline (baseline, spend, adstock, saturation, media effects, aggregate, noise)
- `demantiq/core/noise_model.py`: Gaussian, t-distributed, heteroscedastic, autocorrelated noise
- `demantiq/ground_truth/contributions.py`: True per-channel contributions
- `demantiq/ground_truth/roas_calculator.py`: True ROAS per channel
- `demantiq/orchestration/serializer.py`: Config JSON/YAML serialization
- `demantiq/utils/random.py`: Seeded RNG management
- `demantiq/__init__.py`: Public API (`Simulator`, `SimulationConfig`)

**Acceptance:** SCN-001 (Clean Room) generates valid data. OLS on clean data recovers true betas within 5%.

**Tests:**
- Unit tests for every saturation and adstock function (monotonicity, boundary values, known analytical results)
- Unit tests for spend generator (non-negativity, correlation structure, flighting patterns)
- Unit tests for noise model (distribution shape, autocorrelation, heteroscedasticity)
- Integration test: full pipeline produces valid DataFrame + ground truth dict
- Property tests: contributions sum to total demand (within tolerance), ROAS > 0 for positive betas

### M2: Pricing + Distribution (parallel with M3)

**Goal:** Pricing engine, distribution module, price-media interaction.

**Delivers:**
- `demantiq/config/pricing_config.py`: `PricingConfig` with base_price, elasticity, promo calendar, cost structure
- `demantiq/config/distribution_config.py`: `DistributionConfig` with trajectory, ceiling, stockouts
- `demantiq/generators/pricing_engine.py`: Price/promo simulation with seasonal variation
- `demantiq/generators/distribution_generator.py`: Weighted distribution over time
- `demantiq/core/demand_kernel.py`: Add steps 8-9 (price effect, distribution cap)
- `demantiq/ground_truth/margin_attribution.py`: True margin decomposition
- `demantiq/ground_truth/elasticities.py`: True price elasticity

**Acceptance:** SCN-004 (Pricing Dominant) and SCN-011 (Promotional Trap) generate valid data.

### M3: Endogeneity + Competition (parallel with M2)

**Goal:** Full endogeneity layer, competition module, macro variables.

**Delivers:**
- `demantiq/config/endogeneity_config.py`: `EndogeneityConfig`
- `demantiq/config/competition_config.py`: `CompetitionConfig`
- `demantiq/config/macro_config.py`: `MacroConfig` with regime changes
- `demantiq/core/endogeneity_layer.py`: Feedback loops, seasonal allocation bias, performance chasing, algorithmic targeting, OVB confounders
- `demantiq/generators/competition_generator.py`: Competitor SOV, reactive pricing
- `demantiq/generators/macro_generator.py`: External variable time series (random walk, mean reverting, trending, seasonal)
- `demantiq/core/demand_kernel.py`: Add steps 3, 10-11 (endogeneity, competition, macro)

**Acceptance:** SCN-003 (Adversarial) generates data where naive OLS is demonstrably biased. Bias matches analytical prediction within 10% (Appendix B of PRD).

### M4: Full Interaction Matrix

**Goal:** All pairwise interactions configurable.

**Delivers:**
- `demantiq/config/interaction_config.py`: `InteractionConfig`
- `demantiq/transforms/interactions.py`: Multiplicative and additive interaction functions
- `demantiq/core/demand_kernel.py`: Add step 7 (interactions) — completing all 15 steps
- `demantiq/ground_truth/interaction_effects.py`: True interaction magnitudes
- `demantiq/ground_truth/counterfactuals.py`: True zero-spend counterfactuals accounting for interactions

**Acceptance:** SCN-005 (Interaction Heavy) generates data with known interactions. Ground truth ledger correctly reports interaction magnitudes.

### M5: Scenario Library + Monte Carlo

**Goal:** All 15 named scenarios, Monte Carlo runner, parallel execution.

**Delivers:**
- `demantiq/scenarios/scenario_library.py`: SCN-001 through SCN-015 with full configs
- `demantiq/scenarios/scenario_sampler.py`: Random scenario generation from distribution specs
- `demantiq/scenarios/difficulty_scorer.py`: Difficulty score (0-1) from collinearity, endogeneity, SNR, data length, channels, interactions, structural breaks
- `demantiq/orchestration/monte_carlo.py`: Batch simulation runner
- `demantiq/orchestration/parallel_runner.py`: Multi-core execution (ProcessPoolExecutor)

**Acceptance:** 10,000 simulations complete in < 60 min on CPU. All 15 scenarios pass output validation. Difficulty scores correctly rank Clean Room < Real World < Adversarial.

### M6: Evaluation Framework (API-only)

**Goal:** Full evaluation suite and model comparison.

**Delivers:**
- `demantiq/evaluation/parameter_recovery.py`: Bias, MAPE, coverage, interval width, rank correlation
- `demantiq/evaluation/contribution_accuracy.py`: Per-channel MAPE, channel ranking (Kendall's tau), zero-detection
- `demantiq/evaluation/roas_accuracy.py`: ROAS estimation accuracy
- `demantiq/evaluation/interaction_detection.py`: Interaction identification accuracy
- `demantiq/evaluation/optimization_quality.py`: Recommended vs. true optimal allocation
- `demantiq/evaluation/capability_surface.py`: Multi-dimensional accuracy heatmaps
- `demantiq/evaluation/model_comparison.py`: Head-to-head benchmarking with adapters

**Acceptance:** Head-to-head comparison of PyMC-Marketing on all 15 scenarios produces structured results with confidence intervals.

### M7: Calibration v1 (API-only)

**Goal:** Empirical parameter distributions and realism validation.

**Delivers:**
- `demantiq/calibration/empirical_distributions.py`: Parameter distribution database (JSON-backed)
- `demantiq/calibration/public_data_adapter.py`: Ingest public financial data benchmarks
- `demantiq/calibration/realism_validator.py`: KS tests, ACF comparison, Fourier spectrum, correlation matrix comparison, spend-to-revenue ratio validation

**Acceptance:** Realism validator runs on all 15 scenarios. Parameter ranges align with Appendix C category benchmarks.

### M8: Training Data Pipeline (API-only)

**Goal:** Pipeline producing 1M+ training tuples for neural density estimator.

**Delivers:**
- Training data export format (config + observable data + ground truth as compressed tuples)
- Streaming generation pipeline for large-scale runs
- Output format validated by ML engineer interface spec

**Acceptance:** Pipeline generates 1M+ tuples. Data format is consumable by a training loop.

### Streamlit UI (incremental, per milestone)

Built alongside each milestone:
- **After M1:** `4_Demantiq_Config.py` (basic channel + baseline config), `6_Demantiq_Generate.py` (run + preview)
- **After M2-M3:** Expand config page with pricing, distribution, competition, macro, endogeneity tabs
- **After M5:** `5_Demantiq_Scenarios.py` (browse/select/customize), `7_Monte_Carlo.py` (batch runs)
- **After M6:** `8_Model_Comparison.py` (results visualization)

## 5. Demand Kernel — 15-Step Pipeline

```python
def simulate(config: SimulationConfig) -> SimulationResult:
    rng = create_rng(config.seed)

    # 1. Generate baseline
    base = baseline_generator(config.baseline, config.n_periods, rng)  # trend + seasonality + organic

    # 2. Generate raw spend (pre-endogeneity)
    spend_raw = spend_generator(config.channels, config.n_periods, rng)  # correlated, flighted

    # 3. Apply endogeneity (requires feedback from previous outcome — iterative)
    spend = endogeneity_layer(spend_raw, base, config.endogeneity, rng)

    # 4. Apply adstock per channel
    adstocked = {c.name: adstock(spend[c.name], c.adstock_fn, c.adstock_params) for c in config.channels}

    # 5. Apply saturation per channel
    saturated = {c.name: saturation(adstocked[c.name], c.saturation_fn, c.saturation_params) for c in config.channels}

    # 6. Compute media effects
    media = {c.name: c.beta * saturated[c.name] for c in config.channels}

    # 7. Apply interactions
    media_modified = apply_interactions(media, config.interactions, price, distribution, competition)

    # 8. Compute price effect
    price_effect = pricing_engine(config.pricing, config.n_periods, rng)

    # 9. Apply distribution cap
    dist_cap = distribution_generator(config.distribution, config.n_periods, rng)

    # 10. Compute competition effect
    comp_effect = competition_generator(config.competition, config.n_periods, rng)

    # 11. Compute macro effects
    macro_effect = macro_generator(config.macro, config.n_periods, rng)

    # 12. Aggregate demand
    demand = (base + sum(media_modified.values()) + price_effect + comp_effect + macro_effect) * dist_cap

    # 13. Add noise
    noise = noise_model(config.noise, demand, rng)
    outcome = demand + noise

    # 14. Compute revenue
    revenue = outcome * price

    # 15. Compute margin
    margin = revenue - (outcome * config.pricing.cost_structure.cogs) - total_media_cost

    return SimulationResult(observable_data, ground_truth, summary_truth, config)
```

## 6. Configuration System

All configs are frozen dataclasses with defaults, validation, and JSON/YAML serialization.

```python
@dataclass(frozen=True)
class SimulationConfig:
    n_periods: int = 104
    granularity: str = "weekly"  # "weekly" | "daily"
    channels: list[ChannelConfig] = field(default_factory=list)
    pricing: PricingConfig | None = None
    distribution: DistributionConfig | None = None
    competition: CompetitionConfig | None = None
    macro: MacroConfig | None = None
    endogeneity: EndogeneityConfig | None = None
    interactions: InteractionConfig | None = None
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    seed: int = 42
    metadata: dict = field(default_factory=dict)
```

Optional configs (pricing, distribution, competition, macro, endogeneity, interactions) default to `None` — the demand kernel skips those steps when `None`. This means a minimal config with just channels produces output equivalent to the existing Josiah PyMC engine.

## 7. Ground Truth Ledger

Every simulation produces three outputs:

1. **`observable_data`** (DataFrame) — what the model sees: date, outcome, revenue, per-channel spend, price, distribution, competitor SOV, macro variables, holidays
2. **`ground_truth`** (DataFrame) — per-period truth: true baseline, true per-channel contributions, true price/distribution/competition/macro effects, true interactions, true noise, confounders, exogenous spend
3. **`summary_truth`** (dict) — aggregate truth: true betas, true saturation/adstock params, true ROAS/mROAS, true elasticities, true interactions, true optimal allocation, scenario difficulty score, full config

## 8. Agent Orchestration

### Coordinator Workflow (`docs/workflows/coordinator.yml`)

The coordinator:
1. Runs M1 (core DGP) — blocks until validated
2. Runs M2 and M3 in parallel — blocks until both validated
3. Runs M4 (interactions) — blocks until validated
4. Runs M5 (scenarios + MC) — blocks until validated
5. Runs M6 (evaluation) — blocks until validated
6. Runs M7 (calibration) — blocks until validated
7. Runs M8 (training data) — blocks until validated
8. Builds Streamlit UI incrementally after each relevant milestone

### Per-Milestone Workflow Pattern

Each milestone workflow:
1. **Builder agent(s)**: Implement the code (config, generators, core, tests)
2. **Test agent**: Run the full test suite, ensure no regressions
3. **Validator agent**: Run milestone-specific acceptance criteria
4. **Gate**: Only advance if all tests pass and acceptance criteria met

### Agent Roles

| Agent | Responsibility |
|-------|---------------|
| `coordinator` | Sequences milestones, manages dependencies, reports status |
| `builder` | Writes implementation code for a specific module |
| `test-writer` | Writes unit, integration, and property-based tests |
| `validator` | Runs acceptance criteria, verifies ground truth correctness |
| `ui-builder` | Builds Streamlit pages after backend milestones complete |

## 9. Streamlit UI Design

The app.py landing page gains a toggle: "Simple Generator (Josiah)" vs "Demantiq Simulator."

**Demantiq pages:**

- **Config page** (`4_Demantiq_Config.py`): Tabbed interface — Channels, Baseline, Pricing, Distribution, Competition, Macro, Endogeneity, Interactions, Noise. Each tab maps to one config dataclass. Includes scale presets and "load from scenario" dropdown.
- **Scenarios page** (`5_Demantiq_Scenarios.py`): Browse SCN-001 to SCN-015 with difficulty badges, description, and parameter summary. Select → customize → generate.
- **Generate page** (`6_Demantiq_Generate.py`): Run simulation, show decomposition chart, ground truth summary, download observable data + ground truth + config.
- **Monte Carlo page** (`7_Monte_Carlo.py`): Configure batch (n_scenarios, n_seeds, sampler params), launch run, progress bar, download results ZIP.
- **Model Comparison page** (`8_Model_Comparison.py`): Upload model outputs, compare against ground truth, show accuracy metrics and rankings.

## 10. Testing Strategy

| Test Type | Scope | Framework |
|-----------|-------|-----------|
| Unit tests | Every transform, generator, config validator | pytest |
| Integration tests | Full pipeline for all 15 scenarios | pytest |
| Property-based tests | Contributions sum, ROAS positivity, monotonicity | pytest + hypothesis |
| Statistical tests | Parameter recovery on SCN-001 with OLS oracle | pytest + scipy.stats |
| Performance tests | Single sim < 50ms, batch throughput benchmarks | pytest-benchmark |

## 11. Verification Plan

After all milestones complete:
1. All 15 named scenarios generate valid output (DataFrame structure, no NaNs, correct columns)
2. Ground truth ledger is internally consistent (contributions sum to total, ROAS = contribution/spend)
3. SCN-001 parameter recovery with OLS achieves > 95% coverage
4. SCN-003 endogeneity bias matches analytical prediction (Appendix B) within 10%
5. 10,000 batch simulations complete in < 60 min
6. Streamlit app renders all pages without errors
7. Full test suite passes with > 95% coverage on `demantiq/` package
