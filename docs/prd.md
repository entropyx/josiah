# Demantiq Synthetic Dataset Generator — Product Requirements Document

**Version:** 1.0  
**Author:** Entropy  
**Date:** March 2026  
**Status:** Draft  
**Classification:** Confidential

---

## 1. Overview

### 1.1 Purpose

The Demantiq Synthetic Dataset Generator (hereafter "the Simulator") is a configurable engine that produces realistic synthetic business datasets where every parameter, contribution, and interaction is known with certainty. It serves two primary functions:

1. **Benchmarking engine** — Evaluate and compare the accuracy of any MMM/Demand Architecture model against known ground truth across thousands of scenarios.
2. **Training data factory** — Generate millions of labeled datasets to train the neural density estimator that powers Demantiq's inference engine.

### 1.2 Problem Statement

No existing marketing measurement framework provides a systematic way to validate model accuracy. Practitioners evaluate MMM outputs based on plausibility ("the results look reasonable") rather than ground truth. This makes it impossible to:

- Know when a model is wrong and by how much
- Compare model architectures objectively
- Identify which data conditions cause estimation failure
- Quantify the value of experimental calibration (geo-lifts, brand lift studies)

The Simulator solves this by creating a controlled laboratory where the "right answer" is always known.

### 1.3 Success Criteria

| Criterion | Target |
|-----------|--------|
| Generation speed | ≥100,000 synthetic businesses per hour on a single GPU/CPU node |
| Parameter space coverage | Full combinatorial coverage of all configurable dimensions |
| Realism validation | Simulated data indistinguishable from real client datasets on 10+ statistical properties (spend distributions, autocorrelation, seasonality shape, collinearity structure) |
| Ground truth completeness | Every simulation produces a full ledger of true parameters, true contributions, true ROAS, true marginal effects, and true interaction magnitudes |
| Reproducibility | Any scenario can be exactly reproduced from its configuration seed |
| Extensibility | New variable types, transformation functions, or interaction structures can be added without refactoring core architecture |

---

## 2. Scope

### 2.1 In Scope

- Configurable data-generating process (DGP) covering media, pricing, distribution, competition, macro, and all pairwise interactions
- Endogeneity injection at configurable intensities
- Ground truth computation for all parameters and derived metrics
- Pre-built scenario library for systematic benchmarking
- Monte Carlo orchestration for large-scale simulation runs
- Serialization of configurations for reproducibility
- Validation utilities to compare model outputs against ground truth
- Calibration interface to update simulator parameters from real experimental results
- Public financial data integration for realistic parameter distributions

### 2.2 Out of Scope (Handled by Other Demantiq Components)

- The neural embedding network and density estimator (separate PRD)
- The canonical marketing ontology and evidence database (separate PRD)
- Client-facing dashboard and output layer (separate PRD)
- Sales tooling and commercial reporting

### 2.3 Dependencies

| Dependency | Purpose | Status |
|------------|---------|--------|
| JAX | Differentiable simulation, GPU acceleration | Available |
| NumPy / SciPy | Statistical distributions, signal processing | Available |
| Pandas | Tabular data output | Available |
| Existing Entropy geo-lift results | Realism validation targets | Available (GNC, UTEL) |
| Public financial data pipeline | Calibration of realistic parameter ranges | To be built (Phase 5.3 integration) |

---

## 3. Architecture

### 3.1 High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    SimulationConfig                          │
│  (scenario parameters, channel configs, business context)   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Simulation Engine                         │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │    Spend     │  │   Pricing    │  │  Distribution    │  │
│  │  Generator   │  │   Engine     │  │   Generator      │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │
│         │                 │                    │             │
│         ▼                 ▼                    ▼             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Demand Kernel                           │    │
│  │  (saturation → adstock → interactions → baseline)   │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         │                                   │
│  ┌──────────────┐  ┌────┴──────┐  ┌──────────────────┐     │
│  │ Competition  │  │   Macro   │  │   Endogeneity    │     │
│  │   Module     │  │  Module   │  │     Layer        │     │
│  └──────┬───────┘  └────┬──────┘  └────────┬─────────┘     │
│         │               │                  │                │
│         └───────────────┴──────────────────┘                │
│                         │                                   │
│                         ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Ground Truth Ledger                     │    │
│  │  (true betas, true curves, true contributions,      │    │
│  │   true ROAS, true interactions, true margins)       │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      Outputs                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Observable   │  │ Ground Truth │  │   Evaluation     │  │
│  │  Dataset      │  │   Ledger     │  │   Utilities      │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 File Structure

```
demantiq_simulator/
├── __init__.py
├── config/
│   ├── __init__.py
│   ├── simulation_config.py      # Master config dataclass
│   ├── channel_config.py         # Per-channel parameter config
│   ├── pricing_config.py         # Pricing mechanics config
│   ├── distribution_config.py    # Distribution/availability config
│   ├── competition_config.py     # Competitive dynamics config
│   ├── macro_config.py           # External variables config
│   ├── endogeneity_config.py     # Feedback/confounding config
│   ├── interaction_config.py     # Cross-variable interaction config
│   └── noise_config.py           # Error term configuration
│
├── generators/
│   ├── __init__.py
│   ├── spend_generator.py        # Correlated media spend patterns
│   ├── pricing_engine.py         # Price/promo simulation
│   ├── distribution_generator.py # Weighted distribution over time
│   ├── competition_generator.py  # Competitive SOV and activity
│   ├── macro_generator.py        # External variable time series
│   └── baseline_generator.py     # Trend + seasonality + organic
│
├── transforms/
│   ├── __init__.py
│   ├── saturation.py             # Hill, logistic, power, custom
│   ├── adstock.py                # Geometric, Weibull, delayed, custom
│   └── interactions.py           # Cross-variable interaction functions
│
├── core/
│   ├── __init__.py
│   ├── demand_kernel.py          # Main DGP: combines all components
│   ├── endogeneity_layer.py      # Feedback loops and confounding injection
│   └── noise_model.py            # Heteroscedastic and correlated noise
│
├── ground_truth/
│   ├── __init__.py
│   ├── contributions.py          # True channel/lever contributions
│   ├── roas_calculator.py        # True ROAS per channel
│   ├── margin_attribution.py     # True margin decomposition
│   ├── interaction_effects.py    # True interaction magnitudes
│   ├── elasticities.py           # True price/media elasticities
│   └── counterfactuals.py        # True "what if" scenario outcomes
│
├── scenarios/
│   ├── __init__.py
│   ├── scenario_library.py       # Named pre-built scenarios
│   ├── scenario_sampler.py       # Random scenario generation for Monte Carlo
│   └── difficulty_scorer.py      # Quantifies scenario estimation difficulty
│
├── orchestration/
│   ├── __init__.py
│   ├── monte_carlo.py            # Batch simulation runner
│   ├── parallel_runner.py        # Multi-core/GPU parallelization
│   └── serializer.py             # Config save/load for reproducibility
│
├── evaluation/
│   ├── __init__.py
│   ├── parameter_recovery.py     # Compare estimated vs. true params
│   ├── contribution_accuracy.py  # Channel decomposition accuracy
│   ├── roas_accuracy.py          # ROAS estimation accuracy
│   ├── interaction_detection.py  # Can the model find true interactions?
│   ├── optimization_quality.py   # Does recommended allocation improve outcomes?
│   ├── capability_surface.py     # Map accuracy across scenario dimensions
│   └── model_comparison.py       # Head-to-head engine benchmarking
│
├── calibration/
│   ├── __init__.py
│   ├── empirical_distributions.py # Parameter distributions from real data
│   ├── public_data_adapter.py     # Ingest public financial data benchmarks
│   └── realism_validator.py       # Compare synthetic vs. real data statistics
│
└── utils/
    ├── __init__.py
    ├── distributions.py           # Custom probability distributions
    ├── time_series.py             # Fourier, trend, structural break utils
    ├── correlation.py             # Correlation matrix generation (copulas)
    └── random.py                  # Seeded RNG management
```

---

## 4. Functional Requirements

### 4.1 Configuration System

#### FR-CFG-001: Master Simulation Config

The system shall accept a single `SimulationConfig` object that fully specifies the data-generating process. This config must be serializable to JSON/YAML for reproducibility and shareable across team members.

**Required fields:**

| Field | Type | Description |
|-------|------|-------------|
| `n_periods` | int | Number of time periods to simulate (26–520 weeks) |
| `granularity` | enum | `weekly` or `daily` |
| `channels` | list[ChannelConfig] | 1–30 media channel configurations |
| `pricing` | PricingConfig | Pricing mechanics specification |
| `distribution` | DistributionConfig | Availability/coverage specification |
| `competition` | CompetitionConfig | Competitive dynamics specification |
| `macro` | MacroConfig | External variables specification |
| `endogeneity` | EndogeneityConfig | Feedback/confounding specification |
| `interactions` | InteractionConfig | Cross-variable interaction matrix |
| `noise` | NoiseConfig | Error term specification |
| `baseline` | BaselineConfig | Trend, seasonality, organic level |
| `seed` | int | Random seed for exact reproducibility |
| `metadata` | dict | Optional tags (category, region, scenario name) |

#### FR-CFG-002: Channel Config

Each channel is independently configurable:

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Human-readable channel label |
| `beta` | float | True contribution coefficient |
| `saturation_fn` | enum | `hill`, `logistic`, `power`, `linear`, `custom` |
| `saturation_params` | dict | Function-specific parameters (e.g., `K`, `S` for Hill) |
| `adstock_fn` | enum | `geometric`, `weibull`, `delayed_geometric`, `custom` |
| `adstock_params` | dict | Function-specific parameters (e.g., `alpha`, `theta`, `max_lag`) |
| `spend_pattern` | enum | `always_on`, `pulsed`, `seasonal`, `front_loaded`, `custom` |
| `spend_mean` | float | Average weekly spend |
| `spend_std` | float | Spend standard deviation |
| `spend_floor` | float | Minimum spend (0 for channels that go dark) |
| `correlation_group` | str | Channels in same group are correlated (e.g., "digital", "brand") |
| `properties` | dict | Ontology property vector for cross-client mapping |

#### FR-CFG-003: Pricing Config

| Field | Type | Description |
|-------|------|-------------|
| `base_price` | float | Regular/shelf price |
| `price_elasticity` | float | True price elasticity of demand |
| `promo_calendar` | list[PromoEvent] | Scheduled promotions with depth and duration |
| `promo_frequency` | enum | `weekly`, `biweekly`, `monthly`, `quarterly`, `custom` |
| `promo_depth_mean` | float | Average discount percentage |
| `promo_depth_std` | float | Discount variation |
| `price_media_interaction` | float | Coefficient: how promotions modify media lift (0 = no interaction) |
| `price_seasonality_interaction` | float | Coefficient: how elasticity shifts during peak periods |
| `competitor_price_sensitivity` | float | Cross-price elasticity |
| `cost_structure` | CostConfig | COGS, variable costs, contribution margin at base price |

#### FR-CFG-004: Distribution Config

| Field | Type | Description |
|-------|------|-------------|
| `initial_distribution` | float | Starting weighted distribution (0.0–1.0) |
| `distribution_trajectory` | enum | `stable`, `growing`, `declining`, `step_change`, `custom` |
| `trajectory_params` | dict | Growth rate, step-change date/magnitude, etc. |
| `distribution_ceiling_effect` | float | How strongly distribution caps media-driven demand (0 = no ceiling, 1 = hard ceiling) |
| `distribution_media_interaction` | float | Coefficient: media wasted where no distribution |
| `stockout_probability` | float | Random stockout events probability per period |
| `stockout_demand_loss` | float | Fraction of demand lost during stockout |

#### FR-CFG-005: Competition Config

| Field | Type | Description |
|-------|------|-------------|
| `n_competitors` | int | Number of simulated competitors |
| `competitor_sov_mean` | float | Average competitor share of voice |
| `competitor_sov_pattern` | enum | `stable`, `seasonal`, `reactive`, `random` |
| `sov_suppression_coefficient` | float | How much competitor SOV reduces own media effectiveness |
| `competitor_price_actions` | bool | Whether competitors adjust pricing reactively |
| `competitive_intensity_trend` | enum | `stable`, `increasing`, `decreasing` |

#### FR-CFG-006: Macro Config

| Field | Type | Description |
|-------|------|-------------|
| `variables` | list[MacroVariable] | External variable definitions |
| `regime_changes` | list[RegimeChange] | Structural breaks with timing and magnitude |

Each `MacroVariable`:

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Variable label (e.g., "consumer_confidence", "category_index") |
| `effect_on_demand` | float | True coefficient |
| `time_series_type` | enum | `random_walk`, `mean_reverting`, `trending`, `seasonal`, `from_data` |
| `params` | dict | Type-specific parameters |
| `correlation_with_spend` | float | How much this variable correlates with media spend decisions (creates OVB) |

Each `RegimeChange`:

| Field | Type | Description |
|-------|------|-------------|
| `period` | int | When the break occurs |
| `type` | enum | `level_shift`, `trend_break`, `variance_change`, `parameter_change` |
| `magnitude` | float | Size of the shift |
| `affected_params` | list[str] | Which parameters change (e.g., "baseline", "channel_3.beta") |
| `recovery` | enum | `permanent`, `gradual_recovery`, `v_shaped` |
| `recovery_periods` | int | Periods to recover (if applicable) |

#### FR-CFG-007: Endogeneity Config

| Field | Type | Description |
|-------|------|-------------|
| `overall_strength` | float | Global endogeneity intensity (0.0 = exogenous, 1.0 = fully endogenous) |
| `feedback_lag` | int | How many periods back the outcome feeds into spend decisions |
| `feedback_channels` | list[str] | Which channels react to outcome signals |
| `seasonal_allocation_bias` | float | How much spend increases during high-baseline periods |
| `performance_chasing` | float | How much spend chases recent performance |
| `algorithmic_targeting_bias` | float | Simulates platform optimization (targets likely converters) |
| `price_endogeneity` | float | How much pricing reacts to demand signals |
| `distribution_endogeneity` | float | How much distribution expands into high-demand areas |

#### FR-CFG-008: Interaction Config

| Field | Type | Description |
|-------|------|-------------|
| `price_x_media` | dict[str, float] | Per-channel price×media interaction coefficient |
| `distribution_x_media` | dict[str, float] | Per-channel distribution×media interaction |
| `media_x_media` | dict[tuple, float] | Pairwise media synergy/cannibalization coefficients |
| `price_x_seasonality` | float | How price elasticity varies with seasonal demand |
| `competition_x_media` | dict[str, float] | How competitive pressure modifies per-channel effectiveness |
| `custom_interactions` | list[CustomInteraction] | Arbitrary interaction specifications |

#### FR-CFG-009: Noise Config

| Field | Type | Description |
|-------|------|-------------|
| `noise_type` | enum | `gaussian`, `t_distributed`, `heteroscedastic`, `autocorrelated` |
| `noise_scale` | float | Base noise standard deviation |
| `signal_to_noise_ratio` | float | Alternative specification: SNR target |
| `autocorrelation` | float | AR(1) coefficient for error term |
| `heteroscedasticity_fn` | str | If heteroscedastic: how variance scales with demand level |
| `outlier_probability` | float | Probability of outlier observations per period |
| `outlier_magnitude` | float | Scale factor for outlier observations |

---

### 4.2 Transformation Functions

#### FR-TRN-001: Saturation Functions

All saturation functions must:
- Accept a spend vector and parameter dict
- Return a transformed vector in [0, 1] range (normalized) or [0, ∞) range (unnormalized)
- Be differentiable (JAX-compatible)
- Include the inverse function for counterfactual computation

**Required implementations:**

| Function | Formula | Parameters |
|----------|---------|------------|
| Hill | `x^S / (K^S + x^S)` | `K` (half-saturation), `S` (shape) |
| Logistic | `1 / (1 + exp(-k(x - x0)))` | `k` (steepness), `x0` (midpoint) |
| Power | `x^α` where 0 < α < 1 | `alpha` (diminishing returns rate) |
| Linear | `x` (identity) | None |
| Piecewise linear | Linear segments with breakpoints | `breakpoints`, `slopes` |
| Custom | User-provided JAX function | Arbitrary |

#### FR-TRN-002: Adstock/Carryover Functions

All adstock functions must:
- Accept a spend vector, parameter dict, and max_lag
- Return a transformed vector of same length
- Be differentiable
- Support computation of effective half-life for interpretability

**Required implementations:**

| Function | Description | Parameters |
|----------|-------------|------------|
| Geometric | Exponential decay: `y_t = x_t + α·y_{t-1}` | `alpha` (retention rate, 0–1) |
| Weibull CDF | Delayed peak followed by decay | `shape`, `scale` |
| Weibull PDF | Alternative parameterization with sharper peak | `shape`, `scale` |
| Delayed geometric | Zero effect for `d` periods, then geometric decay | `alpha`, `delay` |
| Custom | User-provided JAX function | Arbitrary |

#### FR-TRN-003: Interaction Functions

All interaction functions must:
- Accept two variable vectors and an interaction coefficient
- Return a modifier vector that scales the effect of the primary variable
- Default to multiplicative interaction: `effect_modified = effect_base × (1 + γ·z_t)`
- Support additive alternative: `effect_modified = effect_base + γ·z_t`
- Be differentiable

---

### 4.3 Demand Kernel

#### FR-DK-001: Core Computation Pipeline

The demand kernel computes the outcome in a defined order:

```
1. Generate baseline:        base_t = trend_t + seasonality_t + organic_level
2. Generate raw spend:       x_ct (from spend generator, pre-endogeneity)
3. Apply endogeneity:        x_ct* = endogeneity_layer(x_ct, y_{t-lag}, base_t)
4. Apply adstock per channel: a_ct = adstock(x_ct*, params_c)
5. Apply saturation per chan: s_ct = saturation(a_ct, params_c)
6. Compute media effects:     m_ct = β_c × s_ct
7. Apply interactions:        m_ct* = m_ct × (1 + Σ γ_interactions × z_variables)
8. Compute price effect:      p_t = β_price × price_transform(price_t)
9. Apply distribution cap:    cap_t = f(distribution_t)
10. Compute competition eff:  comp_t = β_comp × competitor_sov_t
11. Compute macro effects:    macro_t = Σ β_macro_j × macro_j_t
12. Aggregate demand:         D_t = (base_t + Σ m_ct* + p_t + comp_t + macro_t) × cap_t
13. Add noise:                y_t = D_t + ε_t
14. Compute revenue:          rev_t = y_t × price_t
15. Compute margin:           margin_t = rev_t - (y_t × cogs_t) - media_cost_t
```

#### FR-DK-002: Time-Varying Parameters

The kernel must support parameters that change over time:

| Mechanism | Description | Use Case |
|-----------|-------------|----------|
| Structural breaks | Discrete parameter changes at specified periods | COVID, new competitor entry |
| Smooth transitions | Sigmoid transition between parameter values | Gradual market maturation |
| Seasonal variation | Parameters follow seasonal pattern | Price elasticity varies with season |
| Random walks | Parameters drift stochastically | Slow brand equity erosion |

Configuration per parameter:

```
time_varying_spec = {
    "parameter": "channel_3.beta",
    "type": "structural_break",
    "break_period": 104,
    "pre_value": 0.15,
    "post_value": 0.08,
    "transition": "immediate"  # or "sigmoid" with transition_width
}
```

#### FR-DK-003: Multi-Product Support

The kernel must optionally support multiple products with:
- Separate demand functions per product sharing some channels
- Cross-product cannibalization coefficients
- Shared baseline components (brand equity, distribution)
- Product-level pricing with cross-price elasticity
- Portfolio-level margin computation

---

### 4.4 Spend Generator

#### FR-SPN-001: Correlated Spend Patterns

The spend generator must produce realistic media spend data with:

| Requirement | Description |
|-------------|-------------|
| Configurable correlation | Multivariate distribution with user-specified correlation matrix or copula |
| Pattern types | Always-on (low variance), pulsed (on/off flights), seasonal, front-loaded, custom |
| Budget constraints | Total spend per channel per year can be fixed (optimizer-like behavior) |
| Zero inflation | Some channels have periods of zero spend (flights) |
| Non-negativity | All spend values ≥ 0 |
| Realistic distributions | Log-normal or gamma marginals (not normal — spend is right-skewed) |

#### FR-SPN-002: Flighting Patterns

For pulsed/flighted channels:

| Parameter | Description |
|-----------|-------------|
| `flight_duration_weeks` | Average length of active periods |
| `dark_duration_weeks` | Average length of inactive periods |
| `flight_spend_multiplier` | How much higher spend is during flights vs. baseline |
| `flight_timing` | `random`, `seasonal`, `quarterly`, `custom_calendar` |

#### FR-SPN-003: Correlation Groups

Channels assigned to the same `correlation_group` should have their spend correlated. Typical groups:
- "brand" — TV, YouTube, OOH (flighted together for brand campaigns)
- "performance" — Search, Shopping, Retargeting (always-on, budget-linked)
- "seasonal" — All channels spike during holidays but at different magnitudes

The generator should support:
- Within-group correlation coefficient
- Between-group correlation coefficient (lower)
- Lagged correlation (TV spend precedes Search lift by N periods)

---

### 4.5 Endogeneity Layer

#### FR-END-001: Feedback Mechanism

The endogeneity layer modifies generated spend based on outcome history:

```
x_ct_endogenous = x_ct_exogenous + strength × f(y_{t-lag}, ..., y_{t-lag-k})
```

Where `f` can be:
- **Linear feedback**: Spend scales with recent outcome level
- **Threshold feedback**: Spend increases only when outcome exceeds target
- **Momentum feedback**: Spend increases when outcome is trending up
- **Algorithmic feedback**: Simulates platform auto-bidding (spend concentrates on high-propensity periods)

#### FR-END-002: Seasonal Allocation Bias

Spend increases during periods when baseline demand is already high:

```
x_ct_endogenous = x_ct_exogenous × (1 + bias × seasonality_t / max(seasonality))
```

This creates the classic confound where the model cannot distinguish media lift from seasonal lift.

#### FR-END-003: Omitted Variable Confounding

Generate hidden variables that correlate with both spend and outcome but are not included in the observable dataset:

```
z_t ~ AR(1) process
x_ct_confounded = x_ct + α_spend × z_t
y_t_confounded = y_t + α_outcome × z_t
```

The observable dataset contains `x_ct_confounded` and `y_t_confounded` but not `z_t`. The ground truth ledger contains `z_t` and the true unconfounded contributions.

---

### 4.6 Ground Truth Ledger

#### FR-GT-001: Required Ground Truth Outputs

For every simulation run, the system must produce:

| Output | Description | Granularity |
|--------|-------------|-------------|
| `true_betas` | True contribution coefficient per channel/lever | Per channel |
| `true_saturation_params` | True saturation function parameters | Per channel |
| `true_adstock_params` | True adstock function parameters | Per channel |
| `true_contributions_absolute` | True demand units attributable to each lever | Per channel × per period |
| `true_contributions_pct` | Share of total demand per lever | Per channel × per period |
| `true_roas` | True return on ad spend per channel | Per channel (period and total) |
| `true_mroas` | True marginal ROAS at current spend level | Per channel |
| `true_price_elasticity` | True price elasticity of demand | Per period (if time-varying) |
| `true_interaction_effects` | True magnitude of each interaction | Per interaction pair × per period |
| `true_baseline` | True baseline demand (trend + seasonality + organic) | Per period |
| `true_counterfactuals` | Demand/revenue under zero-spend per channel | Per channel |
| `true_margin_decomposition` | Margin attributable to each lever | Per channel × per period |
| `true_optimal_allocation` | Budget allocation that maximizes margin (under true DGP) | Per channel |
| `endogeneity_bias` | Difference between naive OLS estimate and true effect | Per channel |
| `confounders` | Hidden variables generating OVB (not in observable data) | Per period |

#### FR-GT-002: Counterfactual Computation

For each channel, compute the counterfactual: "What would demand have been if this channel had zero spend throughout the simulation?" This must account for:
- Adstock (zeroing out spend removes carryover from previous periods)
- Interactions (removing one channel may reduce the effectiveness of synergistic channels)
- Distribution ceiling (some lost demand may not have been serviceable anyway)

#### FR-GT-003: Optimal Allocation Computation

Using the true DGP, solve the budget optimization problem:
- Given total budget B across all channels
- Find allocation {x_c} that maximizes total margin
- Subject to: Σx_c ≤ B, x_c ≥ 0, and optional per-channel min/max constraints
- Use gradient-based optimization on the differentiable DGP (JAX)

This ground truth optimal allocation is compared against the optimal allocation recommended by whatever model is being evaluated.

---

### 4.7 Scenario Library

#### FR-SCN-001: Pre-Built Named Scenarios

| Scenario ID | Name | Purpose | Key Parameters |
|-------------|------|---------|----------------|
| `SCN-001` | Clean Room | Baseline accuracy test | No endogeneity, low noise, orthogonal spend, 156 weeks, 5 channels |
| `SCN-002` | Real World | Practical accuracy test | Moderate endogeneity (0.3), correlated spend (ρ=0.4), seasonal confounding, 104 weeks, 8 channels |
| `SCN-003` | Adversarial | Robustness test | High endogeneity (0.7), heavy collinearity (ρ=0.8), structural break, low SNR, 52 weeks, 12 channels |
| `SCN-004` | Pricing Dominant | Media attribution accuracy | 60% of revenue variation from pricing, media contributes only 8% | 
| `SCN-005` | Interaction Heavy | Synergy detection | Large price×media (+25%) and distribution×media (+30%) interactions |
| `SCN-006` | Short Data | Minimum data test | Only 26–52 weeks, moderate complexity |
| `SCN-007` | Many Channels | Dimensionality test | 15–20 channels, moderate collinearity, 104 weeks |
| `SCN-008` | Regime Shift | Adaptation test | COVID-like break at week 78, baseline drops 40%, 2 channels go dark |
| `SCN-009` | New Brand | Cold start test | Rapid distribution growth, increasing baseline, short history (52 weeks) |
| `SCN-010` | Mature Market | Saturation detection | Most channels near saturation, incremental returns are small |
| `SCN-011` | Promotional Trap | Price×media confound | Heavy promo calendar correlated with media flights, tests interaction identification |
| `SCN-012` | Platform Bias | Algorithmic targeting | High algorithmic_targeting_bias simulating Performance Max / Meta ASC behavior |
| `SCN-013` | Competitor Entry | External shock | New competitor enters at week 60, SOV doubles, own effectiveness suppressed |
| `SCN-014` | DTC Pure Play | E-commerce specific | No distribution variable, high Search/Social dependency, daily granularity |
| `SCN-015` | Omnichannel Retail | Full complexity | Retail + DTC + Marketplace, distribution, trade spend, 20 channels, all interactions |

#### FR-SCN-002: Scenario Difficulty Scoring

Each scenario must be tagged with a difficulty score (0.0–1.0) computed from:

| Factor | Weight | Calculation |
|--------|--------|-------------|
| Collinearity | 0.20 | Average pairwise |ρ| across channels |
| Endogeneity | 0.20 | Endogeneity strength parameter |
| Signal-to-noise | 0.15 | Inverse of SNR |
| Data length | 0.15 | Inverse of n_periods (normalized) |
| Number of channels | 0.10 | n_channels / 20 |
| Interaction complexity | 0.10 | Number of non-zero interactions / total possible |
| Structural breaks | 0.10 | Number and magnitude of regime changes |

#### FR-SCN-003: Scenario Sampling for Monte Carlo

The `scenario_sampler` generates random scenarios by sampling parameters from configurable distributions:

```python
sampler = ScenarioSampler(
    n_channels=UniformInt(3, 20),
    n_periods=Choice([26, 52, 78, 104, 156]),
    endogeneity_strength=Uniform(0.0, 0.8),
    collinearity=Uniform(0.0, 0.9),
    snr=LogNormal(mu=1.0, sigma=0.5),
    saturation_type=Choice(["hill", "logistic", "power"]),
    adstock_type=Choice(["geometric", "weibull"]),
    has_pricing=Bernoulli(0.7),
    has_distribution=Bernoulli(0.5),
    has_competition=Bernoulli(0.6),
    n_interactions=Poisson(3),
    n_regime_changes=Poisson(0.5),
)

scenarios = sampler.generate(n=10000, seed=42)
```

---

### 4.8 Evaluation Framework

#### FR-EVL-001: Parameter Recovery Metrics

For each parameter θ with true value θ* and estimated posterior p(θ|data):

| Metric | Formula | Target |
|--------|---------|--------|
| Bias | E[θ] - θ* | < 10% of θ* |
| MAPE | |E[θ] - θ*| / |θ*| | < 15% |
| Coverage | P(θ* ∈ 95% CI) across simulations | 90–95% |
| Interval width | Width of 95% CI | Narrower is better (given coverage) |
| Rank correlation | Spearman ρ between true and estimated channel rankings | > 0.85 |

#### FR-EVL-002: Contribution Accuracy Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Total media contribution error | |estimated_total - true_total| / true_total | < 10% |
| Per-channel contribution MAPE | Average |est_c - true_c| / true_c across channels | < 20% |
| Channel ranking accuracy | Kendall's τ between true and estimated rankings | > 0.80 |
| Zero-contribution detection | Correctly identifies channels with true β = 0 | > 90% TPR |

#### FR-EVL-003: Optimization Quality Metric

1. Model recommends optimal allocation A_model given budget B
2. Simulate outcome under true DGP with allocation A_model → revenue_model
3. Compute true optimal allocation A_true → revenue_true
4. Metric: `optimization_efficiency = revenue_model / revenue_true`
5. Target: > 0.90 (model's recommendation captures ≥90% of the true optimal gain)

#### FR-EVL-004: Capability Surface

Map model accuracy across all scenario dimensions simultaneously:

```python
surface = CapabilitySurface(
    dimensions=["collinearity", "endogeneity", "n_channels", "n_periods", "snr"],
    metric="contribution_mape",
    results=monte_carlo_results
)

surface.plot_heatmap(x="collinearity", y="endogeneity", fix={"n_periods": 104})
surface.find_failure_boundary(threshold=0.25)  # Where does accuracy degrade below 25% MAPE?
surface.compare_models(["pymc_marketing", "robyn", "demantiq_v1"])
```

#### FR-EVL-005: Model Comparison Framework

Head-to-head evaluation of multiple engines on identical simulated data:

```python
comparison = ModelComparison(
    models={
        "pymc_marketing": PyMCMarketingAdapter(config),
        "robyn": RobynAdapter(config),
        "meridian": MeridianAdapter(config),
        "demantiq_v1": DemantiqAdapter(config),
    },
    scenarios=scenario_library.get_all(),
    metrics=["contribution_mape", "roas_mape", "coverage", "optimization_efficiency"]
)

comparison.run(n_seeds=50)  # 50 random seeds per scenario
comparison.report()         # Full comparison report with confidence intervals
```

---

### 4.9 Calibration System

#### FR-CAL-001: Empirical Parameter Distributions

Maintain a database of realistic parameter ranges derived from:

| Source | Parameters Calibrated | Update Frequency |
|--------|----------------------|------------------|
| Entropy client experiments (geo-lifts, brand lift) | True channel betas, saturation shapes, adstock decays | After each experiment |
| Public financial data (10-K, 10-Q filings) | Media spend / revenue ratios, category growth rates, margin structures | Quarterly |
| Earnings call NLP | Promotional intensity, competitive dynamics, pricing strategy signals | Quarterly |
| Sector ETFs and market data | Category-level demand trends, macro correlations | Monthly |
| Academic meta-analyses | Advertising elasticity ranges by category | Annually |

#### FR-CAL-002: Realism Validator

Compare synthetic data against real client data on distributional properties:

| Property | Test |
|----------|------|
| Spend distribution shape | KS test against real spend distributions |
| Spend autocorrelation | Compare ACF structure |
| Outcome seasonality shape | Fourier spectrum comparison |
| Channel collinearity pattern | Compare correlation matrices |
| Spend-to-revenue ratio | Within realistic category range |
| Coefficient of variation | Matches observed variability |
| Outlier frequency | Consistent with real data |

If any property fails (p < 0.05 or outside category benchmarks), the validator flags the scenario as unrealistic and suggests parameter adjustments.

---

## 5. Non-Functional Requirements

### 5.1 Performance

| Requirement | Specification |
|-------------|---------------|
| Single simulation (104 weeks, 8 channels) | < 50ms |
| Batch generation (100,000 simulations) | < 60 minutes on a single modern CPU; < 10 minutes with GPU (JAX) |
| Ground truth computation | Adds < 20% overhead to simulation time |
| Monte Carlo with evaluation | < 4 hours for 1,000 scenarios × 50 seeds × 1 model |

### 5.2 Reproducibility

- Every simulation must be exactly reproducible from `SimulationConfig` + `seed`
- Config serialization format: JSON and YAML
- Version-locked RNG implementation (JAX default PRNG)
- All dependencies version-pinned in `pyproject.toml`

### 5.3 Extensibility

- New saturation functions: implement `SaturationFn` protocol, register in `saturation.py`
- New adstock functions: implement `AdstockFn` protocol, register in `adstock.py`
- New variable types: add generator module in `generators/`, add config dataclass in `config/`
- New interaction types: implement `InteractionFn` protocol, register in `interactions.py`
- New evaluation metrics: implement `Metric` protocol, register in `evaluation/`

All extension points use a protocol/interface pattern — no inheritance hierarchies.

### 5.4 Testing

| Test Type | Coverage Target |
|-----------|-----------------|
| Unit tests (per function) | 100% of transformation functions, generators, and ground truth computations |
| Integration tests (full pipeline) | All 15 named scenarios produce valid output |
| Property-based tests | Contributions sum to total demand (within numerical tolerance); ROAS > 0 for positive betas; adstock preserves total spend (approximately); saturation is monotonically non-decreasing |
| Statistical tests | Parameter recovery on SCN-001 (Clean Room) achieves > 95% coverage with a correctly specified oracle model |
| Performance tests | Benchmarks meet NFR targets |

### 5.5 Documentation

- Every config field has a docstring explaining its business meaning, valid range, and default
- Every transformation function has a docstring with formula, parameter descriptions, and example usage
- The scenario library includes a narrative description of each scenario's real-world analog
- A quickstart guide demonstrates generating data, running a model, and evaluating against ground truth in under 20 lines of code

---

## 6. Data Model

### 6.1 Observable Dataset (Model Input)

The output dataset that any MMM engine receives — with NO access to ground truth:

| Column | Type | Description |
|--------|------|-------------|
| `period` | int | Time period index |
| `date` | date | Calendar date |
| `outcome` | float | Target variable (demand units, revenue, or conversions) |
| `revenue` | float | Revenue = outcome × price |
| `channel_{c}_spend` | float | Media spend per channel (post-endogeneity) |
| `channel_{c}_impressions` | float | Optional: impressions (derived from spend + CPM) |
| `price` | float | Effective price after promotions |
| `is_promo` | bool | Whether a promotion is active |
| `promo_depth` | float | Discount percentage (0 if no promo) |
| `distribution` | float | Weighted distribution percentage |
| `competitor_sov` | float | Competitor share of voice (if observable) |
| `macro_{j}` | float | External macro variables |
| `holiday_{h}` | bool | Holiday indicators |
| `month` | int | Month for seasonality controls |
| `week_of_year` | int | Week for finer seasonality |

### 6.2 Ground Truth Ledger (Evaluation Reference)

| Column | Type | Description |
|--------|------|-------------|
| `period` | int | Time period index |
| `true_baseline` | float | True baseline demand (trend + seasonality + organic) |
| `true_channel_{c}_contribution` | float | True demand units from channel c in this period |
| `true_channel_{c}_contribution_pct` | float | True share of demand from channel c |
| `true_price_effect` | float | True demand change from pricing |
| `true_distribution_effect` | float | True demand impact of distribution |
| `true_competition_effect` | float | True demand impact of competitive activity |
| `true_macro_effect` | float | True demand impact of macro variables |
| `true_interaction_{i}_{j}` | float | True interaction effect between variables i and j |
| `true_noise` | float | True error term realization |
| `confounders` | float | Hidden variables not in observable data |
| `exogenous_spend_{c}` | float | Pre-endogeneity spend (what spend would be without feedback) |
| `endogeneity_bias_{c}` | float | Difference between endogenous and exogenous spend |

### 6.3 Summary Ground Truth (Per Simulation)

| Field | Type | Description |
|-------|------|-------------|
| `true_betas` | dict[str, float] | True contribution coefficient per channel |
| `true_saturation_params` | dict[str, dict] | True saturation parameters per channel |
| `true_adstock_params` | dict[str, dict] | True adstock parameters per channel |
| `true_roas` | dict[str, float] | True ROAS per channel (total) |
| `true_mroas` | dict[str, float] | True marginal ROAS per channel (at final spend level) |
| `true_price_elasticity` | float | True price elasticity |
| `true_interactions` | dict[str, float] | True interaction coefficients |
| `true_total_media_contribution_pct` | float | True % of demand from all media |
| `true_optimal_allocation` | dict[str, float] | Margin-maximizing budget split |
| `true_total_margin` | float | Total margin under actual allocation |
| `true_optimal_margin` | float | Total margin under optimal allocation |
| `scenario_difficulty` | float | Computed difficulty score (0–1) |
| `config` | SimulationConfig | Full config for reproducibility |

---

## 7. API Design

### 7.1 Core API

```python
from demantiq_simulator import SimulationConfig, Simulator, ScenarioLibrary

# Option 1: Custom configuration
config = SimulationConfig(
    n_periods=104,
    granularity="weekly",
    channels=[
        ChannelConfig(name="tv", beta=0.15, saturation_fn="hill",
                      saturation_params={"K": 0.5, "S": 2.0},
                      adstock_fn="geometric", adstock_params={"alpha": 0.7},
                      spend_pattern="pulsed", spend_mean=50000, spend_std=20000),
        ChannelConfig(name="search", beta=0.08, saturation_fn="hill",
                      saturation_params={"K": 0.3, "S": 3.0},
                      adstock_fn="geometric", adstock_params={"alpha": 0.2},
                      spend_pattern="always_on", spend_mean=30000, spend_std=5000),
        # ... more channels
    ],
    pricing=PricingConfig(base_price=25.0, price_elasticity=-1.2, ...),
    endogeneity=EndogeneityConfig(overall_strength=0.4, ...),
    interactions=InteractionConfig(price_x_media={"tv": 0.15, "search": 0.05}, ...),
    noise=NoiseConfig(noise_type="gaussian", signal_to_noise_ratio=5.0),
    seed=42
)

sim = Simulator(config)
result = sim.run()

# Access outputs
result.observable_data     # pd.DataFrame — what the model sees
result.ground_truth        # pd.DataFrame — period-level truth
result.summary_truth       # dict — aggregate truth (betas, ROAS, etc.)
result.config              # SimulationConfig — full reproducible config

# Option 2: Named scenario
config = ScenarioLibrary.get("adversarial")
result = Simulator(config).run()

# Option 3: Scenario sampling for Monte Carlo
from demantiq_simulator import ScenarioSampler, MonteCarloRunner

sampler = ScenarioSampler(default_ranges="realistic")
scenarios = sampler.generate(n=1000, seed=42)

runner = MonteCarloRunner(scenarios, n_seeds_per_scenario=10)
results = runner.run(parallel=True, n_workers=8)
```

### 7.2 Evaluation API

```python
from demantiq_simulator.evaluation import (
    ParameterRecovery, ContributionAccuracy,
    OptimizationQuality, CapabilitySurface, ModelComparison
)

# Single model evaluation against one simulation
eval_result = ParameterRecovery.evaluate(
    estimated=model_output,
    truth=result.summary_truth
)
print(eval_result.bias, eval_result.mape, eval_result.coverage)

# Contribution accuracy
contrib_eval = ContributionAccuracy.evaluate(
    estimated_contributions=model_contributions,
    true_contributions=result.ground_truth
)
print(contrib_eval.per_channel_mape, contrib_eval.ranking_correlation)

# Full model comparison across scenarios
comparison = ModelComparison(
    models={"pymc": pymc_adapter, "robyn": robyn_adapter, "demantiq": demantiq_adapter},
    scenarios=scenarios,
    metrics=["contribution_mape", "roas_mape", "coverage", "optimization_efficiency"]
)
report = comparison.run()
report.to_dataframe()
report.plot_comparison()
report.export_html("model_comparison.html")
```

### 7.3 Calibration API

```python
from demantiq_simulator.calibration import (
    EmpiricalDistributions, RealismValidator
)

# Update parameter distributions from a new geo-lift result
empirical = EmpiricalDistributions.load("calibration_db.json")
empirical.add_observation(
    lever_type="paid_social_upper_funnel",
    business_context={"category": "supplements", "price_point": "medium"},
    parameter="beta",
    value=0.12,
    confidence_interval=(0.08, 0.18),
    source="geo_lift",
    client_id="anon_001"
)
empirical.save("calibration_db.json")

# Validate a scenario's realism
validator = RealismValidator(reference_data=real_client_data)
report = validator.validate(result.observable_data)
print(report.pass_fail, report.flagged_properties)
```

---

## 8. Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Simulator too simplistic: real-world dynamics not captured | Engine trained on unrealistic data, poor real-world accuracy | Medium | Realism validator against real client data; continuous calibration from experiments and public data |
| Simulator too complex: configuration becomes unwieldy | Slow adoption, configuration errors, maintenance burden | Medium | Strong defaults per category; named scenarios for common cases; config validation with clear error messages |
| Overfitting to simulator: density estimator learns simulator artifacts | Good synthetic performance, poor real-world transfer | High (critical) | Diverse scenario sampling; regularization; real experimental validation as the ultimate test; deliberately vary DGP functional forms |
| Performance bottleneck at scale | Cannot generate enough training data for neural engine | Low | JAX GPU acceleration; parallel generation; incremental training |
| Ground truth computation errors | False conclusions about model accuracy | Medium | Extensive property-based testing; analytical verification on simple cases; cross-validation with known analytical solutions |
| Calibration data insufficient | Parameter distributions based on too few real experiments | High (early stage) | Supplement with public financial data and academic meta-analyses; explicitly track calibration confidence per parameter |

---

## 9. Milestones

| Milestone | Target Date | Deliverable | Acceptance Criteria |
|-----------|-------------|-------------|---------------------|
| M1: Core DGP | Month 1 | Demand kernel with media channels, saturation, adstock, baseline, noise | SCN-001 (Clean Room) generates valid data; OLS on clean data recovers true betas within 5% |
| M2: Pricing + Distribution | Month 1.5 | Pricing engine, distribution module, price×media interaction | SCN-004 (Pricing Dominant) and SCN-011 (Promotional Trap) generate valid data |
| M3: Endogeneity + Competition | Month 2 | Full endogeneity layer, competition module, macro variables | SCN-003 (Adversarial) generates data where naive OLS is demonstrably biased; bias matches theoretical prediction |
| M4: Full Interaction Matrix | Month 2.5 | All pairwise interactions configurable | SCN-005 (Interaction Heavy) generates data with known interactions; ground truth ledger correct |
| M5: Scenario Library + Monte Carlo | Month 3 | All 15 named scenarios, Monte Carlo runner, parallel execution | 10,000 simulations complete in < 60 min; all scenarios pass realism validation |
| M6: Evaluation Framework | Month 3.5 | Full evaluation suite, model comparison framework, capability surface | Head-to-head comparison of PyMC-Marketing vs. Robyn on all 15 scenarios with publishable results |
| M7: Calibration v1 | Month 4 | Empirical distributions from Entropy experiments + public data | Parameter ranges validated against GNC and UTEL real data |
| M8: Density Estimator Training Data | Month 4 | Pipeline producing 1M+ training tuples for neural engine | Data format validated by ML engineer; training pipeline consumes data successfully |

---

## 10. Open Questions

| # | Question | Owner | Deadline |
|---|----------|-------|----------|
| OQ-1 | Should the simulator support daily granularity from v1, or start with weekly only? Daily increases complexity but is needed for DTC/e-commerce clients. | Arturo | M1 |
| OQ-2 | Should multi-product support be in v1 or deferred? Retail clients (GNC with multiple SKU categories) would benefit, but it significantly increases the state space. | Arturo | M2 |
| OQ-3 | What is the minimum set of named scenarios needed before the evaluation framework is useful for sales conversations? | Arturo | M5 |
| OQ-4 | How should the calibration database handle conflicting evidence (e.g., two geo-lifts for similar channels in similar categories with very different results)? | Arturo + ML Eng | M7 |
| OQ-5 | Should the simulator support geo-level data generation (multiple regions per simulation) for testing geo-lift experiment designs? | Arturo | M3 |
| OQ-6 | What is the licensing model for the open-source simulator release vs. proprietary evaluation framework? | Arturo | M6 |

---

## Appendix A: Transformation Function Reference

### A.1 Hill Saturation

```
f(x) = x^S / (K^S + x^S)

Parameters:
  K (float): Half-saturation point. Spend level at which response = 50% of maximum.
             Higher K = channel needs more spend to saturate.
             Typical range: 0.1–0.9 (normalized spend)
  S (float): Shape/steepness. Controls how quickly diminishing returns set in.
             S=1: gentle curve. S=5: sharp threshold.
             Typical range: 0.5–5.0

Properties:
  - f(0) = 0
  - f(K) = 0.5
  - f(∞) → 1
  - Monotonically increasing
  - Concave for S ≥ 1
```

### A.2 Geometric Adstock

```
y_t = x_t + α × y_{t-1}

Parameters:
  α (float): Retention/decay rate. Fraction of previous period's effect that carries over.
             α=0: no carryover. α=0.9: very long carryover.
             Typical range: 0.0–0.95

Derived:
  Half-life = log(0.5) / log(α) periods
  Mean lag = α / (1 - α) periods

Typical values by channel:
  Search:    α = 0.1–0.3  (half-life < 1 week)
  Social:    α = 0.2–0.5  (half-life 1–2 weeks)
  Display:   α = 0.1–0.4  (half-life < 2 weeks)
  CTV:       α = 0.4–0.7  (half-life 2–4 weeks)
  Linear TV: α = 0.5–0.8  (half-life 2–6 weeks)
  OOH:       α = 0.3–0.6  (half-life 1–3 weeks)
  Print:     α = 0.6–0.85 (half-life 3–8 weeks)
```

### A.3 Weibull Adstock (CDF parameterization)

```
w(t) = 1 - exp(-(t/λ)^k)

Parameters:
  λ (float): Scale. Controls the time scale of the effect.
  k (float): Shape.
             k < 1: peak at t=0, faster-than-exponential decay
             k = 1: equivalent to geometric decay
             k > 1: delayed peak, then decay (useful for TV brand effects)

Typical values:
  Immediate response channels: k=0.5–1.0, λ=1–3
  Delayed response channels:   k=1.5–3.0, λ=3–8
```

---

## Appendix B: Endogeneity Bias Analytical Benchmarks

For validation, the simulator should produce endogeneity bias that matches analytical predictions in simple cases:

### B.1 Simple Feedback Case

With outcome y_t = β·x_t + ε_t and endogenous spend x_t = x_t^exo + γ·y_{t-1}:

```
OLS bias ≈ γ·β·σ²_x / (σ²_x + γ²·σ²_y)
```

The simulator must produce empirical bias (from naive OLS on simulated data) within 10% of this analytical prediction across 1,000 simulation runs with the same configuration.

### B.2 Omitted Variable Case

With hidden confounder z_t affecting both spend (coefficient α_spend) and outcome (coefficient α_outcome):

```
OLS bias ≈ α_outcome × Cov(x, z) / Var(x) ≈ α_outcome × α_spend × σ²_z / σ²_x
```

Same validation requirement: empirical bias within 10% of analytical prediction.

---

## Appendix C: Realism Benchmarks by Category

Target distributional properties derived from real client data and public financial sources:

| Category | Media/Revenue Ratio | Price Elasticity Range | Typical Channels | Seasonal Amplitude |
|----------|--------------------|-----------------------|------------------|--------------------|
| Supplements/Vitamins | 3–8% | -0.8 to -2.0 | 5–12 | Moderate (Jan, Sep peaks) |
| DTC Skincare/Beauty | 15–30% | -1.0 to -2.5 | 4–8 | Moderate (holiday peak) |
| QSR/Fast Food | 4–8% | -0.5 to -1.5 | 6–10 | Low-moderate |
| Financial Services | 5–12% | -0.3 to -1.0 | 8–15 | Low (tax season, Q4) |
| Online Education | 10–25% | -1.5 to -3.0 | 4–8 | High (Jan, Aug enrollment) |
| Consumer Electronics | 3–7% | -1.5 to -3.0 | 8–15 | Very high (holiday, Prime Day) |
| Automotive | 2–4% | -0.5 to -1.5 | 10–20 | Moderate (model year, holidays) |
| CPG/FMCG | 5–12% | -1.0 to -2.5 | 8–15 | Moderate-high |

These benchmarks are used by the Realism Validator and should be updated quarterly from public financial data.
