"""SCN-001 Clean Room integration test.

Acceptance criteria: OLS on clean data recovers true betas within 5%.
This is the M1 acceptance test — if this passes, M1 is complete.
"""

import numpy as np
import pytest
from demantiq import Simulator, SimulationConfig, ChannelConfig, BaselineConfig, NoiseConfig


def _scn001_config() -> SimulationConfig:
    """SCN-001 Clean Room: simple setup, OLS should recover betas."""
    # Each channel gets its own correlation_group for orthogonal (uncorrelated) spend.
    # Betas are well-separated to ensure OLS can distinguish ranking.
    channels = [
        ChannelConfig(
            name="tv", beta=500.0,
            saturation_fn="hill", saturation_params={"K": 0.5, "S": 2.0},
            adstock_fn="geometric", adstock_params={"alpha": 0.3, "max_lag": 4},
            spend_mean=10000, spend_std=3000,
            correlation_group="group_tv",
        ),
        ChannelConfig(
            name="search", beta=350.0,
            saturation_fn="hill", saturation_params={"K": 0.3, "S": 3.0},
            adstock_fn="geometric", adstock_params={"alpha": 0.1, "max_lag": 2},
            spend_mean=8000, spend_std=2000,
            correlation_group="group_search",
        ),
        ChannelConfig(
            name="social", beta=200.0,
            saturation_fn="hill", saturation_params={"K": 0.4, "S": 2.5},
            adstock_fn="geometric", adstock_params={"alpha": 0.2, "max_lag": 4},
            spend_mean=6000, spend_std=1500,
            correlation_group="group_social",
        ),
        ChannelConfig(
            name="display", beta=120.0,
            saturation_fn="hill", saturation_params={"K": 0.6, "S": 1.5},
            adstock_fn="geometric", adstock_params={"alpha": 0.15, "max_lag": 3},
            spend_mean=4000, spend_std=1200,
            correlation_group="group_display",
        ),
        ChannelConfig(
            name="email", beta=80.0,
            saturation_fn="hill", saturation_params={"K": 0.3, "S": 2.0},
            adstock_fn="geometric", adstock_params={"alpha": 0.05, "max_lag": 2},
            spend_mean=3000, spend_std=800,
            correlation_group="group_email",
        ),
    ]

    return SimulationConfig(
        n_periods=260,  # 5 years of weekly data for stable estimation
        granularity="weekly",
        channels=channels,
        noise=NoiseConfig(
            noise_type="gaussian",
            noise_scale=5.0,  # Very low noise for clean recovery
        ),
        baseline=BaselineConfig(
            organic_level=5000.0,
            trend_type="linear",
            trend_params={"slope": 5.0},
            seasonality_n_terms=2,
            seasonality_coefficients=[100, 50, 30, 20],
        ),
        seed=42,
    )


def test_scn001_generates_valid_data():
    """SCN-001 produces valid output with correct structure."""
    config = _scn001_config()
    result = Simulator(config).run()

    assert result.observable_data.shape[0] == config.n_periods
    assert not result.observable_data.isnull().any().any()
    assert not result.ground_truth.isnull().any().any()

    # All 5 channels should be in observable data
    for ch_name in ["tv", "search", "social", "display", "email"]:
        assert f"{ch_name}_spend" in result.observable_data.columns
        assert f"true_{ch_name}_contribution" in result.ground_truth.columns


def test_scn001_contributions_sum():
    """Ground truth contributions + baseline + noise = y."""
    config = _scn001_config()
    result = Simulator(config).run()
    gt = result.ground_truth

    reconstructed = gt["true_baseline"].values.copy()
    for ch in config.channels:
        reconstructed += gt[f"true_{ch.name}_contribution"].values
    reconstructed += gt["true_noise"].values

    np.testing.assert_allclose(reconstructed, gt["y"].values, rtol=1e-10)


def test_scn001_ols_recovers_contribution_ranking():
    """OLS should correctly rank channels by contribution magnitude.

    This is the M1 acceptance test. We use a simpler approach than full
    parameter recovery: regress y on the saturated+adstocked channel features
    to recover the betas, and verify the ranking matches ground truth.

    Since we know the true DGP (adstock -> normalize -> saturate -> beta*saturated),
    we construct the exact features and regress. This is an "oracle" model that
    knows the correct transformations.
    """
    config = _scn001_config()
    result = Simulator(config).run()

    # Rank channels by true beta (this is what OLS recovers)
    true_ranking = sorted(
        [ch.name for ch in config.channels],
        key=lambda name: next(ch.beta for ch in config.channels if ch.name == name),
        reverse=True,
    )

    # Build feature matrix from observable spend using true transforms
    from demantiq.transforms.adstock import get_adstock_fn
    from demantiq.transforms.saturation import get_saturation_fn

    obs = result.observable_data
    features = {}
    for ch in config.channels:
        spend = obs[f"{ch.name}_spend"].values

        # Apply known adstock
        adstock_fn = get_adstock_fn(ch.adstock_fn)
        adstocked = adstock_fn(spend, **ch.adstock_params)

        # Normalize and apply known saturation
        max_abs = np.max(np.abs(adstocked))
        if max_abs > 0:
            normalized = adstocked / max_abs
        else:
            normalized = adstocked

        saturation_fn = get_saturation_fn(ch.saturation_fn)
        features[ch.name] = saturation_fn(normalized, **ch.saturation_params)

    # OLS regression: y = intercept + trend + seasonality + sum(beta_i * feature_i) + noise
    y = obs["y"].values
    n = len(y)

    # Build design matrix matching the true DGP exactly
    t = np.arange(n, dtype=float)

    X_cols = [np.ones(n)]  # intercept (organic_level)
    X_cols.append(t / n)   # trend: slope * t / n_periods

    # Fourier terms: coefficients * sin/cos(2*pi*(i+1)/period * t)
    period = config.baseline.seasonality_period
    for i in range(config.baseline.seasonality_n_terms):
        freq = 2 * np.pi * (i + 1) / period
        X_cols.append(np.sin(freq * t))
        X_cols.append(np.cos(freq * t))

    # Channel features
    channel_names = [ch.name for ch in config.channels]
    for name in channel_names:
        X_cols.append(features[name])

    X = np.column_stack(X_cols)

    # OLS: beta = (X'X)^-1 X'y
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]

    # Extract channel betas (last n_channels coefficients)
    n_baseline_cols = 1 + 1 + 2 * config.baseline.seasonality_n_terms
    channel_betas_hat = beta_hat[n_baseline_cols:]

    # Rank by estimated betas
    estimated_ranking = sorted(
        zip(channel_names, channel_betas_hat),
        key=lambda x: x[1], reverse=True
    )
    estimated_ranking = [name for name, _ in estimated_ranking]

    # Verify ranking matches
    assert true_ranking == estimated_ranking, \
        f"True ranking: {true_ranking}, Estimated: {estimated_ranking}"


def test_scn001_ols_recovers_betas_within_tolerance():
    """Oracle OLS recovers true betas within 5% MAPE.

    Using the exact feature construction (known adstock + saturation),
    OLS should recover betas close to true values. With low noise (20.0)
    and 156 periods, the oracle OLS should achieve very tight recovery.
    """
    config = _scn001_config()
    result = Simulator(config).run()

    from demantiq.transforms.adstock import get_adstock_fn
    from demantiq.transforms.saturation import get_saturation_fn

    obs = result.observable_data
    y = obs["y"].values
    n = len(y)

    # Build design matrix matching DGP exactly
    t = np.arange(n, dtype=float)
    X_cols = [np.ones(n), t / n]

    period = config.baseline.seasonality_period
    for i in range(config.baseline.seasonality_n_terms):
        freq = 2 * np.pi * (i + 1) / period
        X_cols.append(np.sin(freq * t))
        X_cols.append(np.cos(freq * t))

    channel_names = []
    for ch in config.channels:
        spend = obs[f"{ch.name}_spend"].values
        adstock_fn = get_adstock_fn(ch.adstock_fn)
        adstocked = adstock_fn(spend, **ch.adstock_params)
        max_abs = np.max(np.abs(adstocked))
        normalized = adstocked / max_abs if max_abs > 0 else adstocked
        saturation_fn = get_saturation_fn(ch.saturation_fn)
        X_cols.append(saturation_fn(normalized, **ch.saturation_params))
        channel_names.append(ch.name)

    X = np.column_stack(X_cols)
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]

    n_baseline = 1 + 1 + 2 * config.baseline.seasonality_n_terms
    channel_betas_hat = beta_hat[n_baseline:]

    # Compare with true betas
    true_betas = [ch.beta for ch in config.channels]

    errors = []
    for i, (name, true_b) in enumerate(zip(channel_names, true_betas)):
        est_b = channel_betas_hat[i]
        pct_error = abs(est_b - true_b) / abs(true_b)
        errors.append(pct_error)

    mape = np.mean(errors)
    assert mape < 0.05, (
        f"MAPE {mape:.2%} exceeds 5% threshold. "
        f"Per-channel errors: {dict(zip(channel_names, [f'{e:.2%}' for e in errors]))}"
    )


def test_scn001_positive_roas():
    """All channels should have positive ROAS."""
    config = _scn001_config()
    result = Simulator(config).run()
    for ch in config.channels:
        assert result.summary_truth["true_roas"][ch.name] > 0
