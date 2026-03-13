import numpy as np
import pandas as pd

from ..components.adstock import geometric_adstock
from ..components.saturation import logistic_saturation
from ..components.trend import linear_trend, cube_root_trend
from ..components.seasonality import fourier_seasonality
from ..components.channels import generate_spend, channel_effect
from ..components.controls import generate_controls
from ..components.promos import generate_promo_indicators


def generate(config, seed=None):
    """Generate synthetic MMM data using PyMC Marketing-compatible formulas.

    Formula: y = intercept + trend + seasonality + sum(controls)
                 + sum(channels) + sum(promos) + noise

    Channel contribution: beta * logistic_saturation(geometric_adstock(spend / max|spend|, alpha, l_max), lam)
    Promo contribution: promo_coefficient * indicator (0/1)

    Args:
        config: ScenarioConfig dataclass instance.
        seed: Random seed (overrides config.seed if provided).

    Returns:
        Tuple of (DataFrame, ground_truth dict).
    """
    seed = seed if seed is not None else config.seed
    rng = np.random.default_rng(seed)

    # Date range
    start = pd.to_datetime(config.start_date)
    end = pd.to_datetime(config.end_date)
    if config.frequency == "W":
        dates = pd.date_range(start=start, end=end, freq="W-MON")
    else:
        dates = pd.date_range(start=start, end=end, freq="D")
    n = len(dates)

    # Trend
    if config.trend_type == "linear":
        slope = config.trend_params.get("slope", 0.001)
        trend = linear_trend(n, slope)
    elif config.trend_type == "cube_root":
        max_val = config.trend_params.get("max_val", 100)
        offset = config.trend_params.get("offset", 1.0)
        trend = cube_root_trend(n, max_val, offset)
    else:
        trend = np.zeros(n)

    # Seasonality
    seas_coeffs = config.seasonality_coefficients
    if seas_coeffs and len(seas_coeffs) >= 2:
        seasonality = fourier_seasonality(dates, config.seasonality_n_terms, seas_coeffs)
    elif config.seasonality_n_terms > 0:
        seas_rng = np.random.default_rng(seed + 1000)
        seas_amp = config.intercept * 0.05  # ~5% of intercept
        seas_coeffs = seas_rng.uniform(-seas_amp, seas_amp, size=2 * config.seasonality_n_terms).tolist()
        seasonality = fourier_seasonality(dates, config.seasonality_n_terms, seas_coeffs)
    else:
        seasonality = np.zeros(n)
        seas_coeffs = []

    # Build DataFrame
    df = pd.DataFrame({"date": dates})

    # Channels
    channel_contributions = np.zeros(n)
    channel_ground_truth = {}
    channel_scales = {}
    channel_contrib_arrays = {}
    for i, ch in enumerate(config.channels):
        ch_seed = seed + 100 + i
        spend = generate_spend(n, ch.spend_mean, ch.spend_std, seed=ch_seed)
        contribution, spend_scale = channel_effect(spend, ch.alpha, ch.l_max, ch.lam, ch.beta)

        df[f"{ch.name}_spend"] = spend
        channel_contributions += contribution
        channel_scales[ch.name] = float(spend_scale)
        channel_contrib_arrays[ch.name] = contribution

        total_contribution = float(contribution.sum())
        total_spend = float(spend.sum())
        channel_ground_truth[ch.name] = {
            "alpha": ch.alpha,
            "l_max": ch.l_max,
            "lam": ch.lam,
            "beta": ch.beta,
            "spend_mean": ch.spend_mean,
            "spend_std": ch.spend_std,
            "total_contribution": total_contribution,
            "total_spend": total_spend,
            "roas": total_contribution / total_spend if total_spend != 0 else 0.0,
        }

    # Controls
    control_contributions = np.zeros(n)
    control_ground_truth = {}
    control_contrib_arrays = {}
    for i, ctrl in enumerate(config.controls):
        ctrl_seed = seed + 200 + i
        values, contribution = generate_controls(
            n, ctrl.gamma_shape, ctrl.gamma_scale, ctrl.coefficient, seed=ctrl_seed
        )
        df[ctrl.name] = values
        control_contributions += contribution
        control_contrib_arrays[ctrl.name] = contribution

        control_ground_truth[ctrl.name] = {
            "gamma_shape": ctrl.gamma_shape,
            "gamma_scale": ctrl.gamma_scale,
            "coefficient": ctrl.coefficient,
        }

    # Promos (0/1 indicators)
    promo_contributions = np.zeros(n)
    promo_ground_truth = {}
    promo_contrib_arrays = {}
    for i, promo in enumerate(config.promos):
        promo_seed = seed + 300 + i
        indicator, contribution = generate_promo_indicators(dates, promo, seed=promo_seed)
        df[promo.name] = indicator.astype(int)
        promo_contributions += contribution
        promo_contrib_arrays[promo.name] = contribution

        promo_ground_truth[promo.name] = {
            "coefficient": promo.coefficient,
            "n_occurrences": promo.n_occurrences,
            "duration_days": promo.duration_days,
        }

    # Noise
    noise = rng.normal(0, config.noise_std, size=n)

    # Combine: y = intercept + trend + seasonality + controls + channels + promos + noise
    y = (
        config.intercept + trend + seasonality + control_contributions
        + channel_contributions + promo_contributions + noise
    )
    df["y"] = y

    # Ground truth
    ground_truth = {
        "engine": "pymc",
        "seed": seed,
        "intercept": config.intercept,
        "noise_std": config.noise_std,
        "trend_type": config.trend_type,
        "trend_params": config.trend_params,
        "seasonality_n_terms": config.seasonality_n_terms,
        "seasonality_coefficients": seas_coeffs if isinstance(seas_coeffs, list) else seas_coeffs.tolist(),
        "frequency": config.frequency,
        "start_date": config.start_date,
        "end_date": config.end_date,
        "channels": channel_ground_truth,
        "channel_scales": channel_scales,
        "controls": control_ground_truth,
        "promos": promo_ground_truth,
        "total_revenue": float(y.sum()),
        "formula": "y = intercept + trend + seasonality + sum(control_coeff * control_val) + sum(beta * logistic_saturation(geometric_adstock(spend / max|spend|, alpha, l_max), lam)) + sum(promo_coeff * promo_indicator) + noise",
    }

    # Decomposition DataFrame
    decomp = pd.DataFrame({"date": dates})
    decomp["intercept"] = config.intercept
    decomp["trend"] = trend
    decomp["seasonality"] = seasonality
    for ch_name, contrib in channel_contrib_arrays.items():
        decomp[f"{ch_name}_contribution"] = contrib
    for ctrl_name, contrib in control_contrib_arrays.items():
        decomp[f"{ctrl_name}_contribution"] = contrib
    for promo_name, contrib in promo_contrib_arrays.items():
        decomp[f"{promo_name}_contribution"] = contrib
    decomp["noise"] = noise
    decomp["y"] = y

    return df, ground_truth, decomp
