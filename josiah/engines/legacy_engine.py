import numpy as np
import pandas as pd

from ..components.adstock import exponential_adstock
from ..components.saturation import hill_saturation
from ..components.seasonality import sine_seasonality
from ..components.promos import add_promos_legacy


def _generate_baseline(start_date, end_date, baseline_value, growth_rate=0,
                       slope_changes=None, noise=0, preflight_days=0):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    if preflight_days > 0:
        start = start - pd.Timedelta(days=preflight_days)

    dates = pd.date_range(start=start, end=end, freq="D")
    revenue = np.ones(len(dates)) * baseline_value
    days_arr = np.arange(len(dates))
    revenue += days_arr * growth_rate

    if slope_changes:
        for change_date, new_slope in slope_changes.items():
            cd = pd.to_datetime(change_date)
            mask = dates >= cd
            days_since = (dates[mask] - cd).days
            revenue[mask] += days_since * (new_slope - growth_rate)

    if noise > 0:
        revenue += np.random.normal(0, noise * baseline_value, size=len(revenue))

    return pd.DataFrame({"date": dates, "revenue": revenue})


def _hill_cpm(spend, params):
    hill_factor = (spend ** params["hill_coefficient"]) / (
        params["saturation_spend"] ** params["hill_coefficient"]
        + spend ** params["hill_coefficient"]
    )
    return params["base_cpm"] * (1 + hill_factor)


def _generate_media_channels(start_date, end_date, channels_config, preflight_days=30):
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    preflight_start = start_dt - pd.Timedelta(days=preflight_days)
    date_range = pd.date_range(start=preflight_start, end=end_dt, freq="D")

    media_df = pd.DataFrame({"date": date_range})

    for channel in channels_config:
        curve_params = sorted(channel["curve_params"], key=lambda x: pd.to_datetime(x["start_date"]))
        name = channel["name"]

        # Generate spend
        base_spend = channel["base_spend"]
        noise_level = channel["noise_level"]
        spend = np.maximum(base_spend + np.random.normal(0, noise_level * base_spend, len(date_range)), 0)

        # Process periods
        cpm = np.zeros(len(media_df))
        impressions = np.zeros(len(media_df))
        revenue = np.zeros(len(media_df))

        for i, params in enumerate(curve_params):
            cur_start = pd.to_datetime(params["start_date"])
            cur_end = pd.to_datetime(curve_params[i + 1]["start_date"]) if i < len(curve_params) - 1 else end_dt
            mask = (media_df["date"] < cur_end) if i == 0 else (media_df["date"] >= cur_start) & (media_df["date"] < cur_end)

            period_cpm = _hill_cpm(spend[mask], params)
            period_imp = (spend[mask] * 1000) / period_cpm
            period_rev = period_imp * params["conversion_rate"]

            cpm[mask] = period_cpm
            impressions[mask] = period_imp
            revenue[mask] = period_rev

        # Adstock
        revenue = exponential_adstock(revenue, channel["half_life"])

        media_df[f"{name}_spend"] = spend
        media_df[f"{name}_impressions"] = impressions
        media_df[f"{name}_cpm"] = cpm
        media_df[f"{name}_revenue"] = revenue

    media_df["is_preflight"] = media_df["date"] < start_dt
    return media_df


def generate(config, seed=None):
    """Generate synthetic MMM data using the legacy Hill/exponential engine.

    Args:
        config: ScenarioConfig dataclass instance.
        seed: Random seed.

    Returns:
        Tuple of (DataFrame, ground_truth dict).
    """
    seed = seed if seed is not None else config.seed
    np.random.seed(seed)

    # Convert config to legacy format
    baseline_params = {
        "baseline_value": config.intercept,
        "growth_rate": config.trend_params.get("slope", 0),
        "noise_std": config.noise_std,
    }

    # Build channels_config from config.channels (legacy format)
    channels_config = []
    for ch in config.channels:
        channels_config.append({
            "name": ch.name,
            "base_spend": ch.spend_mean,
            "noise_level": ch.spend_std / ch.spend_mean if ch.spend_mean > 0 else 0.2,
            "half_life": ch.l_max,
            "curve_params": [{
                "start_date": config.start_date,
                "hill_coefficient": ch.lam,
                "base_cpm": 8,
                "saturation_spend": ch.spend_mean * 2,
                "conversion_rate": ch.beta * 0.01,
            }],
        })

    baseline_df = _generate_baseline(
        config.start_date, config.end_date,
        baseline_params["baseline_value"],
        growth_rate=baseline_params["growth_rate"],
        noise=baseline_params["noise_std"],
    )

    media_df = _generate_media_channels(
        config.start_date, config.end_date, channels_config
    )

    df = baseline_df.merge(media_df, on="date", how="left")

    # Seasonality
    seas = sine_seasonality(df["date"])
    df["seasonality_revenue"] = df["revenue"] * seas
    df["revenue"] = df["revenue"] + df["seasonality_revenue"]

    # Total revenue
    media_rev_cols = [c for c in df.columns if c.endswith("_revenue") and c not in ("revenue", "seasonality_revenue", "total_revenue")]
    df["total_revenue"] = df["revenue"]
    for col in media_rev_cols:
        df["total_revenue"] += df[col]

    # Rename for consistency
    df = df.rename(columns={"total_revenue": "y"})

    ground_truth = {
        "engine": "legacy",
        "seed": seed,
        "baseline_value": baseline_params["baseline_value"],
        "growth_rate": baseline_params["growth_rate"],
        "noise_std": baseline_params["noise_std"],
        "channels": {ch["name"]: ch for ch in channels_config},
        "formula": "y = baseline + seasonality + sum(adstock(impressions * conversion_rate))",
    }

    return df, ground_truth
