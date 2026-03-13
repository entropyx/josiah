"""Demand kernel: the core data-generating process for Demantiq.

For M1, implements steps 1, 2, 4, 5, 6, 12, 13 of the 15-step pipeline:
  1. Generate baseline
  2. Generate raw spend
  4. Apply adstock per channel
  5. Apply saturation per channel (normalize by max|spend| first)
  6. Compute media effects (beta * saturated)
  12. Aggregate demand (base + sum of media effects)
  13. Add noise
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from demantiq.config.simulation_config import SimulationConfig
from demantiq.generators.baseline_generator import generate_baseline
from demantiq.generators.spend_generator import generate_spend
from demantiq.transforms.adstock import get_adstock_fn
from demantiq.transforms.saturation import get_saturation_fn
from demantiq.core.noise_model import generate_noise
from demantiq.utils.random import create_rng, create_sub_rngs


@dataclass
class SimulationResult:
    """Result of a simulation run.

    Attributes:
        observable_data: DataFrame the model sees (date, y, channel_spend, ...).
        ground_truth: DataFrame with per-period true contributions.
        summary_truth: Dict with aggregate true parameters.
        config: The SimulationConfig used.
    """
    observable_data: pd.DataFrame
    ground_truth: pd.DataFrame
    summary_truth: dict
    config: SimulationConfig


def simulate(config: SimulationConfig) -> SimulationResult:
    """Run a simulation using the demand kernel.

    Args:
        config: Full simulation configuration.

    Returns:
        SimulationResult with observable data, ground truth, and summary.
    """
    rng = create_rng(config.seed)
    sub_rngs = create_sub_rngs(rng, 5)  # baseline, spend, noise, future1, future2

    n = config.n_periods

    # Step 1: Generate baseline
    baseline = generate_baseline(config.baseline, n, sub_rngs[0])

    # Step 2: Generate raw spend
    spend = generate_spend(config.channels, n, sub_rngs[1])

    # Steps 4-6: Per-channel adstock, saturation, media effects
    channel_contributions = {}
    channel_scales = {}
    adstocked = {}
    saturated = {}

    for ch in config.channels:
        ch_spend = spend[ch.name]

        # Step 4: Apply adstock
        adstock_fn = get_adstock_fn(ch.adstock_fn)
        adstock_params = dict(ch.adstock_params)
        ch_adstocked = adstock_fn(ch_spend, **adstock_params)
        adstocked[ch.name] = ch_adstocked

        # Step 5: Normalize by max|spend| then apply saturation
        max_abs = np.max(np.abs(ch_adstocked))
        if max_abs > 0:
            ch_normalized = ch_adstocked / max_abs
        else:
            ch_normalized = ch_adstocked
        channel_scales[ch.name] = float(max_abs)

        saturation_fn = get_saturation_fn(ch.saturation_fn)
        sat_params = dict(ch.saturation_params)
        ch_saturated = saturation_fn(ch_normalized, **sat_params)
        saturated[ch.name] = ch_saturated

        # Step 6: Media effect = beta * saturated
        channel_contributions[ch.name] = ch.beta * ch_saturated

    # Step 12: Aggregate demand
    total_media = np.zeros(n)
    for contrib in channel_contributions.values():
        total_media += contrib

    demand_clean = baseline + total_media

    # Step 13: Add noise
    noise = generate_noise(config.noise, demand_clean, sub_rngs[2])
    y = demand_clean + noise

    # Build observable data
    observable = _build_observable(config, y, spend)

    # Build ground truth
    gt = _build_ground_truth(config, baseline, channel_contributions, noise, y)

    # Build summary truth
    summary = _build_summary(config, channel_contributions, spend, channel_scales, y)

    return SimulationResult(
        observable_data=observable,
        ground_truth=gt,
        summary_truth=summary,
        config=config,
    )


def _build_observable(config: SimulationConfig, y: np.ndarray,
                      spend: dict[str, np.ndarray]) -> pd.DataFrame:
    """Build the observable dataset (what the model sees)."""
    n = config.n_periods

    # Generate dates
    if config.granularity == "weekly":
        dates = pd.date_range("2022-01-01", periods=n, freq="W")
    else:
        dates = pd.date_range("2022-01-01", periods=n, freq="D")

    data = {"date": dates, "y": y}

    # Add spend columns
    for ch in config.channels:
        data[f"{ch.name}_spend"] = spend[ch.name]

    return pd.DataFrame(data)


def _build_ground_truth(config: SimulationConfig, baseline: np.ndarray,
                        contributions: dict[str, np.ndarray],
                        noise: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Build the ground truth DataFrame."""
    n = config.n_periods

    if config.granularity == "weekly":
        dates = pd.date_range("2022-01-01", periods=n, freq="W")
    else:
        dates = pd.date_range("2022-01-01", periods=n, freq="D")

    data = {"date": dates, "true_baseline": baseline}

    for ch_name, contrib in contributions.items():
        data[f"true_{ch_name}_contribution"] = contrib

    data["true_noise"] = noise
    data["y"] = y

    return pd.DataFrame(data)


def _build_summary(config: SimulationConfig,
                   contributions: dict[str, np.ndarray],
                   spend: dict[str, np.ndarray],
                   channel_scales: dict[str, float],
                   y: np.ndarray) -> dict:
    """Build the summary ground truth dict."""
    total_y = float(np.sum(y))

    true_betas = {}
    true_saturation_params = {}
    true_adstock_params = {}
    true_roas = {}
    true_total_contribution = {}
    true_total_spend = {}

    for ch in config.channels:
        true_betas[ch.name] = ch.beta
        true_saturation_params[ch.name] = dict(ch.saturation_params)
        true_adstock_params[ch.name] = dict(ch.adstock_params)

        total_contrib = float(np.sum(contributions[ch.name]))
        total_sp = float(np.sum(spend[ch.name]))

        true_total_contribution[ch.name] = total_contrib
        true_total_spend[ch.name] = total_sp
        true_roas[ch.name] = total_contrib / total_sp if total_sp > 0 else 0.0

    total_media_contrib = sum(true_total_contribution.values())

    return {
        "true_betas": true_betas,
        "true_saturation_params": true_saturation_params,
        "true_adstock_params": true_adstock_params,
        "true_roas": true_roas,
        "true_total_contribution": true_total_contribution,
        "true_total_spend": true_total_spend,
        "true_total_media_contribution_pct": total_media_contrib / total_y if total_y > 0 else 0.0,
        "channel_scales": channel_scales,
        "seed": config.seed,
        "config": config.to_dict(),
    }
