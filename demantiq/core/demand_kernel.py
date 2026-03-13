"""Demand kernel: the core data-generating process for Demantiq.

For M1, implements steps 1, 2, 4, 5, 6, 12, 13 of the 15-step pipeline:
  1. Generate baseline
  2. Generate raw spend
  4. Apply adstock per channel
  5. Apply saturation per channel (normalize by max|spend| first)
  6. Compute media effects (beta * saturated)
  12. Aggregate demand (base + sum of media effects)
  13. Add noise

M2 adds:
  8. Price effect (pricing engine)
  9. Distribution cap (distribution generator)
  14. Revenue = y * price
  15. Margin = revenue - costs - media_cost

M3 adds:
  3. Endogeneity (feedback loops and confounding)
  10. Competition effect (competitor SOV suppression)
  11. Macro effects (external variables and regime changes)

M4 adds:
  7. Interactions (cross-variable modifiers on media effects)
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
    sub_rngs = create_sub_rngs(rng, 8)  # baseline, spend, noise, pricing, distribution, endogeneity, competition, macro

    n = config.n_periods

    # Step 1: Generate baseline
    baseline = generate_baseline(config.baseline, n, sub_rngs[0])

    # Step 2: Generate raw spend
    spend = generate_spend(config.channels, n, sub_rngs[1])

    # Step 3: Apply endogeneity (if configured)
    if config.endogeneity is not None:
        from demantiq.core.endogeneity_layer import apply_endogeneity
        channel_names = [ch.name for ch in config.channels]
        endog_result = apply_endogeneity(
            spend, baseline, config.endogeneity, channel_names, sub_rngs[5]
        )
        spend = endog_result.spend_endogenous
    else:
        endog_result = None

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

    # Generate auxiliary results needed before interactions
    # Step 8: Price effect (if pricing configured)
    pricing_result = None
    if config.pricing is not None:
        from demantiq.generators.pricing_engine import generate_pricing
        pricing_result = generate_pricing(config.pricing, n, sub_rngs[3])

    # Step 9: Distribution cap (if distribution configured)
    distribution_result = None
    if config.distribution is not None:
        from demantiq.generators.distribution_generator import generate_distribution
        distribution_result = generate_distribution(config.distribution, n, sub_rngs[4])

    # Step 10: Competition effect (if configured)
    competition_result = None
    if config.competition is not None:
        from demantiq.generators.competition_generator import generate_competition
        competition_result = generate_competition(config.competition, n, sub_rngs[6])

    # Step 11: Macro effects (if configured)
    macro_result = None
    if config.macro is not None:
        from demantiq.generators.macro_generator import generate_macro
        macro_result = generate_macro(config.macro, n, sub_rngs[7])

    # Step 7: Apply interactions (if configured)
    interaction_details = None
    if config.interactions is not None:
        from demantiq.transforms.interactions import apply_all_interactions
        channel_contributions, interaction_details = apply_all_interactions(
            channel_contributions, config.interactions,
            pricing_result=pricing_result,
            distribution_result=distribution_result,
            competition_result=competition_result,
        )

    # Step 12: Aggregate demand
    total_media = np.zeros(n)
    for contrib in channel_contributions.values():
        total_media += contrib

    demand_clean = baseline + total_media

    # Apply price effect to demand
    if pricing_result is not None:
        demand_clean = demand_clean + pricing_result.price_effect

    # Apply distribution cap to demand
    if distribution_result is not None:
        demand_clean = demand_clean * distribution_result.distribution_cap

    # Add competition effect
    if competition_result is not None:
        demand_clean = demand_clean + competition_result.competition_effect * demand_clean.mean()

    # Add macro effects
    if macro_result is not None:
        demand_clean = demand_clean + macro_result.macro_effect
        demand_clean = demand_clean + macro_result.regime_effects

    # Step 13: Add noise
    noise = generate_noise(config.noise, demand_clean, sub_rngs[2])
    y = demand_clean + noise

    # Steps 14-15: Revenue and margin (if pricing configured)
    revenue = None
    margin_info = None
    if pricing_result is not None:
        revenue = y * pricing_result.price
        total_media_cost = sum(float(np.sum(spend[ch.name])) for ch in config.channels)
        from demantiq.ground_truth.margin_attribution import compute_margin
        margin_info = compute_margin(
            revenue, y,
            config.pricing.cost_structure.cogs_per_unit,
            config.pricing.cost_structure.variable_cost_per_unit,
            np.full(n, total_media_cost / n),
        )

    # Build observable data
    observable = _build_observable(config, y, spend, pricing_result, distribution_result,
                                   revenue, competition_result, macro_result)

    # Build ground truth
    gt = _build_ground_truth(config, baseline, channel_contributions, noise, y,
                             pricing_result, distribution_result,
                             endog_result, competition_result, macro_result,
                             interaction_details)

    # Build summary truth
    summary = _build_summary(config, channel_contributions, spend, channel_scales, y,
                             pricing_result, margin_info,
                             endog_result, competition_result, macro_result,
                             interaction_details)

    return SimulationResult(
        observable_data=observable,
        ground_truth=gt,
        summary_truth=summary,
        config=config,
    )


def _build_observable(config: SimulationConfig, y: np.ndarray,
                      spend: dict[str, np.ndarray],
                      pricing_result=None, distribution_result=None,
                      revenue=None, competition_result=None,
                      macro_result=None) -> pd.DataFrame:
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

    # Add pricing columns
    if pricing_result is not None:
        data["price"] = pricing_result.price
        data["is_promo"] = pricing_result.is_promo
        data["promo_depth"] = pricing_result.promo_depth

    # Add distribution columns
    if distribution_result is not None:
        data["distribution"] = distribution_result.distribution
        data["stockout"] = distribution_result.stockout_mask

    # Add competition columns
    if competition_result is not None:
        data["competitor_sov"] = competition_result.competitor_sov

    # Add macro variable columns
    if macro_result is not None:
        for var_name, var_ts in macro_result.variables.items():
            data[var_name] = var_ts

    # Add revenue
    if revenue is not None:
        data["revenue"] = revenue

    return pd.DataFrame(data)


def _build_ground_truth(config: SimulationConfig, baseline: np.ndarray,
                        contributions: dict[str, np.ndarray],
                        noise: np.ndarray, y: np.ndarray,
                        pricing_result=None, distribution_result=None,
                        endog_result=None, competition_result=None,
                        macro_result=None,
                        interaction_details=None) -> pd.DataFrame:
    """Build the ground truth DataFrame."""
    n = config.n_periods

    if config.granularity == "weekly":
        dates = pd.date_range("2022-01-01", periods=n, freq="W")
    else:
        dates = pd.date_range("2022-01-01", periods=n, freq="D")

    data = {"date": dates, "true_baseline": baseline}

    for ch_name, contrib in contributions.items():
        data[f"true_{ch_name}_contribution"] = contrib

    # Add pricing ground truth
    if pricing_result is not None:
        data["true_price_effect"] = pricing_result.price_effect

    # Add distribution ground truth
    if distribution_result is not None:
        data["true_distribution_cap"] = distribution_result.distribution_cap

    # Add endogeneity ground truth
    if endog_result is not None:
        data["confounders"] = endog_result.confounders
        for ch_name in endog_result.exogenous_spend:
            data[f"exogenous_{ch_name}_spend"] = endog_result.exogenous_spend[ch_name]
            data[f"endogeneity_bias_{ch_name}"] = endog_result.endogeneity_bias[ch_name]

    # Add competition ground truth
    if competition_result is not None:
        data["true_competition_effect"] = competition_result.competition_effect

    # Add macro ground truth
    if macro_result is not None:
        data["true_macro_effect"] = macro_result.macro_effect
        data["true_regime_effects"] = macro_result.regime_effects

    # Add interaction ground truth
    if interaction_details is not None:
        for detail_name, detail_vals in interaction_details.items():
            data[f"true_interaction_{detail_name}"] = detail_vals

    data["true_noise"] = noise
    data["y"] = y

    return pd.DataFrame(data)


def _build_summary(config: SimulationConfig,
                   contributions: dict[str, np.ndarray],
                   spend: dict[str, np.ndarray],
                   channel_scales: dict[str, float],
                   y: np.ndarray,
                   pricing_result=None,
                   margin_info=None,
                   endog_result=None,
                   competition_result=None,
                   macro_result=None,
                   interaction_details=None) -> dict:
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

    summary = {
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

    # Add pricing summary
    if pricing_result is not None:
        from demantiq.ground_truth.elasticities import compute_price_elasticity
        summary["true_price_elasticity"] = config.pricing.price_elasticity
        summary["empirical_price_elasticity"] = compute_price_elasticity(
            pricing_result.price, np.maximum(y, 1e-6)
        )
        summary["base_price"] = config.pricing.base_price

    # Add margin summary
    if margin_info is not None:
        summary["total_revenue"] = margin_info["total_revenue"]
        summary["total_margin"] = margin_info["total_margin"]
        summary["total_media_cost"] = margin_info["total_media_cost"]

    # Add endogeneity summary
    if endog_result is not None:
        summary["endogeneity"] = {
            "overall_strength": config.endogeneity.overall_strength,
            "has_confounders": bool(np.any(endog_result.confounders != 0)),
        }

    # Add competition summary
    if competition_result is not None:
        summary["competition"] = {
            "mean_competitor_sov": float(np.mean(competition_result.competitor_sov)),
            "sov_suppression_coefficient": config.competition.sov_suppression_coefficient,
        }

    # Add macro summary
    if macro_result is not None:
        summary["macro"] = {
            "variables": [v.name for v in config.macro.variables],
            "n_regime_changes": len(config.macro.regime_changes),
        }

    # Add interaction summary
    if interaction_details is not None:
        interaction_summary = {}
        for detail_name, detail_vals in interaction_details.items():
            interaction_summary[detail_name] = {
                "total_effect": float(np.sum(detail_vals)),
                "mean_effect": float(np.mean(detail_vals)),
            }
        summary["interactions"] = interaction_summary

    return summary
