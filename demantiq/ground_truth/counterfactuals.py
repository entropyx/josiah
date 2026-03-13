"""Counterfactual analysis: re-run simulation with channels zeroed."""

import numpy as np
from demantiq.config.simulation_config import SimulationConfig
from demantiq.config.channel_config import ChannelConfig


def compute_counterfactual(config: SimulationConfig, channel_to_zero: str) -> dict:
    """Re-run simulation with one channel zeroed to get counterfactual demand.

    Args:
        config: Original SimulationConfig.
        channel_to_zero: Name of the channel to zero out.

    Returns:
        dict with total_demand_actual, total_demand_counterfactual,
        incremental_demand, and incremental_pct.
    """
    from demantiq.core.demand_kernel import simulate

    # Run actual simulation
    actual_result = simulate(config)
    actual_demand = float(np.sum(actual_result.observable_data["y"].values))

    # Create modified config with zeroed channel
    modified_channels = []
    for ch in config.channels:
        if ch.name == channel_to_zero:
            modified_channels.append(
                ChannelConfig(
                    name=ch.name,
                    beta=0.0,
                    spend_mean=ch.spend_mean,
                    spend_std=ch.spend_std,
                    adstock_fn=ch.adstock_fn,
                    adstock_params=ch.adstock_params,
                    saturation_fn=ch.saturation_fn,
                    saturation_params=ch.saturation_params,
                )
            )
        else:
            modified_channels.append(ch)

    modified_config = SimulationConfig(
        n_periods=config.n_periods,
        granularity=config.granularity,
        channels=modified_channels,
        noise=config.noise,
        baseline=config.baseline,
        seed=config.seed,
        metadata=config.metadata,
        pricing=config.pricing,
        distribution=config.distribution,
        competition=config.competition,
        macro=config.macro,
        endogeneity=config.endogeneity,
        interactions=config.interactions,
    )

    counterfactual_result = simulate(modified_config)
    counterfactual_demand = float(
        np.sum(counterfactual_result.observable_data["y"].values)
    )

    incremental = actual_demand - counterfactual_demand

    return {
        "channel": channel_to_zero,
        "total_demand_actual": actual_demand,
        "total_demand_counterfactual": counterfactual_demand,
        "incremental_demand": incremental,
        "incremental_pct": incremental / actual_demand if actual_demand != 0 else 0.0,
    }
