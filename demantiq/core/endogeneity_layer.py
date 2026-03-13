"""Endogeneity layer: feedback loops and confounding injection."""

import numpy as np
from numpy.random import Generator
from dataclasses import dataclass

from demantiq.config.endogeneity_config import EndogeneityConfig


@dataclass
class EndogeneityResult:
    """Result of endogeneity application."""

    spend_endogenous: dict[str, np.ndarray]
    exogenous_spend: dict[str, np.ndarray]
    endogeneity_bias: dict[str, np.ndarray]
    confounders: np.ndarray  # hidden variable (not in observable data)


def apply_endogeneity(
    spend_raw: dict[str, np.ndarray],
    baseline: np.ndarray,
    config: EndogeneityConfig,
    channel_names: list[str],
    rng: Generator,
) -> EndogeneityResult:
    """Apply endogeneity to raw spend.

    Modifies spend based on baseline (seasonal bias) and hidden confounders.
    """
    n = len(baseline)
    strength = config.overall_strength

    # Determine which channels are affected
    affected = config.feedback_channels if config.feedback_channels else channel_names

    # Generate hidden confounder (OVB)
    confounder = np.zeros(n)
    if config.omitted_variable_strength > 0:
        confounder = _generate_confounder(n, config.omitted_variable_ar, rng)
        confounder = confounder * config.omitted_variable_strength

    spend_endogenous = {}
    exogenous_spend = {}
    endogeneity_bias = {}

    for name in channel_names:
        raw = spend_raw[name].copy()
        exogenous_spend[name] = raw.copy()

        if name not in affected:
            spend_endogenous[name] = raw
            endogeneity_bias[name] = np.zeros(n)
            continue

        modified = raw.copy()

        # Seasonal allocation bias: spend increases during high-baseline periods
        if config.seasonal_allocation_bias > 0:
            max_abs = np.max(np.abs(baseline))
            baseline_norm = baseline / max_abs if max_abs > 0 else baseline
            seasonal_mult = 1.0 + config.seasonal_allocation_bias * strength * baseline_norm
            modified = modified * seasonal_mult

        # Performance chasing: spend reacts to recent baseline (proxy for outcome)
        if config.performance_chasing > 0:
            lag = max(1, config.feedback_lag)
            for t in range(lag, n):
                baseline_signal = np.mean(baseline[t - lag : t])
                baseline_mean = np.mean(baseline)
                if baseline_mean > 0:
                    chase_mult = (
                        1.0
                        + config.performance_chasing
                        * strength
                        * (baseline_signal - baseline_mean)
                        / baseline_mean
                    )
                    modified[t] = modified[t] * max(0.5, chase_mult)

        # OVB: confounder affects spend
        if config.omitted_variable_strength > 0:
            spend_std = np.std(raw) if np.std(raw) > 0 else 1.0
            modified = modified + confounder * spend_std * strength

        modified = np.maximum(modified, 0)
        spend_endogenous[name] = modified
        endogeneity_bias[name] = modified - raw

    return EndogeneityResult(
        spend_endogenous=spend_endogenous,
        exogenous_spend=exogenous_spend,
        endogeneity_bias=endogeneity_bias,
        confounders=confounder,
    )


def _generate_confounder(n: int, ar_coeff: float, rng: Generator) -> np.ndarray:
    """Generate AR(1) hidden confounder."""
    innovations = rng.normal(0, 1, size=n)
    z = np.zeros(n)
    z[0] = innovations[0]
    for t in range(1, n):
        z[t] = ar_coeff * z[t - 1] + innovations[t]
    return z
