"""External macro variable simulation."""

import numpy as np
from numpy.random import Generator
from dataclasses import dataclass

from demantiq.config.macro_config import MacroConfig, MacroVariable, RegimeChange
from demantiq.utils.time_series import apply_structural_break


@dataclass
class MacroResult:
    """Result of macro variable generation."""

    variables: dict[str, np.ndarray]  # name -> time series
    macro_effect: np.ndarray  # total effect on demand
    regime_effects: np.ndarray  # effect of regime changes on baseline


def generate_macro(
    config: MacroConfig, n_periods: int, rng: Generator
) -> MacroResult:
    """Generate external macro variables and regime changes."""
    variables = {}
    total_effect = np.zeros(n_periods)

    for var in config.variables:
        ts = _generate_time_series(var, n_periods, rng)
        variables[var.name] = ts
        # Standardize before applying coefficient
        ts_std = (ts - np.mean(ts)) / (np.std(ts) + 1e-8)
        total_effect += var.effect_on_demand * ts_std

    # Regime changes
    regime_effects = np.zeros(n_periods)
    for rc in config.regime_changes:
        if "baseline" in rc.affected_params:
            regime_effects = apply_structural_break(
                regime_effects,
                rc.period,
                rc.magnitude * 1000,  # scale to demand units
                break_type=rc.change_type,
                recovery=rc.recovery,
                recovery_periods=rc.recovery_periods,
            )

    return MacroResult(
        variables=variables, macro_effect=total_effect, regime_effects=regime_effects
    )


def _generate_time_series(
    var: MacroVariable, n_periods: int, rng: Generator
) -> np.ndarray:
    """Generate a single macro variable time series."""
    params = var.params

    if var.time_series_type == "random_walk":
        steps = rng.normal(0, params.get("step_std", 0.1), size=n_periods)
        return np.cumsum(steps) + params.get("start", 0.0)

    elif var.time_series_type == "mean_reverting":
        mu = params.get("mean", 0.0)
        phi = params.get("phi", 0.9)
        sigma = params.get("sigma", 0.1)
        ts = np.zeros(n_periods)
        ts[0] = mu
        for t in range(1, n_periods):
            ts[t] = mu + phi * (ts[t - 1] - mu) + rng.normal(0, sigma)
        return ts

    elif var.time_series_type == "trending":
        slope = params.get("slope", 0.01)
        sigma = params.get("sigma", 0.05)
        t = np.arange(n_periods, dtype=float)
        return params.get("start", 0.0) + slope * t + rng.normal(0, sigma, n_periods)

    elif var.time_series_type == "seasonal":
        period = params.get("period", 52)
        amplitude = params.get("amplitude", 1.0)
        t = np.arange(n_periods, dtype=float)
        return amplitude * np.sin(2 * np.pi * t / period) + rng.normal(
            0, 0.1, n_periods
        )

    return rng.normal(0, 1, n_periods)
