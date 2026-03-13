"""Correlated media spend generation with flighting patterns."""

import numpy as np
from numpy.random import Generator
from demantiq.config.channel_config import ChannelConfig
from demantiq.utils.correlation import generate_correlation_matrix, gaussian_copula_sample


def generate_spend(channels: list[ChannelConfig] | tuple[ChannelConfig, ...],
                   n_periods: int, rng: Generator) -> dict[str, np.ndarray]:
    """Generate correlated media spend for all channels.

    Args:
        channels: List of channel configurations.
        n_periods: Number of time periods.
        rng: numpy Generator.

    Returns:
        Dict mapping channel name to spend array of shape (n_periods,).
    """
    if not channels:
        return {}

    n_channels = len(channels)

    # Build correlation structure from groups
    groups = _build_correlation_groups(channels)
    corr_matrix = generate_correlation_matrix(
        n_channels,
        within_group_corr=0.5,
        between_group_corr=0.1,
        groups=groups
    )

    # Generate correlated uniform samples via copula
    uniform_samples = gaussian_copula_sample(rng, n_periods, corr_matrix)

    # Transform to spend distributions per channel
    result = {}
    for i, ch in enumerate(channels):
        raw_spend = _transform_to_spend(uniform_samples[:, i], ch, rng)
        raw_spend = _apply_pattern(raw_spend, ch, n_periods, rng)
        raw_spend = np.maximum(raw_spend, ch.spend_floor)
        result[ch.name] = raw_spend

    return result


def _build_correlation_groups(channels: list[ChannelConfig] | tuple[ChannelConfig, ...]) -> list[list[int]]:
    """Build correlation groups from channel configs."""
    group_map: dict[str, list[int]] = {}
    for i, ch in enumerate(channels):
        group_map.setdefault(ch.correlation_group, []).append(i)
    return list(group_map.values())


def _transform_to_spend(uniform: np.ndarray, ch: ChannelConfig,
                        rng: Generator) -> np.ndarray:
    """Transform uniform [0,1] samples to spend distribution."""
    from scipy.stats import lognorm

    # Parameterize log-normal from desired mean/std
    variance = ch.spend_std ** 2
    if ch.spend_mean <= 0:
        return np.zeros_like(uniform)
    mu = np.log(ch.spend_mean ** 2 / np.sqrt(variance + ch.spend_mean ** 2))
    sigma = np.sqrt(np.log(1 + variance / ch.spend_mean ** 2))

    # Use inverse CDF (quantile function) to transform uniforms
    spend = lognorm.ppf(np.clip(uniform, 1e-6, 1 - 1e-6), s=sigma, scale=np.exp(mu))
    return spend


def _apply_pattern(spend: np.ndarray, ch: ChannelConfig,
                   n_periods: int, rng: Generator) -> np.ndarray:
    """Apply spend pattern (flighting, seasonality, etc.)."""
    if ch.spend_pattern == "always_on":
        return spend

    elif ch.spend_pattern == "pulsed":
        # Create on/off flighting pattern
        flight_dur = max(4, n_periods // 10)
        dark_dur = max(2, n_periods // 15)
        mask = np.ones(n_periods, dtype=float)
        t = 0
        on = True
        while t < n_periods:
            if on:
                dur = flight_dur + rng.integers(-1, 2)
                t += max(1, dur)
                on = False
            else:
                dur = dark_dur + rng.integers(-1, 2)
                end = min(t + max(1, dur), n_periods)
                mask[t:end] = 0.0
                t = end
                on = True
        return spend * mask

    elif ch.spend_pattern == "seasonal":
        # Spend follows seasonal pattern
        t = np.arange(n_periods)
        seasonal_mult = 1.0 + 0.5 * np.sin(2 * np.pi * t / min(52, n_periods))
        return spend * seasonal_mult

    elif ch.spend_pattern == "front_loaded":
        # Higher spend in early periods, declining
        t = np.arange(n_periods, dtype=float)
        decay = 1.0 - 0.5 * (t / n_periods)
        return spend * decay

    return spend
