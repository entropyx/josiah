"""Score the difficulty of a SimulationConfig on a 0.0-1.0 scale."""

from __future__ import annotations

from demantiq.config.simulation_config import SimulationConfig


def _collinearity_score(config: SimulationConfig) -> float:
    """Average pairwise |correlation| from channel correlation groups.

    Channels in the same group are considered correlated; those in
    different groups are uncorrelated.  We approximate average pairwise
    correlation as the fraction of channel pairs sharing a group.
    """
    channels = config.channels
    n = len(channels)
    if n < 2:
        return 0.0

    groups: dict[str, int] = {}
    for ch in channels:
        groups[ch.correlation_group] = groups.get(ch.correlation_group, 0) + 1

    # Pairs sharing a group (excluding self-pairs)
    shared_pairs = sum(cnt * (cnt - 1) / 2 for cnt in groups.values())
    total_pairs = n * (n - 1) / 2
    return shared_pairs / total_pairs


def _endogeneity_score(config: SimulationConfig) -> float:
    """Endogeneity overall_strength if present, else 0."""
    if config.endogeneity is None:
        return 0.0
    return min(float(config.endogeneity.overall_strength), 1.0)


def _signal_to_noise_score(config: SimulationConfig) -> float:
    """Higher noise_scale = harder.  Normalize: min(noise_scale/100, 1.0)."""
    return min(config.noise.noise_scale / 100.0, 1.0)


def _data_length_score(config: SimulationConfig) -> float:
    """Inverse normalized: shorter = harder.  (1 - n_periods/520) capped at 1.0."""
    return min(max(1.0 - config.n_periods / 520.0, 0.0), 1.0)


def _channel_count_score(config: SimulationConfig) -> float:
    """len(channels) / 20."""
    return min(len(config.channels) / 20.0, 1.0)


def _interaction_complexity_score(config: SimulationConfig) -> float:
    """Count non-zero interaction coefficients / total possible."""
    if config.interactions is None:
        return 0.0

    n_channels = len(config.channels)
    if n_channels == 0:
        return 0.0

    ix = config.interactions
    non_zero = 0
    total = 0

    # price_x_media: one per channel
    total += n_channels
    non_zero += sum(1 for v in ix.price_x_media.values() if v != 0)

    # distribution_x_media: one per channel
    total += n_channels
    non_zero += sum(1 for v in ix.distribution_x_media.values() if v != 0)

    # media_x_media: n*(n-1)/2 pairs
    n_pairs = n_channels * (n_channels - 1) // 2
    total += max(n_pairs, 1)
    non_zero += sum(1 for v in ix.media_x_media.values() if v != 0)

    # competition_x_media: one per channel
    total += n_channels
    non_zero += sum(1 for v in ix.competition_x_media.values() if v != 0)

    if total == 0:
        return 0.0
    return min(non_zero / total, 1.0)


def _structural_break_score(config: SimulationConfig) -> float:
    """Count regime changes if macro config present."""
    if config.macro is None:
        return 0.0
    n_changes = len(config.macro.regime_changes)
    # Normalize: 1 change = 0.5, 2+ = 1.0
    if n_changes == 0:
        return 0.0
    return min(n_changes / 2.0, 1.0)


# Weights for each component
_WEIGHTS = {
    "collinearity": 0.20,
    "endogeneity": 0.20,
    "signal_to_noise": 0.15,
    "data_length": 0.15,
    "channel_count": 0.10,
    "interaction_complexity": 0.10,
    "structural_breaks": 0.10,
}


def score_difficulty(config: SimulationConfig) -> float:
    """Compute an overall difficulty score for a SimulationConfig.

    Returns a float in [0.0, 1.0] where 0 = trivial and 1 = very hard.
    """
    components = difficulty_components(config)
    return sum(_WEIGHTS[k] * components[k] for k in _WEIGHTS)


def difficulty_components(config: SimulationConfig) -> dict[str, float]:
    """Return individual difficulty component scores."""
    return {
        "collinearity": _collinearity_score(config),
        "endogeneity": _endogeneity_score(config),
        "signal_to_noise": _signal_to_noise_score(config),
        "data_length": _data_length_score(config),
        "channel_count": _channel_count_score(config),
        "interaction_complexity": _interaction_complexity_score(config),
        "structural_breaks": _structural_break_score(config),
    }
