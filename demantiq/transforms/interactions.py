"""Interaction transforms: cross-variable modifiers for media effects."""

import numpy as np


def multiplicative_interaction(effect, modifier, coefficient):
    """Apply multiplicative interaction: effect * (1 + coefficient * modifier)."""
    return effect * (1 + coefficient * modifier)


def additive_interaction(effect, modifier, coefficient):
    """Apply additive interaction: effect + coefficient * modifier."""
    return effect + coefficient * modifier


def apply_all_interactions(media_effects, interaction_config,
                           pricing_result=None, distribution_result=None,
                           competition_result=None):
    """Apply all configured interactions to media effects.

    Args:
        media_effects: dict[str, np.ndarray] of per-channel media contributions.
        interaction_config: InteractionConfig instance.
        pricing_result: PricingResult (has .is_promo, .promo_depth).
        distribution_result: DistributionResult (has .distribution).
        competition_result: CompetitionResult (has .competitor_sov).

    Returns:
        tuple: (modified_media_effects dict, interaction_details dict).
            interaction_details maps interaction name to the delta array.
    """
    modified = {k: v.copy() for k, v in media_effects.items()}
    details = {}

    # 1. price_x_media: media *= (1 + coeff * is_promo)
    if interaction_config.price_x_media and pricing_result is not None:
        for ch_name, coeff in interaction_config.price_x_media.items():
            if ch_name in modified:
                before = modified[ch_name].copy()
                modified[ch_name] = multiplicative_interaction(
                    modified[ch_name], pricing_result.is_promo.astype(float), coeff
                )
                details[f"price_x_{ch_name}"] = modified[ch_name] - before

    # 2. distribution_x_media: media *= (1 + coeff * distribution)
    if interaction_config.distribution_x_media and distribution_result is not None:
        for ch_name, coeff in interaction_config.distribution_x_media.items():
            if ch_name in modified:
                before = modified[ch_name].copy()
                modified[ch_name] = multiplicative_interaction(
                    modified[ch_name], distribution_result.distribution, coeff
                )
                details[f"distribution_x_{ch_name}"] = modified[ch_name] - before

    # 3. media_x_media: media_a *= (1 + coeff * media_b_normalized)
    if interaction_config.media_x_media:
        for (ch_a, ch_b), coeff in interaction_config.media_x_media.items():
            if ch_a in modified and ch_b in media_effects:
                b_vals = media_effects[ch_b]
                b_max = np.max(np.abs(b_vals))
                b_norm = b_vals / b_max if b_max > 0 else b_vals
                before = modified[ch_a].copy()
                modified[ch_a] = multiplicative_interaction(
                    modified[ch_a], b_norm, coeff
                )
                details[f"{ch_a}_x_{ch_b}"] = modified[ch_a] - before

    # 4. competition_x_media: media *= (1 - coeff * competitor_sov)
    if interaction_config.competition_x_media and competition_result is not None:
        for ch_name, coeff in interaction_config.competition_x_media.items():
            if ch_name in modified:
                before = modified[ch_name].copy()
                modified[ch_name] = multiplicative_interaction(
                    modified[ch_name], -competition_result.competitor_sov, coeff
                )
                details[f"competition_x_{ch_name}"] = modified[ch_name] - before

    # 5. Custom interactions (pass-through for now)

    return modified, details
