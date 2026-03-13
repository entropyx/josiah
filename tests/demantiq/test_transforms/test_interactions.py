"""Tests for interaction transforms."""

import numpy as np
from demantiq.transforms.interactions import (
    multiplicative_interaction,
    additive_interaction,
    apply_all_interactions,
)
from demantiq.config.interaction_config import InteractionConfig


def test_multiplicative_interaction():
    effect = np.array([100.0, 200.0, 300.0])
    modifier = np.array([0.0, 0.5, 1.0])
    result = multiplicative_interaction(effect, modifier, 0.5)
    expected = effect * (1 + 0.5 * modifier)
    np.testing.assert_allclose(result, expected)


def test_multiplicative_zero_modifier():
    effect = np.array([100.0, 200.0])
    modifier = np.zeros(2)
    result = multiplicative_interaction(effect, modifier, 0.5)
    np.testing.assert_allclose(result, effect)


def test_additive_interaction():
    effect = np.array([100.0, 200.0, 300.0])
    modifier = np.array([1.0, 2.0, 3.0])
    result = additive_interaction(effect, modifier, 10.0)
    expected = effect + 10.0 * modifier
    np.testing.assert_allclose(result, expected)


def test_apply_all_no_interactions():
    media = {"tv": np.array([100.0, 200.0]), "search": np.array([50.0, 80.0])}
    config = InteractionConfig()
    modified, details = apply_all_interactions(media, config)
    np.testing.assert_array_equal(modified["tv"], media["tv"])
    np.testing.assert_array_equal(modified["search"], media["search"])
    assert details == {}


def test_apply_media_x_media():
    media = {"tv": np.array([100.0, 200.0]), "search": np.array([50.0, 100.0])}
    config = InteractionConfig(media_x_media={("tv", "search"): 0.5})
    modified, details = apply_all_interactions(media, config)
    # search normalized: [0.5, 1.0], so tv *= (1 + 0.5 * [0.5, 1.0])
    expected_tv = np.array([100.0 * 1.25, 200.0 * 1.5])
    np.testing.assert_allclose(modified["tv"], expected_tv)
    assert "tv_x_search" in details
    # search should be unchanged
    np.testing.assert_array_equal(modified["search"], media["search"])


def test_apply_price_x_media():
    """Test price interaction with a mock pricing result."""
    from dataclasses import dataclass

    @dataclass
    class MockPricing:
        is_promo: np.ndarray
        promo_depth: np.ndarray
        price: np.ndarray
        price_effect: np.ndarray

    media = {"tv": np.array([100.0, 200.0, 300.0])}
    pricing = MockPricing(
        is_promo=np.array([False, True, True]),
        promo_depth=np.array([0.0, 0.2, 0.3]),
        price=np.array([10.0, 8.0, 7.0]),
        price_effect=np.zeros(3),
    )
    config = InteractionConfig(price_x_media={"tv": 0.4})
    modified, details = apply_all_interactions(media, config, pricing_result=pricing)
    # Period 0: no promo, unchanged. Period 1,2: boosted by 0.4
    assert modified["tv"][0] == 100.0  # no promo
    assert modified["tv"][1] > 200.0   # promo boost
    assert "price_x_tv" in details


def test_apply_competition_x_media():
    """Test competition interaction with a mock competition result."""
    from dataclasses import dataclass

    @dataclass
    class MockCompetition:
        competitor_sov: np.ndarray
        competition_effect: np.ndarray

    media = {"tv": np.array([100.0, 200.0])}
    competition = MockCompetition(
        competitor_sov=np.array([0.3, 0.5]),
        competition_effect=np.zeros(2),
    )
    config = InteractionConfig(competition_x_media={"tv": 0.5})
    modified, details = apply_all_interactions(
        media, config, competition_result=competition
    )
    # media *= (1 + 0.5 * (-competitor_sov))  →  dampened
    assert modified["tv"][0] < 100.0
    assert modified["tv"][1] < 200.0
    assert "competition_x_tv" in details


def test_does_not_mutate_originals():
    """apply_all_interactions should not mutate the input dict."""
    media = {"tv": np.array([100.0, 200.0])}
    original_tv = media["tv"].copy()
    config = InteractionConfig(media_x_media={("tv", "tv"): 0.5})
    apply_all_interactions(media, config)
    np.testing.assert_array_equal(media["tv"], original_tv)


def test_missing_channel_ignored():
    """Interaction referencing a missing channel is silently skipped."""
    media = {"tv": np.array([100.0])}
    config = InteractionConfig(price_x_media={"nonexistent": 0.5})
    modified, details = apply_all_interactions(media, config)
    np.testing.assert_array_equal(modified["tv"], media["tv"])
    assert details == {}
