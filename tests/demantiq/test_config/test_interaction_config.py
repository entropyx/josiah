"""Tests for InteractionConfig and CustomInteraction."""

from demantiq.config.interaction_config import InteractionConfig, CustomInteraction


def test_defaults():
    config = InteractionConfig()
    assert config.price_x_media == {}
    assert config.distribution_x_media == {}
    assert config.media_x_media == {}
    assert config.competition_x_media == {}
    assert config.custom_interactions == []


def test_to_dict():
    config = InteractionConfig(
        price_x_media={"tv": 0.3},
        media_x_media={("tv", "search"): 0.2},
        custom_interactions=[
            CustomInteraction("a", "b", 0.5, "additive"),
        ],
    )
    d = config.to_dict()
    assert d["price_x_media"] == {"tv": 0.3}
    assert d["media_x_media"] == {"tv_search": 0.2}
    assert len(d["custom_interactions"]) == 1
    assert d["custom_interactions"][0]["interaction_type"] == "additive"


def test_roundtrip():
    config = InteractionConfig(
        price_x_media={"tv": 0.3},
        distribution_x_media={"search": 0.1},
        media_x_media={("tv", "search"): 0.2},
        competition_x_media={"tv": 0.15},
        custom_interactions=[
            CustomInteraction("x", "y", 1.0),
        ],
    )
    d = config.to_dict()
    restored = InteractionConfig.from_dict(d)
    assert restored.price_x_media == {"tv": 0.3}
    assert restored.distribution_x_media == {"search": 0.1}
    assert restored.media_x_media == {("tv", "search"): 0.2}
    assert restored.competition_x_media == {"tv": 0.15}
    assert len(restored.custom_interactions) == 1
    assert restored.custom_interactions[0].coefficient == 1.0


def test_custom_interaction_defaults():
    ci = CustomInteraction("a", "b", 0.5)
    assert ci.interaction_type == "multiplicative"


def test_frozen():
    config = InteractionConfig()
    try:
        config.price_x_media = {"tv": 0.1}
        assert False, "Should raise"
    except AttributeError:
        pass


def test_empty_roundtrip():
    config = InteractionConfig()
    d = config.to_dict()
    restored = InteractionConfig.from_dict(d)
    assert restored.price_x_media == {}
    assert restored.media_x_media == {}
