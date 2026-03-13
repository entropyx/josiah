import numpy as np
from demantiq.config.channel_config import ChannelConfig
from demantiq.generators.spend_generator import generate_spend
from demantiq.utils.random import create_rng


def test_empty_channels():
    rng = create_rng(42)
    result = generate_spend([], 100, rng)
    assert result == {}


def test_single_channel_shape():
    rng = create_rng(42)
    ch = ChannelConfig(name="tv", spend_mean=10000, spend_std=3000)
    result = generate_spend([ch], 104, rng)
    assert "tv" in result
    assert result["tv"].shape == (104,)


def test_non_negative():
    rng = create_rng(42)
    ch = ChannelConfig(name="tv", spend_mean=10000, spend_std=3000, spend_floor=0.0)
    result = generate_spend([ch], 104, rng)
    assert np.all(result["tv"] >= 0)


def test_multiple_channels():
    rng = create_rng(42)
    channels = [
        ChannelConfig(name="tv", spend_mean=50000, spend_std=15000, correlation_group="brand"),
        ChannelConfig(name="search", spend_mean=20000, spend_std=5000, correlation_group="perf"),
        ChannelConfig(name="social", spend_mean=30000, spend_std=8000, correlation_group="brand"),
    ]
    result = generate_spend(channels, 104, rng)
    assert len(result) == 3
    for name in ["tv", "search", "social"]:
        assert result[name].shape == (104,)


def test_pulsed_pattern_has_zeros():
    rng = create_rng(42)
    ch = ChannelConfig(name="tv", spend_mean=10000, spend_std=3000, spend_pattern="pulsed")
    result = generate_spend([ch], 104, rng)
    # Pulsed pattern should have some zero periods
    assert np.any(result["tv"] == 0)


def test_deterministic():
    ch = ChannelConfig(name="tv", spend_mean=10000, spend_std=3000)
    r1 = generate_spend([ch], 104, create_rng(42))
    r2 = generate_spend([ch], 104, create_rng(42))
    np.testing.assert_allclose(r1["tv"], r2["tv"])


def test_spend_floor():
    rng = create_rng(42)
    ch = ChannelConfig(name="tv", spend_mean=10000, spend_std=3000, spend_floor=500.0)
    result = generate_spend([ch], 104, rng)
    assert np.all(result["tv"] >= 500.0)
