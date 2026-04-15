from demantiq.config.channel_config import ChannelConfig


def test_channel_config_has_cpm_and_ctr():
    """CPM and CTR should be configurable per channel with sensible defaults."""
    ch = ChannelConfig(name="facebook")
    assert hasattr(ch, "cpm")
    assert hasattr(ch, "ctr")
    assert ch.cpm > 0
    assert 0 < ch.ctr < 1


def test_channel_config_accepts_custom_cpm_ctr():
    ch = ChannelConfig(name="facebook", cpm=12.0, ctr=0.015)
    assert ch.cpm == 12.0
    assert ch.ctr == 0.015
