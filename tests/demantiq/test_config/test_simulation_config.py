from demantiq.config import SimulationConfig, ChannelConfig, BaselineConfig, NoiseConfig


def test_default_config():
    config = SimulationConfig()
    assert config.n_periods == 104
    assert config.granularity == "weekly"
    assert len(config.channels) == 0
    assert config.seed == 42


def test_config_with_channels():
    ch = ChannelConfig(name="tv", beta=0.15)
    config = SimulationConfig(channels=[ch])
    assert len(config.channels) == 1
    assert config.channels[0].name == "tv"


def test_config_frozen():
    config = SimulationConfig()
    try:
        config.n_periods = 52
        assert False, "Should raise"
    except AttributeError:
        pass


def test_config_to_dict_roundtrip():
    ch = ChannelConfig(name="search", beta=0.08, saturation_fn="logistic",
                       saturation_params={"k": 3.0, "x0": 0.5})
    config = SimulationConfig(
        n_periods=52,
        channels=[ch],
        noise=NoiseConfig(noise_type="t_distributed", t_df=4.0),
        baseline=BaselineConfig(organic_level=500.0),
        seed=123,
    )
    d = config.to_dict()
    restored = SimulationConfig.from_dict(d)
    assert restored.n_periods == 52
    assert restored.channels[0].name == "search"
    assert restored.noise.noise_type == "t_distributed"
    assert restored.baseline.organic_level == 500.0
    assert restored.seed == 123


def test_channel_config_defaults():
    ch = ChannelConfig(name="tv")
    assert ch.saturation_fn == "hill"
    assert ch.adstock_fn == "geometric"
    assert ch.spend_pattern == "always_on"


def test_noise_config_defaults():
    nc = NoiseConfig()
    assert nc.noise_type == "gaussian"
    assert nc.noise_scale == 50.0
    assert nc.signal_to_noise_ratio is None


def test_baseline_config_defaults():
    bc = BaselineConfig()
    assert bc.trend_type == "linear"
    assert bc.organic_level == 1000.0
    assert bc.seasonality_n_terms == 2


def test_channel_list_to_tuple():
    """Channels passed as list should be converted to tuple."""
    ch = ChannelConfig(name="tv")
    config = SimulationConfig(channels=[ch])
    assert isinstance(config.channels, tuple)
