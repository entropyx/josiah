import pytest
from demantiq.config import SimulationConfig, ChannelConfig, BaselineConfig, NoiseConfig
from demantiq.orchestration.serializer import (
    config_to_json, config_from_json, config_to_yaml, config_from_yaml
)


def _test_config():
    return SimulationConfig(
        n_periods=52,
        channels=[
            ChannelConfig(name="tv", beta=0.15, saturation_fn="hill",
                          saturation_params={"K": 0.5, "S": 2.0}),
            ChannelConfig(name="search", beta=0.08),
        ],
        noise=NoiseConfig(noise_type="t_distributed", t_df=4.0),
        baseline=BaselineConfig(organic_level=500.0, seasonality_coefficients=[10, 20]),
        seed=123,
    )


def test_json_roundtrip():
    config = _test_config()
    json_str = config_to_json(config)
    restored = config_from_json(json_str)
    assert restored.n_periods == 52
    assert len(restored.channels) == 2
    assert restored.channels[0].name == "tv"
    assert restored.channels[0].beta == 0.15
    assert restored.noise.noise_type == "t_distributed"
    assert restored.seed == 123


def test_yaml_roundtrip():
    config = _test_config()
    yaml_str = config_to_yaml(config)
    restored = config_from_yaml(yaml_str)
    assert restored.n_periods == 52
    assert len(restored.channels) == 2
    assert restored.channels[1].name == "search"
    assert restored.baseline.organic_level == 500.0


def test_json_is_valid_json():
    import json
    config = _test_config()
    json_str = config_to_json(config)
    parsed = json.loads(json_str)
    assert isinstance(parsed, dict)
    assert parsed["n_periods"] == 52


def test_yaml_is_valid_yaml():
    import yaml
    config = _test_config()
    yaml_str = config_to_yaml(config)
    parsed = yaml.safe_load(yaml_str)
    assert isinstance(parsed, dict)


def test_default_config_roundtrip():
    config = SimulationConfig()
    json_str = config_to_json(config)
    restored = config_from_json(json_str)
    assert restored.n_periods == config.n_periods
    assert restored.seed == config.seed
