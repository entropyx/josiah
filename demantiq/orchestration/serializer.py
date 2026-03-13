"""Config serialization for reproducibility."""

import json
import yaml
from demantiq.config.simulation_config import SimulationConfig


def config_to_json(config: SimulationConfig) -> str:
    """Serialize a SimulationConfig to JSON string."""
    return json.dumps(config.to_dict(), indent=2, default=str)


def config_from_json(json_str: str) -> SimulationConfig:
    """Deserialize a SimulationConfig from JSON string."""
    d = json.loads(json_str)
    return SimulationConfig.from_dict(d)


def config_to_yaml(config: SimulationConfig) -> str:
    """Serialize a SimulationConfig to YAML string."""
    return yaml.dump(config.to_dict(), default_flow_style=False, sort_keys=False)


def config_from_yaml(yaml_str: str) -> SimulationConfig:
    """Deserialize a SimulationConfig from YAML string."""
    d = yaml.safe_load(yaml_str)
    return SimulationConfig.from_dict(d)
