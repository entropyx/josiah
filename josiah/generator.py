import pandas as pd

from .engines.pymc_engine import generate as pymc_generate
from .engines.legacy_engine import generate as legacy_generate
from .scenario import ScenarioConfig


def generate_single(config: ScenarioConfig):
    """Generate a single scenario dataset.

    Args:
        config: ScenarioConfig instance.

    Returns:
        Tuple of (DataFrame, ground_truth dict, decomp DataFrame or None).
    """
    if config.engine == "legacy":
        df, gt = legacy_generate(config, seed=config.seed)
        return df, gt, None
    return pymc_generate(config, seed=config.seed)


def generate_batch(configs: list):
    """Generate datasets for a list of ScenarioConfigs.

    Args:
        configs: List of ScenarioConfig instances.

    Returns:
        List of (DataFrame, ground_truth dict, decomp DataFrame or None) tuples.
    """
    results = []
    for config in configs:
        results.append(generate_single(config))
    return results
