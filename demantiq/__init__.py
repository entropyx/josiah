"""Demantiq Synthetic Dataset Generator.

Public API for generating synthetic marketing mix model data with known ground truth.
"""

from demantiq.config.simulation_config import SimulationConfig
from demantiq.config.channel_config import ChannelConfig
from demantiq.config.baseline_config import BaselineConfig
from demantiq.config.noise_config import NoiseConfig
from demantiq.config.interaction_config import InteractionConfig, CustomInteraction
from demantiq.core.demand_kernel import SimulationResult, simulate
from demantiq.scenarios.scenario_library import ScenarioLibrary
from demantiq.orchestration.monte_carlo import MonteCarloRunner
from demantiq.calibration.realism_validator import RealismValidator


class Simulator:
    """Main entry point for running Demantiq simulations.

    Usage:
        config = SimulationConfig(channels=[...])
        sim = Simulator(config)
        result = sim.run()
        result.observable_data  # DataFrame
        result.ground_truth     # DataFrame
        result.summary_truth    # dict
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

    def run(self) -> SimulationResult:
        """Run the simulation and return results."""
        return simulate(self.config)


__all__ = [
    "Simulator",
    "SimulationConfig",
    "ChannelConfig",
    "BaselineConfig",
    "NoiseConfig",
    "InteractionConfig",
    "CustomInteraction",
    "SimulationResult",
    "ScenarioLibrary",
    "MonteCarloRunner",
    "RealismValidator",
]
