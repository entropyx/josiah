"""Random scenario sampling for Monte Carlo experiments."""

from __future__ import annotations

import numpy as np

from demantiq.config.simulation_config import SimulationConfig
from demantiq.config.channel_config import ChannelConfig
from demantiq.config.baseline_config import BaselineConfig
from demantiq.config.noise_config import NoiseConfig
from demantiq.config.pricing_config import PricingConfig
from demantiq.config.distribution_config import DistributionConfig
from demantiq.config.endogeneity_config import EndogeneityConfig
from demantiq.config.competition_config import CompetitionConfig
from demantiq.config.macro_config import MacroConfig, RegimeChange
from demantiq.config.interaction_config import InteractionConfig


_CHANNEL_NAMES = [
    "facebook", "google", "tiktok", "pinterest", "email",
    "youtube", "snapchat", "linkedin", "twitter", "display",
    "programmatic", "podcast", "influencer", "radio", "ctv",
    "affiliate", "sms", "direct_mail", "ooh", "print",
]

_SATURATION_FNS = ["hill", "logistic"]
_ADSTOCK_FNS = ["geometric", "weibull_cdf"]
_SPEND_PATTERNS = ["always_on", "pulsed", "seasonal", "front_loaded"]


class ScenarioSampler:
    """Generate random SimulationConfigs by sampling parameter distributions.

    Args:
        seed: Master seed for reproducibility.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def sample(self, n: int = 1) -> list[SimulationConfig]:
        """Generate n random SimulationConfigs.

        Args:
            n: Number of configs to generate.

        Returns:
            List of SimulationConfig instances.
        """
        configs = []
        for i in range(n):
            config_seed = int(self.rng.integers(0, 2**31))
            configs.append(self._sample_one(config_seed))
        return configs

    def _sample_one(self, seed: int) -> SimulationConfig:
        """Sample a single random SimulationConfig."""
        rng = self.rng

        # Number of channels: 2-15
        n_channels = int(rng.integers(2, 16))
        channel_names = list(rng.choice(_CHANNEL_NAMES, size=n_channels, replace=False))

        # Number of correlation groups: 1 to n_channels
        n_groups = int(rng.integers(1, max(n_channels, 2)))

        channels = []
        for i, name in enumerate(channel_names):
            sat_fn = str(rng.choice(_SATURATION_FNS))
            if sat_fn == "hill":
                sat_params = {"K": float(rng.uniform(0.2, 0.8)), "S": float(rng.uniform(1.0, 5.0))}
            else:
                sat_params = {"k": float(rng.uniform(0.5, 10.0)), "x0": float(rng.uniform(0.3, 0.7))}

            ads_fn = str(rng.choice(_ADSTOCK_FNS))
            if ads_fn == "geometric":
                ads_params = {"alpha": float(rng.uniform(0.1, 0.9)), "max_lag": int(rng.integers(4, 13))}
            else:
                ads_params = {"shape": float(rng.uniform(0.5, 3.0)),
                              "scale": float(rng.uniform(1.0, 5.0)),
                              "max_lag": int(rng.integers(4, 13))}

            channels.append(ChannelConfig(
                name=name,
                beta=float(rng.uniform(20.0, 500.0)),
                saturation_fn=sat_fn,
                saturation_params=sat_params,
                adstock_fn=ads_fn,
                adstock_params=ads_params,
                spend_pattern=str(rng.choice(_SPEND_PATTERNS)),
                spend_mean=float(rng.uniform(1000.0, 50000.0)),
                spend_std=float(rng.uniform(500.0, 15000.0)),
                spend_floor=float(rng.choice([0.0, 0.0, 100.0])),
                correlation_group=f"group_{i % n_groups}",
            ))

        # Periods: 26-260
        n_periods = int(rng.integers(26, 261))
        granularity = str(rng.choice(["weekly", "daily"], p=[0.8, 0.2]))

        # Noise
        noise = NoiseConfig(
            noise_type=str(rng.choice(["gaussian", "t_distributed", "heteroscedastic"])),
            noise_scale=float(rng.uniform(5.0, 80.0)),
        )

        # Baseline
        baseline = BaselineConfig(
            organic_level=float(rng.uniform(200.0, 3000.0)),
            trend_type="linear",
            trend_params={"slope": float(rng.uniform(-2.0, 5.0))},
            seasonality_n_terms=int(rng.integers(1, 5)),
        )

        # Optional configs (each with ~40% chance)
        pricing = None
        if rng.random() < 0.4:
            pricing = PricingConfig(
                base_price=float(rng.uniform(10.0, 100.0)),
                price_elasticity=float(rng.uniform(-3.0, -0.5)),
                promo_frequency=str(rng.choice(["weekly", "biweekly", "monthly", "quarterly"])),
                promo_depth_mean=float(rng.uniform(0.05, 0.35)),
            )

        distribution = None
        if rng.random() < 0.3:
            distribution = DistributionConfig(
                initial_distribution=float(rng.uniform(0.2, 1.0)),
                distribution_trajectory=str(rng.choice(["stable", "growing", "declining"])),
            )

        endogeneity = None
        if rng.random() < 0.4:
            endogeneity = EndogeneityConfig(
                overall_strength=float(rng.uniform(0.05, 0.8)),
                performance_chasing=float(rng.uniform(0.0, 0.5)),
                algorithmic_targeting_bias=float(rng.uniform(0.0, 0.4)),
            )

        competition = None
        if rng.random() < 0.3:
            competition = CompetitionConfig(
                n_competitors=int(rng.integers(1, 6)),
                competitor_sov_mean=float(rng.uniform(0.1, 0.5)),
                sov_suppression_coefficient=float(rng.uniform(0.05, 0.3)),
            )

        macro = None
        if rng.random() < 0.25:
            n_regime = int(rng.integers(0, 3))
            changes = [
                RegimeChange(
                    period=int(rng.integers(10, max(n_periods - 10, 11))),
                    change_type="level_shift",
                    magnitude=float(rng.uniform(-0.5, 0.3)),
                )
                for _ in range(n_regime)
            ]
            macro = MacroConfig(regime_changes=changes)

        interactions = None
        if pricing is not None and rng.random() < 0.3:
            interactions = InteractionConfig(
                price_x_media={ch.name: float(rng.uniform(-0.1, 0.3))
                               for ch in channels if rng.random() < 0.5},
            )

        return SimulationConfig(
            n_periods=n_periods,
            granularity=granularity,
            channels=channels,
            noise=noise,
            baseline=baseline,
            seed=seed,
            metadata={"scenario": "sampled"},
            pricing=pricing,
            distribution=distribution,
            endogeneity=endogeneity,
            competition=competition,
            macro=macro,
            interactions=interactions,
        )
