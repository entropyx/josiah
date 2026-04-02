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

    def __init__(self, seed: int = 42, rich_context: bool = False, n_fixed_channels: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.rich_context = rich_context
        self.n_fixed_channels = n_fixed_channels

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
        """Sample a single random SimulationConfig.

        Uses a regime system to ensure diverse baseline/media splits:
        ~33% media-dominant (low organic, high betas)
        ~33% balanced
        ~33% baseline-dominant (high organic, low betas)
        """
        rng = self.rng

        # Number of channels: fixed or random 2-15
        if self.n_fixed_channels is not None:
            n_channels = self.n_fixed_channels
        else:
            n_channels = int(rng.integers(2, 16))
        channel_names = list(rng.choice(_CHANNEL_NAMES, size=n_channels, replace=False))

        # Number of correlation groups: 1 to n_channels
        n_groups = int(rng.integers(1, max(n_channels, 2)))

        # Regime: controls baseline/media ratio diversity
        regime = str(rng.choice(["media_dominant", "balanced", "baseline_dominant"]))
        if regime == "media_dominant":
            organic_range = (100.0, 800.0)
            beta_range = (200.0, 800.0)
            spend_mean_range = (5000.0, 80000.0)
        elif regime == "baseline_dominant":
            organic_range = (1500.0, 5000.0)
            beta_range = (10.0, 150.0)
            spend_mean_range = (500.0, 15000.0)
        else:  # balanced
            organic_range = (500.0, 2000.0)
            beta_range = (50.0, 500.0)
            spend_mean_range = (1000.0, 50000.0)

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

            spend_mean = float(rng.uniform(*spend_mean_range))
            channels.append(ChannelConfig(
                name=name,
                beta=float(rng.uniform(*beta_range)),
                saturation_fn=sat_fn,
                saturation_params=sat_params,
                adstock_fn=ads_fn,
                adstock_params=ads_params,
                spend_pattern=str(rng.choice(_SPEND_PATTERNS)),
                spend_mean=spend_mean,
                spend_std=float(rng.uniform(spend_mean * 0.1, spend_mean * 0.5)),
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

        # Baseline — organic_level from regime for diverse baseline/media splits
        baseline = BaselineConfig(
            organic_level=float(rng.uniform(*organic_range)),
            trend_type="linear",
            trend_params={"slope": float(rng.uniform(-2.0, 5.0))},
            seasonality_n_terms=int(rng.integers(1, 5)),
        )

        # Optional configs — probabilities depend on rich_context mode
        # rich_context=True ensures ~39% of samples have full business context
        # (pricing + distribution + interactions), vs ~3.6% in default mode
        p_pricing = 1.0 if self.rich_context else 0.4
        p_distribution = 0.7 if self.rich_context else 0.3
        p_endogeneity = 0.5 if self.rich_context else 0.4
        p_competition = 0.5 if self.rich_context else 0.3
        p_macro = 0.4 if self.rich_context else 0.25
        p_interactions = 0.7 if self.rich_context else 0.3
        p_per_channel_interaction = 0.8 if self.rich_context else 0.5

        pricing = None
        if rng.random() < p_pricing:
            pricing = PricingConfig(
                base_price=float(rng.uniform(10.0, 100.0)),
                price_elasticity=float(rng.uniform(-3.0, -0.5)),
                promo_frequency=str(rng.choice(["weekly", "biweekly", "monthly", "quarterly"])),
                promo_depth_mean=float(rng.uniform(0.05, 0.35)),
            )

        distribution = None
        if rng.random() < p_distribution:
            distribution = DistributionConfig(
                initial_distribution=float(rng.uniform(0.2, 1.0)),
                distribution_trajectory=str(rng.choice(["stable", "growing", "declining"])),
            )

        endogeneity = None
        if rng.random() < p_endogeneity:
            endogeneity = EndogeneityConfig(
                overall_strength=float(rng.uniform(0.05, 0.8)),
                performance_chasing=float(rng.uniform(0.0, 0.5)),
                algorithmic_targeting_bias=float(rng.uniform(0.0, 0.4)),
            )

        competition = None
        if rng.random() < p_competition:
            competition = CompetitionConfig(
                n_competitors=int(rng.integers(1, 6)),
                competitor_sov_mean=float(rng.uniform(0.1, 0.5)),
                sov_suppression_coefficient=float(rng.uniform(0.05, 0.3)),
            )

        macro = None
        if rng.random() < p_macro:
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
        if pricing is not None and rng.random() < p_interactions:
            interaction_dict = {}
            for ch in channels:
                if rng.random() < p_per_channel_interaction:
                    interaction_dict[ch.name] = float(rng.uniform(-0.1, 0.3))
            if interaction_dict:
                dist_interaction_dict = {}
                if distribution is not None:
                    for ch in channels:
                        if rng.random() < p_per_channel_interaction:
                            dist_interaction_dict[ch.name] = float(rng.uniform(-0.1, 0.3))
                interactions = InteractionConfig(
                    price_x_media=interaction_dict,
                    distribution_x_media=dist_interaction_dict,
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
