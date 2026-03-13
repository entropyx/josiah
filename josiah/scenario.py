from dataclasses import dataclass, field
from typing import Optional
import numpy as np


CHANNEL_NAMES = [
    "facebook", "google", "tiktok", "pinterest", "email",
    "youtube", "snapchat", "linkedin", "twitter", "display",
]

PROMO_NAMES = [
    "black_friday", "cyber_monday", "prime_day", "summer_sale",
    "holiday_sale", "flash_sale", "new_year_sale", "back_to_school",
    "valentines", "spring_sale", "labor_day", "memorial_day",
]


@dataclass
class ChannelConfig:
    name: str
    alpha: float        # adstock retention (0-1)
    l_max: int          # max lag
    lam: float          # saturation lambda
    beta: float         # channel coefficient
    spend_mean: float
    spend_std: float


@dataclass
class ControlConfig:
    name: str
    gamma_shape: float
    gamma_scale: float
    coefficient: float


@dataclass
class PromoConfig:
    name: str
    coefficient: float          # lift coefficient (added to y when promo=1)
    n_occurrences: int = 1      # how many times per year
    duration_days: int = 3      # days per occurrence


@dataclass
class ScenarioConfig:
    name: str
    engine: str = "pymc"
    start_date: str = "2022-01-01"
    end_date: str = "2024-12-31"
    frequency: str = "W"
    intercept: float = 1000.0
    noise_std: float = 50.0
    trend_type: str = "linear"
    trend_params: dict = field(default_factory=lambda: {"slope": 0.001})
    seasonality_n_terms: int = 2
    seasonality_coefficients: list = field(default_factory=list)
    channels: list = field(default_factory=list)
    controls: list = field(default_factory=list)
    promos: list = field(default_factory=list)
    seed: int = 42


@dataclass
class BatchConfig:
    n_scenarios: int = 10
    engine: str = "pymc"
    start_date: str = "2022-01-01"
    end_date: str = "2024-12-31"
    frequency: str = "W"
    # Channel params
    n_channels_range: tuple = (2, 5)
    alpha_range: tuple = (0.1, 0.9)
    l_max_range: tuple = (1, 12)
    lam_range: tuple = (0.5, 5.0)
    beta_range: tuple = (200.0, 1500.0)
    spend_mean_range: tuple = (500.0, 5000.0)
    spend_std_ratio_range: tuple = (0.1, 0.5)
    # Scale params
    intercept_range: tuple = (500.0, 2000.0)
    noise_std_range: tuple = (10.0, 100.0)
    trend_slope_range: tuple = (0.0, 3.0)
    seasonality_n_terms_range: tuple = (1, 3)
    seas_coeff_range: tuple = (10.0, 100.0)
    # Controls
    n_controls_range: tuple = (0, 3)
    control_coeff_range: tuple = (50.0, 500.0)
    # Promos
    n_promos_range: tuple = (0, 3)
    promo_coeff_range: tuple = (50.0, 300.0)
    promo_duration_range: tuple = (1, 7)
    promo_occurrences_range: tuple = (1, 3)
    # Seed
    master_seed: int = 42


def generate_batch(batch_config: BatchConfig) -> list:
    """Generate N random ScenarioConfigs by sampling from ranges."""
    rng = np.random.default_rng(batch_config.master_seed)
    configs = []

    for i in range(batch_config.n_scenarios):
        scenario_seed = int(rng.integers(0, 2**31))

        # Channels
        n_ch = rng.integers(batch_config.n_channels_range[0], batch_config.n_channels_range[1] + 1)
        channel_names = rng.choice(CHANNEL_NAMES, size=n_ch, replace=False).tolist()

        channels = []
        for name in channel_names:
            spend_mean = rng.uniform(*batch_config.spend_mean_range)
            channels.append(ChannelConfig(
                name=name,
                alpha=round(rng.uniform(*batch_config.alpha_range), 3),
                l_max=int(rng.integers(*batch_config.l_max_range)),
                lam=round(rng.uniform(*batch_config.lam_range), 3),
                beta=round(rng.uniform(*batch_config.beta_range), 3),
                spend_mean=round(spend_mean, 2),
                spend_std=round(spend_mean * rng.uniform(*batch_config.spend_std_ratio_range), 2),
            ))

        # Controls
        n_ctrl = rng.integers(batch_config.n_controls_range[0], batch_config.n_controls_range[1] + 1)
        controls = []
        for j in range(n_ctrl):
            controls.append(ControlConfig(
                name=f"z{j + 1}",
                gamma_shape=round(rng.uniform(1.0, 5.0), 2),
                gamma_scale=round(rng.uniform(0.5, 3.0), 2),
                coefficient=round(rng.uniform(*batch_config.control_coeff_range), 3),
            ))

        # Promos
        n_promos = rng.integers(batch_config.n_promos_range[0], batch_config.n_promos_range[1] + 1)
        promo_names = rng.choice(PROMO_NAMES, size=min(n_promos, len(PROMO_NAMES)), replace=False).tolist()
        promos = []
        for pname in promo_names:
            promos.append(PromoConfig(
                name=pname,
                coefficient=round(rng.uniform(*batch_config.promo_coeff_range), 3),
                n_occurrences=int(rng.integers(*batch_config.promo_occurrences_range)),
                duration_days=int(rng.integers(*batch_config.promo_duration_range)),
            ))

        # Seasonality
        n_terms = int(rng.integers(*batch_config.seasonality_n_terms_range))
        seas_rng = np.random.default_rng(scenario_seed + 1000)
        seas_amp = rng.uniform(*batch_config.seas_coeff_range)
        seas_coeffs = seas_rng.uniform(-seas_amp, seas_amp, size=2 * n_terms).tolist()

        slope = round(rng.uniform(*batch_config.trend_slope_range), 5)

        configs.append(ScenarioConfig(
            name=f"scenario_{i + 1:03d}",
            engine=batch_config.engine,
            start_date=batch_config.start_date,
            end_date=batch_config.end_date,
            frequency=batch_config.frequency,
            intercept=round(rng.uniform(*batch_config.intercept_range), 2),
            noise_std=round(rng.uniform(*batch_config.noise_std_range), 4),
            trend_type="linear",
            trend_params={"slope": slope},
            seasonality_n_terms=n_terms,
            seasonality_coefficients=seas_coeffs,
            channels=channels,
            controls=controls,
            promos=promos,
            seed=scenario_seed,
        ))

    return configs
