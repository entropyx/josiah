"""Library of 15 named scenarios for testing MMM recovery."""

from __future__ import annotations

from demantiq.config.simulation_config import SimulationConfig
from demantiq.config.channel_config import ChannelConfig
from demantiq.config.baseline_config import BaselineConfig
from demantiq.config.noise_config import NoiseConfig
from demantiq.config.pricing_config import PricingConfig
from demantiq.config.distribution_config import DistributionConfig
from demantiq.config.endogeneity_config import EndogeneityConfig
from demantiq.config.competition_config import CompetitionConfig
from demantiq.config.macro_config import MacroConfig, MacroVariable, RegimeChange
from demantiq.config.interaction_config import InteractionConfig


def _channel(name: str, beta: float = 100.0, group: str = "default",
             sat_fn: str = "logistic", sat_params: dict | None = None,
             ads_fn: str = "geometric", ads_params: dict | None = None,
             spend_mean: float = 10000.0, spend_std: float = 3000.0,
             spend_pattern: str = "always_on",
             spend_floor: float = 0.0) -> ChannelConfig:
    """Helper to create a ChannelConfig with sensible defaults."""
    if sat_params is None:
        sat_params = {"K": 0.5, "S": 2.0} if sat_fn == "hill" else {"k": 3.0, "x0": 0.5}
    if ads_params is None:
        ads_params = {"alpha": 0.5, "max_lag": 8}
    return ChannelConfig(
        name=name,
        beta=beta,
        saturation_fn=sat_fn,
        saturation_params=sat_params,
        adstock_fn=ads_fn,
        adstock_params=ads_params,
        spend_pattern=spend_pattern,
        spend_mean=spend_mean,
        spend_std=spend_std,
        spend_floor=spend_floor,
        correlation_group=group,
    )


_COMMON_CHANNELS = [
    "facebook", "google", "tiktok", "pinterest", "email",
    "youtube", "snapchat", "linkedin", "twitter", "display",
    "programmatic", "podcast", "influencer", "radio", "ctv",
    "affiliate", "sms", "direct_mail", "ooh", "print",
]


class ScenarioLibrary:
    """Collection of 15 named benchmark scenarios.

    Each classmethod returns a fully constructed SimulationConfig.
    """

    @classmethod
    def list_scenarios(cls) -> list[str]:
        """Return names of all available scenarios."""
        return [
            "clean_room", "real_world", "adversarial", "pricing_dominant",
            "interaction_heavy", "short_data", "many_channels", "regime_shift",
            "new_brand", "mature_market", "promotional_trap", "platform_bias",
            "competitor_entry", "dtc_pure_play", "omnichannel_retail",
        ]

    @classmethod
    def get(cls, name: str) -> SimulationConfig:
        """Get a scenario by name."""
        method = getattr(cls, name, None)
        if method is None or not callable(method):
            raise ValueError(f"Unknown scenario: {name}")
        return method()

    @classmethod
    def all_scenarios(cls) -> dict[str, SimulationConfig]:
        """Return all 15 scenarios as a dict."""
        return {name: cls.get(name) for name in cls.list_scenarios()}

    # ── SCN-001 Clean Room ──────────────────────────────────────────────

    @classmethod
    def clean_room(cls) -> SimulationConfig:
        """No endogeneity, low noise, orthogonal spend, 156 weeks."""
        channels = [
            _channel(name, beta=beta, group=f"group_{i}",
                     sat_fn="logistic", sat_params={"k": 3.0, "x0": 0.5})
            for i, (name, beta) in enumerate([
                ("facebook", 200.0), ("google", 500.0), ("tiktok", 150.0),
                ("pinterest", 80.0), ("email", 300.0),
            ])
        ]
        return SimulationConfig(
            n_periods=156,
            channels=channels,
            noise=NoiseConfig(noise_scale=5.0),
            baseline=BaselineConfig(organic_level=1000.0),
            seed=1001,
            metadata={"scenario": "SCN-001", "name": "clean_room"},
        )

    # ── SCN-002 Real World ──────────────────────────────────────────────

    @classmethod
    def real_world(cls) -> SimulationConfig:
        """Moderate endogeneity, correlated channels, 104 weeks."""
        channels = [
            _channel("facebook", beta=250.0, group="social"),
            _channel("google", beta=400.0, group="search"),
            _channel("tiktok", beta=180.0, group="social"),
            _channel("pinterest", beta=100.0, group="social"),
            _channel("email", beta=350.0, group="owned"),
            _channel("youtube", beta=200.0, group="video"),
            _channel("display", beta=120.0, group="programmatic"),
            _channel("programmatic", beta=90.0, group="programmatic"),
        ]
        return SimulationConfig(
            n_periods=104,
            channels=channels,
            noise=NoiseConfig(noise_scale=30.0),
            baseline=BaselineConfig(organic_level=1000.0, seasonality_n_terms=3),
            seed=1002,
            metadata={"scenario": "SCN-002", "name": "real_world"},
            endogeneity=EndogeneityConfig(
                overall_strength=0.3,
                seasonal_allocation_bias=0.2,
            ),
        )

    # ── SCN-003 Adversarial ─────────────────────────────────────────────

    @classmethod
    def adversarial(cls) -> SimulationConfig:
        """High endogeneity, heavy collinearity, low SNR, 52 weeks."""
        channels = [
            _channel(name, beta=float(50 + i * 20), group=f"block_{i // 4}")
            for i, name in enumerate(_COMMON_CHANNELS[:12])
        ]
        return SimulationConfig(
            n_periods=52,
            channels=channels,
            noise=NoiseConfig(noise_scale=50.0),
            baseline=BaselineConfig(organic_level=1000.0),
            seed=1003,
            metadata={"scenario": "SCN-003", "name": "adversarial"},
            endogeneity=EndogeneityConfig(overall_strength=0.7, performance_chasing=0.5),
            macro=MacroConfig(regime_changes=[
                RegimeChange(period=26, change_type="level_shift", magnitude=-0.3),
            ]),
        )

    # ── SCN-004 Pricing Dominant ────────────────────────────────────────

    @classmethod
    def pricing_dominant(cls) -> SimulationConfig:
        """High price elasticity, heavy promos, small channel betas."""
        channels = [
            _channel(name, beta=30.0, group=f"g_{i}")
            for i, name in enumerate(["facebook", "google", "email", "display"])
        ]
        return SimulationConfig(
            n_periods=104,
            channels=channels,
            noise=NoiseConfig(noise_scale=20.0),
            baseline=BaselineConfig(organic_level=1200.0),
            seed=1004,
            metadata={"scenario": "SCN-004", "name": "pricing_dominant"},
            pricing=PricingConfig(
                base_price=30.0,
                price_elasticity=-2.5,
                promo_frequency="weekly",
                promo_depth_mean=0.25,
                promo_depth_std=0.10,
            ),
        )

    # ── SCN-005 Interaction Heavy ───────────────────────────────────────

    @classmethod
    def interaction_heavy(cls) -> SimulationConfig:
        """Large price_x_media and distribution_x_media interactions."""
        ch_names = ["facebook", "google", "tiktok", "email", "youtube"]
        channels = [
            _channel(name, beta=150.0, group=f"g_{i}")
            for i, name in enumerate(ch_names)
        ]
        return SimulationConfig(
            n_periods=104,
            channels=channels,
            noise=NoiseConfig(noise_scale=20.0),
            baseline=BaselineConfig(organic_level=1000.0),
            seed=1005,
            metadata={"scenario": "SCN-005", "name": "interaction_heavy"},
            pricing=PricingConfig(base_price=25.0, price_elasticity=-1.5),
            distribution=DistributionConfig(initial_distribution=0.7),
            interactions=InteractionConfig(
                price_x_media={n: 0.25 for n in ch_names},
                distribution_x_media={n: 0.30 for n in ch_names},
            ),
        )

    # ── SCN-006 Short Data ──────────────────────────────────────────────

    @classmethod
    def short_data(cls) -> SimulationConfig:
        """36 weeks, moderate complexity, 5 channels."""
        channels = [
            _channel(name, beta=beta, group=f"g_{i}")
            for i, (name, beta) in enumerate([
                ("facebook", 200.0), ("google", 350.0), ("tiktok", 120.0),
                ("email", 250.0), ("display", 80.0),
            ])
        ]
        return SimulationConfig(
            n_periods=36,
            channels=channels,
            noise=NoiseConfig(noise_scale=25.0),
            baseline=BaselineConfig(organic_level=1000.0),
            seed=1006,
            metadata={"scenario": "SCN-006", "name": "short_data"},
            endogeneity=EndogeneityConfig(overall_strength=0.15),
        )

    # ── SCN-007 Many Channels ───────────────────────────────────────────

    @classmethod
    def many_channels(cls) -> SimulationConfig:
        """18 channels with moderate collinearity, 104 weeks."""
        channels = [
            _channel(name, beta=float(60 + i * 15), group=f"block_{i // 3}")
            for i, name in enumerate(_COMMON_CHANNELS[:18])
        ]
        return SimulationConfig(
            n_periods=104,
            channels=channels,
            noise=NoiseConfig(noise_scale=30.0),
            baseline=BaselineConfig(organic_level=1200.0),
            seed=1007,
            metadata={"scenario": "SCN-007", "name": "many_channels"},
        )

    # ── SCN-008 Regime Shift ────────────────────────────────────────────

    @classmethod
    def regime_shift(cls) -> SimulationConfig:
        """156 weeks with a macro regime change at period 78."""
        channels = [
            _channel("facebook", beta=200.0, group="social"),
            _channel("google", beta=350.0, group="search"),
            _channel("email", beta=150.0, group="owned"),
        ]
        return SimulationConfig(
            n_periods=156,
            channels=channels,
            noise=NoiseConfig(noise_scale=20.0),
            baseline=BaselineConfig(organic_level=1000.0),
            seed=1008,
            metadata={"scenario": "SCN-008", "name": "regime_shift"},
            macro=MacroConfig(regime_changes=[
                RegimeChange(
                    period=78,
                    change_type="level_shift",
                    magnitude=-0.4,
                    affected_params=["baseline"],
                    recovery="permanent",
                ),
            ]),
        )

    # ── SCN-009 New Brand ───────────────────────────────────────────────

    @classmethod
    def new_brand(cls) -> SimulationConfig:
        """Distribution starts low and grows; increasing baseline trend."""
        channels = [
            _channel("facebook", beta=250.0, group="social"),
            _channel("google", beta=400.0, group="search"),
            _channel("tiktok", beta=200.0, group="social"),
            _channel("email", beta=100.0, group="owned"),
        ]
        return SimulationConfig(
            n_periods=52,
            channels=channels,
            noise=NoiseConfig(noise_scale=20.0),
            baseline=BaselineConfig(
                organic_level=500.0,
                trend_type="linear",
                trend_params={"slope": 5.0},
            ),
            seed=1009,
            metadata={"scenario": "SCN-009", "name": "new_brand"},
            distribution=DistributionConfig(
                initial_distribution=0.2,
                distribution_trajectory="growing",
                trajectory_params={"growth_rate": 0.02},
            ),
        )

    # ── SCN-010 Mature Market ───────────────────────────────────────────

    @classmethod
    def mature_market(cls) -> SimulationConfig:
        """High saturation params so channels are near saturation."""
        channels = [
            _channel(name, beta=beta, group=f"g_{i}",
                     sat_fn="logistic", sat_params={"k": 0.5, "x0": 0.5})
            for i, (name, beta) in enumerate([
                ("facebook", 300.0), ("google", 500.0), ("tiktok", 200.0),
                ("email", 400.0), ("youtube", 250.0),
            ])
        ]
        return SimulationConfig(
            n_periods=104,
            channels=channels,
            noise=NoiseConfig(noise_scale=15.0),
            baseline=BaselineConfig(organic_level=2000.0, trend_params={"slope": 0.5}),
            seed=1010,
            metadata={"scenario": "SCN-010", "name": "mature_market"},
        )

    # ── SCN-011 Promotional Trap ────────────────────────────────────────

    @classmethod
    def promotional_trap(cls) -> SimulationConfig:
        """Heavy promo calendar, price_x_media interaction, correlated spend."""
        ch_names = ["facebook", "google", "tiktok", "display", "email"]
        channels = [
            _channel(name, beta=150.0, group="promo_correlated")
            for name in ch_names
        ]
        return SimulationConfig(
            n_periods=104,
            channels=channels,
            noise=NoiseConfig(noise_scale=25.0),
            baseline=BaselineConfig(organic_level=1000.0),
            seed=1011,
            metadata={"scenario": "SCN-011", "name": "promotional_trap"},
            pricing=PricingConfig(
                base_price=20.0,
                price_elasticity=-1.8,
                promo_frequency="weekly",
                promo_depth_mean=0.30,
                promo_depth_std=0.10,
            ),
            interactions=InteractionConfig(
                price_x_media={n: 0.20 for n in ch_names},
            ),
        )

    # ── SCN-012 Platform Bias ───────────────────────────────────────────

    @classmethod
    def platform_bias(cls) -> SimulationConfig:
        """High algorithmic targeting bias and performance chasing."""
        channels = [
            _channel("facebook", beta=250.0, group="social"),
            _channel("google", beta=400.0, group="search"),
            _channel("tiktok", beta=180.0, group="social"),
            _channel("youtube", beta=200.0, group="video"),
            _channel("display", beta=120.0, group="programmatic"),
        ]
        return SimulationConfig(
            n_periods=104,
            channels=channels,
            noise=NoiseConfig(noise_scale=20.0),
            baseline=BaselineConfig(organic_level=1000.0),
            seed=1012,
            metadata={"scenario": "SCN-012", "name": "platform_bias"},
            endogeneity=EndogeneityConfig(
                overall_strength=0.5,
                algorithmic_targeting_bias=0.6,
                performance_chasing=0.4,
            ),
        )

    # ── SCN-013 Competitor Entry ────────────────────────────────────────

    @classmethod
    def competitor_entry(cls) -> SimulationConfig:
        """Competition SOV doubles at week 60 via regime change."""
        channels = [
            _channel("facebook", beta=250.0, group="social"),
            _channel("google", beta=400.0, group="search"),
            _channel("tiktok", beta=150.0, group="social"),
            _channel("email", beta=200.0, group="owned"),
            _channel("youtube", beta=180.0, group="video"),
        ]
        return SimulationConfig(
            n_periods=104,
            channels=channels,
            noise=NoiseConfig(noise_scale=20.0),
            baseline=BaselineConfig(organic_level=1000.0),
            seed=1013,
            metadata={"scenario": "SCN-013", "name": "competitor_entry"},
            competition=CompetitionConfig(
                n_competitors=3,
                competitor_sov_mean=0.2,
                sov_suppression_coefficient=0.15,
                competitive_intensity_trend="increasing",
            ),
            macro=MacroConfig(regime_changes=[
                RegimeChange(
                    period=60,
                    change_type="level_shift",
                    magnitude=-0.15,
                    affected_params=["baseline"],
                    recovery="permanent",
                ),
            ]),
        )

    # ── SCN-014 DTC Pure Play ───────────────────────────────────────────

    @classmethod
    def dtc_pure_play(cls) -> SimulationConfig:
        """No distribution, daily granularity, 365 periods."""
        channels = [
            _channel("facebook", beta=300.0, group="social",
                     spend_mean=5000.0, spend_std=1500.0),
            _channel("google", beta=500.0, group="search",
                     spend_mean=8000.0, spend_std=2000.0),
            _channel("tiktok", beta=250.0, group="social",
                     spend_mean=4000.0, spend_std=1200.0),
            _channel("email", beta=400.0, group="owned",
                     spend_mean=1000.0, spend_std=300.0),
        ]
        return SimulationConfig(
            n_periods=365,
            granularity="daily",
            channels=channels,
            noise=NoiseConfig(noise_scale=20.0),
            baseline=BaselineConfig(
                organic_level=800.0,
                seasonality_period=365.0,
            ),
            seed=1014,
            metadata={"scenario": "SCN-014", "name": "dtc_pure_play"},
        )

    # ── SCN-015 Omnichannel Retail ──────────────────────────────────────

    @classmethod
    def omnichannel_retail(cls) -> SimulationConfig:
        """20 channels, all features enabled, 156 weeks."""
        channels = [
            _channel(name, beta=float(50 + i * 25), group=f"block_{i // 4}")
            for i, name in enumerate(_COMMON_CHANNELS[:20])
        ]
        ch_names = [ch.name for ch in channels]
        return SimulationConfig(
            n_periods=156,
            channels=channels,
            noise=NoiseConfig(noise_scale=30.0),
            baseline=BaselineConfig(organic_level=1500.0, seasonality_n_terms=3),
            seed=1015,
            metadata={"scenario": "SCN-015", "name": "omnichannel_retail"},
            pricing=PricingConfig(
                base_price=35.0,
                price_elasticity=-1.5,
                promo_frequency="biweekly",
            ),
            distribution=DistributionConfig(
                initial_distribution=0.85,
                distribution_trajectory="stable",
            ),
            endogeneity=EndogeneityConfig(overall_strength=0.25),
            competition=CompetitionConfig(
                n_competitors=4,
                competitor_sov_mean=0.25,
                sov_suppression_coefficient=0.1,
            ),
            macro=MacroConfig(
                variables=[MacroVariable(name="consumer_confidence", effect_on_demand=30.0)],
                regime_changes=[
                    RegimeChange(period=80, change_type="level_shift", magnitude=-0.1),
                ],
            ),
            interactions=InteractionConfig(
                price_x_media={n: 0.10 for n in ch_names[:5]},
                distribution_x_media={n: 0.15 for n in ch_names[:5]},
            ),
        )
