"""Pricing mechanics configuration."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CostConfig:
    """Cost structure for margin computation."""
    cogs_per_unit: float = 5.0
    variable_cost_per_unit: float = 2.0

    def to_dict(self) -> dict:
        return {"cogs_per_unit": self.cogs_per_unit, "variable_cost_per_unit": self.variable_cost_per_unit}

    @classmethod
    def from_dict(cls, d: dict) -> "CostConfig":
        return cls(**d)


@dataclass(frozen=True)
class PromoEvent:
    """A scheduled promotional event."""
    start_period: int = 0
    duration: int = 2
    depth: float = 0.15  # discount percentage

    def to_dict(self) -> dict:
        return {"start_period": self.start_period, "duration": self.duration, "depth": self.depth}

    @classmethod
    def from_dict(cls, d: dict) -> "PromoEvent":
        return cls(**d)


@dataclass(frozen=True)
class PricingConfig:
    """Configuration for pricing mechanics.

    Attributes:
        base_price: Regular/shelf price.
        price_elasticity: True price elasticity of demand (negative = demand drops with price).
        promo_frequency: How often promos occur ('weekly', 'biweekly', 'monthly', 'quarterly').
        promo_depth_mean: Average discount percentage.
        promo_depth_std: Discount variation.
        price_media_interaction: How promotions modify media lift (0 = no interaction).
        cost_structure: COGS and variable costs.
    """
    base_price: float = 25.0
    price_elasticity: float = -1.2
    promo_frequency: str = "monthly"
    promo_depth_mean: float = 0.15
    promo_depth_std: float = 0.05
    price_media_interaction: float = 0.0
    cost_structure: CostConfig = field(default_factory=CostConfig)

    def to_dict(self) -> dict:
        return {
            "base_price": self.base_price,
            "price_elasticity": self.price_elasticity,
            "promo_frequency": self.promo_frequency,
            "promo_depth_mean": self.promo_depth_mean,
            "promo_depth_std": self.promo_depth_std,
            "price_media_interaction": self.price_media_interaction,
            "cost_structure": self.cost_structure.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PricingConfig":
        d = dict(d)  # don't mutate caller's dict
        cost = CostConfig.from_dict(d.pop("cost_structure", {})) if "cost_structure" in d else CostConfig()
        return cls(cost_structure=cost, **d)
