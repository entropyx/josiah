"""Interaction configuration for cross-variable effects."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CustomInteraction:
    """A user-defined interaction between two named variables.

    Attributes:
        variable_a: Name of the first variable (the one being modified).
        variable_b: Name of the second variable (the modifier).
        coefficient: Strength of the interaction.
        interaction_type: 'multiplicative' or 'additive'.
    """

    variable_a: str
    variable_b: str
    coefficient: float
    interaction_type: str = "multiplicative"


@dataclass(frozen=True)
class InteractionConfig:
    """Configuration for variable interactions (Milestone 4).

    Attributes:
        price_x_media: {channel_name: coefficient} — promo amplifies media.
        distribution_x_media: {channel_name: coefficient} — distribution amplifies media.
        media_x_media: {(ch_a, ch_b): coefficient} — cross-channel synergy.
        competition_x_media: {channel_name: coefficient} — competition dampens media.
        custom_interactions: List of CustomInteraction objects.
    """

    price_x_media: dict = field(default_factory=dict)
    distribution_x_media: dict = field(default_factory=dict)
    media_x_media: dict = field(default_factory=dict)
    competition_x_media: dict = field(default_factory=dict)
    custom_interactions: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "price_x_media": dict(self.price_x_media),
            "distribution_x_media": dict(self.distribution_x_media),
            "media_x_media": {
                f"{k[0]}_{k[1]}": v for k, v in self.media_x_media.items()
            },
            "competition_x_media": dict(self.competition_x_media),
            "custom_interactions": [
                {
                    "variable_a": c.variable_a,
                    "variable_b": c.variable_b,
                    "coefficient": c.coefficient,
                    "interaction_type": c.interaction_type,
                }
                for c in self.custom_interactions
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "InteractionConfig":
        media_x_media = {}
        for key, val in d.get("media_x_media", {}).items():
            parts = key.split("_", 1)
            if len(parts) == 2:
                media_x_media[(parts[0], parts[1])] = val

        custom = [
            CustomInteraction(**ci) for ci in d.get("custom_interactions", [])
        ]

        return cls(
            price_x_media=d.get("price_x_media", {}),
            distribution_x_media=d.get("distribution_x_media", {}),
            media_x_media=media_x_media,
            competition_x_media=d.get("competition_x_media", {}),
            custom_interactions=custom,
        )
