"""Public data adapter with hardcoded category benchmarks.

Provides industry-level calibration data (media/revenue ratios, typical
channel mixes, seasonal patterns, price elasticities) drawn from PRD
Appendix C for 8 business categories.
"""

from __future__ import annotations

from typing import Optional


CATEGORY_BENCHMARKS: dict[str, dict] = {
    "supplements": {
        "media_revenue_ratio": (0.10, 0.25),
        "typical_channels": [
            "paid_social", "paid_search", "display", "email", "influencer",
        ],
        "seasonal_pattern": {"amplitude": 0.1, "peak_month": 1},
        "price_elasticity": (-3.0, -1.0),
        "n_channels_typical": (4, 8),
    },
    "dtc_skincare": {
        "media_revenue_ratio": (0.15, 0.35),
        "typical_channels": [
            "paid_social", "paid_search", "display", "email",
            "influencer", "youtube",
        ],
        "seasonal_pattern": {"amplitude": 0.15, "peak_month": 12},
        "price_elasticity": (-2.5, -0.8),
        "n_channels_typical": (5, 10),
    },
    "qsr": {
        "media_revenue_ratio": (0.03, 0.08),
        "typical_channels": [
            "linear_tv", "ctv", "paid_social", "paid_search", "display", "ooh",
        ],
        "seasonal_pattern": {"amplitude": 0.2, "peak_month": 7},
        "price_elasticity": (-2.0, -0.5),
        "n_channels_typical": (6, 12),
    },
    "financial_services": {
        "media_revenue_ratio": (0.05, 0.15),
        "typical_channels": [
            "paid_search", "display", "linear_tv", "ctv",
            "paid_social", "email",
        ],
        "seasonal_pattern": {"amplitude": 0.08, "peak_month": 1},
        "price_elasticity": (-1.5, -0.3),
        "n_channels_typical": (5, 10),
    },
    "online_education": {
        "media_revenue_ratio": (0.12, 0.30),
        "typical_channels": [
            "paid_social", "paid_search", "youtube", "display",
            "email", "influencer",
        ],
        "seasonal_pattern": {"amplitude": 0.25, "peak_month": 9},
        "price_elasticity": (-2.0, -0.6),
        "n_channels_typical": (4, 8),
    },
    "consumer_electronics": {
        "media_revenue_ratio": (0.04, 0.12),
        "typical_channels": [
            "paid_search", "paid_social", "display", "ctv",
            "youtube", "affiliate",
        ],
        "seasonal_pattern": {"amplitude": 0.30, "peak_month": 11},
        "price_elasticity": (-4.0, -1.5),
        "n_channels_typical": (5, 10),
    },
    "automotive": {
        "media_revenue_ratio": (0.02, 0.06),
        "typical_channels": [
            "linear_tv", "ctv", "paid_search", "display",
            "ooh", "paid_social",
        ],
        "seasonal_pattern": {"amplitude": 0.12, "peak_month": 3},
        "price_elasticity": (-1.0, -0.2),
        "n_channels_typical": (6, 12),
    },
    "cpg_fmcg": {
        "media_revenue_ratio": (0.08, 0.20),
        "typical_channels": [
            "linear_tv", "paid_social", "display", "paid_search",
            "ooh", "email",
        ],
        "seasonal_pattern": {"amplitude": 0.18, "peak_month": 12},
        "price_elasticity": (-3.5, -1.0),
        "n_channels_typical": (6, 12),
    },
}


class PublicDataAdapter:
    """Adapter for accessing hardcoded category benchmark data.

    Provides methods to look up media/revenue ratios, typical channels,
    seasonal patterns, and price elasticities for supported business
    categories.
    """

    SUPPORTED_CATEGORIES = list(CATEGORY_BENCHMARKS.keys())

    def ingest_category_benchmarks(self, category: str) -> dict:
        """Return the full benchmark dictionary for a category.

        Args:
            category: One of the supported category keys.

        Returns:
            Dict with media_revenue_ratio, typical_channels,
            seasonal_pattern, price_elasticity, n_channels_typical.

        Raises:
            KeyError: If category is not supported.
        """
        if category not in CATEGORY_BENCHMARKS:
            raise KeyError(
                f"Unknown category '{category}'. "
                f"Supported: {self.SUPPORTED_CATEGORIES}"
            )
        return dict(CATEGORY_BENCHMARKS[category])

    def get_media_revenue_ratio(self, category: str) -> tuple[float, float]:
        """Return (low, high) media-to-revenue ratio for a category."""
        return self.ingest_category_benchmarks(category)["media_revenue_ratio"]

    def get_typical_channels(self, category: str) -> list[str]:
        """Return list of typical advertising channels for a category."""
        return list(self.ingest_category_benchmarks(category)["typical_channels"])

    def get_seasonal_pattern(self, category: str) -> dict:
        """Return seasonal pattern dict (amplitude, peak_month) for a category."""
        return dict(self.ingest_category_benchmarks(category)["seasonal_pattern"])
