"""Tests for PublicDataAdapter."""

import pytest

from demantiq.calibration.public_data_adapter import (
    PublicDataAdapter,
    CATEGORY_BENCHMARKS,
)


ALL_CATEGORIES = [
    "supplements",
    "dtc_skincare",
    "qsr",
    "financial_services",
    "online_education",
    "consumer_electronics",
    "automotive",
    "cpg_fmcg",
]


class TestCategoryCompleteness:
    """All 8 categories are present and well-formed."""

    def test_eight_categories_exist(self):
        assert len(CATEGORY_BENCHMARKS) == 8

    @pytest.mark.parametrize("category", ALL_CATEGORIES)
    def test_category_in_benchmarks(self, category):
        assert category in CATEGORY_BENCHMARKS

    @pytest.mark.parametrize("category", ALL_CATEGORIES)
    def test_required_keys(self, category):
        bench = CATEGORY_BENCHMARKS[category]
        for key in [
            "media_revenue_ratio",
            "typical_channels",
            "seasonal_pattern",
            "price_elasticity",
            "n_channels_typical",
        ]:
            assert key in bench, f"Missing {key} in {category}"


class TestAdapterMethods:
    """PublicDataAdapter accessor methods."""

    @pytest.mark.parametrize("category", ALL_CATEGORIES)
    def test_ingest_returns_dict(self, category):
        adapter = PublicDataAdapter()
        result = adapter.ingest_category_benchmarks(category)
        assert isinstance(result, dict)
        assert "media_revenue_ratio" in result

    @pytest.mark.parametrize("category", ALL_CATEGORIES)
    def test_media_revenue_ratio_range(self, category):
        adapter = PublicDataAdapter()
        lo, hi = adapter.get_media_revenue_ratio(category)
        assert 0 < lo < hi <= 1.0

    @pytest.mark.parametrize("category", ALL_CATEGORIES)
    def test_typical_channels_non_empty(self, category):
        adapter = PublicDataAdapter()
        channels = adapter.get_typical_channels(category)
        assert isinstance(channels, list)
        assert len(channels) >= 3

    @pytest.mark.parametrize("category", ALL_CATEGORIES)
    def test_seasonal_pattern_has_keys(self, category):
        adapter = PublicDataAdapter()
        sp = adapter.get_seasonal_pattern(category)
        assert "amplitude" in sp
        assert "peak_month" in sp
        assert 0 < sp["amplitude"] <= 1.0
        assert 1 <= sp["peak_month"] <= 12

    @pytest.mark.parametrize("category", ALL_CATEGORIES)
    def test_price_elasticity_negative(self, category):
        bench = CATEGORY_BENCHMARKS[category]
        lo, hi = bench["price_elasticity"]
        assert lo < hi < 0

    def test_unknown_category_raises(self):
        adapter = PublicDataAdapter()
        with pytest.raises(KeyError):
            adapter.ingest_category_benchmarks("unknown_category")

    def test_supported_categories_attribute(self):
        adapter = PublicDataAdapter()
        assert set(adapter.SUPPORTED_CATEGORIES) == set(ALL_CATEGORIES)
