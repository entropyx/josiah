"""Tests for counterfactual analysis."""

import numpy as np
from demantiq import SimulationConfig, ChannelConfig
from demantiq.ground_truth.counterfactuals import compute_counterfactual


def _simple_config():
    return SimulationConfig(
        n_periods=52,
        channels=[
            ChannelConfig(name="tv", beta=500.0, spend_mean=10000, spend_std=2000),
            ChannelConfig(name="search", beta=300.0, spend_mean=5000, spend_std=1000),
        ],
        seed=42,
    )


def test_counterfactual_returns_expected_keys():
    result = compute_counterfactual(_simple_config(), "tv")
    assert "total_demand_actual" in result
    assert "total_demand_counterfactual" in result
    assert "incremental_demand" in result
    assert "incremental_pct" in result
    assert result["channel"] == "tv"


def test_counterfactual_positive_incremental():
    """Zeroing a positive-beta channel should reduce demand."""
    result = compute_counterfactual(_simple_config(), "tv")
    assert result["incremental_demand"] > 0
    assert result["incremental_pct"] > 0


def test_counterfactual_zeroed_less_than_actual():
    result = compute_counterfactual(_simple_config(), "tv")
    assert result["total_demand_counterfactual"] < result["total_demand_actual"]


def test_counterfactual_different_channels():
    """TV (beta=500) should have larger incremental than search (beta=300)."""
    tv_result = compute_counterfactual(_simple_config(), "tv")
    search_result = compute_counterfactual(_simple_config(), "search")
    assert tv_result["incremental_demand"] > search_result["incremental_demand"]
