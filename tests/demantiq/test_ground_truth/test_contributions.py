import numpy as np
from demantiq.ground_truth.contributions import compute_contributions


def test_basic_contributions():
    effects = {
        "tv": np.array([100, 200, 150]),
        "search": np.array([50, 80, 70]),
    }
    demand = np.array([1000, 1200, 1100])
    result = compute_contributions(effects, demand)

    assert result["tv"]["total_contribution"] == 450
    assert result["search"]["total_contribution"] == 200
    assert abs(result["tv"]["contribution_pct"] - 450 / 3300) < 1e-10


def test_zero_demand():
    effects = {"tv": np.array([0, 0])}
    demand = np.array([0, 0])
    result = compute_contributions(effects, demand)
    assert result["tv"]["contribution_pct"] == 0.0
