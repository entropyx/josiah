import numpy as np
from demantiq.ground_truth.roas_calculator import compute_roas


def test_basic_roas():
    contributions = {"tv": np.array([100, 200, 150])}
    spend = {"tv": np.array([1000, 2000, 1500])}
    result = compute_roas(contributions, spend)
    assert result["tv"] == 450 / 4500  # 0.1


def test_zero_spend():
    contributions = {"tv": np.array([100])}
    spend = {"tv": np.array([0])}
    result = compute_roas(contributions, spend)
    assert result["tv"] == 0.0
