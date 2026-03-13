import numpy as np
from demantiq.utils.random import create_rng
from demantiq.utils.distributions import sample_lognormal, sample_gamma, sample_truncated_normal


def test_lognormal_positive():
    rng = create_rng(42)
    samples = sample_lognormal(rng, mean=100, std=30, size=1000)
    assert np.all(samples > 0)


def test_lognormal_mean():
    rng = create_rng(42)
    samples = sample_lognormal(rng, mean=100, std=30, size=10000)
    assert abs(np.mean(samples) - 100) < 10  # within 10% of target mean


def test_gamma_positive():
    rng = create_rng(42)
    samples = sample_gamma(rng, mean=50, std=15, size=1000)
    assert np.all(samples > 0)


def test_truncated_normal_bounds():
    rng = create_rng(42)
    samples = sample_truncated_normal(rng, mean=0, std=1, low=-2, high=2, size=1000)
    assert np.all(samples >= -2)
    assert np.all(samples <= 2)
