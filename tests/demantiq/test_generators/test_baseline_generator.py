import numpy as np
from demantiq.config.baseline_config import BaselineConfig
from demantiq.generators.baseline_generator import generate_baseline
from demantiq.utils.random import create_rng


def test_shape():
    config = BaselineConfig()
    rng = create_rng(42)
    result = generate_baseline(config, 104, rng)
    assert result.shape == (104,)


def test_includes_organic_level():
    config = BaselineConfig(organic_level=1000.0, trend_type="linear",
                            trend_params={"slope": 0.0}, seasonality_n_terms=0,
                            seasonality_coefficients=[])
    rng = create_rng(42)
    result = generate_baseline(config, 104, rng)
    # With zero trend and zero seasonality, should be constant at organic_level
    np.testing.assert_allclose(result, 1000.0)


def test_linear_trend_increases():
    config = BaselineConfig(organic_level=0.0, trend_type="linear",
                            trend_params={"slope": 100.0},
                            seasonality_n_terms=0, seasonality_coefficients=[])
    rng = create_rng(42)
    result = generate_baseline(config, 104, rng)
    assert result[-1] > result[0]


def test_seasonality_periodic():
    config = BaselineConfig(organic_level=0.0, trend_type="linear",
                            trend_params={"slope": 0.0}, seasonality_period=52.0,
                            seasonality_n_terms=1, seasonality_coefficients=[100.0, 0.0])
    rng = create_rng(42)
    result = generate_baseline(config, 104, rng)
    # Values at t and t+52 should be equal
    np.testing.assert_allclose(result[:52], result[52:], atol=1e-10)


def test_deterministic():
    config = BaselineConfig()
    r1 = generate_baseline(config, 104, create_rng(42))
    r2 = generate_baseline(config, 104, create_rng(42))
    np.testing.assert_allclose(r1, r2)
