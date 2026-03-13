import numpy as np
from demantiq.utils.time_series import (
    fourier_seasonality, linear_trend, cube_root_trend, apply_structural_break
)
from demantiq.utils.random import create_rng


def test_fourier_seasonality_shape():
    result = fourier_seasonality(104, period=52, n_terms=2, coefficients=[1, 0, 0.5, 0])
    assert result.shape == (104,)


def test_fourier_seasonality_periodic():
    result = fourier_seasonality(104, period=52, n_terms=1, coefficients=[1, 0])
    # Values at t and t+52 should be approximately equal
    np.testing.assert_allclose(result[:52], result[52:], atol=1e-10)


def test_linear_trend_direction():
    result = linear_trend(100, slope=10.0)
    assert result[-1] > result[0]


def test_linear_trend_negative():
    result = linear_trend(100, slope=-5.0)
    assert result[-1] < result[0]


def test_cube_root_trend_shape():
    result = cube_root_trend(100, scale=10.0)
    assert result.shape == (100,)
    assert result[-1] > result[0]


def test_structural_break_level_shift():
    series = np.zeros(100)
    result = apply_structural_break(series, break_period=50, magnitude=10.0)
    assert np.allclose(result[:50], 0)
    assert np.allclose(result[50:], 10.0)


def test_structural_break_gradual_recovery():
    series = np.zeros(100)
    result = apply_structural_break(series, break_period=50, magnitude=10.0,
                                     recovery="gradual_recovery", recovery_periods=20)
    assert result[50] == 10.0  # peak at break
    assert abs(result[70]) < 0.1  # recovered by break + recovery_periods
