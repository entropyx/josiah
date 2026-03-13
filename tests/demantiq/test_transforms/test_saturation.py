import numpy as np
import pytest
from demantiq.transforms.saturation import (
    hill, hill_inverse, logistic, logistic_inverse,
    power, power_inverse, piecewise_linear, get_saturation_fn,
)


class TestHill:
    def test_zero_input(self):
        assert hill(np.array([0.0]), K=0.5, S=2.0)[0] == 0.0

    def test_half_saturation(self):
        """f(K) should equal 0.5."""
        result = hill(np.array([0.5]), K=0.5, S=2.0)
        np.testing.assert_allclose(result, 0.5)

    def test_monotonically_increasing(self):
        x = np.linspace(0, 5, 100)
        y = hill(x, K=0.5, S=2.0)
        assert np.all(np.diff(y) >= 0)

    def test_bounded_0_1(self):
        x = np.linspace(0, 100, 1000)
        y = hill(x, K=0.5, S=2.0)
        assert np.all(y >= 0)
        assert np.all(y <= 1)

    def test_approaches_1(self):
        y = hill(np.array([1000.0]), K=0.5, S=2.0)
        assert y[0] > 0.999

    def test_inverse_roundtrip(self):
        x = np.array([0.1, 0.3, 0.5, 0.8, 1.5])
        y = hill(x, K=0.5, S=2.0)
        x_recovered = hill_inverse(y, K=0.5, S=2.0)
        np.testing.assert_allclose(x_recovered, x, rtol=1e-6)


class TestLogistic:
    def test_midpoint(self):
        """f(x0) should equal 0.5."""
        result = logistic(np.array([0.5]), k=3.0, x0=0.5)
        np.testing.assert_allclose(result, 0.5)

    def test_monotonically_increasing(self):
        x = np.linspace(-5, 5, 100)
        y = logistic(x, k=3.0, x0=0.0)
        assert np.all(np.diff(y) >= 0)

    def test_bounded_0_1(self):
        x = np.linspace(-10, 10, 1000)
        y = logistic(x)
        assert np.all(y >= 0)
        assert np.all(y <= 1)

    def test_inverse_roundtrip(self):
        x = np.array([0.1, 0.3, 0.5, 0.8, 1.5])
        y = logistic(x, k=3.0, x0=0.5)
        x_recovered = logistic_inverse(y, k=3.0, x0=0.5)
        np.testing.assert_allclose(x_recovered, x, rtol=1e-5)


class TestPower:
    def test_zero_input(self):
        assert power(np.array([0.0]), alpha=0.5)[0] == 0.0

    def test_identity_at_alpha_1(self):
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(power(x, alpha=1.0), x)

    def test_diminishing_returns(self):
        """With alpha < 1, second derivative should be negative."""
        x = np.linspace(0.1, 10, 100)
        y = power(x, alpha=0.5)
        dy = np.diff(y)
        ddy = np.diff(dy)
        assert np.all(ddy < 0)  # concave

    def test_inverse_roundtrip(self):
        x = np.array([0.5, 1.0, 2.0, 5.0])
        y = power(x, alpha=0.5)
        x_recovered = power_inverse(y, alpha=0.5)
        np.testing.assert_allclose(x_recovered, x, rtol=1e-6)


class TestPiecewiseLinear:
    def test_zero_input(self):
        result = piecewise_linear(np.array([0.0]))
        assert result[0] == 0.0

    def test_monotonically_increasing(self):
        x = np.linspace(0, 2, 100)
        y = piecewise_linear(x)
        assert np.all(np.diff(y) >= 0)

    def test_custom_breakpoints(self):
        x = np.array([0.0, 0.5, 1.0])
        y = piecewise_linear(x, breakpoints=[0.5], slopes=[2.0, 0.5])
        assert y[0] == 0.0
        np.testing.assert_allclose(y[1], 1.0)  # 0.5 * 2.0
        np.testing.assert_allclose(y[2], 1.25)  # 1.0 + 0.5 * 0.5


class TestRegistry:
    def test_get_hill(self):
        fn = get_saturation_fn("hill")
        assert fn is hill

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_saturation_fn("nonexistent")
