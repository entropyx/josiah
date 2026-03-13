import numpy as np
import pytest
from demantiq.transforms.adstock import (
    geometric, weibull_cdf, weibull_pdf, delayed_geometric, get_adstock_fn,
)


class TestGeometric:
    def test_no_decay(self):
        """With alpha=0, output equals input (normalized)."""
        x = np.array([0, 0, 100, 0, 0], dtype=float)
        result = geometric(x, alpha=0.0, max_lag=4, normalize=True)
        np.testing.assert_allclose(result, x)

    def test_output_shape(self):
        x = np.random.rand(100)
        result = geometric(x, alpha=0.5, max_lag=8)
        assert result.shape == x.shape

    def test_impulse_response(self):
        """Single impulse should produce decaying response."""
        x = np.zeros(20)
        x[0] = 1.0
        result = geometric(x, alpha=0.7, max_lag=10, normalize=False)
        # Result should decay
        assert result[0] > result[1] > result[2]

    def test_deterministic(self):
        x = np.array([100, 200, 150, 50, 300], dtype=float)
        r1 = geometric(x, alpha=0.5, max_lag=4)
        r2 = geometric(x, alpha=0.5, max_lag=4)
        np.testing.assert_allclose(r1, r2)


class TestWeibullCDF:
    def test_output_shape(self):
        x = np.random.rand(100)
        result = weibull_cdf(x, shape=2.0, scale=3.0, max_lag=12)
        assert result.shape == x.shape

    def test_weights_positive(self):
        """All weights should be non-negative."""
        t = np.arange(12, dtype=float)
        cdf_vals = 1 - np.exp(-((t + 1) / 3.0) ** 2.0)
        cdf_prev = np.concatenate([[0], cdf_vals[:-1]])
        weights = cdf_vals - cdf_prev
        assert np.all(weights >= 0)


class TestWeibullPDF:
    def test_output_shape(self):
        x = np.random.rand(100)
        result = weibull_pdf(x, shape=2.0, scale=3.0, max_lag=12)
        assert result.shape == x.shape


class TestDelayedGeometric:
    def test_zero_during_delay(self):
        """Should have no effect during delay period."""
        x = np.zeros(20)
        x[0] = 1.0
        result = delayed_geometric(x, alpha=0.5, delay=3, max_lag=10, normalize=False)
        # First 3 periods should be zero (delay=3 means weights[0:3]=0)
        # The impulse at t=0 gets convolved with weights starting at index 3
        # So result[0], result[1], result[2] should be 0
        assert result[0] == 0.0
        assert result[1] == 0.0
        assert result[2] == 0.0
        # After delay, should have response
        assert result[3] > 0

    def test_output_shape(self):
        x = np.random.rand(50)
        result = delayed_geometric(x, alpha=0.5, delay=2, max_lag=8)
        assert result.shape == x.shape


class TestRegistry:
    def test_get_geometric(self):
        fn = get_adstock_fn("geometric")
        assert fn is geometric

    def test_all_registered(self):
        for name in ["geometric", "weibull_cdf", "weibull_pdf", "delayed_geometric"]:
            fn = get_adstock_fn(name)
            assert callable(fn)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_adstock_fn("nonexistent")
