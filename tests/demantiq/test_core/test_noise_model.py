import numpy as np
from demantiq.config.noise_config import NoiseConfig
from demantiq.core.noise_model import generate_noise
from demantiq.utils.random import create_rng


def test_gaussian_shape():
    config = NoiseConfig(noise_type="gaussian", noise_scale=10.0)
    demand = np.ones(100) * 1000
    noise = generate_noise(config, demand, create_rng(42))
    assert noise.shape == (100,)


def test_gaussian_scale():
    config = NoiseConfig(noise_type="gaussian", noise_scale=10.0)
    demand = np.ones(10000) * 1000
    noise = generate_noise(config, demand, create_rng(42))
    assert abs(np.std(noise) - 10.0) < 2.0  # within 2 of target


def test_t_distributed():
    config = NoiseConfig(noise_type="t_distributed", noise_scale=10.0, t_df=5.0)
    demand = np.ones(100) * 1000
    noise = generate_noise(config, demand, create_rng(42))
    assert noise.shape == (100,)


def test_heteroscedastic():
    config = NoiseConfig(noise_type="heteroscedastic", noise_scale=10.0)
    demand = np.concatenate([np.ones(50) * 100, np.ones(50) * 10000])
    noise = generate_noise(config, demand, create_rng(42))
    # Noise variance should be higher where demand is higher
    low_var = np.var(noise[:50])
    high_var = np.var(noise[50:])
    assert high_var > low_var


def test_autocorrelated():
    config = NoiseConfig(noise_type="autocorrelated", noise_scale=10.0, autocorrelation=0.8)
    demand = np.ones(1000) * 1000
    noise = generate_noise(config, demand, create_rng(42))
    # Check autocorrelation is positive
    autocorr = np.corrcoef(noise[:-1], noise[1:])[0, 1]
    assert autocorr > 0.5


def test_snr_mode():
    config = NoiseConfig(noise_type="gaussian", signal_to_noise_ratio=5.0)
    demand = np.random.default_rng(42).normal(1000, 100, size=10000)
    noise = generate_noise(config, demand, create_rng(42))
    actual_snr = np.std(demand) / np.std(noise)
    assert abs(actual_snr - 5.0) < 1.0  # within 1 of target


def test_outliers():
    config = NoiseConfig(noise_type="gaussian", noise_scale=10.0,
                         outlier_probability=0.1, outlier_magnitude=5.0)
    demand = np.ones(1000) * 1000
    noise = generate_noise(config, demand, create_rng(42))
    # Some values should be much larger than normal
    assert np.max(np.abs(noise)) > 30  # 3x normal scale


def test_deterministic():
    config = NoiseConfig(noise_type="gaussian", noise_scale=10.0)
    demand = np.ones(100) * 1000
    n1 = generate_noise(config, demand, create_rng(42))
    n2 = generate_noise(config, demand, create_rng(42))
    np.testing.assert_allclose(n1, n2)
