import numpy as np
from demantiq.utils.random import create_rng
from demantiq.utils.correlation import generate_correlation_matrix, gaussian_copula_sample


def test_correlation_matrix_shape():
    corr = generate_correlation_matrix(5)
    assert corr.shape == (5, 5)


def test_correlation_matrix_diagonal():
    corr = generate_correlation_matrix(5)
    np.testing.assert_allclose(np.diag(corr), 1.0)


def test_correlation_matrix_symmetric():
    corr = generate_correlation_matrix(5)
    np.testing.assert_allclose(corr, corr.T, atol=1e-10)


def test_correlation_matrix_psd():
    corr = generate_correlation_matrix(5)
    eigvals = np.linalg.eigvalsh(corr)
    assert np.all(eigvals >= -1e-10)


def test_correlation_matrix_groups():
    corr = generate_correlation_matrix(4, within_group_corr=0.8, between_group_corr=0.1,
                                        groups=[[0, 1], [2, 3]])
    assert abs(corr[0, 1] - 0.8) < 0.05
    assert abs(corr[0, 2] - 0.1) < 0.05


def test_copula_sample_shape():
    rng = create_rng(42)
    corr = generate_correlation_matrix(3)
    samples = gaussian_copula_sample(rng, 100, corr)
    assert samples.shape == (100, 3)


def test_copula_sample_range():
    rng = create_rng(42)
    corr = generate_correlation_matrix(3)
    samples = gaussian_copula_sample(rng, 1000, corr)
    assert np.all(samples >= 0)
    assert np.all(samples <= 1)
