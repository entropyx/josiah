"""Correlation matrix generation using Gaussian copula."""

import numpy as np
from numpy.random import Generator


def generate_correlation_matrix(n: int, within_group_corr: float = 0.6,
                                between_group_corr: float = 0.1,
                                groups: list[list[int]] | None = None) -> np.ndarray:
    """Generate a valid correlation matrix with group structure.

    Args:
        n: Number of variables.
        within_group_corr: Correlation within same group.
        between_group_corr: Correlation between different groups.
        groups: List of lists of variable indices in each group.
                If None, all variables in one group.

    Returns:
        Positive semi-definite correlation matrix of shape (n, n).
    """
    if groups is None:
        groups = [list(range(n))]

    corr = np.full((n, n), between_group_corr)
    np.fill_diagonal(corr, 1.0)

    for group in groups:
        for i in group:
            for j in group:
                if i != j:
                    corr[i, j] = within_group_corr

    # Ensure positive semi-definite via nearest PSD
    return _nearest_psd(corr)


def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
    """Find nearest positive semi-definite matrix."""
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, 1e-10)
    result = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Normalize to correlation matrix
    d = np.sqrt(np.diag(result))
    result = result / np.outer(d, d)
    np.fill_diagonal(result, 1.0)
    return result


def gaussian_copula_sample(rng: Generator, n_samples: int,
                           corr_matrix: np.ndarray) -> np.ndarray:
    """Sample from Gaussian copula with given correlation structure.

    Returns uniform marginals with the specified correlation.

    Args:
        rng: numpy Generator.
        n_samples: Number of samples.
        corr_matrix: Correlation matrix.

    Returns:
        Array of shape (n_samples, n_vars) with values in [0, 1].
    """
    from scipy.stats import norm

    n_vars = corr_matrix.shape[0]
    # Sample from multivariate normal
    mean = np.zeros(n_vars)
    samples = rng.multivariate_normal(mean, corr_matrix, size=n_samples)
    # Transform to uniform via CDF
    return norm.cdf(samples)
