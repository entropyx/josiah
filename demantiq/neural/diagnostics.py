"""Diagnostics for the Demantiq neural inference engine.

Simulation-Based Calibration (SBC), coverage probability analysis,
and posterior predictive checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SBCResult:
    """Result of Simulation-Based Calibration check.

    Attributes:
        rank_histograms: Dict mapping parameter name to rank histogram counts.
        n_bins: Number of bins used.
        n_simulations: Number of SBC simulations run.
        uniformity_pvalues: Dict mapping parameter name to KS test p-value
            against uniform distribution.
    """
    rank_histograms: dict[str, np.ndarray]
    n_bins: int
    n_simulations: int
    uniformity_pvalues: dict[str, float]


@dataclass
class CoverageResult:
    """Result of coverage probability analysis.

    Attributes:
        nominal_levels: Array of nominal coverage levels tested (e.g. [0.5, 0.8, 0.9, 0.95]).
        empirical_coverages: Dict mapping parameter name to array of empirical coverages.
        n_samples: Number of test samples used.
    """
    nominal_levels: np.ndarray
    empirical_coverages: dict[str, np.ndarray]
    n_samples: int


def run_sbc(
    posterior,
    prior_samples: np.ndarray,
    observations: np.ndarray,
    n_posterior_samples: int = 1000,
    n_bins: int = 20,
    param_names: list[str] | None = None,
) -> SBCResult:
    """Run Simulation-Based Calibration (SBC).

    For each (theta_true, x_obs) pair from the joint prior:
    1. Draw posterior samples: theta_post ~ q(theta | x_obs)
    2. Compute the rank of theta_true among theta_post for each parameter
    3. Rank histograms should be approximately uniform if the posterior is calibrated

    Args:
        posterior: Trained sbi posterior object.
        prior_samples: (N, D) true parameter values drawn from the prior.
        observations: (N, obs_dim) corresponding observations.
        n_posterior_samples: Number of posterior samples per SBC iteration.
        n_bins: Number of histogram bins.
        param_names: Optional names for each parameter dimension.

    Returns:
        SBCResult with rank histograms and uniformity tests.
    """
    import torch
    from scipy.stats import kstest

    n_sims, n_params = prior_samples.shape
    ranks = np.zeros((n_sims, n_params), dtype=np.int64)

    for i in range(n_sims):
        x_obs = torch.tensor(observations[i], dtype=torch.float32)
        theta_true = prior_samples[i]

        # Draw posterior samples
        samples = posterior.sample((n_posterior_samples,), x=x_obs).cpu().numpy()

        # Compute rank of true value
        for d in range(n_params):
            ranks[i, d] = np.sum(samples[:, d] < theta_true[d])

    # Build histograms
    if param_names is None:
        param_names = [f"param_{d}" for d in range(n_params)]

    rank_histograms = {}
    uniformity_pvalues = {}
    for d, name in enumerate(param_names):
        hist, _ = np.histogram(ranks[:, d], bins=n_bins, range=(0, n_posterior_samples))
        rank_histograms[name] = hist

        # KS test against uniform
        normalized_ranks = ranks[:, d] / n_posterior_samples
        _, pval = kstest(normalized_ranks, "uniform")
        uniformity_pvalues[name] = float(pval)

    return SBCResult(
        rank_histograms=rank_histograms,
        n_bins=n_bins,
        n_simulations=n_sims,
        uniformity_pvalues=uniformity_pvalues,
    )


def compute_coverage(
    posterior,
    theta_true: np.ndarray,
    observations: np.ndarray,
    nominal_levels: np.ndarray | None = None,
    n_posterior_samples: int = 2000,
    param_names: list[str] | None = None,
) -> CoverageResult:
    """Compute empirical coverage probabilities.

    For each nominal level alpha, check what fraction of true parameters
    fall within the alpha-level highest posterior density region.

    Args:
        posterior: Trained sbi posterior object.
        theta_true: (N, D) true parameter values.
        observations: (N, obs_dim) corresponding observations.
        nominal_levels: Coverage levels to test (default: [0.5, 0.8, 0.9, 0.95]).
        n_posterior_samples: Number of posterior samples per observation.
        param_names: Optional names for each parameter dimension.

    Returns:
        CoverageResult with nominal vs empirical coverage.
    """
    import torch

    if nominal_levels is None:
        nominal_levels = np.array([0.5, 0.8, 0.9, 0.95])

    n_obs, n_params = theta_true.shape

    if param_names is None:
        param_names = [f"param_{d}" for d in range(n_params)]

    # For each observation, check if true value is within credible interval
    covered = {name: np.zeros((n_obs, len(nominal_levels)), dtype=bool)
               for name in param_names}

    for i in range(n_obs):
        x_obs = torch.tensor(observations[i], dtype=torch.float32)
        samples = posterior.sample((n_posterior_samples,), x=x_obs).cpu().numpy()

        for d, name in enumerate(param_names):
            for j, level in enumerate(nominal_levels):
                low = np.percentile(samples[:, d], (1 - level) / 2 * 100)
                high = np.percentile(samples[:, d], (1 + level) / 2 * 100)
                covered[name][i, j] = low <= theta_true[i, d] <= high

    empirical_coverages = {
        name: covered[name].mean(axis=0)
        for name in param_names
    }

    return CoverageResult(
        nominal_levels=nominal_levels,
        empirical_coverages=empirical_coverages,
        n_samples=n_obs,
    )


def posterior_predictive_check(
    posterior,
    simulator_fn,
    observation: np.ndarray,
    n_samples: int = 200,
) -> dict[str, np.ndarray]:
    """Run posterior predictive check.

    Sample parameters from the posterior, simulate data for each,
    and return the distribution of simulated data for comparison
    with the observed data.

    Args:
        posterior: Trained sbi posterior object.
        simulator_fn: Function that takes parameter vector and returns simulated y.
        observation: (obs_dim,) packed observation vector.
        n_samples: Number of posterior samples to simulate from.

    Returns:
        Dict with 'posterior_samples' (parameter samples) and
        'simulated_y' (simulated outcomes for each sample).
    """
    import torch

    x_obs = torch.tensor(observation, dtype=torch.float32)
    theta_samples = posterior.sample((n_samples,), x=x_obs).cpu().numpy()

    simulated_ys = []
    for theta in theta_samples:
        try:
            sim_y = simulator_fn(theta)
            simulated_ys.append(sim_y)
        except Exception:
            continue

    return {
        "posterior_samples": theta_samples,
        "simulated_y": np.array(simulated_ys) if simulated_ys else np.array([]),
    }
