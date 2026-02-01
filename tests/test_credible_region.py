"""
Monte Carlo verification of bivariate credible region coverage.

Empirically verifies that credible regions computed from the F-distribution
achieve nominal coverage levels for bivariate t-distributions with various
degrees of freedom.
"""

import numpy as np
import pytest
from scipy.stats import f

from balscd import utils


@pytest.mark.parametrize("nu", [5, 10, 20, 30, 40])
@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10, 0.15])
def test_credible_region_coverage(alpha, nu):
    Sigma = np.array([[1.0, 0.75], [0.75, 3.0]])
    beta_hat = np.array([0.0, 0.0])
    n_sample = 5_000_000
    p = beta_hat.shape[0]

    # Compute F-distribution threshold
    f_crit = f.ppf(1 - alpha, dfn=p, dfd=nu)
    threshold = p * f_crit

    # Draw samples
    samples = utils.sample_multivariate_t(beta_hat, Sigma, nu, n_sample, seed=1)

    # Compute quadratic form for each sample
    diff = samples - beta_hat
    inv_Sigma = np.linalg.inv(Sigma)
    quad_form = np.sum((diff @ inv_Sigma) * diff, axis=1)

    # Check how many fall inside the credible region
    inside = quad_form < threshold
    empirical_coverage = np.mean(inside)

    nominal_coverage = 1 - alpha

    assert np.abs(nominal_coverage - empirical_coverage) < 1e-3
