"""
Unit tests for posterior distribution credible intervals.

Verifies that credible intervals computed from the t-distribution have correct
coverage probabilities for various significance levels and parameters.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_t

from balscd import regression


@pytest.fixture
def posterior():
    """Create a PosteriorDistribution instance for testing."""
    df = pd.read_csv(Path("data") / "argon_marsh.csv")
    return regression.PosteriorDistribution(df, "argon")


@pytest.mark.parametrize(
    "param_index,param_name",
    [(0, "C0"), (1, "S")],
)
@pytest.mark.parametrize(
    "alpha",
    [0.01, 0.05, 0.10],
)
def test_credible_interval_coverage(posterior, param_index, param_name, alpha):
    """Test credible intervals have correct coverage for various alpha levels."""
    tol = 0.0005

    lower, upper = posterior.credible_interval(param_index, alpha)

    rv = multivariate_t(
        loc=posterior.beta_hat[param_index],
        shape=posterior.Sigma[param_index, param_index],
        df=posterior.nu,
    )

    assert np.abs(rv.cdf(lower) - alpha / 2) < tol, (
        f"Lower bound failed for {param_name}"
    )
    assert np.abs(rv.cdf(upper) - (1 - alpha / 2)) < tol, (
        f"Upper bound failed for {param_name}"
    )
