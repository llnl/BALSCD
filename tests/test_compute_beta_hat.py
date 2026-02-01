"""
Unit tests for least squares parameter estimation.

Verifies that computed regression coefficients (C0, S) match expected values
for argon, copper, and nickel datasets from Marsh (1980).
"""

import pytest

from balscd import regression, utils


@pytest.mark.parametrize(
    "dataset,expected_C0,expected_S",
    [
        ("argon_marsh", 1.293, 1.621),
        ("copper_marsh", 3.913, 1.508),
        ("nickel_marsh", 4.578, 1.451),
    ],
)
def test_beta_hat(dataset, expected_C0, expected_S):
    """Test that beta_hat matches expected values for each dataset."""
    df = utils.load_data(dataset)

    # Create RegressionBase instance
    posterior = regression.PosteriorDistribution(df=df, dataset=dataset)

    assert posterior.C0_hat == pytest.approx(expected_C0, rel=1e-3, abs=1e-3)
    assert posterior.S_hat == pytest.approx(expected_S, rel=1e-3, abs=1e-3)
