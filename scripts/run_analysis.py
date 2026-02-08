#!/usr/bin/env python3

"""
Run Bayesian and bootstrap analysis on new shock wave-particle velocity datasets.

This script provides a command-line interface for analyzing new datasets with
optional informative prior distributions. Input data must be CSV files with
the same column structure as Marsh (1980).

Usage
-----
./scripts/run_analysis.py -d pyrolusite_marsh serpentine_marsh toluene_marsh

# With informative prior parameters
./scripts/run_analysis.py \
    -d pyrolusite_marsh serpentine_marsh toluene_marsh \
    --beta0 2.1 1.5 \
    --std_devs0 0.1 0.05 \
    --rho0 0.3 \
    --a0 2.0 \
    --b0 0.5
"""

import argparse
from pathlib import Path

from balscd import regression, utils


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for dataset names and prior parameters.

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing dataset names and optional prior parameters
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate figures and tables for Bayesian analysis of shock wave datasets.",
    )

    parser.add_argument(
        "-d",
        "--datasets",
        type=str,
        nargs="+",
        help="Dataset name(s) without .csv extension (e.g., argon_marsh copper_marsh)",
    )

    # Informative prior parameters
    parser.add_argument(
        "--beta0",
        type=float,
        nargs=2,
        default=None,
        help="Prior mean for beta parameters [beta0_0, beta0_1].",
    )

    parser.add_argument(
        "--std_devs0",
        type=float,
        nargs=2,
        default=None,
        help="Prior standard deviations for beta parameters [std0_0, std0_1].",
    )

    parser.add_argument(
        "--rho0",
        type=float,
        default=None,
        help="Prior correlation coefficient for beta parameters (between -1 and 1).",
    )

    parser.add_argument(
        "--a0",
        type=float,
        default=None,
        help="Prior shape parameter (a0) for inverse gamma distribution of sigma^2.",
    )

    parser.add_argument(
        "--b0",
        type=float,
        default=None,
        help="Prior scale parameter (b0) for inverse gamma distribution of sigma^2.",
    )

    args = parser.parse_args()

    return args


def plot_all(
    dataset: str,
    prior_params: dict[str, float | tuple[float, float]] | None,
) -> None:
    """
    Generate plots and statistics for a single dataset.

    Runs analysis pipeline with optional informative prior and saves all
    figures and summary statistics.

    Parameters
    ----------
    dataset : str
        Dataset name without .csv extension
    prior_params : dict or None
        Dictionary with keys: 'beta0', 'std_devs0', 'rho0', 'a0', 'b0'.
        If None, uses non-informative prior only.
    """
    df = utils.load_data(dataset)

    # Where to save images
    (Path("images") / dataset).mkdir(parents=True, exist_ok=True)

    # Instantiate posterior and bootstrap distribution objects
    posterior = regression.PosteriorDistribution(df, dataset)
    bootstrap = regression.BootstrapDistribution(df, dataset)

    # Raw Us, Up data with least squares fit
    posterior.plot_least_squares_fit()

    # Surface plot of marginal posterior distribution of beta
    posterior.plot_posterior_beta_noninformative()

    # Credible interval for Hugoniot in pressure-volume plane
    posterior.plot_Hugoniot_uncertainty_interval_PV_plane(
        n_sample=100_000, show_legend=True
    )

    # Bootstrap percentile confidence interval for Hugoniot in pressure-volume plane
    bootstrap.plot_Hugoniot_uncertainty_interval_PV_plane(
        n_sample=100_000, show_legend=True
    )

    # Prediction intervals for future Us measurements
    posterior.plot_credible_and_prediction_interval_for_Us()

    # Posterior predictive check of Bayesian calibration
    posterior.plot_posterior_predictive_check()

    # Marginal posterior distribution of sigma^2
    posterior.plot_posterior_variance()

    # Residuals from least squares fit
    posterior.plot_residuals()

    # Marginal posterior distribution of beta with informative prior
    if prior_params is not None:
        posterior.plot_posterior_beta_informative(
            prior_params["beta0"],
            prior_params["std_devs0"],
            prior_params["rho0"],
            prior_params["a0"],
            prior_params["b0"],
            show_legend=True,
        )

    # Create bootstrap confidence interval for mean shock wave velocity and
    # prediction interval for new measurement
    bootstrap.plot_confidence_and_prediction_interval_for_Us()

    print(f"Plots saved to {Path('images') / dataset} directory")

    # Posterior distribution summary statistics
    posterior.save_summary_statistics()

    # Bootstrap distribution summary statistics
    bootstrap.save_summary_statistics()


def main() -> None:
    """
    Parse arguments and run analysis pipeline for all specified datasets.

    Processes each dataset with optional informative prior, generating figures
    and summary statistics tables.
    """
    args = parse_arguments()

    prior_params = None
    if all(
        param is not None
        for param in [args.beta0, args.std_devs0, args.rho0, args.a0, args.b0]
    ):
        prior_params = {
            "beta0": args.beta0,
            "std_devs0": args.std_devs0,
            "rho0": args.rho0,
            "a0": args.a0,
            "b0": args.b0,
        }

    for dataset in args.datasets:
        plot_all(dataset, prior_params)


if __name__ == "__main__":
    main()
