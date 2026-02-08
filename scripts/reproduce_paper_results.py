#!/usr/bin/env python3

"""
Generate all figures and tables for the paper.

This script runs the complete analysis pipeline on argon, copper, and nickel
datasets, producing least squares fits, posterior distributions, bootstrap
comparisons, Hugoniot curves, prediction intervals, and summary statistics.

Usage
-----
./scripts/reproduce_paper_results.py
"""

from balscd import convergence, regression, utils

least_squares_plot_limits = {
    "xlim": (-0.25, 5.5),
    "ylim": (-0.25, 11),
}

residuals_plot_limits = {
    "xlim": (-1, 6.5),
    "ylim": (-0.5, 0.5),
}

informative_prior_configs = {
    "argon_marsh": {
        "beta0": (1.32, 1.5),
        "std_devs": (0.1, 0.2),
        "rho": 0,
        "a0": 5,
        "b0": 5,
    },
    "copper_marsh": {
        "beta0": (3.8, 1.62),
        "std_devs": (0.075, 0.05),
        "rho": -0.2,
        "a0": 5,
        "b0": 5,
    },
    "nickel_marsh": {
        "beta0": (4.7, 1.55),
        "std_devs": (0.08, 0.08),
        "rho": -0.8,
        "a0": 5,
        "b0": 5,
    },
}


def plot_all(dataset: str) -> None:
    """
    Generate all figures and tables for a single dataset.

    Runs complete analysis pipeline: loads data, creates posterior and bootstrap
    distributions, generates all plots, and saves summary statistics.

    Parameters
    ----------
    dataset : str
        Dataset name (e.g., 'argon_marsh', 'copper_marsh', 'nickel_marsh')
    """
    df = utils.load_data(dataset)

    # Determine if legends should be shown (only for nickel_marsh)
    show_legend = dataset == "nickel_marsh"

    # Instantiate posterior and bootstrap distribution objects
    posterior = regression.PosteriorDistribution(df, dataset)
    bootstrap = regression.BootstrapDistribution(df, dataset)

    # Instantiate object for creating comparison plots
    comparison = regression.RegressionComparison(posterior, bootstrap)

    # Fig. 2: Raw Us, Up data with least squares fit
    posterior.plot_least_squares_fit(**least_squares_plot_limits)

    # Fig. 3: Posterior distribution of C0 and histogram of samples
    posterior.plot_posterior_with_samples(show_legend=show_legend)

    # Fig. 4: Surface plot of marginal posterior distribution of beta
    posterior.plot_posterior_beta_noninformative(show_legend=show_legend)

    # Fig. 5: Credible interval for Hugoniot in pressure-volume plane
    posterior.plot_Hugoniot_uncertainty_interval_PV_plane(show_legend=show_legend)

    # Fig. 6: Credible interval for mean shock wave velocity and prediction
    # interval for future Us measurements
    posterior.plot_credible_and_prediction_interval_for_Us(show_legend=show_legend)

    # Fig. 7: Posterior predictive check of Bayesian calibration
    posterior.plot_posterior_predictive_check(show_legend=show_legend)

    # Fig. 8: Posterior and bootstrap distribution of S
    comparison.plot_bootstrap_and_posterior_distributions(show_legend=show_legend)

    # Fig 9a and 9b: Bootstrap and posterior distribution of S w/ and w/ out
    # point with max Up removed
    if dataset == "copper_marsh":
        limits = {"xlim": (1.47, 1.55), "ylim": (0, 65)}
        comparison.plot_bootstrap_distribution_with_and_without_max_Up_point(
            **limits, show_legend=True
        )
        comparison.plot_posterior_distribution_with_and_without_max_Up_point(**limits)

    # Fig. 11: Marginal posterior distribution of sigma^2
    posterior.plot_posterior_variance()

    # Fig. 12: Residuals from least squares fit
    posterior.plot_residuals(**residuals_plot_limits)

    # Fig. 13: Marginal posterior distribution of beta with informative prior
    config = informative_prior_configs[dataset]
    posterior.plot_posterior_beta_informative(**config, show_legend=show_legend)

    # Table 1: Posterior distribution summary statistics
    posterior.save_summary_statistics()

    # Table 2: Bootstrap distribution summary statistics
    bootstrap.save_summary_statistics()

    # Fig. 14: Bootstrap confidence and prediction intervals
    bootstrap.plot_confidence_and_prediction_interval_for_Us(show_legend=show_legend)


def main() -> None:
    """
    Perform complete analysis for all datasets.

    Processes argon, copper, and nickel datasets from Marsh (1980) and generates
    all figures and tables for the paper. Outputs saved to images/ and
    summary_statistics/ directories.
    """
    datasets = ["argon_marsh", "copper_marsh", "nickel_marsh"]

    for dataset in datasets:
        plot_all(dataset)

    # Fig. 10: Convergence of t-distribution to normal distribution
    convergence.plot_t_distribution_convergence()


if __name__ == "__main__":
    main()
