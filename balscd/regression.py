"""
Linear regression analysis for shock wave-particle velocity Hugoniot data.

This module provides Bayesian and bootstrap approaches to analyze the relationship
between particle velocity (Up) and shock wave velocity (Us) in shock compression
experiments. It includes tools for computing posterior distributions, bootstrap
resampling, and visualizing Hugoniot curves in the pressure-volume plane.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy.stats import f, invgamma, multivariate_normal, multivariate_t, t

from balscd import utils
from balscd.plot_config import font_sizes


class RegressionBase(ABC):
    """Base class with shared regression computations."""

    def __init__(self, df: pd.DataFrame, dataset: str):
        """
        Initialize regression with data and compute least squares estimates.

        Parameters
        ----------
        df : pd.DataFrame
            Experimental data with columns: Up_km_s, Us_km_s, rho0_g_cc, P_GPa, V_cc_g
        dataset : str
            Dataset identifier used for organizing output files and plots
        """
        self.df = df
        self.dataset = dataset
        self.n: int = len(df)
        self.x: np.ndarray = df["Up_km_s"].to_numpy().reshape(-1, 1)
        self.y: np.ndarray = df["Us_km_s"].to_numpy().reshape(-1, 1)
        self.X = np.hstack([np.ones((self.n, 1)), self.x])
        self.p: int = self.X.shape[1]  # number of regression coefficients
        self.XTX = self.X.T @ self.X
        self.nu: int = self.n - self.p
        self.beta_hat: np.ndarray = self.compute_beta_hat(self.x, self.y)
        self.C0_hat = self.beta_hat[0]
        self.S_hat = self.beta_hat[1]
        self.y_pred = self.C0_hat + self.S_hat * self.x
        self.residuals = self.y - self.y_pred
        sum_sq_res = np.sum(self.residuals**2)
        self.s_squared = sum_sq_res / self.nu

        # Compute r-squared
        sum_sq_tot = np.sum((self.y - self.y.mean()) ** 2)
        self.r_squared = 1 - sum_sq_res / sum_sq_tot

    def __repr__(self) -> str:
        """Return string representation of the regression object."""
        return (
            f"{self.__class__.__name__}"
            f"(df=<DataFrame: {self.n} rows>, "
            f"dataset={self.dataset!r})"
        )

    @staticmethod
    def compute_beta_hat(
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Compute least squares estimate of beta parameters.

        Parameters
        ----------
        x : np.ndarray
            Predictor variable (particle velocity), shape (n, 1)
        y : np.ndarray
            Response variable (shock wave velocity), shape (n, 1)

        Returns
        -------
        beta_hat : np.ndarray
            Regression coefficients [C0, S]
        """
        n = len(x)
        X = np.hstack([np.ones((n, 1)), x])
        XTX = X.T @ X
        beta_hat = np.linalg.solve(XTX, X.T @ y).flatten()
        return beta_hat

    @abstractmethod
    def sample_beta(self, n_sample: int, seed: int = 1) -> np.ndarray:
        """
        Sample beta parameters from the distribution.

        This is an abstract method that must be implemented by subclasses.
        PosteriorDistribution samples from a multivariate t-distribution,
        while BootstrapDistribution uses bootstrap resampling.

        Parameters
        ----------
        n_sample : int
            Number of samples to generate
        seed : int
            Random seed for reproducibility

        Returns
        -------
        betas : np.ndarray
            Samples of beta, shape (n_sample, 2)
        """
        pass

    def plot_least_squares_fit(
        self,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
    ) -> None:
        """
        Plot data with least squares regression line and summary statistics.

        Parameters
        ----------
        xlim : tuple[float, float], optional
            x-axis limits (Up range). If None, computed from data.
        ylim : tuple[float, float], optional
            y-axis limits (Us range). If None, computed from data.
        """
        ax = utils.setup_figure()
        ax.scatter(
            self.x,
            self.y,
            color="blue",
            facecolors="none",
            edgecolors="blue",
            s=50,
            alpha=0.95,
        )
        ax.set_xlabel("Particle Velocity [km/s]", fontsize=font_sizes["label"])
        ax.set_ylabel("Shock Wave Velocity [km/s]", fontsize=font_sizes["label"])

        # Calculate sensible defaults from data if not provided
        if xlim is None:
            x_range = self.x.max() - self.x.min()
            xlim = (self.x.min() - 0.1 * x_range, self.x.max() + 0.1 * x_range)
        if ylim is None:
            y_range = self.y.max() - self.y.min()
            ylim = (self.y.min() - 0.1 * y_range, self.y.max() + 0.1 * y_range)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        # Plot the regression line and add text
        ax.plot(self.x, self.y_pred, color="red")

        # Add summary stats to lower right corner
        ax.text(
            0.95,
            0.05,
            f"$n={self.n}$\n"
            f"$\\hat{{C}}_0={self.C0_hat:.3f}$\n"
            f"$\\hat{{S}}={self.S_hat:.3f}$\n"
            f"$R^2={self.r_squared:.3f}$",
            fontsize=font_sizes["text"],
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
        )
        utils.save_and_close_figure("ls_fit", self.dataset)

    def plot_residuals(
        self,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
    ) -> None:
        """
        Plot shock wave velocity residuals from least squares fit.

        Parameters
        ----------
        xlim : tuple[float, float], optional
            x-axis limits. If None, computed from data.
        ylim : tuple[float, float], optional
            y-axis limits. If None, computed from data.
        """
        ax = utils.setup_figure()
        ax.scatter(
            self.x,
            self.residuals,
            color="blue",
            facecolors="none",
            edgecolors="blue",
            s=50,
            alpha=0.95,
        )
        ax.set_xlabel(
            "Particle Velocity [km/s]",
            fontsize=font_sizes["label"],
        )
        ax.set_ylabel(
            "Residual Shock Wave Velocity [km/s]",
            fontsize=font_sizes["label"],
        )
        # Calculate default limits from data if not provided
        if xlim is None:
            x_range = self.x.max() - self.x.min()
            xlim = (
                self.x.min() - 0.1 * x_range,
                self.x.max() + 0.1 * x_range,
            )
        if ylim is None:
            residual_range = self.residuals.max() - self.residuals.min()
            ylim = (
                self.residuals.min() - 0.2 * residual_range,
                self.residuals.max() + 0.2 * residual_range,
            )

        # Add text box showing sample standard deviation
        ax.text(
            x=0.96,
            y=0.04,
            s=f"Std. dev. = {np.sqrt(self.s_squared):.3f}",
            fontsize=font_sizes["label"],
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
        )

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        utils.save_and_close_figure("residuals", self.dataset)

    def compute_Hugoniot_PV_plane(
        self,
        Up_grid: np.ndarray,
        beta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Hugoniot curve in the pressure-volume plane.

        Uses Rankine-Hugoniot equations to project Us-Up into pressure-volume plane.

        Parameters
        ----------
        Up_grid : np.ndarray
            Particle velocity grid, shape (n_grid,)
        beta : np.ndarray
            Model parameters [C0, S]. Shape (2,) for single curve or
            (n_sample, 2) for multiple curves.

        Returns
        -------
        P : np.ndarray
            Pressure values [GPa]. Shape (n_grid,) or (n_sample, n_grid).
        V : np.ndarray
            Specific volume values [cm^3/g]. Shape (n_grid,) or (n_sample, n_grid).
        """
        # Evaluate shock wave velocity on Up_grid
        # Works for both beta shapes (2,) and (n_sample, 2) via broadcasting
        Us_grid = beta[..., 0:1] + beta[..., 1:2] * Up_grid

        # Compute initial density, volume, and pressure
        rho_mean = self.df["rho0_g_cc"].mean()
        V0 = 1 / rho_mean
        P0_GPa = 0.0001  # Initial pressure is 1 bar for ambient pressure

        # Compute Hugoniot for sampled beta
        V = V0 * (Us_grid - Up_grid) / Us_grid
        P = P0_GPa + rho_mean * Us_grid * Up_grid

        return P, V

    def sample_Hugoniot_PV_plane(
        self,
        n_sample: int,
        seed: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample Hugoniot curves in the pressure-volume plane.

        Parameters
        ----------
        n_sample : int
            Number of Hugoniot curves to sample
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        V_samples : np.ndarray
            Specific volume samples, shape (n_sample, n_grid)
        P_samples : np.ndarray
            Pressure samples, shape (n_sample, n_grid)
        Up_grid : np.ndarray
            Particle velocity grid used for evaluation, shape (n_grid,)
        """
        # Up grid on which to evaluate Hugoniot
        Up_grid = np.linspace(
            start=0.001 * self.x.min(),
            stop=1.999 * self.x.max(),
            num=300,
        )

        betas = self.sample_beta(n_sample, seed)

        P_samples, V_samples = self.compute_Hugoniot_PV_plane(Up_grid, betas)

        return V_samples, P_samples, Up_grid

    def plot_Hugoniot_uncertainty_interval_PV_plane(
        self,
        n_sample: int = 100_000,
        n_grid: int = 400,
        show_legend: bool = False,
        alpha: float = 0.05,
    ) -> None:
        """
        Plot Hugoniot curve with credible interval in pressure-volume plane.

        Parameters
        ----------
        n_sample : int, optional
            Number of curves to sample for credible interval
        n_grid : int, optional
            Number of volume grid points for interpolation
        show_legend : bool, optional
            Whether to display legend
        alpha : float, optional
            Significance level for credible interval
        """
        # Sample Hugoniot curves in pressure-volume plane
        V_samples, P_samples, Up_grid = self.sample_Hugoniot_PV_plane(n_sample)

        # Create common volume grid
        V_min = self.df["V_cc_g"].min()
        V_max = self.df["V_cc_g"].max()
        V_grid = np.linspace(V_min, V_max, num=n_grid)

        # Interpolate each curve onto the grid
        interpolated_pressures = np.empty((n_sample, n_grid))
        for idx in range(n_sample):
            V = V_samples[idx]
            P = P_samples[idx]

            # Sort by V since np.interp requires sorted x values
            sort_idx = np.argsort(V)
            V_sorted = V[sort_idx]
            P_sorted = P[sort_idx]

            # np.interp extrapolates to np.nan outside the data range
            interpolated_pressures[idx] = np.interp(V_grid, V_sorted, P_sorted)

        # Compute percentiles
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        lower = np.percentile(interpolated_pressures, lower_percentile, axis=0)
        upper = np.percentile(interpolated_pressures, upper_percentile, axis=0)

        # Determine class type for labeling
        class_suffix = self.__class__.__name__.lower().replace("distribution", "")
        interval_type = "Confidence" if class_suffix == "bootstrap" else "Credible"

        # Plot
        ax = utils.setup_figure()

        # Add credible or confidence interval
        ax.fill_between(
            V_grid,
            lower,
            upper,
            color="gray",
            alpha=0.3,
            label=f"{int((1 - alpha) * 100)}% {interval_type} interval",
        )

        # Add Hugoniot for least squares estimate of beta
        P_ls, V_ls = self.compute_Hugoniot_PV_plane(Up_grid, self.beta_hat)
        # Clip to the volume_grid range
        vmin, vmax = V_grid[0], V_grid[-1]
        mask = (V_ls >= vmin) & (V_ls <= vmax)
        ax.plot(
            V_ls[mask],
            P_ls[mask],
            c="red",
            lw=1,
            label="Least squares estimate",
        )

        # Add points
        ax.scatter(
            self.df["V_cc_g"],
            self.df["P_GPa"],
            marker="o",
            zorder=2,
            facecolor="none",
            edgecolors="blue",
            s=50,
            alpha=0.95,
            label="Measurements",
        )

        ax.set_xlabel("Volume [cm$^3$/g]", fontsize=font_sizes["label"])
        ax.set_ylabel("Pressure [GPa]", fontsize=font_sizes["label"])
        if show_legend:
            ax.legend(fontsize=font_sizes["legend"])
        filename = f"Hugoniot_{interval_type.lower()}_interval_PV_plane"
        utils.save_and_close_figure(filename, self.dataset)

    def _plot_interval_for_Us(
        self,
        x_pred: np.ndarray,
        y_pred_mean: np.ndarray,
        cred_lower: np.ndarray,
        cred_upper: np.ndarray,
        pred_lower: np.ndarray,
        pred_upper: np.ndarray,
        interval_type: str,
        filename: str,
        show_legend: bool = False,
        alpha: float = 0.05,
        mean_label: str = "Pred. mean",
    ) -> None:
        """
        Plot credible/confidence and prediction intervals for shock wave velocity.

        Parameters
        ----------
        x_pred : np.ndarray
            Particle velocity grid for predictions
        y_pred_mean : np.ndarray
            Mean predictions
        cred_lower : np.ndarray
            Lower bound for mean interval
        cred_upper : np.ndarray
            Upper bound for mean interval
        pred_lower : np.ndarray
            Lower bound for new point prediction interval
        pred_upper : np.ndarray
            Upper bound for new point prediction interval
        interval_type : str
            Either "Cred." or "Conf." for labeling
        filename : str
            Base filename for saving plot
        show_legend : bool, optional
            Whether to display legend
        alpha : float, optional
            Significance level
        """
        ax = utils.setup_figure()

        # Add prediction interval for new Us measurement
        ax.fill_between(
            x_pred,
            pred_lower,
            pred_upper,
            color="gray",
            alpha=0.3,
            zorder=1,
            label=f"{100 * (1 - alpha):.0f}% Pred. interval\nfor new point",
        )

        # Add credible/confidence interval for mean Us
        ax.fill_between(
            x_pred,
            cred_lower,
            cred_upper,
            color="yellow",
            alpha=0.6,
            zorder=2,
            label=f"{100 * (1 - alpha):.0f}% {interval_type} interval\nfor mean $U_\\mathrm{{s}}$",
        )

        # Add experimental data
        ax.scatter(
            self.x,
            self.y,
            marker="o",
            facecolor="none",
            edgecolors="blue",
            s=50,
            alpha=0.9,
            zorder=3,
            label="Experimental data",
        )

        # Add line for prediction mean
        ax.plot(
            x_pred,
            y_pred_mean,
            linewidth=1,
            color="red",
            zorder=4,
            label=mean_label,
        )

        if show_legend:
            ax.legend(
                loc="upper left",
                fontsize=font_sizes["legend"],
            )

        ax.set_xlabel("Particle Velocity [km/s]", fontsize=font_sizes["label"])
        ax.set_ylabel("Shock Wave Velocity [km/s]", fontsize=font_sizes["label"])
        utils.save_and_close_figure(filename, self.dataset)

    def _save_parameter_summary_table(
        self,
        params: list[tuple[str, float, float, float, float]],
        interval_type: str,
        subdirectory: str,
    ) -> None:
        """
        Save parameter summary statistics to text file.

        Parameters
        ----------
        params : list[tuple[str, float, float, float, float]]
            List of tuples containing (name, mean, std_dev, lower_bound, upper_bound)
        interval_type : str
            Type of interval ("Credible Interval" or "Confidence Interval")
        subdirectory : str
            Subdirectory name for organizing output files
        """
        # Specify where to save results
        output_dir = Path("summary_statistics") / subdirectory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine the mean column header based on interval type
        mean_header = (
            "Posterior Mean" if "Credible" in interval_type else "Bootstrap Mean"
        )

        # Build output string
        lines = []
        lines.append(
            f"{'Parameter':<10} | {mean_header:>14} | {'Std Dev':>8} | {interval_type:>18}"
        )
        lines.append("-" * 60)
        for name, mean, sd, lower, upper in params:
            lines.append(
                f"{name:<10} | {mean:>14.3f} | {sd:>8.3f} | ({lower:.3f}, {upper:.3f})"
            )

        output_text = "\n".join(lines) + "\n"

        # Write to file
        output_path = output_dir / f"{self.dataset}.txt"
        output_path.write_text(output_text, encoding="utf-8")


class PosteriorDistribution(RegressionBase):
    """Posterior distribution for Bayesian linear regression with non-informative prior."""

    def __init__(
        self,
        df: pd.DataFrame,
        dataset: str,
    ):
        """
        Initialize posterior distribution from data.

        Computes posterior parameters assuming non-informative prior:
        p(beta, sigma^2) propto 1/sigma^2

        Parameters
        ----------
        df : pd.DataFrame
            Experimental data
        dataset : str
            Dataset name for file naming
        """
        super().__init__(df, dataset)
        # Posterior distribution parameters
        self.Sigma: np.ndarray = self.s_squared * np.linalg.inv(self.XTX)
        self.covariance_matrix = self.nu / (self.nu - 2) * self.Sigma
        self.standard_deviations = np.sqrt(np.diag(self.covariance_matrix))

    def evaluate_t_pdf_on_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate bivariate t-distribution PDF on a grid for contour plotting.

        Returns
        -------
        beta1_mesh : np.ndarray
            Grid of C0 values, shape (100, 100)
        beta2_mesh : np.ndarray
            Grid of S values, shape (100, 100)
        pdf : np.ndarray
            PDF values at grid points, shape (100, 100)
        """
        # Set-up grid for evaluating posterior PDF
        beta1_range = np.linspace(
            self.C0_hat - 3.25 * self.standard_deviations[0],
            self.C0_hat + 3.25 * self.standard_deviations[0],
            num=100,
        )
        beta2_range = np.linspace(
            self.S_hat - 3.25 * self.standard_deviations[1],
            self.S_hat + 3.25 * self.standard_deviations[1],
            num=100,
        )

        beta1_mesh, beta2_mesh = np.meshgrid(beta1_range, beta2_range)

        # Combine into grid points for evaluation
        pos = np.dstack((beta1_mesh, beta2_mesh))

        # Compute PDF of bivariate t-distribution
        rv = multivariate_t(loc=self.beta_hat, shape=self.Sigma, df=self.nu)
        pdf = rv.pdf(pos)

        return beta1_mesh, beta2_mesh, pdf

    def sample_beta(
        self,
        n_sample: int,
        seed: int = 1,
    ) -> np.ndarray:
        """
        Sample parameters from posterior t-distribution.

        Parameters
        ----------
        n_sample : int
            Number of samples to draw
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        np.ndarray
            Parameter samples from multivariate t-distribution, shape (n_sample, 2)
        """
        # Sample betas from posterior distribution
        rv = multivariate_t(
            loc=self.beta_hat,
            shape=self.Sigma,
            df=self.nu,
        )
        betas = rv.rvs(size=n_sample, random_state=seed)
        return betas

    def credible_interval(
        self,
        param_index: int,
        alpha: float = 0.05,
    ) -> tuple[float, float]:
        """
        Compute marginal credible interval for a parameter.

        Parameters
        ----------
        param_index : int
            Parameter index (0 for C0, 1 for S)
        alpha : float, optional
            Significance level

        Returns
        -------
        lower : float
            Lower bound of 100(1-alpha)% credible interval
        upper : float
            Upper bound of 100(1-alpha)% credible interval
        """
        t_crit_low = t.ppf(alpha / 2, df=self.nu)
        t_crit_high = t.ppf(1 - alpha / 2, df=self.nu)
        mean = self.beta_hat[param_index]
        scale = self.Sigma[param_index, param_index]
        lower = mean + t_crit_low * np.sqrt(scale)
        upper = mean + t_crit_high * np.sqrt(scale)
        return lower, upper

    @staticmethod
    def create_credible_region(
        beta: np.ndarray,
        Sigma: np.ndarray,
        nu: int,
        alpha: float = 0.05,
        edgecolor: str = "red",
    ) -> Ellipse:
        """
        Create ellipse representing credible region for bivariate t-distribution.

        Uses F-distribution to determine ellipse size for specified credible level.

        Parameters
        ----------
        beta : np.ndarray
            Center of ellipse [C0, S], shape (2,)
        Sigma : np.ndarray
            Scale matrix, shape (2, 2)
        nu : int
            Degrees of freedom
        alpha : float, optional
            Significance level (1-alpha is credible level)
        edgecolor : str, optional
            Ellipse edge color

        Returns
        -------
        Ellipse
            Matplotlib ellipse patch
        """
        # Compute ellipse orientation in (C0, S) Plane
        eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

        # Compute ellipse dimensions
        f_crit = f.ppf(1 - alpha, dfn=2, dfd=nu)
        semi_axes = np.sqrt(2 * f_crit * eigenvalues)
        major_axis, minor_axis = 2 * semi_axes

        ellipse = Ellipse(
            xy=beta,
            width=major_axis,
            height=minor_axis,
            angle=angle,
            edgecolor=edgecolor,
            facecolor="none",
        )
        return ellipse

    def plot_posterior_with_samples(
        self,
        n_sample=100_000,
        show_legend: bool = False,
        seed: int = 1,
    ) -> None:
        """
        Plot posterior t-distribution density of C0 with histogram of samples.

        Parameters
        ----------
        n_sample : int, optional
            Number of Monte Carlo samples
        show_legend : bool, optional
            Whether to display legend
        seed : int, optional
            Random seed for reproducibility
        """
        # Use univariate t-distribution to compute quantiles for axis limits
        rv_marginal = t(df=self.nu, loc=self.C0_hat, scale=np.sqrt(self.Sigma[0, 0]))
        lower = rv_marginal.ppf(0.0001)
        upper = rv_marginal.ppf(0.9999)

        # Compute marginal posterior distribution of beta
        rv = multivariate_t(loc=self.C0_hat, shape=self.Sigma[0, 0], df=self.nu)
        x_grid = np.linspace(start=lower, stop=upper, num=500)
        pdf = rv.pdf(x_grid)

        # Get Monte Carlo samples of betas
        rv = multivariate_t(loc=self.beta_hat, shape=self.Sigma, df=self.nu)
        betas_mc = rv.rvs(size=n_sample, random_state=seed)
        C0_mc = betas_mc[:, 0]

        # Compute bin edges based on data range and desired bin width
        bin_width = x_grid[10] - x_grid[0]
        bins = np.arange(
            start=min(x_grid),
            stop=max(x_grid) + bin_width,
            step=bin_width,
        )

        # Make plot
        ax = utils.setup_figure()
        ax.hist(
            C0_mc,
            color="black",
            histtype="step",
            bins=bins,
            linewidth=2,
            density=True,
            zorder=1,
        )
        ax.plot(
            x_grid,
            pdf,
            color="red",
            linewidth=2,
            label="Posterior",
            zorder=2,
        )
        ax.set_xlim(x_grid[0], x_grid[-1])
        ax.set_xlabel("$c_0$ [km/s]", fontsize=font_sizes["label"])
        ax.set_ylabel("Density", fontsize=font_sizes["label"])
        if show_legend:
            # Create custom legend with line for histogram
            legend_elements = [
                Line2D([0], [0], color="red", linewidth=2, label="Posterior"),
                Line2D([0], [0], color="black", linewidth=2, label="Samples"),
            ]
            ax.legend(
                handles=legend_elements, loc="upper left", fontsize=font_sizes["legend"]
            )
        utils.save_and_close_figure("marginal_density_C0", self.dataset)

    def plot_posterior_beta_noninformative(
        self,
        show_legend: bool = True,
        alpha: float = 0.05,
    ) -> None:
        """
        Plot bivariate posterior distribution with credible ellipse.

        Shows contour plot of joint posterior for (C0, S) with least squares
        estimate and 100(1-alpha)% credible region.

        Parameters
        ----------
        show_legend : bool, optional
            Whether to display legend
        alpha : float, optional
            Significance level for credible region
        """
        # Evaluate posterior distribution on a grid
        beta1_mesh, beta2_mesh, pdf = self.evaluate_t_pdf_on_grid()

        # Plot the density function with the least squares estimate of beta
        ax = utils.setup_figure(grid=False)
        contour = ax.contourf(beta1_mesh, beta2_mesh, pdf, levels=50, cmap="viridis")
        cb = plt.colorbar(contour, ax=ax)
        cb.ax.tick_params(labelsize=13)
        ax.scatter(self.C0_hat, self.S_hat, color="red", label="Posterior mean")

        # Plot elliptical credible region
        ellipse = self.create_credible_region(
            self.beta_hat,
            self.Sigma,
            self.nu,
            alpha,
        )
        ax.add_patch(ellipse)

        # Add plot labels and finish plot
        ax.set_xlabel("$c_0$ [km/s]", fontsize=font_sizes["label"])
        ax.set_ylabel("$s$", fontsize=font_sizes["label"])
        if show_legend:
            ax.legend(
                loc="upper right",
                fontsize=font_sizes["legend"],
            )
        utils.save_and_close_figure("posterior_beta_noninformative", self.dataset)

    def plot_credible_and_prediction_interval_for_Us(
        self,
        show_legend: bool = False,
        alpha: float = 0.05,
    ) -> None:
        """
        Plot 100(1-alpha)% credible interval for mean shock wave velocity and
        prediction interval for new shock wave velocity measurements.

        Parameters
        ----------
        show_legend : bool, optional
            Whether to display legend
        alpha : float, optional
            Significance level
        """
        # Prediction mean and scale matrix
        x_pred = np.linspace(self.x.min(), self.x.max(), num=100)
        X_pred = np.vstack([np.ones_like(x_pred), x_pred]).T
        y_pred_mean = X_pred @ self.beta_hat
        y_pred_scale = np.sum(X_pred @ self.Sigma * X_pred, axis=1) + self.s_squared

        # Prediction interval for new Us measurement
        t_multiplier = t.ppf(q=1 - alpha / 2, df=self.nu)
        y_pred_scale_sqrt = np.sqrt(y_pred_scale)
        pred_lower = y_pred_mean - t_multiplier * y_pred_scale_sqrt
        pred_upper = y_pred_mean + t_multiplier * y_pred_scale_sqrt

        # Credible interval for mean shock wave velocity
        cred_scale = np.sum(X_pred @ self.Sigma * X_pred, axis=1)
        cred_scale_sqrt = np.sqrt(cred_scale)
        cred_lower = y_pred_mean - t_multiplier * cred_scale_sqrt
        cred_upper = y_pred_mean + t_multiplier * cred_scale_sqrt

        # Use shared plotting method
        self._plot_interval_for_Us(
            x_pred=x_pred,
            y_pred_mean=y_pred_mean,
            cred_lower=cred_lower,
            cred_upper=cred_upper,
            pred_lower=pred_lower,
            pred_upper=pred_upper,
            interval_type="Cred.",
            filename="credible_and_prediction_interval",
            show_legend=show_legend,
            alpha=alpha,
            mean_label="Posterior mean",
        )

    def plot_posterior_predictive_check(
        self,
        show_legend: bool = False,
        seed: int = 1,
    ) -> None:
        """
        Generate posterior predictive check plot.

        Simulates new data from posterior predictive distribution and plots
        against observed data to assess model fit.

        Parameters
        ----------
        show_legend : bool, optional
            Whether to display legend
        seed : int, optional
            Random seed for reproducibility
        """
        rng = np.random.default_rng(seed)

        # Sample sigma^2 ~ IG(nu / 2, nu * s^2 / 2)
        sigma_sq = invgamma.rvs(
            a=self.nu / 2,
            scale=self.nu * self.s_squared / 2,
            random_state=rng,
        )

        # Sample beta from N(beta_hat, sigma^2 * (X'X)^{-1})
        cov_beta = sigma_sq * np.linalg.inv(self.XTX)
        pred_beta_hat = multivariate_normal.rvs(
            mean=self.beta_hat,
            cov=cov_beta,
            random_state=rng,
        )

        # Simulate new y values at x values
        error = rng.normal(0, scale=np.sqrt(sigma_sq), size=self.x.shape)
        y_new = pred_beta_hat[0] + pred_beta_hat[1] * self.x + error

        # Compute min and max of actual and simulated y measurements
        all_y = np.concatenate([self.y, y_new])
        min_y = all_y.min()
        max_y = all_y.max()

        ax = utils.setup_figure()
        ax.scatter(
            self.y,
            y_new,
            marker="o",
            zorder=2,
            facecolor="none",
            edgecolors="blue",
            s=50,
            alpha=0.95,
        )
        lims = [min_y * 0.95, max_y * 1.05]
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.plot(
            lims,
            lims,
            color="red",
            linestyle="--",
            linewidth=2,
            label=r"Reference line, $y=x$",
        )
        ax.set_xlabel(
            "Original Shock Wave Velocity [km/s]", fontsize=font_sizes["label"]
        )
        ax.set_ylabel(
            "Simulated Shock Wave Velocity [km/s]", fontsize=font_sizes["label"]
        )
        if show_legend:
            ax.legend(loc="upper left", fontsize=font_sizes.get("legend", 10))
        utils.save_and_close_figure("post_pred_check", self.dataset)

    def plot_posterior_variance(self) -> None:
        """Plot posterior distribution of variance parameter sigma^2."""
        # Parameters for inverse gamma distribution
        shape = self.nu / 2
        scale = self.nu * self.s_squared / 2

        # Compute posterior mean and variance
        posterior_mean = scale / (shape - 1)
        posterior_variance = scale**2 / (shape - 1) ** 2 / (shape - 2)
        posterior_std_dev = np.sqrt(posterior_variance)

        # Compute PDF
        x_grid = np.linspace(
            start=invgamma.ppf(q=0.000001, a=shape, scale=scale),
            stop=invgamma.ppf(q=0.99999, a=shape, scale=scale),
            num=500,
        )
        pdf = invgamma.pdf(x_grid, a=shape, scale=scale)

        # Make plot
        ax = utils.setup_figure()
        ax.plot(x_grid, pdf)
        ax.text(
            x=max(x_grid),
            y=max(pdf),
            s=f"Mean = {posterior_mean:.3f}\nStd. dev. = {posterior_std_dev:.3f}",
            fontsize=font_sizes["text"],
            ha="right",
            va="top",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
        )
        ax.set_xlabel(r"$\sigma^2$", fontsize=font_sizes["label"])
        ax.set_ylabel("Posterior Density", fontsize=font_sizes["label"])
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
        utils.save_and_close_figure("posterior_sigma_sq", self.dataset)

    def save_summary_statistics(self, alpha: float = 0.05) -> None:
        """
        Save posterior summary statistics to text file.

        Parameters
        ----------
        alpha : float, optional
            Significance level for credible intervals
        """
        # Compute credible intervals
        lower_C0, upper_C0 = self.credible_interval(0, alpha)
        lower_S, upper_S = self.credible_interval(1, alpha)

        # Extract posterior means and standard deviations
        C0_hat, S_hat = self.beta_hat
        sd_C0_hat, sd_S_hat = self.standard_deviations

        # Define parameters with their values
        params = [
            ("C0", C0_hat, sd_C0_hat, lower_C0, upper_C0),
            ("S", S_hat, sd_S_hat, lower_S, upper_S),
        ]

        # Use shared method
        self._save_parameter_summary_table(
            params=params, interval_type="Credible Interval", subdirectory="posterior"
        )

    def plot_posterior_beta_informative(
        self,
        beta0: tuple[float, float],
        std_devs: tuple[float, float],
        rho: float,
        a0: float,
        b0: float,
        alpha: float = 0.05,
        show_legend: bool = False,
    ) -> None:
        """
        Plot posterior with informative prior and compare to non-informative case.

        Shows prior mean, posterior mean, least squares estimate, and credible
        ellipses for both prior and posterior distributions.

        Parameters
        ----------
        beta0 : tuple[float, float]
            Prior mean for [C0, S]
        std_devs : tuple[float, float]
            Prior standard deviations for [C0, S]
        rho : float
            Prior correlation between C0 and S
        a0 : float
            Prior shape parameter for inverse gamma distribution of sigma_sq
        b0 : float
            Prior scale parameter for inverse gamma distribution of sigma_sq
        alpha : float, optional
            Significance level for credible regions
        show_legend : bool, optional
            Whether to display legend
        """
        # Specify prior mean and covariance matrix
        Sigma0 = utils.construct_covariance_matrix(*std_devs, rho)

        # Posterior precision matrix, up to multiplicative sigma**2 factor
        G = self.XTX + np.linalg.inv(Sigma0)

        # Weighted combination of prior and data
        beta0_array = np.array(beta0)
        gamma = self.X.T @ self.y + np.linalg.solve(Sigma0, beta0_array).reshape(-1, 1)

        # Posterior mean
        beta_posterior = np.linalg.solve(G, gamma).flatten()

        # Shape and scale of posterior distribution
        a_tilde = a0 + self.n / 2
        b_tilde = (
            b0
            + (
                np.sum(self.y * self.y)
                + beta0_array.T @ np.linalg.solve(Sigma0, beta0_array)
                - gamma.T @ np.linalg.solve(G, gamma)
            )
            / 2
        )

        # Posterior covariance matrix
        Sigma_posterior = b_tilde / a_tilde * np.linalg.inv(G)

        # Plot the PDF
        ax = utils.setup_figure()

        # Plot prior mean, posterior mean, and least squares estimate
        ax.scatter(
            self.beta_hat[0],
            self.beta_hat[1],
            color="blue",
            marker="o",
            s=80,
            label="Least squares est.",
        )
        ax.scatter(
            beta_posterior[0],
            beta_posterior[1],
            color="red",
            marker="^",
            s=80,
            label="Posterior mean",
        )
        ax.scatter(
            beta0[0],
            beta0[1],
            color="green",
            marker="s",
            s=80,
            label="Prior mean",
        )

        # Plot elliptical credible region for posterior distribution
        posterior_ellipse = self.create_credible_region(
            beta_posterior,
            Sigma_posterior,
            2 * a_tilde,
            alpha,
            "red",
        )
        ax.add_patch(posterior_ellipse)

        # Plot elliptical credible region for prior distribution
        Sigma_prior = (b0 / a0) * Sigma0
        prior_ellipse = self.create_credible_region(
            beta0,
            Sigma_prior,
            2 * a0,
            alpha,
            "green",
        )
        ax.add_patch(prior_ellipse)

        # Add plot labels and finish plot
        ax.set_xlabel("$c_0$ [km/s]", fontsize=font_sizes["label"])
        ax.set_ylabel("$s$", fontsize=font_sizes["label"])
        if show_legend:
            ax.legend(fontsize=font_sizes["legend"])
        utils.save_and_close_figure("posterior_beta_informative", self.dataset)


class BootstrapDistribution(RegressionBase):
    """Bootstrap distribution for linear regression parameters."""

    def __init__(self, df: pd.DataFrame, dataset: str):
        """
        Initialize bootstrap distribution.

        Parameters
        ----------
        df : pd.DataFrame
            Experimental data
        dataset : str
            Dataset name for file naming
        """
        super().__init__(df, dataset)

    def sample_beta(
        self,
        n_sample: int,
        seed: int = 1,
    ) -> np.ndarray:
        """
        Sample parameters using bootstrap resampling.

        Resamples data with replacement and computes least squares estimates
        for each resample.

        Parameters
        ----------
        n_sample : int
            Number of bootstrap samples
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        np.ndarray
            Bootstrap parameter samples, shape (n_sample, 2)
        """
        rng = np.random.default_rng(seed)
        indices = rng.integers(
            low=0,
            high=self.n,
            size=(n_sample, self.n),
        )
        betas = np.zeros((n_sample, 2))
        for i in range(n_sample):
            x_resample = self.x[indices[i]]
            y_resample = self.y[indices[i]]
            beta_hat_resample = self.compute_beta_hat(x_resample, y_resample)
            betas[i, :] = beta_hat_resample.flatten()

        return betas

    def plot_confidence_and_prediction_interval_for_Us(
        self,
        n_sample: int = 100_000,
        show_legend: bool = False,
        alpha: float = 0.05,
        seed: int = 1,
    ) -> None:
        """
        Plot 100(1-alpha)% confidence interval for mean shock wave velocity and
        prediction interval for new shock wave velocity measurements using bootstrap.

        Parameters
        ----------
        n_sample : int, optional
            Number of bootstrap samples
        show_legend : bool, optional
            Whether to display legend
        alpha : float, optional
            Significance level
        seed : int, optional
            Random seed for reproducibility
        """
        # Generate bootstrap samples
        betas = self.sample_beta(n_sample, seed)

        # Prediction grid
        n_pred = 100
        x_pred = np.linspace(self.x.min(), self.x.max(), num=n_pred)
        X_pred = np.vstack([np.ones_like(x_pred), x_pred]).T

        # Compute predicted Us for each bootstrap sample
        y_pred_samples = X_pred @ betas.T  # shape: (n_pred, n_sample)

        # Confidence interval for mean Us (percentile method)
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        cred_lower = np.percentile(y_pred_samples, lower_percentile, axis=1)
        cred_upper = np.percentile(y_pred_samples, upper_percentile, axis=1)

        # Prediction interval for new Us measurement
        rng = np.random.default_rng(seed)
        error = rng.normal(scale=np.sqrt(self.s_squared), size=(n_pred, n_sample))
        pred_samples = y_pred_samples + error
        pred_lower = np.percentile(pred_samples, lower_percentile, axis=1)
        pred_upper = np.percentile(pred_samples, upper_percentile, axis=1)

        # Mean prediction
        y_pred_mean = np.mean(y_pred_samples, axis=1)

        # Use shared plotting method
        self._plot_interval_for_Us(
            x_pred=x_pred,
            y_pred_mean=y_pred_mean,
            cred_lower=cred_lower,
            cred_upper=cred_upper,
            pred_lower=pred_lower,
            pred_upper=pred_upper,
            interval_type="Conf.",
            filename="confidence_and_prediction_interval",
            show_legend=show_legend,
            alpha=alpha,
            mean_label="Bootstrap mean",
        )

    def save_summary_statistics(
        self,
        n_sample: int = 100_000,
        alpha: float = 0.05,
        seed: int = 1,
    ) -> None:
        """
        Save bootstrap summary statistics to text file.

        Parameters
        ----------
        n_sample : int, optional
            Number of bootstrap samples
        alpha : float, optional
            Significance level for confidence intervals
        seed : int, optional
            Random seed for reproducibility
        """
        # Generate bootstrap samples
        betas = self.sample_beta(n_sample, seed)

        # Compute bootstrap statistics
        C0_bootstrap = betas[:, 0]
        S_bootstrap = betas[:, 1]

        # Means and standard deviations
        C0_mean = np.mean(C0_bootstrap)
        S_mean = np.mean(S_bootstrap)
        C0_std = np.std(C0_bootstrap)
        S_std = np.std(S_bootstrap)

        # Confidence intervals using percentile method
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_C0 = np.percentile(C0_bootstrap, lower_percentile)
        upper_C0 = np.percentile(C0_bootstrap, upper_percentile)
        lower_S = np.percentile(S_bootstrap, lower_percentile)
        upper_S = np.percentile(S_bootstrap, upper_percentile)

        # Define parameters with their values
        params = [
            ("C0", C0_mean, C0_std, lower_C0, upper_C0),
            ("S", S_mean, S_std, lower_S, upper_S),
        ]

        # Use shared method
        self._save_parameter_summary_table(
            params=params,
            interval_type="Confidence Interval",
            subdirectory="bootstrap",
        )
        print("")


class RegressionComparison:
    """Tools for comparing posterior and bootstrap distributions."""

    def __init__(
        self,
        posterior: PosteriorDistribution,
        bootstrap: BootstrapDistribution,
    ):
        """
        Initialize comparison object.

        Parameters
        ----------
        posterior : PosteriorDistribution
            Bayesian posterior distribution
        bootstrap : BootstrapDistribution
            Bootstrap distribution
        Raises
        ------
        ValueError
            If posterior and bootstrap distributions are from different datasets

        Notes
        -----
        Both distributions must be constructed from the same dataset to ensure
        valid statistical comparisons.
        """
        if posterior.dataset != bootstrap.dataset:
            raise ValueError(
                f"Posterior and bootstrap must be from same dataset. "
                f"Got posterior='{posterior.dataset}', bootstrap='{bootstrap.dataset}'"
            )

        self.posterior = posterior
        self.bootstrap = bootstrap
        self.dataset = posterior.dataset

    def plot_bootstrap_and_posterior_distributions(
        self,
        n_sample: int = 100_000,
        show_legend: bool = False,
    ) -> None:
        """
        Plot bootstrap and posterior t-distributions of C0 and S.

        Plots histograms of bootstrap samples overlaid with posterior
        t-distribution densities of both parameters.

        Parameters
        ----------
        n_sample : int, optional
            Number of bootstrap samples
        show_legend : bool, optional
            Whether to display legend
        """
        # Get bootstrap samples
        betas_bootstrap = self.bootstrap.sample_beta(n_sample)

        for i, (name, units) in enumerate(zip(["c_0", "s"], ["km/s", ""])):
            # Grid for evaluating PDFs of beta
            beta_grid = np.linspace(
                start=min(betas_bootstrap[:, i]),
                stop=max(betas_bootstrap[:, i]),
                num=1_000,
            )

            # Evaluate t-distribution pdf to overlay
            t_pdf = multivariate_t(
                loc=self.posterior.beta_hat[i],
                shape=self.posterior.Sigma[i, i],
                df=self.posterior.nu,
            ).pdf(x=beta_grid)

            # Plot bootstrap samples with posterior t distribution
            ax = utils.setup_figure()
            ax.hist(
                betas_bootstrap[:, i],
                color="black",
                histtype="step",
                bins=80,
                linewidth=2,
                density=True,
            )
            line_posterior = ax.plot(
                beta_grid,
                t_pdf,
                c="red",
                linewidth=2,
                label="Posterior",
            )
            xlab = f"${name}$ [{units}]" if units else f"${name}$"
            ax.set_xlabel(xlab, fontsize=font_sizes["label"])
            ax.set_ylabel("Density", fontsize=font_sizes["label"])
            if show_legend:
                # Create proxy artist for bootstrap as a black line
                bootstrap_line = Line2D(
                    [0], [0], color="black", linewidth=2, label="Bootstrap"
                )
                ax.legend(
                    handles=[bootstrap_line, line_posterior[0]],
                    loc="upper left",
                    fontsize=font_sizes["legend"],
                )
            utils.save_and_close_figure(
                f"comparison_bootstrap_tdist_{name}", self.dataset
            )

    def plot_bootstrap_distribution_with_and_without_max_Up_point(
        self,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        n_sample: int = 100_000,
        show_legend: bool = False,
    ) -> None:
        """
        Compare bootstrap distributions of S with and without max Up point.

        Removes the point with maximum Up and compares bootstrap distributions
        to assess its influence.

        Parameters
        ----------
        xlim : tuple[float, float]
            x-axis limits for S values
        ylim : tuple[float, float]
            y-axis limits for density
        n_sample : int, optional
            Number of bootstrap samples
        show_legend : bool, optional
            Whether to display legend
        """
        # Compute parameter estimates with full dataset
        betas1 = self.bootstrap.sample_beta(n_sample)

        # Compute parameter estimates with max Up point removed
        df2 = utils.remove_max_particle_velocity_point(self.bootstrap.df)
        bootstrap2 = BootstrapDistribution(df2, self.dataset)
        betas2 = bootstrap2.sample_beta(n_sample)

        # Plot bootstrap samples
        ax = utils.setup_figure()
        ax.hist(
            betas1[:, 1],
            color="black",
            histtype="step",
            bins=80,
            linewidth=2,
            density=True,
        )
        ax.hist(
            betas2[:, 1],
            color="green",
            histtype="step",
            bins=80,
            linewidth=2,
            density=True,
        )
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlabel("$s$", fontsize=font_sizes["label"])
        ax.set_ylabel("Density", fontsize=font_sizes["label"])
        if show_legend:
            legend_elements = [
                Line2D([0], [0], color="black", linewidth=2, label="All data"),
                Line2D(
                    [0],
                    [0],
                    color="green",
                    linewidth=2,
                    label=r"Max $U_\mathrm{p}$ point removed",
                ),
            ]
            ax.legend(
                handles=legend_elements, loc="upper left", fontsize=font_sizes["legend"]
            )
        utils.save_and_close_figure("compare_bootstrap_s", self.dataset)

    def plot_posterior_distribution_with_and_without_max_Up_point(
        self,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
    ) -> None:
        """
        Compare posterior t-distribution of S with and without max Up point.

        Removes the point with maximum Up and compares posterior distributions
        to assess its influence.

        Parameters
        ----------
        xlim : tuple[float, float]
            x-axis limits for S values
        ylim : tuple[float, float]
            y-axis limits for density
        """
        # Get t-distribution for data with max particle velocity point removed
        df_reduced = utils.remove_max_particle_velocity_point(self.posterior.df)
        posterior_reduced = PosteriorDistribution(df_reduced, self.dataset)

        # Create grid spanning plus/minus 4 std devs around both posterior means
        std_full = self.posterior.standard_deviations[1]
        std_reduced = posterior_reduced.standard_deviations[1]

        lower = min(
            self.posterior.S_hat - 4 * std_full,
            posterior_reduced.S_hat - 4 * std_reduced,
        )
        upper = max(
            self.posterior.S_hat + 4 * std_full,
            posterior_reduced.S_hat + 4 * std_reduced,
        )

        beta_grid = np.linspace(lower, upper, num=1_000)

        # Evaluate t-distribution posterior PDFs
        t_pdf1 = multivariate_t(
            loc=self.posterior.S_hat,
            shape=self.posterior.Sigma[1, 1],
            df=self.posterior.nu,
        ).pdf(x=beta_grid)

        t_pdf2 = multivariate_t(
            loc=posterior_reduced.S_hat,
            shape=posterior_reduced.Sigma[1, 1],
            df=posterior_reduced.nu,
        ).pdf(x=beta_grid)

        # Plot
        ax = utils.setup_figure()
        ax.plot(
            beta_grid,
            t_pdf1,
            c="black",
            linewidth=2,
            label="All data",
        )
        ax.plot(
            beta_grid,
            t_pdf2,
            c="green",
            linewidth=2,
            label=r"Max $U_\mathrm{p}$ point removed",
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("$s$", fontsize=font_sizes["label"])
        ax.set_ylabel("Density", fontsize=font_sizes["label"])
        utils.save_and_close_figure("compare_tdist_s", self.dataset)
