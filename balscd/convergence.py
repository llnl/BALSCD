"""
Visualization of t-distribution convergence to normal distribution.

Generates a figure showing how the t-distribution converges to a normal
distribution as degrees of freedom increase.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_t, norm

from balscd.plot_config import font_sizes as font_sizes
from balscd.utils import setup_figure


def plot_t_distribution_convergence() -> None:
    """
    Plot t-distributions converging to normal as degrees of freedom increase.

    Creates figure showing t-distributions with nu = 2, 5, 10, 15, 20 alongside
    a standard normal distribution to illustrate asymptotic convergence.
    Saves to images/t_distribution_convergence.png.
    """
    # Parameters
    mu = 0
    Sigma = 1.5  # Scale matrix for t-distribution
    nus = [2, 5, 10, 15, 20]
    x_grid = np.linspace(-7, 7, 500)

    # Plotting
    fig, ax = setup_figure(figsize=(8, 5))

    # Normal distribution
    normal_pdf = norm.pdf(x_grid, loc=mu, scale=np.sqrt(Sigma))
    ax.plot(
        x_grid,
        normal_pdf,
        label=r"$\nu \rightarrow \infty$",
        color="black",
        linestyle="--",
        linewidth=1.5,
    )

    # t-distributions
    loc_vec = np.array([mu], dtype=float)
    scale_mat = np.array([[Sigma]], dtype=float)

    for nu in reversed(nus):
        rv = multivariate_t(loc=loc_vec, shape=scale_mat, df=nu)
        pdf = rv.pdf(x_grid)
        ax.plot(x_grid, pdf, label=rf"$ \nu = {nu} $")

    ax.set_xlabel("x", fontsize=font_sizes["label"])
    ax.set_ylabel("Probability Density Function", fontsize=font_sizes["label"])
    ax.legend(fontsize=font_sizes["legend"])
    fig.tight_layout()
    fig.savefig(os.path.join("images", "t_distribution_convergence.png"))
    plt.close()
