"""
Data manipulation and plotting utilities for shock wave-particle velocity data
analysis.

This module provides helper functions for loading and validating experimental data,
configuring matplotlib figures, and performing statistical computations.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from balscd.plot_config import font_sizes


def load_data(dataset: str) -> pd.DataFrame:
    """
    Load and validate experimental data from CSV file.

    Removes duplicate rows and validates that required columns exist and
    sufficient data is present for regression analysis.

    Parameters
    ----------
    dataset : str
        Dataset filename without .csv extension (e.g., 'argon_marsh')

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns: rho0_g_cc, Us_km_s, Up_km_s, P_GPa, V_cc_g

    Raises
    ------
    FileNotFoundError
        If CSV file does not exist in data/ directory
    ValueError
        If dataset is empty, has fewer than 3 unique rows, or missing required columns
    """
    if not dataset:
        raise ValueError("Dataset name cannot be empty")

    filepath = Path("data") / f"{dataset}.csv"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {filepath}\n"
            f"Please ensure the file exists in the 'data' directory."
        )

    print(f"Loading {dataset} dataset")

    try:
        df = pd.read_csv(filepath)
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV file {filepath}: {e}")

    # Remove duplicate rows
    num_duplicates = df.duplicated().sum()
    if num_duplicates > 0:
        print(
            f"Found {num_duplicates} duplicated row(s) in {dataset}. Removing duplicates."
        )
        df = df.drop_duplicates()

    if len(df) < 3:
        raise ValueError(
            f"Dataset {dataset} has insufficient data for regression: "
            f"{len(df)} rows found, minimum 3 required"
        )

    # Require subset of columns in Marsh (1980)
    required_cols = ["rho0_g_cc", "Us_km_s", "Up_km_s", "P_GPa", "V_cc_g"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def remove_max_particle_velocity_point(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove observation with maximum particle velocity.

    Used for posterior and bootstrap distribution sensitivity analysis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing Up_km_s column

    Returns
    -------
    pd.DataFrame
        DataFrame with maximum Up_km_s observation removed
    """
    Up_max_index = df["Up_km_s"].idxmax()
    return df.drop(index=Up_max_index)


def save_and_close_figure(
    filename: str,
    dataset: str,
) -> None:
    """
    Save current matplotlib figure and close.

    Saves to images/{dataset}/{filename}.png with tight layout.

    Parameters
    ----------
    filename : str
        Output filename without extension
    dataset : str
        Dataset name (used as subdirectory)
    """
    plt.tight_layout()
    plt.savefig(Path("images") / dataset / f"{filename}.png")
    plt.close()


def setup_figure(
    figsize: tuple[int, int] = (5, 5),
    grid: bool = True,
) -> tuple[Figure, Axes]:
    """
    Create matplotlib figure with standard formatting.

    Parameters
    ----------
    figsize : tuple[int, int], optional
        Figure dimensions (width, height) in inches
    grid : bool, optional
        Whether to display grid lines

    Returns
    -------
    fig : Figure
        Matplotlib figure object
    ax : Axes
        Matplotlib axes object with tick size 13
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.tick_params(axis="both", labelsize=font_sizes["tick"])
    ax.grid(grid)
    return fig, ax


def sample_multivariate_t(
    mu: np.ndarray,
    Sigma: np.ndarray,
    nu: int,
    n_sample: int,
    seed: int = 1,
) -> np.ndarray:
    """
    Draw samples from multivariate t-distribution.

    Implements algorithm from Hofert (2013) using chi-square and normal draws.

    Parameters
    ----------
    mu : np.ndarray
        Location vector (mean), shape (p,)
    Sigma : np.ndarray
        Scale matrix (not covariance), shape (p, p)
    nu : int
        Degrees of freedom
    n_sample : int
        Number of samples to draw
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Samples from multivariate t-distribution, shape (n_sample, p)

    References
    ----------
    Hofert, M. (2013). On sampling from the multivariate t distribution.
        The R Journal, 5(2), 129-136.
    """
    rng = np.random.default_rng(seed)
    p = mu.size
    L = np.linalg.cholesky(Sigma)
    W = rng.chisquare(df=nu, size=n_sample)
    Z = rng.multivariate_normal(np.zeros(p), np.eye(p), size=n_sample)
    samples = mu + (Z @ L.T) / np.sqrt(W / nu)[:, None]
    return samples


def construct_covariance_matrix(
    sd1: float,
    sd2: float,
    rho: float,
) -> np.ndarray:
    """
    Construct a 2x2 covariance matrix from standard deviations and correlation.

    Parameters
    ----------
    sd1 : float
        Standard deviation of first variable (must be positive)
    sd2 : float
        Standard deviation of second variable (must be positive)
    rho : float
        Correlation coefficient (must be in [-1, 1])

    Returns
    -------
    np.ndarray
        Covariance matrix, shape (2, 2)

    Raises
    ------
    ValueError
        If sd1 or sd2 are non-positive, or if rho is outside [-1, 1]
    """
    if sd1 <= 0:
        raise ValueError(f"sd1 must be positive, got {sd1}")
    if sd2 <= 0:
        raise ValueError(f"sd2 must be positive, got {sd2}")
    if not -1 <= rho <= 1:
        raise ValueError(f"Correlation must be in [-1, 1], got {rho}")

    return np.array(
        [
            [sd1**2, rho * sd1 * sd2],
            [rho * sd1 * sd2, sd2**2],
        ]
    )
