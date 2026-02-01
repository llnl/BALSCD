"""
Unit tests for data loading and validation.

Tests error handling for missing files, empty datasets, insufficient data rows,
and duplicate removal in the load_data function.
"""

import pandas as pd
import pytest

from balscd.utils import load_data


@pytest.mark.parametrize("dataset", ["nonexistent_dataset", "does_not_exist"])
def test_file_not_found_error(dataset):
    """Test that non-existent dataset names raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_data(dataset)


@pytest.mark.parametrize(
    "dataset,expected_length",
    [
        ("copper_marsh", 144),
        ("argon_marsh", 13),
        ("nickel_marsh", 19),
    ],
)
def test_load_data_returns_correct_length(dataset, expected_length):
    """Test that load_data returns a DataFrame with the correct length for each dataset."""
    df = load_data(dataset)

    # Check type
    assert isinstance(df, pd.DataFrame)

    # Check length
    assert len(df) == expected_length, f"Expected {expected_length} rows, got {len(df)}"


def test_load_data_empty_dataset(tmp_path, monkeypatch):
    """Test that ValueError is raised when dataset is empty (no data rows)."""
    # Create temporary data directory
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create CSV with only headers, no data
    csv_content = """\
rho0_g_cc,Us_km_s,Up_km_s,P_GPa,V_cc_g
"""

    csv_file = data_dir / "empty_data.csv"
    csv_file.write_text(csv_content)

    # Change working directory to temp path
    monkeypatch.chdir(tmp_path)

    with pytest.raises(
        ValueError,
        match=r"insufficient data for regression.*0 rows found, minimum 3 required",
    ):
        load_data("empty_data")


def test_load_data_insufficient_rows(tmp_path, monkeypatch):
    """Test that ValueError is raised when dataset has fewer than 3 rows."""
    # Create temporary data directory
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create CSV with only 2 data rows
    csv_content = """\
rho0_g_cc,Us_km_s,Up_km_s,P_GPa,V_cc_g
1.784,5.193,1.260,11.681,0.757
1.784,6.212,1.890,20.974,0.696
"""

    csv_file = data_dir / "insufficient_data.csv"
    csv_file.write_text(csv_content)

    # Change working directory to temp path
    monkeypatch.chdir(tmp_path)

    with pytest.raises(
        ValueError,
        match=r"insufficient data for regression.*2 rows found, minimum 3 required",
    ):
        load_data("insufficient_data")


def test_duplicate_rows_insufficient_data(tmp_path, monkeypatch):
    """Test that ValueError is raised when dataset has 4 rows but only 2 unique rows."""
    # Create temporary data directory
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create CSV with 4 rows but only 2 unique (duplicates)
    csv_content = """\
rho0_g_cc,Us_km_s,Up_km_s,P_GPa,V_cc_g
1.784,5.193,1.260,11.681,0.757
1.784,5.193,1.260,11.681,0.757
1.784,6.212,1.890,20.974,0.696
1.784,6.212,1.890,20.974,0.696
"""

    csv_file = data_dir / "duplicate_data.csv"
    csv_file.write_text(csv_content)

    # Change working directory to temp path
    monkeypatch.chdir(tmp_path)

    with pytest.raises(
        ValueError,
        match=r"insufficient data for regression.*2 rows found, minimum 3 required",
    ):
        load_data("duplicate_data")
