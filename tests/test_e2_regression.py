"""Regression test of E2: the script run on the raw data reproduces the reference tables."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from edmdc_tank_observables.paths import DEFAULT_RAW_DIR, PROJECT_ROOT

REFERENCE_DIR = PROJECT_ROOT / "tests" / "reference_data" / "E2"
TABLES = ("models", "errors", "error_summary", "improvements")
RTOL = 1e-9
ATOL = 1e-12


def load_script():
    """Import experiments/E2_dictionaries.py as a module, without running it."""
    path = PROJECT_ROOT / "experiments" / "E2_dictionaries.py"
    spec = importlib.util.spec_from_file_location("e2_dictionaries", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def e2_tables_dir(tmp_path_factory):
    """Run E2 on every Baseline run of the raw data, writing into a temporary folder; skip if any run or the E1 metadata is missing."""
    expected_runs = pd.read_csv(REFERENCE_DIR / "models.csv")["run"].unique()
    missing = [r for r in expected_runs if not (DEFAULT_RAW_DIR / "Baseline" / f"{r}.csv").is_file()]
    if missing:
        pytest.skip(f"raw data not available: missing {missing} in {DEFAULT_RAW_DIR / 'Baseline'} (see data/README.md)")
    e2 = load_script()
    if not e2.E1_METADATA.is_file():
        pytest.skip(f"E1 results not available: {e2.E1_METADATA.relative_to(PROJECT_ROOT).as_posix()}")
    out_dir = tmp_path_factory.mktemp("E2_dictionaries")
    e2.run_experiment(e2.load_runs(), out_dir, e2.load_acquisition_dates())
    return out_dir / "tables"


@pytest.mark.parametrize("name", TABLES)
def test_e2_reproduces_the_reference_table(e2_tables_dir, name):
    """The table has the reference rows and columns; text and integers match exactly and numbers within tolerance."""
    result = pd.read_csv(e2_tables_dir / f"{name}.csv")
    reference = pd.read_csv(REFERENCE_DIR / f"{name}.csv")
    assert list(result.columns) == list(reference.columns)
    assert len(result) == len(reference)
    for column in reference.columns:
        assert result[column].dtype == reference[column].dtype, column
        if pd.api.types.is_float_dtype(reference[column]):
            np.testing.assert_allclose(
                result[column], reference[column], rtol=RTOL, atol=ATOL, err_msg=f"{name}.csv, column {column}"
            )
        else:
            assert result[column].tolist() == reference[column].tolist(), f"{name}.csv, column {column}"
