"""Tests of the Baseline loader."""

import numpy as np
import pandas as pd
import pytest

from edmdc_tank_observables.data.loader import load_baseline, load_baseline_file
from edmdc_tank_observables.paths import DEFAULT_RAW_DIR

needs_raw_data = pytest.mark.skipif(
    not (DEFAULT_RAW_DIR / "Baseline").is_dir(),
    reason="raw data not available (see data/README.md)",
)


def write_baseline_csv(folder, opening, level, setpoint):
    """Write a minimal Baseline CSV with the columns the loader requires."""
    path = folder / "Baseline" / f"Baseline_{opening}pct.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Valve_Closure_pct": 100 - opening,
            "Level_pct": level,
            "Control_Signal_pct": 50.0,
            "Setpoint_pct": setpoint,
        }
    ).to_csv(path, index=False)
    return path


@needs_raw_data
def test_finds_the_ten_baseline_files():
    """The loader finds the ten Baseline files, ordered from lowest to highest valve opening."""
    datasets = load_baseline()
    assert len(datasets) == 10
    assert [d.valve_opening_pct for d in datasets] == [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


@needs_raw_data
def test_time_advances_one_second_per_row():
    """In the loaded data, time moves forward exactly one second from each row to the next."""
    for d in load_baseline(drop_faults=False):
        assert np.all(np.diff(d.df["t_s"].to_numpy()) == 1.0)


@needs_raw_data
def test_closure_column_is_100_minus_opening():
    """In every Baseline file, the valve closure column equals 100 minus the opening in the file name."""
    for d in load_baseline():
        assert (d.df["Valve_Closure_pct"] == 100 - d.valve_opening_pct).all()


@needs_raw_data
def test_level_is_recorded_in_steps_of_100_over_255():
    """In every Baseline run, every level value is a whole multiple of 100/255 % (an 8-bit reading)."""
    for d in load_baseline(drop_faults=False):
        counts = d.df["Level_pct"].to_numpy() * 255 / 100
        np.testing.assert_allclose(counts, np.round(counts), rtol=0, atol=1e-6)


def test_synthetic_sensor_fault_is_marked(tmp_path):
    """A reading of zero level while the setpoint is positive is flagged as a sensor fault and counted."""
    level = [10.0, 10.0, 0.0, 10.0, 10.0]  # fault at row 2: level 0 with setpoint > 0
    path = write_baseline_csv(tmp_path, 30, level, setpoint=20.0)
    d = load_baseline_file(path, drop_faults=False)
    assert d.n_raw_rows == 5
    assert d.n_sensor_faults == 1
    assert d.df["sensor_ok"].tolist() == [True, True, False, True, True]


def test_sensor_fault_is_dropped_by_default_and_time_keeps_the_gap(tmp_path):
    """Faulty samples are removed by default, and the time of the remaining samples still shows where they were."""
    path = write_baseline_csv(tmp_path, 30, [10.0, 10.0, 0.0, 10.0, 10.0], setpoint=20.0)
    d = load_baseline_file(path)
    assert d.n_sensor_faults == 1
    assert d.df["t_s"].tolist() == [0.0, 1.0, 3.0, 4.0]


def test_zero_level_with_zero_setpoint_is_not_a_fault(tmp_path):
    """A zero level is not a fault when the setpoint is also zero."""
    path = write_baseline_csv(tmp_path, 30, [0.0, 0.0, 0.0], setpoint=0.0)
    assert load_baseline_file(path).n_sensor_faults == 0


def test_opening_is_read_from_the_file_name(tmp_path):
    """The valve opening of a run is the number in its file name."""
    path = write_baseline_csv(tmp_path, 70, [10.0, 10.0], setpoint=20.0)
    assert load_baseline_file(path).valve_opening_pct == 70


def test_data_dir_argument_replaces_the_default(tmp_path):
    """Giving another data folder loads the files from there, ordered by opening."""
    for opening in (20, 10):
        write_baseline_csv(tmp_path, opening, [10.0, 10.0], setpoint=20.0)
    assert [d.valve_opening_pct for d in load_baseline(tmp_path)] == [10, 20]


def test_missing_data_dir_gives_a_clear_error(tmp_path):
    """A data folder that does not exist gives an error that points to the data instructions."""
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        load_baseline(tmp_path / "nonexistent")


def test_empty_baseline_folder_gives_a_clear_error(tmp_path):
    """A Baseline folder with no CSV files gives an error saying so."""
    (tmp_path / "Baseline").mkdir()
    with pytest.raises(FileNotFoundError, match="No CSV files"):
        load_baseline(tmp_path)


def test_missing_required_column_fails_loudly(tmp_path):
    """A CSV that lacks a required column is rejected with an error naming the missing columns."""
    path = tmp_path / "Baseline_10pct.csv"
    pd.DataFrame({"Level_pct": [1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_baseline_file(path)
