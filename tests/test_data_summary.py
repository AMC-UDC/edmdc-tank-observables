"""Tests of the summary of the raw data, on synthetic runs with known answers and on the real Baseline runs."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from edmdc_tank_observables.data.loader import RawDataset, load_baseline, load_baseline_file
from edmdc_tank_observables.paths import DEFAULT_RAW_DIR, PROJECT_ROOT


def load_script():
    """Import experiments/data_summary.py as a module, without running it."""
    path = PROJECT_ROOT / "experiments" / "data_summary.py"
    spec = importlib.util.spec_from_file_location("data_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ds = load_script()
STEP = ds.STEP

needs_raw_data = pytest.mark.skipif(
    not (DEFAULT_RAW_DIR / "Baseline").is_dir(),
    reason="raw data not available (see data/README.md)",
)


def write_run_csv(folder, level, setpoint=20.0):
    """Write a 1 Hz Baseline CSV recorded on one day, with the columns the loader requires."""
    path = folder / "Baseline_30pct.csv"
    pd.DataFrame(
        {
            "Timestamp": "2025/10/13 10:19:00",
            "Valve_Closure_pct": 70,
            "Level_pct": level,
            "Control_Signal_pct": 50.0,
            "Setpoint_pct": setpoint,
        }
    ).to_csv(path, index=False)
    return path


def valid_run(level):
    """A run of valid samples at 1 Hz, starting at t = 0, with the given level in %."""
    level = np.asarray(level, dtype=float)
    n = len(level)
    df = pd.DataFrame(
        {
            "Level_pct": level,
            "Control_Signal_pct": 50.0,
            "Setpoint_pct": 20.0,
            "t_s": np.arange(n, dtype=float),
            "sensor_ok": True,
        }
    )
    return RawDataset(name="synthetic", valve_opening_pct=30.0, df=df, n_raw_rows=n, n_sensor_faults=0)


def test_sensor_faults_are_counted_and_their_rows_listed(tmp_path):
    """Two readings of zero level with a positive setpoint are counted as two faults, at the rows where they are."""
    level = [10.0] * 12
    level[3] = 0.0
    level[7] = 0.0
    raw = load_baseline_file(write_run_csv(tmp_path, level), drop_faults=False)
    row = ds.run_row(raw)
    assert row["n_samples_raw"] == 12
    assert row["duration_s"] == 12.0
    assert row["n_sensor_faults"] == 2
    assert row["sensor_fault_rows"] == "3;7"
    assert row["acquisition_date"] == "2025-10-13"


def test_a_fault_on_a_five_second_sample_leaves_one_interior_gap(tmp_path):
    """A fault at t = 10 s removes a sample of the 5 s grid, so the decimated series has one gap; a fault at t = 7 s has no effect."""
    level = [10.0] * 21
    level[10] = 0.0
    raw = load_baseline_file(write_run_csv(tmp_path, level), drop_faults=False)
    row = ds.run_row(raw)
    assert row["n_samples_ts5"] == 4  # t = 0, 5, 15, 20
    assert row["n_interior_gaps_ts5"] == 1

    level = [10.0] * 21
    level[7] = 0.0
    raw = load_baseline_file(write_run_csv(tmp_path, level), drop_faults=False)
    row = ds.run_row(raw)
    assert row["n_samples_ts5"] == 5
    assert row["n_interior_gaps_ts5"] == 0


def test_levels_off_the_recording_grid_are_detected(tmp_path):
    """A run whose levels are all multiples of 100/255 % is on the grid; one value in between puts it off."""
    level = [10 * STEP, 11 * STEP, 12 * STEP]
    raw = load_baseline_file(write_run_csv(tmp_path, level), drop_faults=False)
    assert ds.quantization_row(raw)["all_levels_on_grid"]

    level[1] = 10.5 * STEP
    raw = load_baseline_file(write_run_csv(tmp_path, level), drop_faults=False)
    assert not ds.quantization_row(raw)["all_levels_on_grid"]


def test_only_pairs_inside_the_transient_are_measured_in_recording_steps():
    """With a transient from t = 2 to t = 6 s, only the four 1 s pairs inside it are taken, each |dh| in recording steps."""
    counts = [0, 0, 0, 1, 1, 3, 3, 5, 5, 5]  # level in recording steps, one value per second
    raw = valid_run(np.array(counts) * STEP)
    steps = pd.DataFrame({"t_change_s": [2.0], "t95_s": [4.0]})
    increments, n_across_gaps = ds.transient_increments(raw, steps, 1.0)
    np.testing.assert_allclose(increments, [1, 0, 2, 0])  # pairs (2,3), (3,4), (4,5), (5,6)
    assert n_across_gaps == 0


def test_a_slower_period_keeps_only_its_own_samples_inside_the_transient():
    """At Ts = 2 s the same transient holds the samples t = 2, 4, 6, so two pairs with |dh| of 1 and 2 steps."""
    counts = [0, 0, 0, 1, 1, 3, 3, 5, 5, 5]
    raw = valid_run(np.array(counts) * STEP)
    steps = pd.DataFrame({"t_change_s": [2.0], "t95_s": [4.0]})
    increments, _ = ds.transient_increments(raw, steps, 2.0)
    np.testing.assert_allclose(increments, [1, 2])


def test_a_pair_that_crosses_a_gap_is_not_measured():
    """A sample removed inside the transient leaves a pair 2 s apart at Ts = 1 s; it is counted apart and not measured."""
    raw = valid_run(np.arange(10) * STEP)
    raw.df = raw.df.drop(index=4).reset_index(drop=True)  # t = 4 s removed as a sensor fault
    steps = pd.DataFrame({"t_change_s": [2.0], "t95_s": [4.0]})
    increments, n_across_gaps = ds.transient_increments(raw, steps, 1.0)
    np.testing.assert_allclose(increments, [1, 1])  # pairs (2,3) and (5,6); (3,5) crosses the gap
    assert n_across_gaps == 1


def test_a_step_without_t95_has_no_transient():
    """A step whose level never reaches 95 % has no transient, so it gives no pairs."""
    raw = valid_run(np.arange(10) * STEP)
    steps = pd.DataFrame({"t_change_s": [2.0], "t95_s": [np.nan]})
    increments, _ = ds.transient_increments(raw, steps, 1.0)
    assert len(increments) == 0


def test_fraction_below_one_step_counts_the_pairs_with_no_change():
    """Of the increments 0, 0, 1, 2 and 0 steps, three are below one step: a fraction of 0.6, with median 0."""
    row = ds.sampling_bounds_row(1.0, np.array([0.0, 0.0, 1.0, 2.0, 0.0]), np.array([100.0]))
    assert row["n_pairs"] == 5
    assert row["n_pairs_below_one_step"] == 3
    assert row["frac_below_one_step"] == pytest.approx(0.6)
    assert row["median_abs_dh_steps"] == 0.0


def test_an_increment_of_one_step_is_not_below_one_step():
    """An increment of exactly one recording step, computed in floating point, is not counted as below one step."""
    increment = np.abs(np.diff([10 * STEP, 11 * STEP])) / STEP
    row = ds.sampling_bounds_row(1.0, increment, np.array([100.0]))
    assert row["n_pairs_below_one_step"] == 0


def test_samples_per_rise_leave_out_steps_with_t95_zero_or_undefined_and_count_them():
    """With t95 of 100, 50, 0 s and one undefined, at Ts = 5 s the rises have 20 and 10 samples and two steps are excluded."""
    row = ds.sampling_bounds_row(5.0, np.array([0.0]), np.array([100.0, 50.0, 0.0, np.nan]))
    assert row["median_samples_per_rise"] == pytest.approx(15.0)
    assert row["min_samples_per_rise"] == pytest.approx(10.0)
    assert row["n_steps_used"] == 2
    assert row["n_steps_excluded"] == 2


@needs_raw_data
def test_real_data_ten_runs_with_no_gaps_at_five_seconds_and_levels_on_the_grid():
    """The ten Baseline runs have no interior gap at Ts = 5 s and every level on the 100/255 % grid."""
    raws = load_baseline(drop_faults=False)
    assert len(raws) == 10
    for raw in raws:
        assert ds.run_row(raw)["n_interior_gaps_ts5"] == 0
        assert ds.quantization_row(raw)["all_levels_on_grid"]


@needs_raw_data
def test_real_data_has_160_steps():
    """The ten Baseline runs have 16 setpoint steps each, 160 in total."""
    steps = ds.all_steps(load_baseline(drop_faults=False))
    assert len(steps) == 160
