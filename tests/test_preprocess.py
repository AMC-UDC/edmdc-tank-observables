"""Tests of the preprocessing core: resampling, gaps, normalization and split."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from edmdc_tank_observables.data.loader import RawDataset, load_baseline
from edmdc_tank_observables.data.preprocess import (
    PreprocessedDataset,
    common_end_time,
    denormalize,
    denormalize_delta,
    find_gaps,
    normalize,
    prepare,
    resample,
    temporal_split,
)
from edmdc_tank_observables.paths import DEFAULT_RAW_DIR

needs_raw_data = pytest.mark.skipif(
    not (DEFAULT_RAW_DIR / "Baseline").is_dir(),
    reason="raw data not available (see data/README.md)",
)


def make_raw(n_rows, dropped_rows=()):
    """Build a 1 Hz run whose signals equal known multiples of the row index.

    Level = row, control = 2 x row, setpoint = 3 x row, so every retained
    sample can be checked by hand. Rows in dropped_rows are removed, keeping
    their original t_s, as the loader does with sensor faults.
    """
    rows = np.array([r for r in range(n_rows) if r not in dropped_rows], dtype=float)
    df = pd.DataFrame(
        {
            "Level_pct": rows,
            "Control_Signal_pct": 2 * rows,
            "Setpoint_pct": 3 * rows,
            "t_s": rows,
        }
    )
    return RawDataset(
        name="synthetic",
        valve_opening_pct=50.0,
        df=df,
        n_raw_rows=n_rows,
        n_sensor_faults=len(dropped_rows),
    )


def test_resample_keeps_one_sample_out_of_five():
    """Resampling to 5 s keeps one sample out of every five, with the values that belong to those times."""
    h, u, sp, t = resample(make_raw(21), ts=5.0)
    np.testing.assert_array_equal(t, [0, 5, 10, 15, 20])
    np.testing.assert_array_equal(h, [0, 5, 10, 15, 20])
    np.testing.assert_array_equal(u, [0, 10, 20, 30, 40])
    np.testing.assert_array_equal(sp, [0, 15, 30, 45, 60])


def test_resample_rejects_a_period_that_is_not_a_whole_number_of_seconds():
    """A sampling period of 2.5 s is rejected because it is not a whole number of seconds."""
    with pytest.raises(ValueError):
        resample(make_raw(21), ts=2.5)


def test_resample_rejects_a_non_positive_period():
    """A sampling period of zero is rejected."""
    with pytest.raises(ValueError):
        resample(make_raw(21), ts=0.0)


def test_failure_in_row_0_just_moves_the_start_to_t_5():
    """If the first sample is lost, the series simply starts at 5 s and has no gap."""
    _, _, _, t = resample(make_raw(21, dropped_rows=(0,)), ts=5.0)
    np.testing.assert_array_equal(t, [5, 10, 15, 20])
    assert len(find_gaps(t, 5.0)) == 0
    assert prepare(make_raw(21, dropped_rows=(0,)), ts=5.0).t[0] == 5.0


def test_failure_off_the_grid_has_no_effect():
    """Losing a sample that falls between the kept times changes nothing."""
    _, _, _, t_clean = resample(make_raw(21), ts=5.0)
    _, _, _, t_fault = resample(make_raw(21, dropped_rows=(3,)), ts=5.0)
    np.testing.assert_array_equal(t_fault, t_clean)


def test_failure_on_the_grid_inside_the_series_makes_prepare_fail():
    """Losing a kept sample in the middle of the series leaves a gap, and preparing the run fails with an error that reports it."""
    raw = make_raw(21, dropped_rows=(10,))
    _, _, _, t = resample(raw, ts=5.0)
    np.testing.assert_array_equal(find_gaps(t, 5.0), [1])  # pair (t=5, t=15)
    with pytest.raises(ValueError, match=r"1 gap.*t = 5"):
        prepare(raw, ts=5.0)


def test_normalize_maps_50_to_one_half_and_denormalize_returns_it():
    """Normalizing 0, 50 and 100 % gives 0, 0.5 and 1, and undoing it returns the original values."""
    x = np.array([0.0, 50.0, 100.0])
    x_norm = normalize(x, (0.0, 100.0))
    np.testing.assert_array_equal(x_norm, [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(denormalize(x_norm, (0.0, 100.0)), x)


def test_normalize_rejects_a_degenerate_scale():
    """A range whose minimum equals its maximum is rejected."""
    with pytest.raises(ValueError):
        normalize(np.array([1.0]), (5.0, 5.0))


def test_denormalize_delta_only_rescales_by_the_range():
    """A normalized difference is scaled by the range of the variable, without adding the minimum."""
    np.testing.assert_allclose(denormalize_delta(np.array([0.02]), (20.0, 70.0)), [1.0])


def test_denormalize_delta_keeps_a_zero_error_at_zero():
    """An error of zero stays zero when converted back, even if the range does not start at 0."""
    np.testing.assert_array_equal(denormalize_delta(np.zeros(3), (20.0, 70.0)), np.zeros(3))
    assert denormalize(np.zeros(3), (20.0, 70.0))[0] == 20.0  # a level of 0 is not an error of 0


def test_difference_of_denormalized_signals_is_the_denormalized_difference():
    """Subtracting two signals after converting them back equals converting back their difference."""
    a, b = np.array([0.3, 0.9]), np.array([0.1, 0.4])
    scale = (20.0, 70.0)
    np.testing.assert_allclose(denormalize(a, scale) - denormalize(b, scale), denormalize_delta(a - b, scale))


def test_prepare_normalizes_h_u_and_sp_with_the_physical_range():
    """Preparing a run scales level, control and setpoint to 0-1 (setpoint like the level), and the level can be converted back to %."""
    d = prepare(make_raw(21), ts=5.0)
    np.testing.assert_allclose(d.h, np.array([0, 5, 10, 15, 20]) / 100)
    np.testing.assert_allclose(d.u, np.array([0, 10, 20, 30, 40]) / 100)
    np.testing.assert_allclose(d.sp, np.array([0, 15, 30, 45, 60]) / 100)  # same scale as h
    assert d.h_scale == (0.0, 100.0)
    assert d.u_scale == (0.0, 100.0)
    assert d.ts == 5.0
    assert d.name == "synthetic"
    assert len(d) == 5
    assert d.n_pairs == 4
    np.testing.assert_allclose(d.denormalize_h(d.h), [0, 5, 10, 15, 20])


def make_dataset(n):
    """Dataset of n samples with t = 5 x index and h = index."""
    k = np.arange(n, dtype=float)
    return PreprocessedDataset(
        h=k, u=2 * k, sp=3 * k, t=5.0 * k, ts=5.0,
        h_scale=(0.0, 100.0), u_scale=(0.0, 100.0), name="run",
    )


def make_timed_dataset(t_first, t_last, name="run"):
    """Dataset at Ts = 5 s whose samples go from t_first to t_last (s), with h = t."""
    t = np.arange(t_first, t_last + 5.0, 5.0)
    return PreprocessedDataset(
        h=t.copy(), u=t.copy(), sp=t.copy(), t=t, ts=5.0,
        h_scale=(0.0, 100.0), u_scale=(0.0, 100.0), name=name,
    )


def test_split_of_10_samples_at_70_percent_gives_7_and_3():
    """Splitting 10 samples at 70 % gives 7 for fitting and 3 for evaluation, named accordingly, with no sample in both and the original times kept."""
    train, test = temporal_split(make_dataset(10), train_frac=0.7)
    assert (len(train), len(test)) == (7, 3)
    assert (train.name, test.name) == ("run_train", "run_test")
    np.testing.assert_array_equal(train.t, 5.0 * np.arange(7))
    np.testing.assert_array_equal(test.t, 5.0 * np.arange(7, 10))  # absolute time kept
    np.testing.assert_array_equal(test.h, [7, 8, 9])
    np.testing.assert_array_equal(test.sp, [21, 24, 27])
    assert set(train.t).isdisjoint(test.t)
    assert test.ts == 5.0 and test.h_scale == (0.0, 100.0)


def test_split_shares_no_transition():
    """No pair of consecutive samples appears in both parts, and the pair that crosses the cut belongs to neither."""
    train, test = temporal_split(make_dataset(10), train_frac=0.7)
    pairs_train = set(zip(train.t[:-1], train.t[1:]))
    pairs_test = set(zip(test.t[:-1], test.t[1:]))
    assert pairs_train.isdisjoint(pairs_test)
    assert train.n_pairs + test.n_pairs == 9 - 1  # the pair crossing the cut is in neither


@pytest.mark.parametrize("t_first", [0.0, 5.0, 10.0])
def test_split_cuts_at_the_same_instant_whatever_the_first_sample(t_first):
    """Runs that end at 95 s are cut at 70 % of 100 s: the evaluation part starts at t = 70 s even if the run starts at 0, 5 or 10 s."""
    train, test = temporal_split(make_timed_dataset(t_first, 95.0), train_frac=0.7)
    assert train.t[-1] == 65.0
    assert test.t[0] == 70.0


def test_split_rounds_the_cut_instant_to_the_sampling_grid():
    """55 % of 100 s is 55.00000000000001 in floating point; the cut is still at the sample of 55 s, not one sample later."""
    assert 0.55 * 100.0 > 55.0  # the floating-point case this test is about
    train, test = temporal_split(make_timed_dataset(0.0, 95.0), train_frac=0.55)
    assert train.t[-1] == 50.0
    assert test.t[0] == 55.0


def test_split_parts_cover_the_run_with_no_overlap_and_no_gap():
    """Fit and evaluation parts, one after the other, give back every sample of the run exactly once."""
    d = make_timed_dataset(5.0, 95.0)
    train, test = temporal_split(d, train_frac=0.7)
    np.testing.assert_array_equal(np.concatenate([train.t, test.t]), d.t)
    np.testing.assert_array_equal(np.concatenate([train.h, test.h]), d.h)


def test_common_end_time_returns_the_shared_final_instant():
    """Runs that start at different samples but end at the same one share their final instant."""
    runs = [make_timed_dataset(t0, 95.0, f"run_{t0:g}") for t0 in (0.0, 5.0, 10.0)]
    assert common_end_time(runs) == 95.0


def test_common_end_time_rejects_runs_with_different_end_times():
    """If one run ends at a different instant, the error names the runs and their end times."""
    runs = [make_timed_dataset(0.0, 95.0, "run_a"), make_timed_dataset(0.0, 90.0, "run_b")]
    with pytest.raises(ValueError, match="run_b.*90"):
        common_end_time(runs)


@pytest.mark.parametrize("frac", [0.0, 1.0, -0.1, 1.5])
def test_split_rejects_a_fraction_outside_0_1(frac):
    """A fit fraction of 0 or below, or of 1 or above, is rejected."""
    with pytest.raises(ValueError):
        temporal_split(make_dataset(10), train_frac=frac)


@pytest.mark.parametrize("frac", [0.04, 0.96])
def test_split_rejects_a_fraction_that_leaves_a_half_empty(frac):
    """A fit fraction so extreme that one part would have no samples is rejected."""
    with pytest.raises(ValueError):
        temporal_split(make_dataset(10), train_frac=frac)


def test_dataset_denormalizes_a_level_difference_with_its_own_scale():
    """The dataset converts a level difference back to % using the scale it was normalized with."""
    d = replace(make_dataset(4), h_scale=(20.0, 70.0))
    np.testing.assert_allclose(d.denormalize_h_delta(np.array([0.02, -0.04])), [1.0, -2.0])


@needs_raw_data
def test_real_baseline_has_no_gaps_at_5_s_and_is_cut_at_the_same_instant():
    """The ten real Baseline runs have no gaps at 5 s, end at the same instant, and their evaluation part starts at 21420 s."""
    runs = [prepare(raw, ts=5.0) for raw in load_baseline()]  # prepare raises if there is any interior gap
    assert len(runs) == 10
    assert common_end_time(runs) == 30595.0
    for d in runs:
        assert len(find_gaps(d.t, 5.0)) == 0
        train, test = temporal_split(d, train_frac=0.7)
        assert train.t[-1] == 21415.0 and test.t[0] == 21420.0
        assert len(train) + len(test) == len(d)
