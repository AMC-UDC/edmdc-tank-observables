"""Tests of the evaluation protocol: model rows, error rows, the sweep and the aggregates across runs."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from edmdc_tank_observables.data.preprocess import PreprocessedDataset
from edmdc_tank_observables.evaluation.protocol import (
    COEFFICIENT_COLUMNS,
    ERROR_COLUMNS,
    MODEL_COLUMNS,
    check_same_windows,
    coefficient_rows,
    count_improvements,
    evaluate,
    model_from_coefficients,
    model_row,
    summarize_across_runs,
    sweep,
    window_table,
    window_trajectory,
    windows_shared_across_runs,
)
from edmdc_tank_observables.models.edmdc import fit_edmdc
from edmdc_tank_observables.observables.dictionaries import make_dictionary


def make_dataset(n, step_samples, seed, name):
    """A synthetic run with a positive level, a varying control and a setpoint that rises at the given samples."""
    rng = np.random.default_rng(seed)
    u = 0.5 + 0.3 * np.sin(0.4 * np.arange(n) + seed) + 0.02 * rng.standard_normal(n)
    h = [0.3]
    for k in range(n - 1):
        h.append(0.9 * h[-1] + 0.1 * u[k] + 0.04)
    sp = np.full(n, 0.2)
    for p in step_samples:
        sp[p:] += 0.1
    return PreprocessedDataset(
        h=np.array(h), u=u, sp=sp, t=5.0 * np.arange(n), ts=5.0,
        h_scale=(0.0, 100.0), u_scale=(0.0, 100.0), name=name,
    )


def make_runs():
    """Two synthetic runs as (name, opening, fit part, evaluation part)."""
    return [
        ("run_a", 30.0, make_dataset(80, (), 1, "run_a_train"), make_dataset(40, (10, 20, 30), 2, "run_a_test")),
        ("run_b", 60.0, make_dataset(80, (), 3, "run_b_train"), make_dataset(40, (10, 20, 30), 4, "run_b_test")),
    ]


def test_model_row_has_exactly_the_model_columns():
    """The summary row of a fitted model has the agreed columns, with the right values."""
    _, _, train, _ = make_runs()[0]
    model = fit_edmdc(make_dictionary("D1c", n_delays=2), train)
    row = model_row(model, "run_a", 30.0)
    assert list(row) == list(MODEL_COLUMNS)
    assert row["run"] == "run_a" and row["opening_pct"] == 30.0 and row["ts"] == 5.0
    assert row["dictionary"] == "D1c" and row["p"] == 3 and row["n_delays"] == 2
    assert row["n_pairs"] == 80 - 2 - 1
    assert row["cond"] == model.cond and row["spectral_radius"] == model.spectral_radius
    assert row["fit_rmse_pct"] == model.fit_rmse_pct


# Every configuration, with small hyperparameters.
ALL_CONFIGS = {
    "D0": {}, "D1a": {"order": 3}, "D1b": {"n_centers": 3}, "D1c": {"n_delays": 2},
    "D2": {}, "D3a": {"order": 3}, "D3b": {"n_centers": 3}, "D3c": {"n_delays": 2},
}


def test_coefficient_rows_list_a_b_and_c_with_their_names():
    """A model with p observables gives p * (p + 2) rows, by output and then by term, with the values of A, B and c."""
    _, _, train, _ = make_runs()[0]
    model = fit_edmdc(make_dictionary("D1c", n_delays=2), train)
    rows = coefficient_rows(model, "run_a", 30.0)
    assert len(rows) == 3 * 5
    assert all(list(r) == list(COEFFICIENT_COLUMNS) for r in rows)
    assert all(r["run"] == "run_a" and r["opening_pct"] == 30.0 and r["dictionary"] == "D1c" for r in rows)
    terms = ["h", "h_lag_1", "h_lag_2", "u", "const"]
    assert [(r["output"], r["term"]) for r in rows] == [(o, t) for o in ["h", "h_lag_1", "h_lag_2"] for t in terms]
    # The row predicting h_lag_1 is the second row of [A | B | c].
    second = [r["coefficient"] for r in rows[5:10]]
    assert second == [*model.A[1], model.B[1, 0], model.c[1]]


@pytest.mark.parametrize("dict_id", list(ALL_CONFIGS))
def test_coefficients_written_to_csv_and_read_back_give_the_same_model(dict_id, tmp_path):
    """Writing the coefficients table to CSV and reading it back with round-trip precision rebuilds A, B and c exactly, for every dictionary."""
    runs = make_runs()
    _, _, coefficients = sweep(
        runs, dict_ids=[dict_id], hyperparams=ALL_CONFIGS, horizons_s=(10,), keep_coefficients=True
    )
    coefficients.to_csv(tmp_path / "coefficients.csv", index=False)
    # The default float parser of read_csv can be one unit in the last place off.
    table = pd.read_csv(tmp_path / "coefficients.csv", float_precision="round_trip")
    dictionary = make_dictionary(dict_id, **ALL_CONFIGS[dict_id])
    for name, _, train, _ in runs:
        model = fit_edmdc(dictionary, train)
        A, B, c = model_from_coefficients(table, name, dictionary)
        assert np.array_equal(A, model.A) and np.array_equal(B, model.B) and np.array_equal(c, model.c)


def test_model_from_coefficients_rejects_missing_or_repeated_rows():
    """Rebuilding a model fails if a coefficient is missing, repeated, or the run is not in the table."""
    _, _, coefficients = sweep(make_runs(), dict_ids=["D2"], horizons_s=(10,), keep_coefficients=True)
    dictionary = make_dictionary("D2")
    with pytest.raises(ValueError, match="missing"):
        model_from_coefficients(coefficients.drop(index=3), "run_a", dictionary)
    with pytest.raises(ValueError, match="repeated"):
        model_from_coefficients(pd.concat([coefficients, coefficients.iloc[[0]]]), "run_a", dictionary)
    with pytest.raises(ValueError, match="missing"):
        model_from_coefficients(coefficients, "run_c", dictionary)


def test_sweep_without_coefficients_returns_the_same_two_tables():
    """By default the sweep returns only the models and errors tables, and asking for the coefficients does not change them."""
    kwargs = {"dict_ids": ["D0", "D1c"], "hyperparams": {"D1c": {"n_delays": 2}}, "horizons_s": (10, 20)}
    default = sweep(make_runs(), **kwargs)
    without = sweep(make_runs(), keep_coefficients=False, **kwargs)
    with_coefficients = sweep(make_runs(), keep_coefficients=True, **kwargs)
    assert len(default) == 2 and len(without) == 2 and len(with_coefficients) == 3
    for table_default, table_without, table_with in zip(default, without, with_coefficients):
        pd.testing.assert_frame_equal(table_default, table_without)
        pd.testing.assert_frame_equal(table_default, table_with)
    assert list(with_coefficients[2].columns) == list(COEFFICIENT_COLUMNS)
    assert len(with_coefficients[2]) == 2 * (1 * 3 + 3 * 5)


def test_evaluate_gives_one_row_per_horizon_with_the_error_columns():
    """Evaluating a model returns one row per horizon, with the agreed columns."""
    _, _, train, test = make_runs()[0]
    model = fit_edmdc(make_dictionary("D0"), train)
    rows = evaluate(model, test, horizons_s=(10, 20), run="run_a", opening_pct=30.0)
    assert [list(r) for r in rows] == [list(ERROR_COLUMNS)] * 2
    assert [r["horizon_s"] for r in rows] == [10, 20]
    assert [r["horizon_steps"] for r in rows] == [2, 4]
    assert all(r["series"] == "D0" and r["n_windows"] == 3 and r["ts"] == 5.0 for r in rows)


def test_sweep_with_two_runs_and_three_dictionaries():
    """The sweep fits and evaluates every dictionary on every run, with one error row per run, dictionary and horizon."""
    models, errors = sweep(
        make_runs(), dict_ids=["D0", "D2", "D1c"], hyperparams={"D1c": {"n_delays": 2}}, horizons_s=(10, 20)
    )
    assert list(models.columns) == list(MODEL_COLUMNS)
    assert list(errors.columns) == list(ERROR_COLUMNS)
    assert len(models) == 6
    assert sorted(models["dictionary"]) == ["D0", "D0", "D1c", "D1c", "D2", "D2"]
    assert models.set_index(["run", "dictionary"])["p"].to_dict() == {
        ("run_a", "D0"): 1, ("run_a", "D2"): 2, ("run_a", "D1c"): 3,
        ("run_b", "D0"): 1, ("run_b", "D2"): 2, ("run_b", "D1c"): 3,
    }
    # Per run: 3 dictionaries, each at 2 horizons.
    assert len(errors) == 2 * 3 * 2
    assert sorted(errors["series"].unique()) == ["D0", "D1c", "D2"]
    assert (errors["n_windows"] == 3).all()
    assert errors.groupby("run")["opening_pct"].first().to_dict() == {"run_a": 30.0, "run_b": 60.0}


def test_sweep_offset_reaches_the_windows():
    """The offset given to the sweep shifts the windows, so a large one makes it fail instead of being ignored."""
    sweep(make_runs(), dict_ids=["D0"], horizons_s=(10,), offset=1)
    with pytest.raises(ValueError):
        sweep(make_runs(), dict_ids=["D0"], horizons_s=(10,), offset=-20)


def hand_errors():
    """An errors table written by hand: three runs, three series and two horizons.

    At 30 s the RMSE per run (a, b, c) is D0 = 1, 2, 4; D2 = 0.5, 3, 3.9;
    D1a = 2, 2, 2. At 120 s every value is ten times larger.
    """
    values = {"D0": (1.0, 2.0, 4.0), "D2": (0.5, 3.0, 3.9), "D1a": (2.0, 2.0, 2.0)}
    rows = []
    for horizon, factor in ((30.0, 1.0), (120.0, 10.0)):
        for series, rmse in values.items():
            for run, value in zip(("a", "b", "c"), rmse):
                rows.append({"run": run, "series": series, "horizon_s": horizon, "rmse": value * factor})
    return pd.DataFrame(rows)


def test_summary_across_runs_gives_median_minimum_and_maximum():
    """For each series and horizon the summary is the median, the minimum and the maximum over the runs."""
    summary = summarize_across_runs(hand_errors()).set_index(["series", "horizon_s"])
    assert list(summary.columns) == ["median", "min", "max", "n_runs"]
    assert summary.loc[("D0", 30.0)].tolist() == [2.0, 1.0, 4.0, 3]
    assert summary.loc[("D2", 30.0)].tolist() == [3.0, 0.5, 3.9, 3]
    assert summary.loc[("D1a", 30.0)].tolist() == [2.0, 2.0, 2.0, 3]
    assert summary.loc[("D0", 120.0)].tolist() == [20.0, 10.0, 40.0, 3]
    assert len(summary) == 6


def test_summary_across_runs_can_use_another_error_column():
    """The summary can be taken over a column other than rmse."""
    errors = hand_errors().rename(columns={"rmse": "max_abs"})
    summary = summarize_across_runs(errors, column="max_abs").set_index(["series", "horizon_s"])
    assert summary.loc[("D2", 120.0), "median"] == 30.0


def test_improvements_over_d0_are_counted_run_by_run():
    """A series improves on D0 in a run when its error is lower than that of D0 in the same run and horizon."""
    counts = count_improvements(hand_errors(), reference="D0").set_index(["series", "horizon_s"])
    assert list(counts.columns) == ["n_better", "n_compared"]
    assert counts.loc[("D2", 30.0)].tolist() == [2, 3]  # runs a and c
    assert counts.loc[("D1a", 30.0)].tolist() == [1, 3]  # run c only
    assert "D0" not in counts.index.get_level_values("series")


def test_a_tie_with_the_reference_is_not_an_improvement():
    """An error equal to that of the reference does not count as an improvement."""
    counts = count_improvements(hand_errors(), reference="D1a").set_index(["series", "horizon_s"])
    assert counts.loc[("D0", 30.0)].tolist() == [1, 3]  # run a; run b ties at 2
    assert counts.loc[("D2", 120.0)].tolist() == [1, 3]  # run a only


def test_improvements_with_an_absent_reference_fail():
    """Asking for a reference series that is not in the table is an error."""
    with pytest.raises(ValueError):
        count_improvements(hand_errors(), reference="D1c")


def test_window_trajectory_of_an_exact_model_follows_the_measured_level():
    """A model that fits the data exactly predicts, over a window, the level that was measured."""
    _, _, train, test = make_runs()[0]
    model = fit_edmdc(make_dictionary("D0"), train)
    t, h_pred = window_trajectory(model, test, start=9, horizon_steps=4)
    assert np.allclose(t, [5.0, 10.0, 15.0, 20.0])
    assert np.allclose(h_pred, 100.0 * test.h[10:14])


def test_window_trajectory_uses_the_history_of_a_model_with_delays():
    """With delays the trajectory starts from the measured levels that end at the window start."""
    _, _, train, test = make_runs()[0]
    model = fit_edmdc(make_dictionary("D1c", n_delays=2), train)
    _, h_pred = window_trajectory(model, test, start=9, horizon_steps=4)
    expected = model.simulate(test.h[7:10], test.u[9:13])
    assert np.allclose(h_pred, 100.0 * expected)


def test_window_trajectory_without_history_or_room_fails():
    """A window that starts before the delays of the model or runs past the segment end is an error."""
    _, _, train, test = make_runs()[0]
    model = fit_edmdc(make_dictionary("D1c", n_delays=2), train)
    with pytest.raises(ValueError):
        window_trajectory(model, test, start=1, horizon_steps=4)
    with pytest.raises(ValueError):
        window_trajectory(model, test, start=36, horizon_steps=4)
    window_trajectory(model, test, start=35, horizon_steps=4)


def make_shifted_runs():
    """Two runs whose evaluation parts start one sample apart in time, with their steps at the same absolute instants (150, 200 and 250 s)."""
    run_a = replace(make_dataset(40, (10, 20, 30), 2, "run_a_test"), t=100.0 + 5.0 * np.arange(40))
    run_b = replace(make_dataset(41, (11, 21, 31), 4, "run_b_test"), t=95.0 + 5.0 * np.arange(41))
    return [
        ("run_a", 30.0, make_dataset(80, (), 1, "run_a_train"), run_a),
        ("run_b", 60.0, make_dataset(80, (), 3, "run_b_train"), run_b),
    ]


def test_window_table_gives_the_absolute_start_time_of_each_window():
    """The windows of both runs start at 145, 195 and 245 s, although their local sample index differs by one."""
    windows = window_table(make_shifted_runs(), horizons_s=(10.0,), max_delays=2)
    assert windows["run"].tolist() == ["run_a", "run_b"]
    assert windows["n_windows"].tolist() == [3, 3]
    assert windows["starts"].tolist() == ["9;19;29", "10;20;30"]
    assert (windows["start_time_s"] == "145;195;245").all()
    assert (windows["sp_after_pct"] == windows["sp_after_pct"].iloc[0]).all()
    assert windows_shared_across_runs(windows) == {10.0: True}


def test_check_same_windows_accepts_every_series_on_the_table_windows_and_rejects_a_mismatch():
    """Every series scored by the sweep matches the windows of its run; a row with one window fewer is an error."""
    runs = make_shifted_runs()
    _, errors = sweep(runs, ("D0", "D1c"), hyperparams={"D1c": {"n_delays": 2}}, horizons_s=(10.0,))
    windows = window_table(runs, horizons_s=(10.0,), max_delays=2)
    check_same_windows(errors, windows)
    broken = errors.copy()
    broken.loc[broken.index[0], "n_windows"] = 2
    with pytest.raises(ValueError):
        check_same_windows(broken, windows)


def test_window_table_and_the_error_give_one_step_for_a_horizon_below_half_the_sampling_period():
    """A 2 s horizon with Ts = 5 s rounds to 0 steps but counts as 1 in both the windows table and the errors, which then match."""
    runs = make_shifted_runs()
    windows = window_table(runs, horizons_s=(2.0,), max_delays=2)
    _, errors = sweep(runs, ("D0", "D1c"), hyperparams={"D1c": {"n_delays": 2}}, horizons_s=(2.0,))
    assert windows["horizon_steps"].tolist() == [1, 1]
    assert (errors["horizon_steps"] == 1).all()
    check_same_windows(errors, windows)
