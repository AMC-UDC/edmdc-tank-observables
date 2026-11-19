"""Tests of the one-step error of the E2 models on toy models and series whose answers can be worked out by hand."""

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from edmdc_tank_observables.data.preprocess import PreprocessedDataset
from edmdc_tank_observables.evaluation.protocol import coefficient_rows, sweep, window_table
from edmdc_tank_observables.models.edmdc import EDMDcModel, fit_edmdc
from edmdc_tank_observables.observables.dictionaries import make_dictionary
from edmdc_tank_observables.paths import PROJECT_ROOT


def load_script():
    """Import experiments/E2_one_step.py as a module, without running it."""
    path = PROJECT_ROOT / "experiments" / "E2_one_step.py"
    spec = importlib.util.spec_from_file_location("e2_one_step", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


one = load_script()

HYPERPARAMS = {"D1a": {"order": 2}, "D1b": {"n_centers": 1}, "D1c": {"n_delays": 1},
               "D3a": {"order": 2}, "D3b": {"n_centers": 1}, "D3c": {"n_delays": 1}}


def dataset(h, u):
    """Segment with the given normalized level and control, 1 s apart, on a 0-100 % scale."""
    h, u = np.asarray(h, dtype=float), np.asarray(u, dtype=float)
    return PreprocessedDataset(
        h=h, u=u, sp=np.zeros_like(h), t=np.arange(len(h), dtype=float), ts=1.0,
        h_scale=(0.0, 100.0), u_scale=(0.0, 100.0), name="toy",
    )


def toy_model(dict_id, A, B, c, **kwargs):
    """EDMDc model with the given matrices on a dictionary."""
    return EDMDcModel(A=np.array(A, dtype=float), B=np.array(B, dtype=float), c=np.array(c, dtype=float),
                      dictionary=make_dictionary(dict_id, **kwargs))


def first_order_series(n=200, seed=0):
    """A level that follows a noisy first-order response to a random control, kept inside (0, 1)."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.2, 0.8, n)
    h = np.empty(n)
    h[0] = 0.5
    for k in range(n - 1):
        h[k + 1] = 0.9 * h[k] + 0.1 * u[k] + 0.01 * rng.standard_normal()
    return dataset(np.clip(h, 0.05, 0.95), u)


@pytest.mark.parametrize("dict_id", ["D0", "D1a", "D1b", "D1c", "D2", "D3a", "D3b", "D3c"])
def test_first_instant_of_each_window_is_the_same_one_step_and_multi_step(dict_id):
    """At the first instant of a window both predictions start from the same measured history, so they coincide."""
    data = first_order_series()
    model = fit_edmdc(make_dictionary(dict_id, **HYPERPARAMS.get(dict_id, {})), data)
    n_delays = model.dictionary.n_delays
    for i in (10, 50, 120):
        one_step = one.window_one_step_errors(model, data, [i], 6)[0]
        multi_step = model.simulate(data.h[i - n_delays : i + 1], data.u[i : i + 1])[0] - data.h[i + 1]
        assert one_step == pytest.approx(data.denormalize_h_delta(multi_step), rel=1e-12)


def test_window_errors_by_hand():
    """With h(k+1) = 0.5 h(k) + u(k), window at 1 of 2 steps: predictions 0.2 and 0.35 against 0.3 and 0.5, errors -10 and -15 %."""
    data = dataset([0.2, 0.4, 0.3, 0.5, 0.1], [0.1, 0.0, 0.2, 0.0, 0.0])
    model = toy_model("D0", [[0.5]], [[1.0]], [0.0])
    np.testing.assert_allclose(one.window_one_step_errors(model, data, [1], 2), [-10.0, -15.0])


def test_errors_of_several_windows_are_pooled_window_after_window():
    """Two windows of two steps give four errors, those of the first window first."""
    data = dataset([0.2, 0.4, 0.3, 0.5, 0.1], [0.1, 0.0, 0.2, 0.0, 0.0])
    model = toy_model("D0", [[0.5]], [[1.0]], [0.0])
    # Window at 0: predictions 0.2 and 0.2 against 0.4 and 0.3.
    np.testing.assert_allclose(one.window_one_step_errors(model, data, [0, 1], 2), [-20.0, -10.0, -10.0, -15.0])


def test_one_step_errors_over_a_whole_segment_by_hand():
    """With h(k+1) = 0.5 h(k) + u(k) the four pairs of the toy segment give errors 0, -20, -15 and +25 %."""
    data = dataset([0.2, 0.2, 0.3, 0.5, 0.0], [0.1, 0.0, 0.2, 0.0, 0.0])
    model = toy_model("D0", [[0.5]], [[1.0]], [0.0])
    # Predictions 0.2, 0.1, 0.35, 0.25 against the measured 0.2, 0.3, 0.5, 0.0.
    np.testing.assert_allclose(one.one_step_errors(model, data), [0.0, -20.0, -15.0, 25.0], atol=1e-12)


def test_delayed_level_comes_from_the_measurement_not_from_a_prediction():
    """A D1c model that predicts h(k+1) = h(k-1) gives the measured h(k-1) at every instant; open-loop simulation does not."""
    h = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    data = dataset(h, np.zeros(6))
    # Lifted state [h(k), h(k-1)]; A swaps the two, so simulation feeds back its own predictions.
    model = toy_model("D1c", [[0.0, 1.0], [1.0, 0.0]], [[0.0], [0.0]], [0.0, 0.0], n_delays=1)
    errors = one.window_one_step_errors(model, data, [2], 3)
    # Instants 3, 4, 5 predicted as the measured h(2 - 1), h(3 - 1), h(4 - 1).
    np.testing.assert_allclose(errors, data.denormalize_h_delta(np.array([0.2, 0.3, 0.4]) - np.array([0.4, 0.5, 0.6])))
    simulated = model.simulate(np.array([0.2, 0.3]), np.zeros(3))
    np.testing.assert_allclose(simulated, [0.2, 0.3, 0.2])


def errors_table():
    """One-step errors of A and B in three runs, at fit and in the window segment at 120 s."""
    rows = []
    rmse = {("fit", np.nan): {"A": [1.0, 1.0, 1.0], "B": [2.0, 2.0, 2.0]},
            ("window", 120.0): {"A": [1.0, 2.0, 3.0], "B": [2.0, 2.0, 1.0]}}
    for (segment, horizon_s), by_dict in rmse.items():
        for d, values in by_dict.items():
            for run, value in zip(("r1", "r2", "r3"), values):
                rows.append({"run": run, "opening_pct": 10.0, "dictionary": d, "segment": segment,
                             "horizon_s": horizon_s, "n_errors": 5, "rmse": value})
    return pd.DataFrame(rows)


def multistep_table():
    """Multi-step errors of E2 for A and B in the same runs at 120 s: A is lower in all three, by 1."""
    return pd.DataFrame([{"run": run, "series": d, "horizon_s": 120.0, "rmse": value}
                         for d, value in (("A", 1.0), ("B", 2.0)) for run in ("r1", "r2", "r3")])


def test_pair_counts_and_median_difference_with_multi_step_alongside():
    """In the window at 120 s, A - B is -1, 0 and +2: one win each, one exact tie, median 0; multi-step 3/0, median -1."""
    counts = one.pair_counts(errors_table(), [("A", "B")], multistep_table())
    window = counts[counts["segment"] == "window"].iloc[0]
    assert (window["n_a_lower"], window["n_b_lower"], window["n_ties"], window["n_runs"]) == (1, 1, 1, 3)
    assert window["median_diff"] == 0.0
    assert (window["multistep_n_a_lower"], window["multistep_n_b_lower"], window["multistep_median_diff"]) == (3, 0, -1.0)
    fit = counts[counts["segment"] == "fit"].iloc[0]
    assert (fit["n_a_lower"], fit["n_b_lower"], fit["median_diff"]) == (3, 0, -1.0)
    assert pd.isna(fit["horizon_s"]) and pd.isna(fit["multistep_n_a_lower"])


def test_e2_metadata_must_be_clean_and_have_the_coefficients_table(tmp_path):
    """E2 results not produced from a clean commit, or whose metadata lacks coefficients_table, are rejected."""
    path = tmp_path / "metadata.json"
    config = {"coefficients_table": "tables/coefficients.csv"}
    path.write_text(json.dumps({"git": {"commit": "abc1234", "clean": False}, "config": config}), encoding="utf-8")
    with pytest.raises(ValueError, match="clean"):
        one.read_e2_metadata(path)
    path.write_text(json.dumps({"git": {"commit": "abc1234", "clean": True}, "config": {"x": 1}}), encoding="utf-8")
    with pytest.raises(ValueError, match="coefficients_table"):
        one.read_e2_metadata(path)
    path.write_text(json.dumps({"git": {"commit": "abc1234", "clean": True}, "config": config}), encoding="utf-8")
    assert one.read_e2_metadata(path) == ("abc1234", config)


def test_fit_rmse_different_from_e2_is_rejected():
    """An in-sample RMSE that differs from fit_rmse_pct of E2 beyond 1e-10 relative, another number of errors, or a missing model, is rejected."""
    errors = errors_table()
    models = pd.DataFrame([{"run": r, "dictionary": d, "fit_rmse_pct": v, "n_pairs": 5}
                           for d, v in (("A", 1.0), ("B", 2.0)) for r in ("r1", "r2", "r3")])
    one.check_fit_rmse(errors, models)
    off = models.copy()
    off.loc[0, "fit_rmse_pct"] = 1.0 + 1e-8
    with pytest.raises(ValueError, match="differs from fit_rmse_pct"):
        one.check_fit_rmse(errors, off)
    with pytest.raises(ValueError, match="n_pairs"):
        one.check_fit_rmse(errors, models.assign(n_pairs=6))
    with pytest.raises(ValueError, match="do not match"):
        one.check_fit_rmse(errors, models.iloc[1:])


ALL_DICTIONARIES = ["D0", "D1a", "D1b", "D1c", "D2", "D3a", "D3b", "D3c"]


@pytest.mark.parametrize("dict_id", ALL_DICTIONARIES)
def test_models_read_back_from_a_coefficients_csv_give_exactly_the_same_one_step_errors(dict_id, tmp_path):
    """A fitted model written to a coefficients CSV and read back with round-trip precision gives the same one-step errors, bit for bit."""
    data = first_order_series()
    fitted = fit_edmdc(make_dictionary(dict_id, **HYPERPARAMS.get(dict_id, {})), data)
    pd.DataFrame(coefficient_rows(fitted, "toy", 10.0)).to_csv(tmp_path / "coefficients.csv", index=False)
    table = pd.read_csv(tmp_path / "coefficients.csv", float_precision="round_trip")
    model = one.read_models(table, [("toy", 10.0, data, data)], [dict_id], HYPERPARAMS)[("toy", dict_id)]
    assert np.array_equal(one.one_step_errors(model, data), one.one_step_errors(fitted, data))
    assert np.array_equal(one.window_one_step_errors(model, data, [10, 50], 6),
                          one.window_one_step_errors(fitted, data, [10, 50], 6))


def test_a_model_missing_from_the_coefficients_table_is_rejected():
    """If the coefficients of a run and dictionary are not in the table, reading the models fails."""
    data = first_order_series()
    fitted = fit_edmdc(make_dictionary("D0"), data)
    table = pd.DataFrame(coefficient_rows(fitted, "toy", 10.0))
    runs = [("toy", 10.0, data, data), ("other", 20.0, data, data)]
    with pytest.raises(ValueError, match="missing"):
        one.read_models(table, runs, ["D0"], {})
    with pytest.raises(ValueError, match="missing"):
        one.read_models(table, runs[:1], ["D0", "D2"], {})


def synthetic_dataset(n, step_samples, seed, name):
    """A synthetic run at Ts = 5 s with a positive level, a varying control and a setpoint that rises at the given samples."""
    rng = np.random.default_rng(seed)
    u = 0.5 + 0.3 * np.sin(0.4 * np.arange(n) + seed) + 0.02 * rng.standard_normal(n)
    h = [0.3]
    for k in range(n - 1):
        h.append(0.9 * h[-1] + 0.1 * u[k] + 0.04 + 0.002 * rng.standard_normal())
    sp = np.full(n, 0.6)
    for p in step_samples:
        sp[p:] += 0.05
    return PreprocessedDataset(
        h=np.array(h), u=u, sp=sp, t=5.0 * np.arange(n), ts=5.0,
        h_scale=(0.0, 100.0), u_scale=(0.0, 100.0), name=name,
    )


def synthetic_runs():
    """Two synthetic runs whose evaluation part has two steps 130 samples apart, so a 600 s window fits after each."""
    return [
        (name, opening, synthetic_dataset(400, (), seed, f"{name}_train"),
         synthetic_dataset(300, (10, 140), seed + 1, f"{name}_test"))
        for name, opening, seed in (("run_a", 30.0, 1), ("run_b", 50.0, 3))
    ]


def write_e2_results(folder, runs, fit_rmse_factor=1.0):
    """Write synthetic E2 results for the eight dictionaries in folder, with fit_rmse_pct of the models table multiplied by fit_rmse_factor."""
    models, errors, coefficients = sweep(
        runs, ALL_DICTIONARIES, hyperparams=HYPERPARAMS, horizons_s=one.HORIZONS_S, keep_coefficients=True
    )
    models["fit_rmse_pct"] = models["fit_rmse_pct"] * fit_rmse_factor
    tables = folder / "tables"
    tables.mkdir(parents=True)
    models.to_csv(tables / "models.csv", index=False)
    errors.to_csv(tables / "errors.csv", index=False)
    coefficients.to_csv(tables / "coefficients.csv", index=False)
    window_table(runs, one.HORIZONS_S, max_delays=1).to_csv(tables / "windows.csv", index=False)
    config = {"hyperparameters": HYPERPARAMS, "coefficients_table": "tables/coefficients.csv"}
    metadata = {"git": {"commit": "abc1234", "clean": True}, "config": config}
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return folder


def test_fit_rows_written_are_those_of_the_e2_models_table(tmp_path):
    """With fit_rmse_pct of E2 off by 1e-13 relative the check passes, and the fit rows written hold the E2 values, not the recomputed ones."""
    runs = synthetic_runs()
    e2_dir = write_e2_results(tmp_path / "E2", runs, fit_rmse_factor=1.0 + 1e-13)
    one.run(runs, out_dir=tmp_path / "out", e2_dir=e2_dir)

    written = pd.read_csv(tmp_path / "out" / "one_step_errors.csv", float_precision="round_trip")
    e2_models = pd.read_csv(e2_dir / "tables" / "models.csv", float_precision="round_trip")
    fit = written[written["segment"] == "fit"].merge(e2_models, on=["run", "dictionary"])
    assert len(fit) == 2 * 8
    assert (fit["rmse"] == fit["fit_rmse_pct"]).all()
    assert (fit["n_errors"] == fit["n_pairs"]).all()

    # The recomputed value is a different float: what was written is not it.
    _, _, train, _ = runs[0]
    model = fit_edmdc(make_dictionary("D0"), train)
    recomputed = float(np.sqrt(np.mean(one.one_step_errors(model, train) ** 2)))
    written_d0 = fit[(fit["run"] == "run_a") & (fit["dictionary"] == "D0")]["rmse"].item()
    assert written_d0 != recomputed and written_d0 == pytest.approx(recomputed, rel=1e-12)

    metadata = json.loads((tmp_path / "out" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["config"]["e2_commit"] == "abc1234"
    assert "copied from E2 models.csv" in metadata["config"]["fit_rows"]


def test_run_stops_when_the_e2_fit_rmse_differs_beyond_the_tolerance(tmp_path):
    """With fit_rmse_pct of E2 off by 1e-8 relative the script stops before writing anything."""
    runs = synthetic_runs()
    e2_dir = write_e2_results(tmp_path / "E2", runs, fit_rmse_factor=1.0 + 1e-8)
    with pytest.raises(ValueError, match="differs from fit_rmse_pct"):
        one.run(runs, out_dir=tmp_path / "out", e2_dir=e2_dir)
    assert not (tmp_path / "out").exists()


def test_windows_different_from_e2_are_rejected():
    """Window starts that differ from those of the E2 windows table are rejected."""
    windows = pd.DataFrame({"run": ["r1"], "horizon_s": [120.0], "horizon_steps": [24], "starts": ["35;395"]})
    one.check_windows(windows, windows.copy())
    with pytest.raises(ValueError, match="differ"):
        one.check_windows(windows, windows.assign(starts="35;396"))
