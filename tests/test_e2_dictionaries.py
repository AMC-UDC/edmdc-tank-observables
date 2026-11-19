"""Smoke test of the E2 script (comparison of dictionaries) on synthetic runs."""

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from edmdc_tank_observables.data.loader import RawDataset
from edmdc_tank_observables.data.preprocess import PreprocessedDataset
from edmdc_tank_observables.paths import PROJECT_ROOT


def load_script():
    """Import experiments/E2_dictionaries.py as a module, without running it."""
    path = PROJECT_ROOT / "experiments" / "E2_dictionaries.py"
    spec = importlib.util.spec_from_file_location("e2_dictionaries", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_dataset(n, step_samples, seed, name):
    """A synthetic run at Ts = 5 s with a positive level, a varying control and a setpoint that rises at the given samples."""
    rng = np.random.default_rng(seed)
    u = 0.5 + 0.3 * np.sin(0.4 * np.arange(n) + seed) + 0.02 * rng.standard_normal(n)
    h = [0.3]
    for k in range(n - 1):
        h.append(0.9 * h[-1] + 0.1 * u[k] + 0.04)
    sp = np.full(n, 0.6)
    for p in step_samples:
        sp[p:] += 0.05
    return PreprocessedDataset(
        h=np.array(h), u=u, sp=sp, t=5.0 * np.arange(n), ts=5.0,
        h_scale=(0.0, 100.0), u_scale=(0.0, 100.0), name=name,
    )


def make_runs():
    """Two synthetic runs whose evaluation part has two steps 130 samples apart, so a 600 s window fits after each."""
    return [
        (name, opening, make_dataset(400, (), seed, f"{name}_train"), make_dataset(300, (10, 140), seed + 1, f"{name}_test"))
        for name, opening, seed in (("Baseline_30pct", 30.0, 1), ("Baseline_50pct", 50.0, 3))
    ]


# Dates on which the synthetic runs were recorded.
SYNTHETIC_DATES = {"Baseline_30pct": "2025-10-15", "Baseline_50pct": "2025-10-17"}


# Values of the synthetic E1 metadata: on purpose different from those of the
# real E1 run, so a test passes only if they are read from the file.
SYNTHETIC_SELECTED = {"order": 4, "n_centers": 2, "n_delays": 5}


def write_e1_metadata(folder, selected=None, clean=True, commit="abc1234", warnings=(), drop=None):
    """Write a synthetic E1 metadata.json in folder with the given selected values and git state, and return its path."""
    values = dict(SYNTHETIC_SELECTED if selected is None else selected)
    families = {"D1a": "order", "D1b": "n_centers", "D1c": "n_delays", "D3a": "order", "D3b": "n_centers", "D3c": "n_delays"}
    chosen = {family: {name: values[name]} for family, name in families.items()}
    if drop is not None:
        del chosen[drop]
    metadata = {
        "script": "experiments/E1_hyperparameters.py",
        "git": {"commit": commit, "branch": "main", "clean": clean},
        "config": {},
        "summary": {"selected_hyperparameters": chosen, "upper_edge_warnings": list(warnings)},
    }
    path = folder / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def e2_output(tmp_path_factory):
    """Run the E2 script on two synthetic runs and three dictionaries, with a synthetic E1 metadata, writing into a temporary folder."""
    e2 = load_script()
    e1_metadata = write_e1_metadata(tmp_path_factory.mktemp("E1_hyperparameters"))
    out_dir = tmp_path_factory.mktemp("E2_dictionaries")
    models, errors = e2.run_experiment(
        make_runs(), out_dir, SYNTHETIC_DATES, dict_ids=("D0", "D1c", "D2"), e1_metadata=e1_metadata
    )
    return e2, out_dir, models, errors


def test_e2_writes_the_tables_the_figures_and_the_metadata(e2_output):
    """The script leaves its CSV tables, both figures in PDF and PNG and a metadata.json in the output folder."""
    _, out_dir, _, _ = e2_output
    tables = {
        "models", "errors", "error_summary", "improvements", "fit_indicators", "errors_by_run_120s", "windows",
        "level_ranges",
    }
    for name in tables:
        assert (out_dir / "tables" / f"{name}.csv").is_file()
    for name in ("error_vs_horizon", "trajectory"):
        for extension in ("pdf", "png"):
            assert (out_dir / "figures" / f"{name}.{extension}").is_file()
    assert (out_dir / "metadata.json").is_file()


def test_e2_evaluates_every_dictionary_once_per_run(e2_output):
    """Each run has one model per dictionary and one error row per dictionary and horizon."""
    _, _, models, errors = e2_output
    assert len(models) == 2 * 3
    assert len(errors) == 2 * 3 * 3
    assert set(errors["series"]) == {"D0", "D1c", "D2"}
    assert errors.groupby(["run", "series", "horizon_s"]).size().eq(1).all()
    assert (errors["n_windows"] == 2).all()


def test_e2_summary_table_has_one_row_per_series_and_horizon(e2_output):
    """The summary table has median, minimum and maximum for the three series at the three horizons, with readable names."""
    _, out_dir, _, errors = e2_output
    summary = pd.read_csv(out_dir / "tables" / "error_summary.csv")
    assert len(summary) == 3 * 3
    assert list(summary["series"].unique()) == ["D0", "D1c", "D2"]
    assert summary["name"].notna().all()
    row = summary[(summary["series"] == "D2") & (summary["horizon_s"] == 120.0)].iloc[0]
    by_run = errors[(errors["series"] == "D2") & (errors["horizon_s"] == 120.0)]["rmse"]
    assert row["median"] == pytest.approx(by_run.median())
    assert row["min"] == pytest.approx(by_run.min()) and row["max"] == pytest.approx(by_run.max())


def test_e2_improvements_compare_the_dictionaries_with_d0(e2_output):
    """The improvement counts are given against D0, over the two runs, for every other dictionary."""
    _, out_dir, _, _ = e2_output
    counts = pd.read_csv(out_dir / "tables" / "improvements.csv")
    assert list(counts.columns) == ["reference", "series", "name", "horizon_s", "n_better", "n_compared"]
    assert set(counts["reference"]) == {"D0"}
    assert set(counts["series"]) == {"D1c", "D2"}
    assert len(counts) == 2 * 3
    assert (counts["n_compared"] == 2).all()


def test_e2_by_run_table_has_one_row_per_run(e2_output):
    """The 120 s table has one row per run and one column per series."""
    _, out_dir, _, _ = e2_output
    by_run = pd.read_csv(out_dir / "tables" / "errors_by_run_120s.csv")
    assert by_run["run"].tolist() == ["Baseline_30pct", "Baseline_50pct"]
    assert list(by_run.columns) == ["run", "opening_pct", "D0", "D1c", "D2"]


def test_e2_metadata_records_the_configuration_used(e2_output):
    """The metadata keeps the dictionaries, horizons and runs actually used, and a summary of the results."""
    _, out_dir, _, _ = e2_output
    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    config = metadata["config"]
    assert config["dictionaries"] == ["D0", "D1c", "D2"]
    assert config["hyperparameters"] == {"D1c": {"n_delays": 5}}
    assert config["horizons_s"] == [30.0, 120.0, 600.0]
    assert config["runs"] == ["Baseline_30pct", "Baseline_50pct"]
    assert metadata["summary"]["n_runs"] == 2 and metadata["summary"]["n_models"] == 6


def test_e2_windows_table_gives_the_windows_of_each_run_and_horizon(e2_output):
    """The windows table has one row per run and horizon, with two windows at the same absolute instants in both runs."""
    _, out_dir, _, _ = e2_output
    windows = pd.read_csv(out_dir / "tables" / "windows.csv")
    assert len(windows) == 2 * 3
    assert (windows["n_windows"] == 2).all()
    assert windows["start_time_s"].nunique() == 1


# --- Hyperparameters read from the E1 metadata ---------------------------------


def test_e2_has_no_hyperparameter_constants():
    """The script defines no HYPERPARAMS constant of its own."""
    e2 = load_script()
    assert not hasattr(e2, "HYPERPARAMS")


def test_read_e1_hyperparameters_returns_the_three_selected_values_and_the_e1_commit(tmp_path):
    """The order, number of centres and number of delays are the ones written in the E1 metadata, with its commit."""
    e2 = load_script()
    values, commit = e2.read_e1_hyperparameters(write_e1_metadata(tmp_path))
    assert values == {"order": 4, "n_centers": 2, "n_delays": 5}
    assert commit == "abc1234"


def test_combined_dictionaries_take_the_value_of_their_generic_family():
    """Each D3 configuration gets the same value as its D1 family; D0 and D2 get none."""
    e2 = load_script()
    hyperparams = e2.dictionary_hyperparameters({"order": 4, "n_centers": 2, "n_delays": 5})
    assert hyperparams == {
        "D1a": {"order": 4}, "D1b": {"n_centers": 2}, "D1c": {"n_delays": 5},
        "D3a": {"order": 4}, "D3b": {"n_centers": 2}, "D3c": {"n_delays": 5},
    }


def test_read_e1_hyperparameters_fails_if_the_file_is_missing(tmp_path):
    """Without the E1 metadata there is no value to fall back on: the error says which file is missing."""
    e2 = load_script()
    with pytest.raises(FileNotFoundError, match="metadata.json"):
        e2.read_e1_hyperparameters(tmp_path / "metadata.json")


@pytest.mark.parametrize("family, name", [("D1a", "order"), ("D1b", "n_centers"), ("D1c", "n_delays")])
def test_read_e1_hyperparameters_fails_if_a_value_is_missing(tmp_path, family, name):
    """If the E1 metadata lacks the value of one family, the error names it."""
    e2 = load_script()
    with pytest.raises(ValueError, match=f"{family}.*{name}"):
        e2.read_e1_hyperparameters(write_e1_metadata(tmp_path, drop=family))


@pytest.mark.parametrize("bad", [0, -1, 2.5, "3", True, None])
def test_read_e1_hyperparameters_fails_if_a_value_is_not_a_positive_integer(tmp_path, bad):
    """Zero, negative, fractional, text, boolean or empty values are rejected."""
    e2 = load_script()
    path = write_e1_metadata(tmp_path, selected={"order": 2, "n_centers": bad, "n_delays": 1})
    with pytest.raises(ValueError, match="n_centers"):
        e2.read_e1_hyperparameters(path)


@pytest.mark.parametrize("clean", [False, "unknown"])
def test_read_e1_hyperparameters_fails_if_e1_did_not_run_from_a_clean_commit(tmp_path, clean):
    """Values from an E1 run with uncommitted changes, or an unknown git state, are not used."""
    e2 = load_script()
    with pytest.raises(ValueError, match="clean"):
        e2.read_e1_hyperparameters(write_e1_metadata(tmp_path, clean=clean))


def test_read_e1_hyperparameters_fails_if_e1_reported_an_upper_edge(tmp_path):
    """A value that E1 flagged at the upper end of its grid is not an optimum, so E2 does not use it."""
    e2 = load_script()
    path = write_e1_metadata(tmp_path, warnings=["D1c: the selected n_delays = 8 is the upper end of the grid 1-8"])
    with pytest.raises(ValueError, match="upper end"):
        e2.read_e1_hyperparameters(path)


def test_e2_metadata_records_where_the_hyperparameters_came_from(e2_output):
    """The E2 metadata keeps the E1 metadata file it read, the E1 commit in it and the three values, equal to those of E1."""
    _, out_dir, _, _ = e2_output
    source = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))["config"]["hyperparameter_source"]
    assert source["e1_metadata"].endswith("metadata.json")
    assert source["e1_commit"] == "abc1234"
    assert source["values"] == SYNTHETIC_SELECTED


# --- Level ranges --------------------------------------------------------------


def level_runs():
    """One synthetic run with known levels: fit part 10-30 %, evaluation part 40-95 %, steps at samples 3 and 9.

    With horizons of 2 and 3 steps the windows start at 2 and 8, and the
    compared samples are 3-5 and 9-11: levels 50, 60, 70 and 80, 85, 90. The
    evaluation part also has 40 % (before the first window) and 95 % (after
    the last one), which are outside every window.
    """
    def dataset(h, sp, name):
        h = np.asarray(h, dtype=float) / 100.0
        return PreprocessedDataset(
            h=h, u=np.zeros(len(h)), sp=np.asarray(sp, dtype=float), t=5.0 * np.arange(len(h)), ts=5.0,
            h_scale=(0.0, 100.0), u_scale=(0.0, 100.0), name=name,
        )

    train = dataset([10, 20, 30, 20], [0.1] * 4, "Baseline_30pct_train")
    h_eval = [40, 45, 45, 50, 60, 70, 75, 75, 75, 80, 85, 90, 95]
    sp_eval = [0.4] * 3 + [0.5] * 6 + [0.6] * 4
    return [("Baseline_30pct", 30.0, train, dataset(h_eval, sp_eval, "Baseline_30pct_test"))]


def test_level_ranges_give_the_fit_evaluation_and_window_ranges_of_each_run():
    """The fit and evaluation ranges cover their whole parts, and the window range only the samples compared at any horizon."""
    e2 = load_script()
    runs = level_runs()
    windows = e2.window_table(runs, (10.0, 15.0))
    table = e2.level_range_table(runs, windows, {"Baseline_30pct": "2025-10-15"})
    assert list(table.columns) == [
        "run", "opening_pct", "acquisition_date", "h_fit_min", "h_fit_max",
        "h_eval_min", "h_eval_max", "h_windows_min", "h_windows_max",
    ]
    row = table.iloc[0]
    assert (row["run"], row["opening_pct"], row["acquisition_date"]) == ("Baseline_30pct", 30.0, "2025-10-15")
    assert (row["h_fit_min"], row["h_fit_max"]) == pytest.approx((10.0, 30.0))
    assert (row["h_eval_min"], row["h_eval_max"]) == pytest.approx((40.0, 95.0))
    assert (row["h_windows_min"], row["h_windows_max"]) == pytest.approx((50.0, 90.0))


def test_level_ranges_need_the_date_of_every_run():
    """A run without an acquisition date is an error, not an empty cell."""
    e2 = load_script()
    runs = level_runs()
    with pytest.raises(ValueError, match="acquisition date"):
        e2.level_range_table(runs, e2.window_table(runs, (10.0,)), {})


def test_e2_writes_one_level_range_row_per_run_with_its_date(e2_output):
    """The level ranges table written by E2 has one row per run, its date, and windows inside the evaluation range."""
    _, out_dir, _, _ = e2_output
    table = pd.read_csv(out_dir / "tables" / "level_ranges.csv")
    assert table["run"].tolist() == ["Baseline_30pct", "Baseline_50pct"]
    assert table["acquisition_date"].tolist() == ["2025-10-15", "2025-10-17"]
    assert (table["h_eval_min"] <= table["h_windows_min"]).all()
    assert (table["h_windows_max"] <= table["h_eval_max"]).all()


def raw_run(timestamps):
    """A raw run with the given Timestamp values and nothing else needed."""
    df = pd.DataFrame({"Timestamp": timestamps})
    return RawDataset(name="Baseline_30pct", valve_opening_pct=30.0, df=df, n_raw_rows=len(df), n_sensor_faults=0)


def test_acquisition_date_is_read_from_the_timestamp_column():
    """A run whose timestamps all fall on 15 October 2025 is dated 2025-10-15."""
    e2 = load_script()
    run = raw_run(["2025/10/15 10:24:00", "2025/10/15 10:24:00", "2025/10/15 18:54:00"])
    assert e2.acquisition_date(run) == "2025-10-15"


def test_acquisition_date_of_a_run_over_two_dates_or_without_timestamps_fails():
    """A run that crosses midnight, or has no Timestamp column, has no single date and is an error."""
    e2 = load_script()
    with pytest.raises(ValueError, match="2 dates"):
        e2.acquisition_date(raw_run(["2025/10/15 23:59:00", "2025/10/16 00:00:00"]))
    no_timestamp = raw_run(["2025/10/15 10:24:00"])
    no_timestamp.df = no_timestamp.df.rename(columns={"Timestamp": "Time"})
    with pytest.raises(ValueError, match="Timestamp"):
        e2.acquisition_date(no_timestamp)
