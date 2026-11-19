"""E2 - Comparison of the eight dictionary configurations.

On each run of the Baseline subset one model per dictionary is fitted on the
fit part (the first TRAIN_FRAC of the acquisition time, cut at the same instant
in every run) and scored on the evaluation part of the same run, with the
multi-step error on the windows anchored at the setpoint steps. One model is
fitted per run because the electrovalve opening is constant within a run, so it
cannot be an input of the model.

The runs were recorded on different days, with the same setpoint profile and
a different electrovalve opening. Here they are summarized with the median and
the range, and the regularity of the result is reported by counting in how
many runs each dictionary improves on D0. The paired tests are in
E2_comparisons.py, with descriptive p-values because the independence between
runs cannot be verified.

The hyperparameters (order, number of RBF centres, number of delays) are read
from the metadata.json that E1 writes, not set in this script; E2 stops if
that file is missing, incomplete, flags a value at the upper end of its grid
or was not produced from a clean commit. The file read, its E1 commit and the
values used are recorded in the metadata of E2.

The windows of each run and horizon are saved with their absolute start time,
and every series is checked to be scored on them. The coefficients A, B and c
of every fitted model are saved too, so later analyses can read the models
instead of fitting them again.

The level range of each run is saved too: the minimum and maximum level of the
fit part, of the whole evaluation part and of the samples compared in the
evaluation windows, with the date on which the run was recorded.

Outputs in results/E2_dictionaries/: CSV tables in tables/, figures in
figures/ and metadata.json.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window: the script only writes files

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from edmdc_tank_observables.data.loader import RawDataset, load_baseline
from edmdc_tank_observables.data.preprocess import common_end_time, prepare, temporal_split
from edmdc_tank_observables.evaluation.metrics import window_starts
from edmdc_tank_observables.evaluation.protocol import (
    check_same_windows,
    count_improvements,
    summarize_across_runs,
    sweep,
    window_table,
    window_trajectory,
    windows_shared_across_runs,
)
from edmdc_tank_observables.models.edmdc import fit_edmdc
from edmdc_tank_observables.observables.dictionaries import make_dictionary
from edmdc_tank_observables.paths import PROJECT_ROOT
from edmdc_tank_observables.provenance import write_metadata
from edmdc_tank_observables.reporting import save_figure, set_style

# --- Configuration -------------------------------------------------------------

SUBSET = "Baseline"
TS_S = 5.0
TRAIN_FRAC = 0.7

DICTIONARIES = ("D0", "D1a", "D1b", "D1c", "D2", "D3a", "D3b", "D3c")

# The hyperparameters are not set here: they are read from the metadata that
# E1 writes when it selects them. Each combined configuration (D3x) takes the
# value of its generic family (D1x), so that D1x versus D3x measures the effect
# of adding the physical observable and not that of retuning the
# hyperparameter.
E1_METADATA = PROJECT_ROOT / "results" / "E1_hyperparameters" / "metadata.json"

# Hyperparameter of each generic family, and the configurations that use it.
FAMILY_HYPERPARAMETERS = {"D1a": "order", "D1b": "n_centers", "D1c": "n_delays"}
USES_VALUE_OF = {"D1a": "D1a", "D1b": "D1b", "D1c": "D1c", "D3a": "D1a", "D3b": "D1b", "D3c": "D1c"}

# Prediction horizons, in seconds: from the pure transient after a step to a
# window that is mostly settled.
HORIZONS_S = (30.0, 120.0, 600.0)

# Horizon of the run-by-run table.
BY_RUN_HORIZON_S = 120.0

# Run, horizon and dictionaries of the trajectory figure. The window is the
# one anchored at the first setpoint step of the evaluation part.
FIGURE_RUN = "Baseline_50pct"
FIGURE_HORIZON_S = 600.0
FIGURE_DICTIONARIES = ("D0", "D2")

SERIES_NAMES = {
    "D0": "State only",
    "D1a": "Polynomial",
    "D1b": "RBF",
    "D1c": "Delays",
    "D2": "Physical",
    "D3a": "Polynomial + physical",
    "D3b": "RBF + physical",
    "D3c": "Delays + physical",
}

# Black is kept for the measured level in the trajectory figure.
COLORS = {
    "D0": "#0072B2",
    "D1a": "#009E73",
    "D1b": "#56B4E9",
    "D1c": "#CC79A7",
    "D2": "#D55E00",
    "D3a": "#117733",
    "D3b": "#E69F00",
    "D3c": "#7F3F98",
}

OUT_DIR = PROJECT_ROOT / "results" / "E2_dictionaries"


# --- Hyperparameters -----------------------------------------------------------


def read_e1_hyperparameters(path) -> tuple[dict, str]:
    """Read and validate the hyperparameters selected by E1 from its metadata.json.

    Only the values of the generic families are read (D1a order, D1b
    n_centers, D1c n_delays); the combined configurations take them in
    dictionary_hyperparameters.

    Parameters
    ----------
    path : pathlib.Path or str
        The metadata.json written by E1.

    Returns
    -------
    values : dict
        {"order": int, "n_centers": int, "n_delays": int}.
    e1_commit : str
        Commit from which E1 was run, as recorded in the file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If E1 was not run from a clean commit (git.clean is not true), if it
        flagged a selection at the upper end of a grid, if a value is missing
        from summary.selected_hyperparameters, or if a value is not a positive
        integer.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"The E1 metadata {path} does not exist: run E1 first.")
    metadata = json.loads(path.read_text(encoding="utf-8"))

    git = metadata.get("git", {})
    if git.get("clean") is not True:
        raise ValueError(f"{path}: E1 was not run from a clean commit (git.clean = {git.get('clean')!r}).")
    warnings = metadata.get("summary", {}).get("upper_edge_warnings")
    if warnings:
        raise ValueError(f"{path}: E1 selected values at the upper end of their grid, not optima: {warnings}.")

    selected = metadata.get("summary", {}).get("selected_hyperparameters", {})
    values = {}
    for family, name in FAMILY_HYPERPARAMETERS.items():
        value = selected.get(family, {}).get(name)
        if value is None:
            raise ValueError(f"{path}: summary.selected_hyperparameters.{family}.{name} is missing.")
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{path}: {family} {name} = {value!r} is not a positive integer.")
        values[name] = value
    return values, str(git.get("commit"))


def dictionary_hyperparameters(values: dict) -> dict:
    """Keyword arguments of make_dictionary for each configuration that has a hyperparameter.

    Parameters
    ----------
    values : dict
        {"order": int, "n_centers": int, "n_delays": int}, as returned by
        read_e1_hyperparameters.

    Returns
    -------
    dict
        Configuration id -> {name: value}, for D1a/b/c and D3a/b/c; each D3x
        gets the value of its D1x family.
    """
    return {
        dict_id: {FAMILY_HYPERPARAMETERS[family]: values[FAMILY_HYPERPARAMETERS[family]]}
        for dict_id, family in USES_VALUE_OF.items()
    }


def print_hyperparameters(values, e1_metadata, e1_commit, hyperparams, dict_ids) -> None:
    """Print the hyperparameters read, their source and the number of observables p of each dictionary."""
    print(f"Hyperparameters read from {e1_metadata} (E1 commit {e1_commit}):")
    print("  " + ", ".join(f"{name} = {value}" for name, value in values.items()))
    print("Number of observables p per dictionary:")
    for dict_id in dict_ids:
        p = make_dictionary(dict_id, **hyperparams.get(dict_id, {})).dimension
        print(f"  {dict_id:<4} p = {p:<3} {hyperparams.get(dict_id, '')}")


def _relative_to_root(path) -> str:
    """A path as text, relative to the repository root when it is inside it, so the metadata holds no absolute path."""
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


# --- Data and invariants -------------------------------------------------------


def load_runs() -> list[tuple]:
    """Load, preprocess and split every Baseline run.

    Returns
    -------
    list of tuple
        One tuple (name, opening_pct, train, test) per run, from lowest to
        highest opening, as expected by sweep.

    Raises
    ------
    ValueError
        If a run has gaps after resampling to TS_S (see prepare), or if the
        runs do not end at the same instant (see common_end_time), so they
        would not be cut at the same instant.
    """
    raws = load_baseline()
    datasets = [prepare(raw, TS_S) for raw in raws]
    common_end_time(datasets)
    runs = []
    for raw, dataset in zip(raws, datasets):
        train, test = temporal_split(dataset, TRAIN_FRAC)
        runs.append((raw.name, raw.valve_opening_pct, train, test))
    return runs


def acquisition_date(raw: RawDataset) -> str:
    """Date on which a run was recorded, read from its Timestamp column.

    Parameters
    ----------
    raw : RawDataset
        The run as loaded from its CSV, with a Timestamp column in the format
        "YYYY/MM/DD HH:MM:SS".

    Returns
    -------
    str
        The date in ISO format, "YYYY-MM-DD".

    Raises
    ------
    ValueError
        If the run has no Timestamp column, if a value does not follow the
        format, or if the run spans more than one date.
    """
    if "Timestamp" not in raw.df.columns:
        raise ValueError(f"{raw.name}: no Timestamp column.")
    dates = pd.to_datetime(raw.df["Timestamp"], format="%Y/%m/%d %H:%M:%S").dt.date.unique()
    if len(dates) != 1:
        raise ValueError(f"{raw.name}: the run spans {len(dates)} dates, {sorted(str(d) for d in dates)}.")
    return dates[0].isoformat()


def load_acquisition_dates() -> dict:
    """Read the date on which each Baseline run was recorded.

    Returns
    -------
    dict
        Run name -> date in ISO format, see acquisition_date.

    Raises
    ------
    ValueError
        If a run has no valid single date; see acquisition_date.
    """
    return {raw.name: acquisition_date(raw) for raw in load_baseline()}


def print_level_ranges(runs) -> None:
    """Print the minimum and maximum level of the fit and evaluation parts of each run, in %.

    Parameters
    ----------
    runs : list of tuple
        Runs as returned by load_runs.
    """
    print("\nLevel range (% of level), fit part vs evaluation part")
    fit_all, eval_all = [], []
    for name, _, train, test in runs:
        h_fit, h_eval = train.denormalize_h(train.h), test.denormalize_h(test.h)
        fit_all.append(h_fit)
        eval_all.append(h_eval)
        print(
            f"  {name:<16} fit h [{h_fit.min():6.2f}, {h_fit.max():6.2f}]"
            f"   eval h [{h_eval.min():6.2f}, {h_eval.max():6.2f}]"
        )
    fit_all, eval_all = np.concatenate(fit_all), np.concatenate(eval_all)
    print(
        f"  {'all runs':<16} fit h [{fit_all.min():6.2f}, {fit_all.max():6.2f}]"
        f"   eval h [{eval_all.min():6.2f}, {eval_all.max():6.2f}]"
    )


# --- Tables --------------------------------------------------------------------


def series_order(series) -> list[str]:
    """Sort series ids in DICTIONARIES order."""
    return sorted(series, key=DICTIONARIES.index)


def sort_by_series(table: pd.DataFrame, column: str = "series", then: tuple = ()) -> pd.DataFrame:
    """Sort a table by its series column in series_order, then by the columns in `then`."""
    return table.sort_values(
        [column, *then], key=lambda c: c.map(DICTIONARIES.index) if c.name == column else c
    ).reset_index(drop=True)


def error_summary_table(errors: pd.DataFrame) -> pd.DataFrame:
    """RMSE per series and horizon across runs: median, minimum and maximum."""
    summary = summarize_across_runs(errors)
    summary.insert(1, "name", summary["series"].map(SERIES_NAMES))
    return sort_by_series(summary, then=("horizon_s",))


def improvements_table(errors: pd.DataFrame) -> pd.DataFrame:
    """Runs in which each dictionary has a lower RMSE than D0."""
    counts = count_improvements(errors, reference="D0")
    counts.insert(0, "reference", "D0")
    counts.insert(2, "name", counts["series"].map(SERIES_NAMES))
    return sort_by_series(counts, then=("horizon_s",))


def fit_table(models: pd.DataFrame) -> pd.DataFrame:
    """Fit indicators per dictionary: dimension and median across runs of the others."""
    table = (
        models.groupby("dictionary")
        .agg(
            p=("p", "first"),
            cond=("cond", "median"),
            spectral_radius=("spectral_radius", "median"),
            fit_rmse_pct=("fit_rmse_pct", "median"),
            n_pairs=("n_pairs", "median"),
        )
        .reset_index()
    )
    table.insert(1, "name", table["dictionary"].map(SERIES_NAMES))
    return sort_by_series(table, column="dictionary")


def by_run_table(errors: pd.DataFrame, horizon_s: float) -> pd.DataFrame:
    """RMSE of every series in every run at one horizon, one row per run."""
    selected = errors[errors["horizon_s"] == horizon_s]
    table = selected.pivot_table(index=["run", "opening_pct"], columns="series", values="rmse")
    table = table[series_order(table.columns)].reset_index()
    table.columns.name = None
    return table.sort_values("opening_pct").reset_index(drop=True)


def level_range_table(runs, windows: pd.DataFrame, acquisition_dates: dict) -> pd.DataFrame:
    """Level range of the fit part, the evaluation part and the evaluation windows of each run, in % of level.

    Parameters
    ----------
    runs : list of tuple
        Runs as returned by load_runs.
    windows : pandas.DataFrame
        Windows table of the runs, as returned by window_table: the windows
        the series are scored on.
    acquisition_dates : dict
        Run name -> date on which it was recorded, as returned by
        load_acquisition_dates.

    Returns
    -------
    pandas.DataFrame
        One row per run with run, opening_pct, acquisition_date, and the
        minimum and maximum level of the fit part (h_fit_min, h_fit_max), of
        the whole evaluation part (h_eval_min, h_eval_max) and of every sample
        compared in a window at any horizon of the windows table, that is
        h[X + 1 : X + 1 + H] for each start X and horizon H in steps
        (h_windows_min, h_windows_max).

    Raises
    ------
    ValueError
        If a run has no date in acquisition_dates or no window in the windows
        table.
    """
    rows = []
    for name, opening_pct, train, test in runs:
        if name not in acquisition_dates:
            raise ValueError(f"{name}: no acquisition date.")
        compared = []
        for _, row in windows[windows["run"] == name].iterrows():
            steps = int(row["horizon_steps"])
            for x in str(row["starts"]).split(";"):
                if x:
                    compared.append(test.h[int(x) + 1 : int(x) + 1 + steps])
        if not compared:
            raise ValueError(f"{name}: no evaluation window in the windows table.")
        h_fit, h_eval = train.denormalize_h(train.h), test.denormalize_h(test.h)
        h_windows = test.denormalize_h(np.concatenate(compared))
        rows.append(
            {
                "run": name,
                "opening_pct": opening_pct,
                "acquisition_date": acquisition_dates[name],
                "h_fit_min": h_fit.min(),
                "h_fit_max": h_fit.max(),
                "h_eval_min": h_eval.min(),
                "h_eval_max": h_eval.max(),
                "h_windows_min": h_windows.min(),
                "h_windows_max": h_windows.max(),
            }
        )
    return pd.DataFrame(rows)


def print_tables(summary, improvements, fit, by_run, windows) -> None:
    """Print the tables of the experiment to the console."""
    print("\n" + "=" * 78)
    print("E2 - Comparison of dictionaries")
    print("=" * 78)

    print("\nWindows of the evaluation part (run x horizon)")
    print(windows.to_string(index=False))
    same = windows_shared_across_runs(windows)
    print("Same window start times in every run: " + ", ".join(f"{h:g} s: {v}" for h, v in same.items()))

    print("\nRMSE (% of level), median across runs [min, max]")
    cells = summary.assign(
        cell=summary.apply(lambda r: f"{r['median']:.2f} [{r['min']:.2f}, {r['max']:.2f}]", axis=1)
    )
    table = cells.pivot(index="series", columns="horizon_s", values="cell")
    table = table.loc[series_order(table.index)]
    table.insert(0, "name", [SERIES_NAMES[s] for s in table.index])
    print(table.to_string())

    print("\nRuns in which the series has a lower RMSE than D0")
    cells = improvements.assign(cell=improvements["n_better"].astype(str) + "/" + improvements["n_compared"].astype(str))
    table = cells.pivot(index="series", columns="horizon_s", values="cell")
    print(table.loc[series_order(table.index)].to_string())

    print("\nFit indicators (median across runs)")
    print(fit.to_string(index=False))

    print(f"\nRMSE (% of level) by run at {BY_RUN_HORIZON_S:g} s")
    print(by_run.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print()


# --- Figures -------------------------------------------------------------------


def figure_error_vs_horizon(summary: pd.DataFrame) -> plt.Figure:
    """Median RMSE against the horizon, one line per series."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for series in series_order(summary["series"].unique()):
        block = summary[summary["series"] == series].sort_values("horizon_s")
        ax.plot(
            block["horizon_s"],
            block["median"],
            marker="o",
            markersize=4,
            color=COLORS[series],
            linewidth=2.0 if series in ("D0", "D2") else 1.2,
            label=series,
        )
    horizons = sorted(summary["horizon_s"].unique())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(horizons)
    ax.set_xticklabels([f"{h:g}" for h in horizons])
    ax.set_xlabel("Prediction horizon (s)")
    ax.set_ylabel("RMSE (% of level)")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def figure_trajectory(runs, hyperparams) -> plt.Figure:
    """Measured and predicted level over the window of the first setpoint step of FIGURE_RUN.

    Parameters
    ----------
    runs : list of tuple
        Runs as returned by load_runs.
    hyperparams : dict
        Hyperparameters per configuration, as returned by
        dictionary_hyperparameters.

    Raises
    ------
    ValueError
        If FIGURE_RUN is not among the runs, or if its window is not valid.
    """
    matches = [r for r in runs if r[0] == FIGURE_RUN]
    if not matches:
        raise ValueError(f"The figure run {FIGURE_RUN!r} is not among the runs.")
    _, _, train, test = matches[0]

    horizon_steps = int(round(FIGURE_HORIZON_S / test.ts))
    start = int(window_starts(test, horizon_steps)[0])
    window = slice(start + 1, start + 1 + horizon_steps)
    t = np.arange(1, horizon_steps + 1) * test.ts

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(t, test.denormalize_h(test.sp[window]), color="#000000", linewidth=0.8, linestyle=":", label="Setpoint")
    ax.plot(t, test.denormalize_h(test.h[window]), color="#000000", linewidth=2.0, label="Measured level")
    for dict_id in FIGURE_DICTIONARIES:
        model = fit_edmdc(make_dictionary(dict_id, **hyperparams.get(dict_id, {})), train)
        t_pred, h_pred = window_trajectory(model, test, start, horizon_steps)
        ax.plot(t_pred, h_pred, color=COLORS[dict_id], linewidth=1.6, label=f"{dict_id} ({SERIES_NAMES[dict_id].lower()})")

    ax.set_xlabel("Time from the window start (s)")
    ax.set_ylabel("Level (%)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# --- Run -----------------------------------------------------------------------


def run_experiment(
    runs, out_dir, acquisition_dates, dict_ids=DICTIONARIES, e1_metadata=E1_METADATA
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit and score the dictionaries on the runs and write tables (model coefficients included), figures and metadata.

    Parameters
    ----------
    runs : list of tuple
        Runs as returned by load_runs. FIGURE_RUN must be among them.
    out_dir : pathlib.Path or str
        Output folder. Tables go to out_dir/tables, figures to out_dir/figures.
    acquisition_dates : dict
        Run name -> date on which it was recorded, as returned by
        load_acquisition_dates. Every run must have one.
    dict_ids : iterable of str, default DICTIONARIES
        Dictionary configurations to compare.
    e1_metadata : pathlib.Path or str, default E1_METADATA
        The metadata.json of E1 the hyperparameters are read from.

    Returns
    -------
    models, errors : pandas.DataFrame
        The models and errors tables returned by sweep.

    Raises
    ------
    FileNotFoundError
        If the E1 metadata does not exist.
    ValueError
        If the E1 metadata is not valid (see read_e1_hyperparameters), if a
        dictionary cannot be fitted, if the windows are not valid for some
        run, if some series was not scored on the windows of the windows
        table, if a run has no acquisition date, if a run has no evaluation
        window (see level_range_table), or if FIGURE_RUN is not among the runs
        or its window is not valid (see window_trajectory).
    """
    out_dir = Path(out_dir)
    dict_ids = tuple(dict_ids)
    values, e1_commit = read_e1_hyperparameters(e1_metadata)
    hyperparams = dictionary_hyperparameters(values)
    print_hyperparameters(values, _relative_to_root(e1_metadata), e1_commit, hyperparams, dict_ids)
    print(f"Runs used: {len(runs)}")
    print_level_ranges(runs)

    max_delays = max((hyperparams.get(d, {}).get("n_delays", 0) for d in dict_ids), default=0)
    windows = window_table(runs, HORIZONS_S, max_delays)
    models, errors, coefficients = sweep(
        runs, dict_ids, hyperparams=hyperparams, horizons_s=HORIZONS_S, keep_coefficients=True
    )
    check_same_windows(errors, windows)

    summary = error_summary_table(errors)
    improvements = improvements_table(errors)
    fit = fit_table(models)
    by_run = by_run_table(errors, BY_RUN_HORIZON_S)
    level_ranges = level_range_table(runs, windows, acquisition_dates)
    print_tables(summary, improvements, fit, by_run, windows)

    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    models.to_csv(tables_dir / "models.csv", index=False)
    errors.to_csv(tables_dir / "errors.csv", index=False)
    summary.to_csv(tables_dir / "error_summary.csv", index=False)
    improvements.to_csv(tables_dir / "improvements.csv", index=False)
    fit.to_csv(tables_dir / "fit_indicators.csv", index=False)
    by_run.to_csv(tables_dir / f"errors_by_run_{BY_RUN_HORIZON_S:g}s.csv", index=False)
    windows.to_csv(tables_dir / "windows.csv", index=False)
    coefficients.to_csv(tables_dir / "coefficients.csv", index=False)
    level_ranges.to_csv(tables_dir / "level_ranges.csv", index=False)

    set_style()
    save_figure(figure_error_vs_horizon(summary), out_dir / "figures", "error_vs_horizon")
    save_figure(figure_trajectory(runs, hyperparams), out_dir / "figures", "trajectory")

    median_rmse = {
        series: {f"{h:g}": float(v) for h, v in block.set_index("horizon_s")["median"].items()}
        for series, block in summary.groupby("series", sort=False)
    }
    write_metadata(
        out_dir,
        config={
            "subset": SUBSET,
            "ts_s": TS_S,
            "train_frac": TRAIN_FRAC,
            "dictionaries": list(dict_ids),
            "hyperparameters": {k: v for k, v in hyperparams.items() if k in dict_ids},
            "hyperparameter_source": {
                "e1_metadata": _relative_to_root(e1_metadata),
                "e1_commit": e1_commit,
                "values": values,
            },
            "horizons_s": list(HORIZONS_S),
            "window_offset": 0,
            "runs": [r[0] for r in runs],
            "figure_run": FIGURE_RUN,
            "figure_horizon_s": FIGURE_HORIZON_S,
            "figure_dictionaries": list(FIGURE_DICTIONARIES),
            "coefficients_table": (
                "tables/coefficients.csv: coefficients of Psi_{k+1} = A Psi_k + B u_k + c in normalized "
                "units, one row per element of [A | B | c] of each model. Columns: run, opening_pct, "
                "dictionary, output (observable predicted, the row of the matrices), term (regressor it "
                "multiplies: an observable name, 'u' or 'const'), coefficient."
            ),
        },
        summary={
            "n_runs": len(runs),
            "n_models": int(len(models)),
            "median_rmse_pct": median_rmse,
        },
    )
    print(f"Results in {out_dir}")
    return models, errors


def main() -> None:
    """Run E2 on every Baseline run and write the outputs to OUT_DIR."""
    run_experiment(load_runs(), OUT_DIR, load_acquisition_dates())


if __name__ == "__main__":
    main()
