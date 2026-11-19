"""E2 one-step - One-step-ahead error of the E2 models, in and out of sample.

E2 scores each model by simulating it in open loop over windows anchored at
the setpoint steps of the evaluation part (multi-step error). Its in-sample
residual is a one-step error. This script measures the missing case, the
one-step error out of sample, so that leaving the fit sample and lengthening
the horizon can be looked at separately.

Every prediction here starts from the measured level: the lifted state at
instant k is built from the measured h_k (and its measured delays), advanced
one step with the input u_k, and its first component is compared with the
measured h_{k+1}. Errors are prediction minus measurement, in % of level, as
in E2.

Models
    Nothing is fitted here. The models are those of E2: their coefficients A,
    B and c are read from tables/coefficients.csv of the E2 results (with
    float_precision="round_trip", so they are the fitted values exactly), and
    each dictionary is built with the hyperparameters recorded in the E2
    metadata. The runs, split and cut are those of E2 (load_runs of E2).
    Before anything is written the script checks that the E2 results come
    from a clean commit and include the coefficients table, that every run
    and dictionary has its model, that the in-sample one-step RMSE recomputed
    with predict_step equals fit_rmse_pct of the E2 models table within the
    relative tolerance FIT_RMSE_RTOL (and its number of errors equals
    n_pairs), and that the window starts equal those of the E2 windows table.

Segments
    fit
        Every pair of consecutive samples of the fit part (the in-sample
        residual of E2). Its rows are copied from the E2 models table (rmse =
        fit_rmse_pct, n_errors = n_pairs), so that this figure has a single
        source; the value recomputed here with predict_step is used only for
        the check above. The two can differ in the last digits because
        fit_edmdc computes the same sums in another order.
    eval_all
        Every pair of consecutive samples of the evaluation part.
    window
        The instants that the multi-step error of E2 predicts in each window,
        h[i + 1 .. i + H] for a window starting at i with H steps, each one
        predicted from the measured data at the instant before. The errors of
        all windows of a run are pooled into one RMSE per run, dictionary and
        horizon, as in E2.

The runs were recorded on different days, with the same setpoint profile and
a different electrovalve opening; the independence between runs cannot be
verified. The summaries are descriptive (median and range across runs, and counts of
runs in which one dictionary has the lower RMSE), with no statistical test.

Outputs in results/E2_one_step/: CSV tables and metadata.json.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from edmdc_tank_observables.evaluation.protocol import model_from_coefficients, window_table
from edmdc_tank_observables.models.edmdc import EDMDcModel
from edmdc_tank_observables.observables.dictionaries import make_dictionary
from edmdc_tank_observables.paths import PROJECT_ROOT
from edmdc_tank_observables.provenance import write_metadata


def _load_e2():
    """Import experiments/E2_dictionaries.py as a module, without running it, to reuse its data loading and settings."""
    path = Path(__file__).resolve().parent / "E2_dictionaries.py"
    spec = importlib.util.spec_from_file_location("E2_dictionaries", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e2 = _load_e2()

# --- Configuration -------------------------------------------------------------

E2_DIR = PROJECT_ROOT / "results" / "E2_dictionaries"
OUT_DIR = PROJECT_ROOT / "results" / "E2_one_step"

# Dictionaries and horizons are those of E2.
DICTIONARIES = e2.DICTIONARIES
HORIZONS_S = e2.HORIZONS_S

SEGMENTS = ("fit", "eval_all", "window")

# Pairs (a, b) whose run-by-run counts are reported.
PAIRS = (
    ("D2", "D0"),
    ("D1a", "D3a"),
    ("D1b", "D3b"),
    ("D3c", "D1c"),
    ("D1c", "D0"),
    ("D1a", "D2"),
)

# Relative tolerance of the check against fit_rmse_pct of E2. predict_step and
# fit_edmdc compute the same one-step prediction with the sums in another
# order, so the two values agree only up to rounding.
FIT_RMSE_RTOL = 1e-10

ERROR_COLUMNS = ("run", "opening_pct", "dictionary", "segment", "horizon_s", "n_errors", "rmse")


# --- Checks against the E2 results ---------------------------------------------


def read_e2_metadata(path) -> tuple[str, dict]:
    """Read the metadata.json of the E2 results and require a clean commit and the coefficients table.

    Parameters
    ----------
    path : pathlib.Path or str
        The metadata.json written by E2.

    Returns
    -------
    commit : str
        Commit from which E2 was run.
    config : dict
        The config section of the metadata.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If E2 was not run from a clean commit (git.clean is not true), or if
        its config has no coefficients_table, so that E2 run did not store
        the coefficients of its models.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"The E2 metadata {path} does not exist: run E2 first.")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    git = metadata.get("git", {})
    if git.get("clean") is not True:
        raise ValueError(f"{path}: E2 was not run from a clean commit (git.clean = {git.get('clean')!r}).")
    config = metadata.get("config", {})
    if "coefficients_table" not in config:
        raise ValueError(f"{path}: config.coefficients_table is missing, so that E2 run did not store its coefficients.")
    return str(git.get("commit")), config


def read_models(coefficients: pd.DataFrame, runs, dict_ids, hyperparams: dict) -> dict:
    """Build the E2 model of every run and dictionary from the coefficients table.

    Parameters
    ----------
    coefficients : pandas.DataFrame
        Coefficients table of E2 (tables/coefficients.csv), read with
        float_precision="round_trip" so that the values are exact.
    runs : list of tuple
        Runs as returned by load_runs of E2; only their names are used.
    dict_ids : iterable of str
        Dictionaries to build.
    hyperparams : dict
        Keyword arguments of make_dictionary for each dictionary id, as
        recorded in the E2 metadata.

    Returns
    -------
    dict
        (run name, dictionary id) -> EDMDcModel with the A, B and c read and
        its dictionary. The other fields of the model (ts, n_pairs, cond,
        fit_rmse_pct) keep their defaults: they are not used here.

    Raises
    ------
    ValueError
        If a dictionary cannot be built (see make_dictionary), or if the
        coefficients of some run and dictionary are missing, repeated or have
        unknown names (see model_from_coefficients).
    """
    models = {}
    for name, _, _, _ in runs:
        for dict_id in dict_ids:
            dictionary = make_dictionary(dict_id, **hyperparams.get(dict_id, {}))
            A, B, c = model_from_coefficients(coefficients, name, dictionary)
            models[(name, dict_id)] = EDMDcModel(A=A, B=B, c=c, dictionary=dictionary)
    return models


def check_fit_rmse(errors: pd.DataFrame, e2_models: pd.DataFrame, rtol: float = FIT_RMSE_RTOL) -> None:
    """Check that the recomputed in-sample one-step RMSE equals fit_rmse_pct of E2 for every run and dictionary.

    The number of errors must also equal n_pairs of E2 exactly.

    Parameters
    ----------
    errors : pandas.DataFrame
        One-step errors table (ERROR_COLUMNS) with the fit rows recomputed
        here; its rows with segment "fit" are checked.
    e2_models : pandas.DataFrame
        Models table written by E2, with the columns run, dictionary,
        fit_rmse_pct and n_pairs.
    rtol : float, default FIT_RMSE_RTOL
        Relative tolerance of the RMSE comparison.

    Raises
    ------
    ValueError
        If the two tables do not have the same (run, dictionary) pairs, if
        some RMSE differs beyond the tolerance, or if some number of errors
        differs from n_pairs.
    """
    fit = errors[errors["segment"] == "fit"][["run", "dictionary", "n_errors", "rmse"]]
    e2_fit = e2_models[["run", "dictionary", "fit_rmse_pct", "n_pairs"]]
    joined = fit.merge(e2_fit, on=["run", "dictionary"], how="outer")
    if joined[["rmse", "fit_rmse_pct"]].isna().any().any():
        missing = joined[joined[["rmse", "fit_rmse_pct"]].isna().any(axis=1)][["run", "dictionary"]]
        raise ValueError(f"The fit models do not match the E2 models table, e.g. {missing.head(5).to_dict('records')}.")
    bad = joined[~np.isclose(joined["rmse"], joined["fit_rmse_pct"], rtol=rtol, atol=0.0)]
    if len(bad) > 0:
        raise ValueError(
            f"The in-sample one-step RMSE differs from fit_rmse_pct of E2, e.g. {bad.head(5).to_dict('records')}."
        )
    bad = joined[joined["n_errors"] != joined["n_pairs"]]
    if len(bad) > 0:
        raise ValueError(f"The number of fit errors differs from n_pairs of E2, e.g. {bad.head(5).to_dict('records')}.")


def fit_rows_from_e2(errors: pd.DataFrame, e2_models: pd.DataFrame) -> pd.DataFrame:
    """Replace the fit rows of the errors table with the values of the E2 models table.

    The in-sample residual is then written with a single value, the one of
    E2 (rmse = fit_rmse_pct, n_errors = n_pairs). Call check_fit_rmse first,
    so that the recomputed values are known to agree with it.

    Parameters
    ----------
    errors : pandas.DataFrame
        One-step errors table (ERROR_COLUMNS).
    e2_models : pandas.DataFrame
        Models table written by E2, with the columns run, dictionary,
        fit_rmse_pct and n_pairs, one row per run and dictionary.

    Returns
    -------
    pandas.DataFrame
        A copy of errors, with the same rows in the same order, in which the
        rows with segment "fit" take rmse and n_errors from e2_models.

    Raises
    ------
    ValueError
        If some fit row has no model in e2_models.
    """
    out = errors.copy()
    is_fit = out["segment"] == "fit"
    e2 = e2_models.set_index(["run", "dictionary"])
    keys = pd.MultiIndex.from_frame(out.loc[is_fit, ["run", "dictionary"]])
    missing = keys.difference(e2.index)
    if len(missing) > 0:
        raise ValueError(f"Fit rows without a model in the E2 models table, e.g. {missing[:5].tolist()}.")
    out.loc[is_fit, "rmse"] = e2.loc[keys, "fit_rmse_pct"].to_numpy()
    out.loc[is_fit, "n_errors"] = e2.loc[keys, "n_pairs"].to_numpy()
    return out


def check_windows(windows: pd.DataFrame, e2_windows: pd.DataFrame) -> None:
    """Check that the window starts per run and horizon equal those of the E2 windows table.

    Parameters
    ----------
    windows : pandas.DataFrame
        Windows table built here with window_table.
    e2_windows : pandas.DataFrame
        Windows table written by E2.

    Raises
    ------
    ValueError
        If the two tables do not have the same (run, horizon_s) pairs, or if
        the horizon in steps or the starts differ for any of them.
    """
    keys = ["run", "horizon_s"]
    columns = [*keys, "horizon_steps", "starts"]
    joined = windows[columns].merge(e2_windows[columns], on=keys, how="outer", suffixes=("", "_e2"))
    same = (joined["horizon_steps"] == joined["horizon_steps_e2"]) & (
        joined["starts"].astype(str) == joined["starts_e2"].astype(str)
    )
    if not same.all():
        raise ValueError(f"The windows differ from those of E2, e.g. {joined[~same].head(5).to_dict('records')}.")


# --- One-step errors -----------------------------------------------------------


def one_step_errors(model, dataset) -> np.ndarray:
    """One-step error at every pair of consecutive samples of a segment, from the measured level.

    The lifted states and inputs are built as in fit_edmdc (transform and
    align_input on the segment), so on the fit part the RMSE of these errors
    is the in-sample residual, up to the order of the sums.

    Parameters
    ----------
    model : EDMDcModel
        Fitted model.
    dataset : PreprocessedDataset
        Segment to score.

    Returns
    -------
    numpy.ndarray
        Predicted minus measured level, in % of level, one value per pair of
        consecutive samples after the first n_delays.
    """
    psi = model.dictionary.transform(dataset.h)
    u = model.dictionary.align_input(dataset.u)
    prediction = model.predict_step(psi[:-1], u[:-1])[:, 0]
    return dataset.denormalize_h_delta(prediction - psi[1:, 0])


def window_one_step_errors(model, dataset, starts, horizon_steps: int) -> np.ndarray:
    """One-step error at the instants that the multi-step error predicts in each window, pooled.

    For a window starting at i, the instants i + 1 .. i + H are predicted,
    each from the measured data at the instant before: the lifted state at
    i + j is built from the measured levels h[i + j - n_delays .. i + j] and
    advanced with u[i + j]. Each row of transform depends only on the measured
    level at its instant and its delays, so row j of
    transform(h[i - n_delays : i + H]) is that lifted state.

    Parameters
    ----------
    model : EDMDcModel
        Fitted model.
    dataset : PreprocessedDataset
        Evaluation segment.
    starts : iterable of int
        Window starts i, as given by window_starts.
    horizon_steps : int
        Number of instants H predicted in each window.

    Returns
    -------
    numpy.ndarray
        Predicted minus measured level, in % of level, for every instant of
        every window, window after window.
    """
    n_delays = model.dictionary.n_delays
    errors = []
    for i in starts:
        psi = model.dictionary.transform(dataset.h[i - n_delays : i + horizon_steps])
        prediction = model.predict_step(psi, dataset.u[i : i + horizon_steps].reshape(-1, 1))[:, 0]
        measured = dataset.h[i + 1 : i + 1 + horizon_steps]
        errors.append(dataset.denormalize_h_delta(prediction - measured))
    return np.concatenate(errors)


def _row(name, opening_pct, dict_id, segment, horizon_s, errors) -> dict:
    """One row of the errors table from a set of errors."""
    return {
        "run": name,
        "opening_pct": opening_pct,
        "dictionary": dict_id,
        "segment": segment,
        "horizon_s": horizon_s,
        "n_errors": len(errors),
        "rmse": float(np.sqrt(np.mean(errors**2))),
    }


def one_step_table(runs, dict_ids, models: dict, windows: pd.DataFrame) -> pd.DataFrame:
    """Compute the one-step RMSE of each model in every segment.

    Parameters
    ----------
    runs : list of tuple
        One tuple (name, opening_pct, train, test) per run, as returned by
        load_runs of E2.
    dict_ids : iterable of str
        Dictionaries to score.
    models : dict
        (run name, dictionary id) -> EDMDcModel, as returned by read_models.
    windows : pandas.DataFrame
        Windows table of the runs (window_table), whose starts are used for
        the window segment.

    Returns
    -------
    pandas.DataFrame
        Errors table with the columns ERROR_COLUMNS: one row per run and
        dictionary for fit and eval_all (horizon_s empty), and one per run,
        dictionary and horizon for window. The fit rows hold the values
        recomputed here; run replaces them with those of E2 after the check
        (see fit_rows_from_e2).

    Raises
    ------
    KeyError
        If some run and dictionary has no entry in models.
    """
    rows = []
    for name, opening_pct, train, test in runs:
        run_windows = windows[windows["run"] == name]
        for dict_id in dict_ids:
            model = models[(name, dict_id)]
            rows.append(_row(name, opening_pct, dict_id, "fit", np.nan, one_step_errors(model, train)))
            rows.append(_row(name, opening_pct, dict_id, "eval_all", np.nan, one_step_errors(model, test)))
            for _, w in run_windows.iterrows():
                starts = [int(x) for x in str(w["starts"]).split(";")]
                errors = window_one_step_errors(model, test, starts, int(w["horizon_steps"]))
                rows.append(_row(name, opening_pct, dict_id, "window", w["horizon_s"], errors))
    return pd.DataFrame(rows, columns=list(ERROR_COLUMNS))


# --- Summaries -----------------------------------------------------------------


def summary_table(errors: pd.DataFrame) -> pd.DataFrame:
    """Median, minimum and maximum across runs of the one-step RMSE, per dictionary, segment and horizon.

    Parameters
    ----------
    errors : pandas.DataFrame
        Errors table returned by one_step_table.

    Returns
    -------
    pandas.DataFrame
        One row per (dictionary, segment, horizon_s) with the columns median,
        min, max and n_runs. horizon_s is empty for fit and eval_all.
    """
    return (
        errors.groupby(["dictionary", "segment", "horizon_s"], sort=False, dropna=False)["rmse"]
        .agg(median="median", min="min", max="max", n_runs="count")
        .reset_index()
    )


def _compare(a: pd.Series, b: pd.Series) -> dict:
    """Counts of runs in which a or b has the lower RMSE, exact ties, and the median of a - b."""
    if set(a.index) != set(b.index):
        raise ValueError(f"The two series do not have the same runs: {sorted(set(a.index) ^ set(b.index))}.")
    b = b.loc[a.index]
    return {
        "n_a_lower": int((a < b).sum()),
        "n_b_lower": int((b < a).sum()),
        "n_ties": int((a == b).sum()),
        "n_runs": len(a),
        "median_diff": float((a - b).median()),
    }


def pair_counts(errors: pd.DataFrame, pairs, multistep: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run-by-run comparison of each pair, per segment and horizon, with the multi-step counts of E2 next to the window rows.

    Parameters
    ----------
    errors : pandas.DataFrame
        Errors table returned by one_step_table.
    pairs : iterable of tuple of str
        Pairs (a, b) of dictionaries.
    multistep : pandas.DataFrame, optional
        Errors table written by E2 (columns run, series, horizon_s, rmse). If
        given, the window rows get the same counts for the multi-step RMSE at
        the same horizon.

    Returns
    -------
    pandas.DataFrame
        One row per (pair, segment, horizon_s) with the columns a, b, segment,
        horizon_s, n_a_lower, n_b_lower, n_ties (exactly equal RMSE), n_runs,
        median_diff (median over runs of rmse_a - rmse_b) and, if multistep is
        given, multistep_n_a_lower, multistep_n_b_lower and
        multistep_median_diff (empty outside the window segment).

    Raises
    ------
    ValueError
        If a dictionary of a pair is missing, or if the two dictionaries of a
        pair (or the one-step and multi-step tables) do not have the same runs.
    """
    rows = []
    for a, b in pairs:
        for (segment, horizon_s), block in errors.groupby(["segment", "horizon_s"], sort=False, dropna=False):
            by_dict = {d: block[block["dictionary"] == d].set_index("run")["rmse"] for d in (a, b)}
            if any(len(v) == 0 for v in by_dict.values()):
                raise ValueError(f"Pair ({a}, {b}): a dictionary has no rows in segment {segment}.")
            row = {"a": a, "b": b, "segment": segment, "horizon_s": horizon_s, **_compare(by_dict[a], by_dict[b])}
            if multistep is not None:
                row.update(multistep_n_a_lower=np.nan, multistep_n_b_lower=np.nan, multistep_median_diff=np.nan)
                if segment == "window":
                    at_horizon = multistep[multistep["horizon_s"] == horizon_s]
                    ms = {d: at_horizon[at_horizon["series"] == d].set_index("run")["rmse"] for d in (a, b)}
                    if set(ms[a].index) != set(by_dict[a].index):
                        raise ValueError(f"Pair ({a}, {b}) at {horizon_s:g} s: the multi-step table has other runs.")
                    counts = _compare(ms[a], ms[b])
                    row.update(
                        multistep_n_a_lower=counts["n_a_lower"],
                        multistep_n_b_lower=counts["n_b_lower"],
                        multistep_median_diff=counts["median_diff"],
                    )
            rows.append(row)
    return pd.DataFrame(rows)


# --- Run -----------------------------------------------------------------------


def print_tables(summary: pd.DataFrame, counts: pd.DataFrame, e2_commit: str) -> None:
    """Print the summary and the pair counts to the console."""
    print("=" * 78)
    print(f"E2 one-step error (models read from the coefficients of the clean E2 run {e2_commit})")
    print("=" * 78)
    print("\nOne-step RMSE (% of level): median [min, max] across runs")
    cells = summary.assign(
        column=summary["segment"] + summary["horizon_s"].map(lambda h: "" if pd.isna(h) else f" {h:g} s"),
        cell=summary.apply(lambda r: f"{r['median']:.3f} [{r['min']:.3f}, {r['max']:.3f}]", axis=1),
    )
    table = cells.pivot(index="dictionary", columns="column", values="cell")
    print(table.loc[list(DICTIONARIES), cells["column"].drop_duplicates()].to_string())

    print("\nRuns with the lower RMSE in each pair (a / b / exact ties), median of a - b; multi-step of E2 next to it")
    cells = counts.assign(
        pair=counts["a"] + " vs " + counts["b"],
        where=counts["segment"] + counts["horizon_s"].map(lambda h: "" if pd.isna(h) else f" {h:g} s"),
        one_step=counts.apply(lambda r: f"{r['n_a_lower']}/{r['n_b_lower']}/{r['n_ties']}  {r['median_diff']:+.4f}", axis=1),
        multi_step=counts.apply(
            lambda r: "" if pd.isna(r["multistep_n_a_lower"])
            else f"{r['multistep_n_a_lower']:.0f}/{r['multistep_n_b_lower']:.0f}  {r['multistep_median_diff']:+.4f}",
            axis=1,
        ),
    )
    print(cells[["pair", "where", "one_step", "multi_step"]].to_string(index=False))
    print()


def run(runs, out_dir=OUT_DIR, e2_dir=E2_DIR) -> dict:
    """Read the E2 models, check them against the E2 results, compute the one-step errors and write them.

    Parameters
    ----------
    runs : list of tuple
        Runs as returned by load_runs of E2.
    out_dir : pathlib.Path or str, default OUT_DIR
        Output folder.
    e2_dir : pathlib.Path or str, default E2_DIR
        Output folder of E2, with metadata.json and tables/coefficients.csv,
        tables/models.csv, tables/windows.csv and tables/errors.csv.

    Returns
    -------
    dict
        The tables, keyed errors, summary and pair_counts. The fit rows of
        errors, and so the fit rows of the summary and pair counts, hold the
        values of the E2 models table.

    Raises
    ------
    FileNotFoundError
        If the E2 metadata or any of the four E2 tables does not exist.
    ValueError
        If E2 did not run from a clean commit or did not store its
        coefficients (see read_e2_metadata), if the windows are not valid for
        some run (see window_table), if a model is missing from the
        coefficients table (see read_models), if the windows or the in-sample
        residuals do not match those of E2 (see check_windows and
        check_fit_rmse), or if a pair cannot be compared.
    """
    out_dir, e2_dir = Path(out_dir), Path(e2_dir)
    e2_commit, e2_config = read_e2_metadata(e2_dir / "metadata.json")
    hyperparams = {k: v for k, v in e2_config.get("hyperparameters", {}).items() if k in DICTIONARIES}

    max_delays = max((hyperparams.get(d, {}).get("n_delays", 0) for d in DICTIONARIES), default=0)
    windows = window_table(runs, HORIZONS_S, max_delays)
    check_windows(windows, pd.read_csv(e2_dir / "tables" / "windows.csv"))

    # round_trip: the default float parser of read_csv can be one unit in the
    # last place off, and the coefficients and fit_rmse_pct must be exact.
    coefficients = pd.read_csv(e2_dir / "tables" / "coefficients.csv", float_precision="round_trip")
    e2_models = pd.read_csv(e2_dir / "tables" / "models.csv", float_precision="round_trip")
    models = read_models(coefficients, runs, DICTIONARIES, hyperparams)

    errors = one_step_table(runs, DICTIONARIES, models, windows)
    check_fit_rmse(errors, e2_models)
    errors = fit_rows_from_e2(errors, e2_models)

    summary = summary_table(errors)
    counts = pair_counts(errors, PAIRS, pd.read_csv(e2_dir / "tables" / "errors.csv"))
    print_tables(summary, counts, e2_commit)

    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {"errors": errors, "summary": summary, "pair_counts": counts}
    for key, table in tables.items():
        table.to_csv(out_dir / f"one_step_{key}.csv", index=False)

    try:
        e2_source = e2_dir.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        e2_source = str(e2_dir)
    write_metadata(
        out_dir,
        config={
            "e2_results": e2_source,
            "e2_commit": e2_commit,
            "models_source": {
                "coefficients": f"{e2_source}/tables/coefficients.csv",
                "e2_commit": e2_commit,
                "read": "pandas.read_csv with float_precision=round_trip",
                "hyperparameters": "config.hyperparameters of the E2 metadata",
            },
            "dictionaries": list(DICTIONARIES),
            "hyperparameters": hyperparams,
            "horizons_s": list(HORIZONS_S),
            "segments": {
                "fit": "every pair of consecutive samples of the fit part (in-sample residual)",
                "eval_all": "every pair of consecutive samples of the evaluation part",
                "window": (
                    "for each E2 window start i and horizon H steps, the instants i+1..i+H, each predicted from the"
                    " measured level at the instant before; errors of all windows of a run pooled into one RMSE"
                ),
            },
            "fit_rows": (
                "fit rows copied from E2 models.csv (rmse = fit_rmse_pct, n_errors = n_pairs); recomputed with"
                f" predict_step only as a check, rtol {FIT_RMSE_RTOL:g}"
            ),
            "prediction": "lifted state built from the measured level (and measured delays), advanced one step with predict_step",
            "error_sign": "prediction - measurement, in % of level",
            "pairs": [list(pair) for pair in PAIRS],
            "ties": "exactly equal RMSE, counted apart from both sides",
            "multistep_source": "tables/errors.csv of the E2 results (column rmse, same horizon)",
            "checks": [
                "E2 metadata clean and with config.coefficients_table",
                "a model for every run and dictionary in tables/coefficients.csv of E2",
                f"in-sample one-step RMSE recomputed with predict_step equals fit_rmse_pct of E2 (rtol {FIT_RMSE_RTOL:g}),"
                " and its number of errors equals n_pairs",
                "window starts equal tables/windows.csv of E2",
            ],
            "analysis": "descriptive, no tests",
        },
    )
    print(f"Results in {out_dir}")
    return tables


def main() -> None:
    """Compute the one-step errors of the E2 models on every Baseline run and write them to OUT_DIR."""
    run(e2.load_runs())


if __name__ == "__main__":
    main()
