"""Fitting and evaluation protocol.

It makes every experiment fit and evaluate in the same way. The functions
return dictionary rows ready to build a ``DataFrame``, or the ``DataFrame``
itself, with four fixed schemas:

``models``
    One row per (run, dictionary): dimension of the dictionary, condition
    number, spectral radius and in-sample fit error (MODEL_COLUMNS).

``errors``
    One row per (run, series, horizon), where the series is a dictionary id
    (ERROR_COLUMNS). The errors are those of the windows anchored at
    the setpoint steps of the evaluation segment.

``windows``
    One row per (run, horizon) with the windows themselves: their number,
    their start as a local sample index and as an absolute time, and the
    setpoint at the window start X and at X + 1 (before and after the step
    when the offset is 0). Every series of a run is checked against it.

``coefficients``
    One row per element of [A | B | c] of each fitted model
    (COEFFICIENT_COLUMNS): the observable it predicts (output) and the
    regressor it multiplies (term: an observable name, "u" or "const"). It is
    optional in sweep, and model_from_coefficients rebuilds A, B and c from it.

The aggregates at the end summarize the errors table across runs. The runs
were recorded on different days, with the same setpoint profile and a
different electrovalve opening, and the independence between them cannot be
verified: the summary of this module is descriptive (median and range, and the number of
runs in which a series improves on a reference), with no statistical test. The paired
tests between dictionaries are computed in experiments/E2_comparisons.py.
"""

import numpy as np
import pandas as pd

from edmdc_tank_observables.data.preprocess import PreprocessedDataset
from edmdc_tank_observables.evaluation.metrics import MultistepError, _horizon_to_steps, multistep_error, window_starts
from edmdc_tank_observables.models.edmdc import EDMDcModel, fit_edmdc
from edmdc_tank_observables.observables.dictionaries import Dictionary, make_dictionary

MODEL_COLUMNS = (
    "run", "opening_pct", "ts", "dictionary", "p", "n_delays", "cond", "spectral_radius",
    "fit_rmse_pct", "n_pairs",
)
ERROR_COLUMNS = (
    "run", "opening_pct", "ts", "series", "horizon_s", "horizon_steps", "n_windows", "rmse",
    "rmse_final", "max_abs",
)
COEFFICIENT_COLUMNS = ("run", "opening_pct", "dictionary", "output", "term", "coefficient")


def model_row(model: EDMDcModel, run: str, opening_pct: float) -> dict:
    """Summarize a fitted model as one row of the models table.

    Parameters
    ----------
    model : EDMDcModel
        Fitted model.
    run : str
        Name of the run the model was fitted on.
    opening_pct : float
        Electrovalve opening of that run, in %.

    Returns
    -------
    dict
        Row with the keys of MODEL_COLUMNS.
    """
    return {
        "run": run,
        "opening_pct": opening_pct,
        "ts": model.ts,
        "dictionary": model.dictionary.id,
        "p": model.p,
        "n_delays": model.dictionary.n_delays,
        "cond": model.cond,
        "spectral_radius": model.spectral_radius,
        "fit_rmse_pct": model.fit_rmse_pct,
        "n_pairs": model.n_pairs,
    }


def coefficient_rows(model: EDMDcModel, run: str, opening_pct: float) -> list[dict]:
    """List the coefficients of a fitted model as rows of the coefficients table.

    The model is Psi_{k+1} = A Psi_k + B u_k + c in normalized units. Each row
    of [A | B | c] predicts one observable (output) and each column multiplies
    one regressor (term).

    Parameters
    ----------
    model : EDMDcModel
        Fitted model.
    run : str
        Name of the run the model was fitted on.
    opening_pct : float
        Electrovalve opening of that run, in %.

    Returns
    -------
    list of dict
        p * (p + 2) rows with the keys of COEFFICIENT_COLUMNS, ordered by
        output and then by term: the observables in column order, "u" and
        "const". Output and observable terms are named as in
        Dictionary.observable_names.
    """
    names = model.dictionary.observable_names
    terms = [*names, "u", "const"]
    rows = []
    for i, output in enumerate(names):
        values = [*model.A[i], model.B[i, 0], model.c[i]]
        for term, value in zip(terms, values):
            rows.append(
                {
                    "run": run,
                    "opening_pct": opening_pct,
                    "dictionary": model.dictionary.id,
                    "output": output,
                    "term": term,
                    "coefficient": float(value),
                }
            )
    return rows


def model_from_coefficients(
    table: pd.DataFrame, run: str, dictionary: Dictionary
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild A, B and c of one model from the coefficients table.

    Parameters
    ----------
    table : pandas.DataFrame
        Coefficients table with the columns COEFFICIENT_COLUMNS, as returned
        by sweep or read back from its CSV. The values are rebuilt exactly
        only if the CSV is read with ``float_precision="round_trip"``; the
        default parser of pandas.read_csv can be one unit in the last place
        off.
    run : str
        Run the model was fitted on.
    dictionary : Dictionary
        Dictionary of the model. Its id selects the rows and its
        observable_names give the expected outputs and terms.

    Returns
    -------
    A : numpy.ndarray, shape (p, p)
    B : numpy.ndarray, shape (p, 1)
    c : numpy.ndarray, shape (p,)

    Raises
    ------
    ValueError
        If some (output, term) pair appears more than once for this run and
        dictionary, if some expected pair is missing (including when the run
        or dictionary is not in the table), or if there are pairs with names
        the dictionary does not have.
    """
    names = dictionary.observable_names
    terms = [*names, "u", "const"]
    p = len(names)
    block = table[(table["run"] == run) & (table["dictionary"] == dictionary.id)]

    found = pd.MultiIndex.from_frame(block[["output", "term"]])
    if found.has_duplicates:
        repeated = found[found.duplicated()].unique().tolist()[:5]
        raise ValueError(f"{run}, {dictionary.id}: repeated coefficients, e.g. {repeated}.")
    expected = pd.MultiIndex.from_product([names, terms], names=["output", "term"])
    missing = expected.difference(found).tolist()
    unexpected = found.difference(expected).tolist()
    if missing:
        raise ValueError(f"{run}, {dictionary.id}: {len(missing)} coefficients missing, e.g. {missing[:5]}.")
    if unexpected:
        raise ValueError(f"{run}, {dictionary.id}: coefficients with unknown names, e.g. {unexpected[:5]}.")

    wide = block.set_index(["output", "term"])["coefficient"].unstack("term")
    matrix = wide.loc[names, terms].to_numpy(dtype=float)
    return matrix[:, :p], matrix[:, p : p + 1], matrix[:, p + 1]


def _error_row(
    error: MultistepError, run: str, opening_pct: float, ts: float, series: str
) -> dict:
    """Turn a MultistepError into one row of the errors table."""
    return {
        "run": run,
        "opening_pct": opening_pct,
        "ts": ts,
        "series": series,
        "horizon_s": error.horizon_s,
        "horizon_steps": error.horizon_steps,
        "n_windows": error.n_windows,
        "rmse": error.rmse,
        "rmse_final": error.rmse_final,
        "max_abs": error.max_abs,
    }


def evaluate(
    model: EDMDcModel,
    test: PreprocessedDataset,
    horizons_s,
    run: str,
    opening_pct: float,
    offset: int = 0,
) -> list[dict]:
    """Evaluate a fitted model on an evaluation segment, one row per horizon.

    Parameters
    ----------
    model : EDMDcModel
        Fitted model.
    test : PreprocessedDataset
        Evaluation segment.
    horizons_s : iterable of float
        Prediction horizons, in seconds.
    run : str
        Label of the run that `test` belongs to.
    opening_pct : float
        Electrovalve opening of that run, in %.
    offset : int, default 0
        Samples by which every window start is moved.

    Returns
    -------
    list of dict
        Rows with the keys of ERROR_COLUMNS, one per horizon, with the
        dictionary id as series.

    Raises
    ------
    ValueError
        If the windows are not valid for this segment, horizon or model; see
        :func:`edmdc_tank_observables.evaluation.metrics.window_starts`.
    """
    return [
        _error_row(
            multistep_error(test, horizon, model=model, offset=offset),
            run, opening_pct, test.ts, model.dictionary.id,
        )
        for horizon in horizons_s
    ]


def sweep(
    runs,
    dict_ids,
    hyperparams: dict | None = None,
    horizons_s=(30, 120, 600),
    offset: int = 0,
    keep_coefficients: bool = False,
) -> tuple[pd.DataFrame, ...]:
    """Fit and evaluate a set of dictionaries on several runs.

    For each run every dictionary is fitted on the fit part and evaluated on
    the evaluation part.

    Parameters
    ----------
    runs : iterable of tuple
        One tuple (name, opening_pct, train, test) per run: its name, the
        electrovalve opening in %, and its fit and evaluation parts as
        PreprocessedDataset.
    dict_ids : sequence of str
        Dictionary configurations to evaluate; read again for every run.
    hyperparams : dict, optional
        Keyword arguments of make_dictionary for each dictionary id, e.g.
        {"D1c": {"n_delays": 3}}. Ids without an entry get none.
    horizons_s : sequence of float, default (30, 120, 600)
        Prediction horizons, in seconds; read again for every model.
    offset : int, default 0
        Samples by which every window start is moved.
    keep_coefficients : bool, default False
        Whether to also return the coefficients of every fitted model.

    Returns
    -------
    models, errors : pandas.DataFrame
        The models table (columns MODEL_COLUMNS) and the errors table
        (columns ERROR_COLUMNS), in long format.
    coefficients : pandas.DataFrame
        Only if keep_coefficients is True, as a third element: the
        coefficients table (columns COEFFICIENT_COLUMNS), see
        coefficient_rows.

    Raises
    ------
    ValueError
        If a dictionary cannot be built or fitted, or if the windows are not
        valid for some run; see :func:`edmdc_tank_observables.models.edmdc.fit_edmdc` and
        :func:`edmdc_tank_observables.evaluation.metrics.window_starts`.
    """
    hyperparams = hyperparams or {}
    model_rows, error_rows, coef_rows = [], [], []

    for name, opening_pct, train, test in runs:
        for dict_id in dict_ids:
            dictionary = make_dictionary(dict_id, **hyperparams.get(dict_id, {}))
            model = fit_edmdc(dictionary, train)
            model_rows.append(model_row(model, name, opening_pct))
            error_rows.extend(evaluate(model, test, horizons_s, name, opening_pct, offset))
            if keep_coefficients:
                coef_rows.extend(coefficient_rows(model, name, opening_pct))

    models = pd.DataFrame(model_rows, columns=list(MODEL_COLUMNS))
    errors = pd.DataFrame(error_rows, columns=list(ERROR_COLUMNS))
    if not keep_coefficients:
        return models, errors
    return models, errors, pd.DataFrame(coef_rows, columns=list(COEFFICIENT_COLUMNS))


def window_table(runs, horizons_s, max_delays: int = 0, offset: int = 0) -> pd.DataFrame:
    """Windows of the evaluation segment of each run at each horizon.

    The local sample index of a window start depends on where the segment
    begins; its absolute time does not. Runs whose steps happen at the same
    instants have the same start_time_s even if their starts differ.

    Parameters
    ----------
    runs : iterable of tuple
        One tuple (name, opening_pct, train, test) per run, as in sweep. Only
        the evaluation segment `test` is used.
    horizons_s : sequence of float
        Prediction horizons, in seconds; read again for every run.
    max_delays : int, default 0
        Largest n_delays among the models to be scored: the windows must leave
        room for its history.
    offset : int, default 0
        Samples by which every window start is moved.

    Returns
    -------
    pandas.DataFrame
        One row per (run, horizon_s) with horizon_steps, n_windows, and, as
        text joined by ";", starts (window starts X, local sample index),
        start_time_s (absolute time of each X, in s), sp_before_pct and
        sp_after_pct (setpoint at X and at X + 1, in %).

    Raises
    ------
    ValueError
        If the windows are not valid; see
        :func:`edmdc_tank_observables.evaluation.metrics.window_starts`.
    """
    rows = []
    for name, _, _, test in runs:
        sp = test.denormalize_h(test.sp)
        for horizon_s in horizons_s:
            # Same rounding as the multi-step error, so both tables agree.
            horizon_steps = _horizon_to_steps(horizon_s, test.ts)
            starts = window_starts(test, horizon_steps, n_delays=max_delays, offset=offset)
            rows.append(
                {
                    "run": name,
                    "horizon_s": horizon_s,
                    "horizon_steps": horizon_steps,
                    "n_windows": len(starts),
                    "starts": ";".join(str(int(x)) for x in starts),
                    "start_time_s": ";".join(f"{test.t[x]:g}" for x in starts),
                    "sp_before_pct": ";".join(f"{sp[x]:g}" for x in starts),
                    "sp_after_pct": ";".join(f"{sp[x + 1]:g}" for x in starts),
                }
            )
    return pd.DataFrame(rows)


def check_same_windows(errors: pd.DataFrame, windows: pd.DataFrame) -> None:
    """Check that every series of a run and horizon was scored on the windows of the windows table.

    The windows of a run depend only on its setpoint, the horizon and the
    offset, so, for tables built with the same offset, the same number of
    windows and the same horizon in steps mean the same windows.

    Parameters
    ----------
    errors : pandas.DataFrame
        Errors table, as returned by sweep.
    windows : pandas.DataFrame
        Windows table of the same runs, as returned by window_table.

    Raises
    ------
    ValueError
        If some row of `errors` has no matching run and horizon in `windows`,
        or a different number of windows or horizon in steps.
    """
    joined = errors.merge(windows, on=["run", "horizon_s"], how="left", suffixes=("", "_expected"))
    bad = joined[
        (joined["n_windows"] != joined["n_windows_expected"])
        | (joined["horizon_steps"] != joined["horizon_steps_expected"])
    ]
    if len(bad) > 0:
        cases = bad[["run", "series", "horizon_s"]].drop_duplicates().head(5).to_dict("records")
        raise ValueError(f"Series not scored on the expected windows, e.g. {cases}.")


def windows_shared_across_runs(windows: pd.DataFrame) -> dict:
    """Whether every run has its windows at the same absolute instants, per horizon.

    Parameters
    ----------
    windows : pandas.DataFrame
        Table returned by window_table.

    Returns
    -------
    dict
        Horizon in s -> True if start_time_s is the same in every run.
    """
    return {h: bool(block["start_time_s"].nunique() == 1) for h, block in windows.groupby("horizon_s")}


def summarize_across_runs(errors: pd.DataFrame, column: str = "rmse") -> pd.DataFrame:
    """Median, minimum and maximum across runs of an error column, for each series and horizon.

    The median is used instead of the mean so that a single run with a poor fit
    does not drag the summary; the minimum and maximum give its range.

    Parameters
    ----------
    errors : pandas.DataFrame
        Errors table with at least the columns series, horizon_s and `column`,
        one row per run, series and horizon.
    column : str, default "rmse"
        Column to summarize ("rmse", "rmse_final" or "max_abs").

    Returns
    -------
    pandas.DataFrame
        One row per (series, horizon_s) with the columns median, min, max and
        n_runs (number of rows summarized).
    """
    return (
        errors.groupby(["series", "horizon_s"])[column]
        .agg(median="median", min="min", max="max", n_runs="count")
        .reset_index()
    )


def count_improvements(errors: pd.DataFrame, reference: str, column: str = "rmse") -> pd.DataFrame:
    """Count in how many runs each series has a lower error than a reference series.

    The comparison is run by run at the same horizon. A tie is not an
    improvement.

    Parameters
    ----------
    errors : pandas.DataFrame
        Errors table with at least the columns run, series, horizon_s and
        `column`, one row per run, series and horizon.
    reference : str
        Series to compare with, e.g. "D0".
    column : str, default "rmse"
        Error column to compare.

    Returns
    -------
    pandas.DataFrame
        One row per (series, horizon_s) for every series except the reference,
        with the columns n_better (runs in which the series has a lower error
        than the reference) and n_compared (runs in which the reference has an
        error; a run in which the series has none counts as compared and not
        better).

    Raises
    ------
    ValueError
        If the reference series is not in the table.
    """
    if reference not in set(errors["series"]):
        raise ValueError(f"The reference series {reference!r} is not in the errors table.")

    keys = ["run", "horizon_s"]
    ref = errors[errors["series"] == reference].set_index(keys)[column].rename("reference")
    joined = errors[errors["series"] != reference].join(ref, on=keys).dropna(subset=["reference"])
    joined["better"] = joined[column] < joined["reference"]
    return (
        joined.groupby(["series", "horizon_s"])["better"]
        .agg(n_better="sum", n_compared="count")
        .reset_index()
        .astype({"n_better": int, "n_compared": int})
    )


def window_trajectory(
    model: EDMDcModel, test: PreprocessedDataset, start: int, horizon_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Predicted level over one window, for a figure.

    The model is initialized and fed exactly as in the multi-step error, so the
    trajectory drawn is the one that is scored: history h[start - n_delays :
    start + 1], control u[start : start + horizon_steps].

    Parameters
    ----------
    model : EDMDcModel
        Fitted model.
    test : PreprocessedDataset
        Evaluation segment.
    start : int
        Window start X, the last sample the model sees.
    horizon_steps : int
        Number of steps to predict.

    Returns
    -------
    t, h_pred : numpy.ndarray
        Time from the window start, in s (ts, 2 ts, ..., horizon_steps ts),
        and predicted level, in % of level. h_pred[j] is compared with
        h[start + 1 + j].

    Raises
    ------
    ValueError
        If start < n_delays of the model, so there is not enough history, if
        the window does not fit in the segment, or, from
        :meth:`EDMDcModel.simulate`, if the dictionary is physical and the
        history has a negative level.
    """
    n_delays = model.dictionary.n_delays
    if start < n_delays or start + horizon_steps + 1 > len(test):
        raise ValueError(
            f"{test.name}: a window of {horizon_steps} steps from sample {start} needs "
            f"{n_delays} delays of history and samples up to {start + horizon_steps}, "
            f"but the segment has {len(test)} samples."
        )
    history = test.h[start - n_delays : start + 1]
    prediction = model.simulate(history, test.u[start : start + horizon_steps])
    t = np.arange(1, horizon_steps + 1) * test.ts
    return t, test.denormalize_h(prediction)
