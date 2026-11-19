"""E1 - Selection of the hyperparameter of each dictionary family.

What is configured, and why
---------------------------
Data. Each Baseline run is split as in E2: the first TRAIN_FRAC of the
acquisition time is the fit part and the rest the evaluation part. E1 never
uses the evaluation part. The fit part is split again with the same temporal
cut: its first INNER_FIT_FRAC is the inner fit part and the rest the inner
validation part. Both cuts are made at an absolute instant, so they fall at
the same instants in every run (the runs are checked to end at the same
instant).
Every candidate is fitted on the inner fit part and scored on the inner
validation part, so the evaluation part stays unseen by the selection.

Grids. Every integer of order 2-6 (D1a, D3a), n_centers 1-9 (D1b, D3b) and
n_delays 1-8 (D1c, D3c). The lower ends are the limits of each family: order 1
is D0, and fewer than one centre or one delay is not that family. The same
stopping rule applies to the three families instead of the same number of
candidates: if the selected value is the last value of its grid, the script
writes its outputs, warns, exits with an error and does not report the value
as optimal; the grid must then be extended and E1 run again. A selection at the
lower end is not flagged.

Scoring. The multi-step error on the windows anchored at the setpoint steps of
the inner validation part, the same windows and code as in E2. D0 is scored on
the same windows as a reference. The number of windows per run and horizon,
their starts and setpoint levels are printed and saved, and every series is
checked to be scored on the same windows.

Aggregation. The median across runs of the RMSE, with the minimum and maximum,
as in E2: one run with a poor fit does not drag the median.

Selection. The criterion is the median RMSE at SELECTION_HORIZON_S (120 s): a
single horizon, so that the criterion is one figure. S1 reports the winner at
every horizon of HORIZONS_S (30, 120 and 600 s) only to check whether it
changes with the horizon. The selected value of each family is the strict
minimum: the candidate with the lowest median, however small the difference with the next one (an exact tie
goes to the candidate with fewer observables). It is the standard criterion
and needs no threshold. For D1a/b/c it is the value E2 uses.

Parsimony, as a sensitivity only. The script also reports which candidate a
parsimony rule would keep: among the candidates whose median does not exceed
the maximum across runs of the selected one, the one with fewest observables.
It is a heuristic and does not decide: its threshold, the maximum across runs
of the selected candidate, reflects how much the runs differ from each other
(different valve openings) and not the uncertainty of the selection.

Combined dictionaries. D3a/b/c take the value selected for D1a/b/c, so that
D1x versus D3x measures the effect of adding sqrt(h) and not that of retuning
the hyperparameter. D3a/b/c are still selected on their own grid, and their own
value is reported against the inherited one as a sensitivity.

Sensitivities, printed and saved
--------------------------------
S1  Selected value at 30, 120 and 600 s per family, and whether it
    differs between horizons.
S2  Selected value against the value the parsimony heuristic would keep, with
    its threshold and the candidates within it.
S3  For D3a/b/c, own selection against the value inherited from D1a/b/c, with
    the median RMSE of both at 120 s.
S4  In how many runs the selected candidate (and, for D3a/b/c, the inherited
    one) has the lowest RMSE of its family at 120 s. A count, with no
    statistical test.
S5  Windows of the inner validation part per run and horizon: number, starts
    (local sample index and absolute time) and setpoint levels before and
    after each step, and whether the start times are the same in every run
    (this last flag is printed only).
S6  Fit indicators per candidate: p, condition number, spectral radius,
    in-sample one-step RMSE, number of pairs and, for RBF, the kernel width.

Limitations
-----------
- The inner validation part is small: it holds only the setpoint steps that
  fall in the last (1 - INNER_FIT_FRAC) of the fit part (their number is
  printed in S5), so the selection rests on a few step responses per run.
- The runs were recorded on different days, with the same setpoint profile
  and a different electrovalve opening; the independence between them cannot
  be verified, so the median across them is descriptive.
- The setpoint staircase rises, so the inner validation part covers higher
  levels than the inner fit part: like the evaluation in E2, the selection
  measures extrapolation in level, not interpolation. The level ranges of both
  parts are printed.
- In the RBF dictionaries the centres are equally spaced in [0, 1] and the
  width is the spacing between them, so M changes the number and the width of
  the basis functions together; the curve over M measures both, and the width
  is not tuned on its own.
- With a few step responses, neighbouring values may not be distinguishable,
  and the minimum over a wide grid has more chances to win by chance. No
  tie-breaking rule fixes this; S1, S2 and S4 make it visible.

Outputs in results/E1_hyperparameters/: CSV tables in tables/, one figure per
family in figures/ and metadata.json.
"""

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window: the script only writes files

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from edmdc_tank_observables.data.loader import load_baseline
from edmdc_tank_observables.data.preprocess import common_end_time, prepare, temporal_split
from edmdc_tank_observables.evaluation.protocol import (
    check_same_windows,
    sweep,
    window_table,
    windows_shared_across_runs,
)
from edmdc_tank_observables.paths import PROJECT_ROOT
from edmdc_tank_observables.provenance import write_metadata
from edmdc_tank_observables.reporting import save_figure, set_style

# --- Configuration -------------------------------------------------------------

SUBSET = "Baseline"
TS_S = 5.0
# Fit / evaluation split, the same as in E2. The evaluation part is not used.
TRAIN_FRAC = 0.7
# Inner split of the fit part: inner fit / inner validation.
INNER_FIT_FRAC = 0.75

# Hyperparameter and grid of each family: every integer of the range. The D1
# families come first because the D3 families inherit their values.
GRIDS = {
    "D1a": ("order", tuple(range(2, 7))),
    "D1b": ("n_centers", tuple(range(1, 10))),
    "D1c": ("n_delays", tuple(range(1, 9))),
    "D3a": ("order", tuple(range(2, 7))),
    "D3b": ("n_centers", tuple(range(1, 10))),
    "D3c": ("n_delays", tuple(range(1, 9))),
}

# Family whose selected value each combined dictionary uses in E2.
INHERITS_FROM = {"D3a": "D1a", "D3b": "D1b", "D3c": "D1c"}

# Dictionaries scored as references, without hyperparameter.
REFERENCE_DICTIONARIES = ("D0",)

SELECTION_HORIZON_S = 120.0
HORIZONS_S = (30.0, 120.0, 600.0)

SELECTION_RULE = (
    "Selected: the candidate with the lowest median RMSE across runs at the selection "
    "horizon (exact ties to fewer observables). A selection at the last value of a grid "
    "is flagged and not reported as optimal; the first value is the limit of the family. "
    "Parsimony is reported as a sensitivity only and does not decide (a heuristic): among "
    "the candidates whose median does not exceed the maximum across runs of the selected "
    "one, the one with fewest observables."
)

# Placeholder for the hyperparameter of the reference series.
NO_HYPERPARAMETER = "none"

OUT_DIR = PROJECT_ROOT / "results" / "E1_hyperparameters"


# --- Data ----------------------------------------------------------------------


def split_run(dataset):
    """Split a full run into its inner fit and inner validation parts.

    The run is cut at TRAIN_FRAC of its acquisition time as in E2 and the
    evaluation part is dropped; the fit part is cut again at INNER_FIT_FRAC of
    its own (see temporal_split).

    Parameters
    ----------
    dataset : PreprocessedDataset
        Full run, as returned by prepare.

    Returns
    -------
    fit, validation : PreprocessedDataset
        Inner fit and inner validation parts, named with the suffixes
        _inner_fit and _inner_validation.

    Raises
    ------
    ValueError
        If a cut leaves a part empty; see temporal_split.
    """
    train, _ = temporal_split(dataset, TRAIN_FRAC)
    fit, validation = temporal_split(train, INNER_FIT_FRAC)
    return (
        replace(fit, name=dataset.name + "_inner_fit"),
        replace(validation, name=dataset.name + "_inner_validation"),
    )


def load_runs() -> list[tuple]:
    """Load, preprocess and split every Baseline run into its inner parts.

    Returns
    -------
    list of tuple
        One tuple (name, opening_pct, fit, validation) per run, from lowest to
        highest opening, as expected by sweep.

    Raises
    ------
    ValueError
        If a run has gaps after resampling to TS_S (see prepare), or if the
        runs do not end at the same instant (see common_end_time), so they
        would not be cut at the same instants.
    """
    raws = load_baseline()
    datasets = [prepare(raw, TS_S) for raw in raws]
    common_end_time(datasets)
    return [(raw.name, raw.valve_opening_pct, *split_run(d)) for raw, d in zip(raws, datasets)]


def print_level_ranges(runs) -> None:
    """Print the minimum and maximum level of the inner fit and inner validation parts of each run, in %.

    Parameters
    ----------
    runs : list of tuple
        Runs as returned by load_runs.
    """
    print("\nLevel range (% of level), inner fit part vs inner validation part")
    for name, _, fit, validation in runs:
        h_fit, h_val = fit.denormalize_h(fit.h), validation.denormalize_h(validation.h)
        print(
            f"  {name:<16} fit h [{h_fit.min():6.2f}, {h_fit.max():6.2f}]"
            f"   validation h [{h_val.min():6.2f}, {h_val.max():6.2f}]"
        )


# --- Scoring -------------------------------------------------------------------


def grid_candidates(grids) -> list[tuple]:
    """List every candidate of the grids.

    Parameters
    ----------
    grids : dict
        Family id -> (hyperparameter name, values), as GRIDS.

    Returns
    -------
    list of tuple
        One (family, hyperparameter, value) per value, family by family in the
        order of `grids`.
    """
    return [(family, name, value) for family, (name, values) in grids.items() for value in values]


def rbf_width(n_centers: int) -> float:
    """Width of the Gaussian kernels of an RBF dictionary with default range and width.

    It is the rule the dictionary applies: the spacing between the centres,
    equally spaced in [0, 1], or 1 for a single centre.

    Parameters
    ----------
    n_centers : int
        Number of centres M.

    Returns
    -------
    float
        1 / (M - 1) if M > 1, else 1.0, in normalized level.
    """
    return 1.0 / (n_centers - 1) if n_centers > 1 else 1.0


def _label(table: pd.DataFrame, position: int, name: str, value) -> None:
    """Insert the hyperparameter name and value columns of a candidate into a sweep table."""
    table.insert(position, "hyperparameter", name)
    table.insert(position + 1, "value", value)


def score_candidates(runs, grids=GRIDS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every candidate of the grids and the references, and score them on the inner validation parts.

    Parameters
    ----------
    runs : list of tuple
        Runs as returned by load_runs: (name, opening_pct, fit, validation).
    grids : dict, default GRIDS
        Family id -> (hyperparameter name, values).

    Returns
    -------
    models, errors : pandas.DataFrame
        The models and errors tables of sweep for every candidate, the
        references (REFERENCE_DICTIONARIES), with two more columns: hyperparameter (NO_HYPERPARAMETER for the
        references) and value (NaN for the references). The models table also
        has rbf_width (NaN except for RBF dictionaries).

    Raises
    ------
    ValueError
        If a candidate cannot be fitted or the windows are not valid for some
        run; see sweep.
    """
    model_tables, error_tables = [], []

    for family, name, value in grid_candidates(grids):
        models, errors = sweep(runs, (family,), hyperparams={family: {name: value}}, horizons_s=HORIZONS_S)
        _label(models, 4, name, value)
        models["rbf_width"] = rbf_width(value) if name == "n_centers" else np.nan
        _label(errors, 4, name, value)
        model_tables.append(models)
        error_tables.append(errors)

    models, errors = sweep(runs, REFERENCE_DICTIONARIES, horizons_s=HORIZONS_S)
    _label(models, 4, NO_HYPERPARAMETER, np.nan)
    models["rbf_width"] = np.nan
    _label(errors, 4, NO_HYPERPARAMETER, np.nan)
    model_tables.append(models)
    error_tables.append(errors)

    return pd.concat(model_tables, ignore_index=True), pd.concat(error_tables, ignore_index=True)


# --- Aggregation and selection -------------------------------------------------


def summary_table(errors: pd.DataFrame, models: pd.DataFrame) -> pd.DataFrame:
    """RMSE per candidate and horizon across runs: median, minimum and maximum, with the number of observables.

    Parameters
    ----------
    errors : pandas.DataFrame
        Errors table returned by score_candidates.
    models : pandas.DataFrame
        Models table returned by score_candidates.

    Returns
    -------
    pandas.DataFrame
        One row per (series, hyperparameter, value, horizon_s) with p, median,
        min, max and n_runs.
    """
    keys = ["series", "hyperparameter", "value"]
    summary = (
        errors.groupby([*keys, "horizon_s"], dropna=False)["rmse"]
        .agg(median="median", min="min", max="max", n_runs="count")
        .reset_index()
    )
    p = models.groupby(["dictionary", "hyperparameter", "value"], dropna=False)["p"].first()
    p = p.rename_axis(keys).reset_index()
    summary = summary.merge(p, on=keys, how="left")
    # Float, like value (e.g. 1.0), so that a series without a model gets NaN.
    summary["p"] = summary["p"].astype(float)
    return summary[[*keys, "p", "horizon_s", "median", "min", "max", "n_runs"]]


def select(table: pd.DataFrame) -> dict:
    """Select the candidate of one family at one horizon, and report the parsimony sensitivity.

    Parameters
    ----------
    table : pandas.DataFrame
        One row per candidate of the family with the columns value, p,
        median and max (median and maximum RMSE across runs).

    Returns
    -------
    dict
        selected_value and selected_median: candidate with the lowest median
        (exact ties go to the smaller p, then the smaller value);
        selected_at_upper_edge: whether it is the largest value of the table;
        parsimony_threshold: maximum across runs of the selected candidate;
        parsimony_tied_values: values whose median does not exceed that
        threshold, in increasing order;
        parsimony_value and parsimony_median: among those, the one with the
        smallest p (then the smaller value). The parsimony keys are a
        sensitivity and do not change the selection.
    """
    best = table.sort_values(["median", "p", "value"]).iloc[0]
    threshold = float(best["max"])
    tied = table[table["median"] <= threshold].sort_values(["p", "value"])
    parsimonious = tied.iloc[0]
    return {
        "selected_value": int(best["value"]),
        "selected_median": float(best["median"]),
        "selected_at_upper_edge": bool(best["value"] == table["value"].max()),
        "parsimony_threshold": threshold,
        "parsimony_tied_values": sorted(int(v) for v in tied["value"]),
        "parsimony_value": int(parsimonious["value"]),
        "parsimony_median": float(parsimonious["median"]),
    }


def runs_where_best(errors: pd.DataFrame, family: str, value: int, horizon_s: float) -> tuple[int, int]:
    """Count the runs in which a candidate has the lowest RMSE of its family (S4).

    Parameters
    ----------
    errors : pandas.DataFrame
        Errors table returned by score_candidates.
    family : str
        Family id, e.g. "D1a".
    value : int
        Value of the candidate.
    horizon_s : float
        Horizon, in s.

    Returns
    -------
    n_best, n_runs : int
        Runs in which the candidate's RMSE equals the lowest of the family
        (a tie counts as best), and runs compared.
    """
    block = errors[(errors["series"] == family) & (errors["horizon_s"] == horizon_s)]
    lowest = block.groupby("run")["rmse"].transform("min")
    is_best = block[block["value"] == value]["rmse"] == lowest[block["value"] == value]
    return int(is_best.sum()), int(block["run"].nunique())


def _family_rows(summary: pd.DataFrame, family: str, horizon_s: float) -> pd.DataFrame:
    """Summary rows of one family at one horizon."""
    return summary[(summary["series"] == family) & (summary["horizon_s"] == horizon_s)]


def selection_table(summary: pd.DataFrame, errors: pd.DataFrame, grids=GRIDS) -> pd.DataFrame:
    """Selection per family with the sensitivities S1 to S4.

    Parameters
    ----------
    summary : pandas.DataFrame
        Table returned by summary_table.
    errors : pandas.DataFrame
        Errors table returned by score_candidates.
    grids : dict, default GRIDS
        Family id -> (hyperparameter name, values). Families in INHERITS_FROM
        must come after the family they inherit from.

    Returns
    -------
    pandas.DataFrame
        One row per family with the keys of select (parsimony_tied_values as
        text joined by ";"; S2), the selected value at each horizon in
        winner_<h>s and winner_differs (S1), the run counts of S4, and for the
        families in INHERITS_FROM the inherited value, its median at the
        selection horizon and whether the own selection differs (S3).
        e2_value is the value E2 uses: the inherited one if any, else the
        selected one.
    """
    rows = {}
    for family, (name, _) in grids.items():
        result = select(_family_rows(summary, family, SELECTION_HORIZON_S))
        row = {"family": family, "hyperparameter": name, **result}
        row["parsimony_tied_values"] = ";".join(str(v) for v in result["parsimony_tied_values"])

        winners = [select(_family_rows(summary, family, h))["selected_value"] for h in HORIZONS_S]
        for horizon_s, winner in zip(HORIZONS_S, winners):
            row[f"winner_{horizon_s:g}s"] = winner
        row["winner_differs"] = len(set(winners)) > 1

        n_best, n_runs = runs_where_best(errors, family, result["selected_value"], SELECTION_HORIZON_S)
        row["n_runs_selected_best"] = n_best
        row["n_runs"] = n_runs

        parent = INHERITS_FROM.get(family)
        if parent is None:
            row.update(inherited_from=None, inherited_value=None, inherited_median=None, own_differs=None)
            row["e2_value"] = result["selected_value"]
        else:
            inherited = rows[parent]["selected_value"]
            rows_120 = _family_rows(summary, family, SELECTION_HORIZON_S)
            row["inherited_from"] = parent
            row["inherited_value"] = inherited
            row["inherited_median"] = float(rows_120[rows_120["value"] == inherited]["median"].iloc[0])
            row["own_differs"] = result["selected_value"] != inherited
            row["n_runs_inherited_best"] = runs_where_best(errors, family, inherited, SELECTION_HORIZON_S)[0]
            row["e2_value"] = inherited
        rows[family] = row
    return pd.DataFrame(list(rows.values()))


def fit_indicators_table(models: pd.DataFrame) -> pd.DataFrame:
    """Fit indicators per candidate, median across runs (S6).

    Parameters
    ----------
    models : pandas.DataFrame
        Models table returned by score_candidates.

    Returns
    -------
    pandas.DataFrame
        One row per (dictionary, hyperparameter, value) with p, rbf_width and
        the median across runs of cond, spectral_radius, fit_rmse_pct and
        n_pairs.
    """
    return (
        models.groupby(["dictionary", "hyperparameter", "value"], dropna=False, sort=False)
        .agg(
            p=("p", "first"),
            rbf_width=("rbf_width", "first"),
            cond=("cond", "median"),
            spectral_radius=("spectral_radius", "median"),
            fit_rmse_pct=("fit_rmse_pct", "median"),
            n_pairs=("n_pairs", "median"),
        )
        .reset_index()
    )


def edge_warnings(selection: pd.DataFrame, grids=GRIDS) -> list[str]:
    """Warnings for every family whose selected value falls on the last value of its grid.

    Parameters
    ----------
    selection : pandas.DataFrame
        Table returned by selection_table.
    grids : dict, default GRIDS
        Family id -> (hyperparameter name, values).

    Returns
    -------
    list of str
        One message per family whose selected value is the upper end of the
        grid. Empty if there is none.
    """
    warnings = []
    for _, row in selection[selection["selected_at_upper_edge"]].iterrows():
        name, values = grids[row["family"]]
        warnings.append(
            f"{row['family']}: the selected {name} = {row['selected_value']} is the upper end of "
            f"the grid {min(values)}-{max(values)}; it is not an optimum. Extend the grid and run E1 again."
        )
    return warnings


# --- Console output ------------------------------------------------------------


def print_results(summary, selection, windows, fit, warnings) -> None:
    """Print the windows, the error curves, the selection and the sensitivities to the console."""
    print("\n" + "=" * 78)
    print("E1 - Hyperparameter selection on the inner validation part")
    print("=" * 78)

    print("\nS5 - Windows of the inner validation part (run x horizon)")
    print(windows.to_string(index=False))
    same = windows_shared_across_runs(windows)
    print("Same window start times in every run: " + ", ".join(f"{h:g} s: {v}" for h, v in same.items()))

    print("\nRMSE (% of level), median across runs [min, max]")
    cells = summary.assign(
        candidate=summary["series"] + summary["value"].map(lambda v: "" if pd.isna(v) else f" {int(v)}"),
        cell=summary.apply(lambda r: f"{r['median']:.3f} [{r['min']:.3f}, {r['max']:.3f}]", axis=1),
    )
    table = cells.pivot(index="candidate", columns="horizon_s", values="cell")
    print(table.loc[cells["candidate"].drop_duplicates()].to_string())

    columns = [
        "family", "hyperparameter", "selected_value", "selected_median",
        "parsimony_threshold", "parsimony_tied_values", "parsimony_value", "parsimony_median",
    ]
    print(f"\nS2 - Selected value (lowest median) vs parsimony heuristic (sensitivity only) at {SELECTION_HORIZON_S:g} s")
    print(selection[columns].to_string(index=False))

    print("\nS1 - Selected value per horizon")
    winners = [f"winner_{h:g}s" for h in HORIZONS_S]
    print(selection[["family", *winners, "winner_differs"]].to_string(index=False))

    print("\nS3 - Combined dictionaries: own selection vs inherited value")
    combined = selection[selection["inherited_from"].notna()]
    print(
        combined[
            ["family", "selected_value", "selected_median", "inherited_from", "inherited_value", "inherited_median", "own_differs"]
        ].to_string(index=False)
    )

    print(f"\nS4 - Runs in which the candidate has the lowest RMSE of its family at {SELECTION_HORIZON_S:g} s")
    s4 = selection[["family", "selected_value", "n_runs_selected_best", "n_runs"]]
    print(s4.to_string(index=False))
    print(combined[["family", "inherited_value", "n_runs_inherited_best"]].to_string(index=False))

    print("\nS6 - Fit indicators per candidate (median across runs)")
    print(fit.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    print("\nValues for E2 (e2_value): " + ", ".join(f"{r.family} {r.hyperparameter}={r.e2_value}" for r in selection.itertuples()))
    for message in warnings:
        print(f"WARNING: {message}")
    print()


# --- Figures -------------------------------------------------------------------


def figure_family(summary: pd.DataFrame, selection_row: pd.Series) -> plt.Figure:
    """Median RMSE (with the range across runs) against the hyperparameter of one family, one panel per horizon.

    D0 is drawn as a horizontal reference; the selected value and the parsimony sensitivity at the selection horizon are
    marked.
    """
    family, name = selection_row["family"], selection_row["hyperparameter"]
    fig, axes = plt.subplots(1, len(HORIZONS_S), figsize=(10.0, 3.4))
    for ax, horizon_s in zip(axes, HORIZONS_S):
        rows = _family_rows(summary, family, horizon_s).sort_values("value")
        ax.fill_between(rows["value"], rows["min"], rows["max"], color="#0072B2", alpha=0.2, linewidth=0)
        ax.plot(rows["value"], rows["median"], marker="o", markersize=4, color="#0072B2", label=family)
        ref = summary[(summary["series"] == "D0") & (summary["horizon_s"] == horizon_s)]
        if len(ref) > 0:
            ax.axhline(ref["median"].iloc[0], color="#D55E00", linestyle="--", linewidth=1.0, label="D0")
        if horizon_s == SELECTION_HORIZON_S:
            for key, marker in (("selected_value", "*"), ("parsimony_value", "o")):
                point = rows[rows["value"] == selection_row[key]]
                ax.plot(point["value"], point["median"], marker=marker, markersize=12, markerfacecolor="none",
                        color="#000000", linestyle="none", label=key.replace("_value", ""))
        ax.set_title(f"{horizon_s:g} s")
        ax.set_xlabel(name)
        ax.set_xticks(rows["value"])
    axes[0].set_ylabel("RMSE (% of level)")
    axes[-1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


# --- Run -----------------------------------------------------------------------


def run_experiment(runs, out_dir) -> dict:
    """Score the grids on the inner validation parts, select per family and write tables, figures and metadata.

    Parameters
    ----------
    runs : list of tuple
        Runs as returned by load_runs.
    out_dir : pathlib.Path or str
        Output folder. Tables go to out_dir/tables, figures to out_dir/figures.

    Returns
    -------
    dict
        The tables (keys models, errors, summary, selection, windows, fit) and
        the list of upper-edge warnings (key warnings).

    Raises
    ------
    ValueError
        If a candidate cannot be fitted, if the windows are not valid for some
        run, or if some series was not scored on the expected windows.
    """
    out_dir = Path(out_dir)
    print(f"Runs used: {len(runs)}")
    print_level_ranges(runs)

    max_delays = max(max(values) for name, values in GRIDS.values() if name == "n_delays")
    windows = window_table(runs, HORIZONS_S, max_delays)
    models, errors = score_candidates(runs)
    check_same_windows(errors, windows)

    summary = summary_table(errors, models)
    selection = selection_table(summary, errors)
    fit = fit_indicators_table(models)
    warnings = edge_warnings(selection)
    print_results(summary, selection, windows, fit, warnings)

    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    models.to_csv(tables_dir / "models.csv", index=False)
    errors.to_csv(tables_dir / "errors.csv", index=False)
    summary.to_csv(tables_dir / "error_summary.csv", index=False)
    selection.to_csv(tables_dir / "selection.csv", index=False)
    windows.to_csv(tables_dir / "windows.csv", index=False)
    fit.to_csv(tables_dir / "fit_indicators.csv", index=False)

    set_style()
    for _, row in selection.iterrows():
        save_figure(figure_family(summary, row), out_dir / "figures", f"error_vs_{row['family']}")

    by_family = selection.set_index("family")
    write_metadata(
        out_dir,
        config={
            "subset": SUBSET,
            "ts_s": TS_S,
            "train_frac": TRAIN_FRAC,
            "inner_fit_frac": INNER_FIT_FRAC,
            "evaluation_part_used": False,
            "grids": {f: {"hyperparameter": n, "values": list(v)} for f, (n, v) in GRIDS.items()},
            "inherits_from": INHERITS_FROM,
            "references": list(REFERENCE_DICTIONARIES),
            "horizons_s": list(HORIZONS_S),
            "selection_horizon_s": SELECTION_HORIZON_S,
            "aggregation": "median across runs, with minimum and maximum",
            "selection_rule": SELECTION_RULE,
            "window_offset": 0,
            "runs": [r[0] for r in runs],
        },
        summary={
            "n_runs": len(runs),
            "n_candidates": len(grid_candidates(GRIDS)),
            "selected_hyperparameters": {
                f: {GRIDS[f][0]: int(by_family.loc[f, "e2_value"])} for f in GRIDS
            },
            "own_selection": {f: {GRIDS[f][0]: int(by_family.loc[f, "selected_value"])} for f in GRIDS},
            "parsimony_sensitivity": {f: {GRIDS[f][0]: int(by_family.loc[f, "parsimony_value"])} for f in GRIDS},
            "n_windows_per_run": {
                f"{h:g}": sorted(int(n) for n in block["n_windows"].unique())
                for h, block in windows.groupby("horizon_s")
            },
            "upper_edge_warnings": warnings,
        },
    )
    print(f"Results in {out_dir}")
    return {
        "models": models, "errors": errors, "summary": summary, "selection": selection,
        "windows": windows, "fit": fit, "warnings": warnings,
    }


def main() -> None:
    """Run E1 on every Baseline run, write the outputs to OUT_DIR and stop with an error if a selection is at an upper edge."""
    result = run_experiment(load_runs(), OUT_DIR)
    if result["warnings"]:
        raise SystemExit(
            "E1 finished with selections at the upper end of their grid; they are not optima:\n  "
            + "\n  ".join(result["warnings"])
        )


if __name__ == "__main__":
    main()
