"""Paper assets - LaTeX table fragments and figures for the manuscript, written from saved results.

The script does not fit or analyse anything: it reads CSV tables written by
E2_dictionaries, E2_comparisons and E2_one_step and formats some of their
columns as LaTeX, and it plots one raw run with the evaluation windows of E2.
It writes only in results/paper_assets/; the .tex files and the figure are
copied into the manuscript by hand.

Tables
------
table_errors.tex
    One row per dictionary: median across runs of the in-sample one-step RMSE
    (fit_rmse_pct of fit_indicators.csv, already a median across runs), with
    FIT_DECIMALS decimals, and the multi-step RMSE at each horizon of
    ERROR_HORIZONS_S as "median [min, max]" across runs (error_summary.csv).
table_paired_120s.tex
    Paired comparisons at PAIRED_HORIZON_S from paired_tests.csv, in the groups
    of PAIRED_GROUPS: for each pair "A vs B", the runs where each has the lower
    RMSE, the median of d = RMSE_A - RMSE_B with its bootstrap 95 % interval,
    and the sign-test and Wilcoxon p-values. Groups with a Holm family use the
    p-values adjusted within that family; the others use the unadjusted ones.
    A pair stored as (B, A) is turned round: the counts are swapped and the
    median and interval change sign, with the bounds swapped.
table_one_step.tex
    For each pair of ONE_STEP_PAIRS, written as in the paired table: counts
    and median difference of the one-step RMSE at the window instants at
    ONE_STEP_HORIZON_S, of the multi-step RMSE at that horizon and of the
    one-step RMSE on the whole evaluation part, in that column order
    (one_step_pair_counts.csv). A pair stored as (B, A)
    is turned round: the counts are swapped and the medians change sign. The
    pairs must be those of the E2_one_step metadata, in either orientation.

Each file holds only a tabular environment with booktabs rules; the caption,
label and float are written in the manuscript. Dictionaries are written with
the manuscript macros of MACROS. The first line of each file is a comment with
its source tables, the commit of the results read (from their metadata.json)
and the commit of this script.

Figure
------
figure_run_windows.pdf
    Level and setpoint of FIGURE_RUN against time, sampled at the Ts of E2,
    zoomed on the end of the fit part and the first N_WINDOWS_SHOWN evaluation
    windows: from MARGIN_BEFORE_SPLIT_S before the split to
    MARGIN_AFTER_LAST_WINDOW_S after the end of the longest horizon of the last
    window shown. A dotted vertical line marks the first sample of the
    evaluation part (the split of E2, at its train_frac); "Fit" is centred on
    the fit part shown and "Evaluation" on the longest band-free interval of
    the evaluation part shown, both in italics at the top. Each window shown
    has one band per horizon of E2, from its start X (the last sample before the step) to X + horizon,
    lighter for longer horizons and drawn longest first so the shorter ones
    stay visible. The Ts, train_frac, horizons and window starts are read from
    the E2 results, not copied. Vector PDF, FIGURE_SIZE_CM, grayscale, with
    the fonts of FIGURE_FONT_RC (Times-like, embedded as TrueType). A
    provenance line in the format of the tables (sources: the raw file of
    FIGURE_RUN and windows.csv; E2 results commit; script commit) goes into the
    PDF Subject field.

Invariants: every source metadata.json must come from a clean commit, every
dictionary, horizon and pair must be found exactly once, the Holm column of a
group must not be empty, there must be no ties (the "n_X/n_Y" cells could not
show them), ONE_STEP_PAIRS must be the pairs of the E2_one_step metadata, the
window start times of windows.csv must be the same at every horizon and be the
times of their start samples in the evaluation part rebuilt here, the figure
needs at least N_WINDOWS_SHOWN windows and no more horizons than BAND_GREYS,
and each segment label must fit inside its interval.

Outputs: the three .tex files and metadata.json in OUT_DIR
(results/paper_assets/tables/); the figure and metadata.json in FIGURE_DIR
(results/paper_assets/figures/).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from edmdc_tank_observables.data.loader import load_baseline_file
from edmdc_tank_observables.data.preprocess import prepare, temporal_split
from edmdc_tank_observables.paths import DEFAULT_RAW_DIR, PROJECT_ROOT
from edmdc_tank_observables.provenance import git_state, write_metadata
from edmdc_tank_observables.reporting import set_style

# --- Configuration -------------------------------------------------------------

E2_DIR = PROJECT_ROOT / "results" / "E2_dictionaries"
CMP_DIR = PROJECT_ROOT / "results" / "E2_comparisons"
ONE_STEP_DIR = PROJECT_ROOT / "results" / "E2_one_step"
# Outputs stay inside the repository; they are copied into the manuscript by hand.
OUT_DIR = PROJECT_ROOT / "results" / "paper_assets" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "paper_assets" / "figures"

DICTIONARIES = ("D0", "D1a", "D1b", "D1c", "D2", "D3a", "D3b", "D3c")
MACROS = {
    "D0": r"\dLin",
    "D1a": r"\dPoly",
    "D1b": r"\dRBF",
    "D1c": r"\dDelay",
    "D2": r"\dSqrt",
    "D3a": r"\dPolySqrt",
    "D3b": r"\dRBFSqrt",
    "D3c": r"\dDelaySqrt",
}

ERROR_HORIZONS_S = (30.0, 120.0, 600.0)
PAIRED_HORIZON_S = 120.0
ONE_STEP_HORIZON_S = 120.0

# Decimals of RMSE values and their differences, of the in-sample one-step RMSE
# and the one-step table differences (both small), and of p-values. P-values
# below P_FLOOR are written as a bound.
RMSE_DECIMALS = 2
FIT_DECIMALS = 3
ONE_STEP_DECIMALS = 3
P_DECIMALS = 3
P_FLOOR = 0.001

# Groups of the paired table: label row, Holm family of paired_tests.csv whose
# adjusted p-values are shown (None: unadjusted) and pairs written as (A, B).
PAIRED_GROUPS = (
    (
        r"Curvature term vs \dLin",
        "curvature_vs_D0",
        (("D1a", "D0"), ("D1b", "D0"), ("D2", "D0"), ("D3a", "D0"), ("D3b", "D0"), ("D3c", "D0")),
    ),
    (
        r"Adding $\sqrt{h_k}$",
        "sqrt_h",
        (("D2", "D0"), ("D3c", "D1c"), ("D3a", "D1a"), ("D3b", "D1b")),
    ),
    (
        "Single comparisons",
        None,
        (("D1c", "D0"), ("D1a", "D2"), ("D3c", "D2")),
    ),
)

# Pairs of the one-step table, written as in the paired table: the sqrt(h)
# pairs with sqrt(h) first, then two of the single comparisons (E2_one_step
# does not compute D3c vs D2).
ONE_STEP_PAIRS = PAIRED_GROUPS[1][2] + (("D1c", "D0"), ("D1a", "D2"))

# Figure: run, number of evaluation windows shown, margins of the time axis
# (s) around the split and after the longest horizon of the last window shown,
# and size (single column).
FIGURE_RUN = "Baseline_50pct"
N_WINDOWS_SHOWN = 2
MARGIN_BEFORE_SPLIT_S = 1000.0
MARGIN_AFTER_LAST_WINDOW_S = 400.0
FIGURE_SIZE_CM = (12.0, 5.0)
# Grey of the window bands from the shortest to the longest horizon (darkest first).
BAND_GREYS = ("0.55", "0.72", "0.88")
# Fonts of the figure: STIX, a Times-like serif shipped with matplotlib, to match
# the Times text of the manuscript, embedded as TrueType (pdf.fonttype 42)
# instead of the default Type 3, which some publisher systems reject.
FIGURE_FONT_RC = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
}


# --- Formatting ----------------------------------------------------------------


def latex_number(x: float, decimals: int) -> str:
    """Write a number with a fixed number of decimals and a math minus sign.

    Parameters
    ----------
    x : float
        Value to write.
    decimals : int
        Number of decimals.

    Returns
    -------
    str
        The rounded value with a decimal point; negative values start with
        ``$-$``. A negative value that rounds to zero is written without sign.
    """
    text = f"{x:.{decimals}f}"
    if float(text) == 0:
        text = text.lstrip("-")
    return text.replace("-", "$-$")


def latex_p(p: float) -> str:
    """Write a p-value with P_DECIMALS decimals, or as ``$<$0.001`` below P_FLOOR.

    Parameters
    ----------
    p : float
        The p-value.

    Returns
    -------
    str
        The formatted p-value.
    """
    if p < P_FLOOR:
        return f"$<${P_FLOOR:.{P_DECIMALS}f}"
    return f"{p:.{P_DECIMALS}f}"


def dictionary_macro(dictionary: str) -> str:
    """Manuscript macro of a dictionary ID.

    Parameters
    ----------
    dictionary : str
        Dictionary ID, for example "D1a".

    Returns
    -------
    str
        The macro, for example ``\\dPoly``.

    Raises
    ------
    KeyError
        If the ID is not in MACROS.
    """
    return MACROS[dictionary]


def _comparison(a: str, b: str) -> str:
    """Cell "A vs B" with the macros of both dictionaries."""
    return f"{dictionary_macro(a)}{{}} vs {dictionary_macro(b)}"


def _tabular(column_spec: str, header: list, body: list) -> str:
    """Bare tabular with booktabs rules around the header and body lines."""
    lines = [rf"\begin{{tabular}}{{{column_spec}}}", r"\toprule", *header, r"\midrule", *body]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _row(cells: list) -> str:
    """Table row from its cells."""
    return " & ".join(cells) + r" \\"


def provenance_comment(sources: list, results_commit: str, script_commit: str) -> str:
    """First line of a table file: source tables, results commit and script commit.

    Parameters
    ----------
    sources : list of str
        Paths of the source tables, as they should be shown.
    results_commit : str
        Commit of the results read, from their metadata.json.
    script_commit : str
        Commit of this script.

    Returns
    -------
    str
        A LaTeX comment line, without the newline.
    """
    return f"% Sources: {', '.join(sources)}; results commit {results_commit}; script commit {script_commit}"


# --- Inputs --------------------------------------------------------------------


def read_results_metadata(metadata_path) -> dict:
    """Read a results metadata.json and check that it comes from a clean commit.

    Parameters
    ----------
    metadata_path : pathlib.Path or str
        The metadata.json next to the results.

    Returns
    -------
    dict
        The metadata.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If git.clean is not true.
    """
    metadata_path = Path(metadata_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"The metadata {metadata_path} does not exist: run the script that writes it first.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    clean = metadata.get("git", {}).get("clean")
    if clean is not True:
        raise ValueError(f"{metadata_path}: the results were not produced from a clean commit (git.clean = {clean!r}).")
    return metadata


def _single_row(table: pd.DataFrame, mask, what: str) -> pd.Series:
    """The one row of table selected by mask; ValueError naming what if there is none or more than one."""
    rows = table[mask]
    if len(rows) != 1:
        raise ValueError(f"Expected one row for {what}, found {len(rows)}.")
    return rows.iloc[0]


# --- Tables --------------------------------------------------------------------


def table_errors(fit_indicators: pd.DataFrame, error_summary: pd.DataFrame) -> str:
    """Table of one-step in-sample RMSE and multi-step RMSE per dictionary.

    Parameters
    ----------
    fit_indicators : pandas.DataFrame
        fit_indicators.csv of E2, with columns dictionary and fit_rmse_pct (the
        median across runs).
    error_summary : pandas.DataFrame
        error_summary.csv of E2, with columns series, horizon_s, median, min
        and max (RMSE across runs, % of level).

    Returns
    -------
    str
        The tabular, rows in the order of DICTIONARIES.

    Raises
    ------
    ValueError
        If a dictionary, or a dictionary and horizon, is not found exactly once.
    """
    body = []
    for d in DICTIONARIES:
        fit = _single_row(fit_indicators, fit_indicators["dictionary"] == d, f"{d} in fit_indicators")
        cells = [dictionary_macro(d), latex_number(fit["fit_rmse_pct"], FIT_DECIMALS)]
        for h in ERROR_HORIZONS_S:
            s = _single_row(
                error_summary, (error_summary["series"] == d) & (error_summary["horizon_s"] == h), f"{d} at {h:g} s"
            )
            low, high = latex_number(s["min"], RMSE_DECIMALS), latex_number(s["max"], RMSE_DECIMALS)
            cells.append(f"{latex_number(s['median'], RMSE_DECIMALS)} [{low}, {high}]")
        body.append(_row(cells))
    n = len(ERROR_HORIZONS_S)
    header = [
        _row(["", "One-step", rf"\multicolumn{{{n}}}{{c}}{{Multi-step, median [min, max]}}"]),
        rf"\cmidrule(lr){{3-{2 + n}}}",
        _row(["Dictionary", "in-sample", *(f"{h:g}~s" for h in ERROR_HORIZONS_S)]),
    ]
    return _tabular("l" + "c" * (1 + n), header, body)


def paired_values(tests: pd.DataFrame, a: str, b: str, family) -> dict:
    """Counts, median difference with its interval and p-values of A vs B at PAIRED_HORIZON_S.

    Parameters
    ----------
    tests : pandas.DataFrame
        paired_tests.csv of E2_comparisons.
    a, b : str
        The pair, in the orientation wanted: d = RMSE_A - RMSE_B.
    family : str or None
        Holm family whose adjusted p-values are returned (columns
        p_holm_<family> and p_wilcoxon_holm_<family>); None for the unadjusted
        p_sign and p_wilcoxon.

    Returns
    -------
    dict
        Keys n_a and n_b (runs where A, resp. B, has the lower RMSE), n_ties,
        median, ci_low and ci_high (of d, % of level), p_sign and p_wilcoxon.
        If the pair is stored as (B, A), the counts are swapped and the median
        and the interval bounds change sign, with the bounds swapped.

    Raises
    ------
    ValueError
        If the pair is not found exactly once, in either orientation, or if
        the adjusted p-values of the family are empty for it.
    """
    either = ((tests["a"] == a) & (tests["b"] == b)) | ((tests["a"] == b) & (tests["b"] == a))
    r = _single_row(tests, either & (tests["horizon_s"] == PAIRED_HORIZON_S), f"{a} vs {b} at {PAIRED_HORIZON_S:g} s")
    if family is None:
        p_sign, p_wilcoxon = r["p_sign"], r["p_wilcoxon"]
    else:
        p_sign, p_wilcoxon = r[f"p_holm_{family}"], r[f"p_wilcoxon_holm_{family}"]
        if pd.isna(p_sign) or pd.isna(p_wilcoxon):
            raise ValueError(f"{a} vs {b} has no p-value adjusted in the Holm family {family}.")
    values = {
        "n_a": int(r["n_wins_a"]),
        "n_b": int(r["n_wins_b"]),
        "n_ties": int(r["n_ties"]),
        "median": r["diff_median"],
        "ci_low": r["diff_median_ci_low"],
        "ci_high": r["diff_median_ci_high"],
        "p_sign": p_sign,
        "p_wilcoxon": p_wilcoxon,
    }
    if r["a"] != a:
        values.update(
            n_a=values["n_b"],
            n_b=values["n_a"],
            median=-values["median"],
            ci_low=-values["ci_high"],
            ci_high=-values["ci_low"],
        )
    return values


def table_paired(tests: pd.DataFrame) -> str:
    """Table of the paired comparisons at PAIRED_HORIZON_S, one block per group of PAIRED_GROUPS.

    Parameters
    ----------
    tests : pandas.DataFrame
        paired_tests.csv of E2_comparisons.

    Returns
    -------
    str
        The tabular.

    Raises
    ------
    ValueError
        As :func:`paired_values`, or if a pair has ties.
    """
    body = []
    for i, (label, family, pairs) in enumerate(PAIRED_GROUPS):
        if i > 0:
            body.append(r"\midrule")
        body.append(rf"\multicolumn{{5}}{{l}}{{\emph{{{label}}}}} \\")
        for a, b in pairs:
            v = paired_values(tests, a, b, family)
            if v["n_ties"]:
                raise ValueError(f"{a} vs {b} has {v['n_ties']} tie(s), which n_X/n_Y would not show.")
            interval = f"[{latex_number(v['ci_low'], RMSE_DECIMALS)}, {latex_number(v['ci_high'], RMSE_DECIMALS)}]"
            cells = [
                _comparison(a, b),
                f"{v['n_a']}/{v['n_b']}",
                f"{latex_number(v['median'], RMSE_DECIMALS)} {interval}",
                latex_p(v["p_sign"]),
                latex_p(v["p_wilcoxon"]),
            ]
            body.append(_row(cells))
    header = [_row(["Comparison", "$n_X/n_Y$", r"Median $\delta$ [95\% CI]", "$p$ sign", "$p$ Wilcoxon"])]
    return _tabular("lcccc", header, body)


def one_step_values(counts: pd.DataFrame, a: str, b: str) -> dict:
    """Counts and median differences of A vs B, one-step and multi-step, at ONE_STEP_HORIZON_S.

    Parameters
    ----------
    counts : pandas.DataFrame
        one_step_pair_counts.csv of E2_one_step.
    a, b : str
        The pair, in the orientation wanted: differences are A - B.

    Returns
    -------
    dict
        For each of the keys "eval_all" (one-step, whole evaluation part),
        "window" (one-step, window instants) and "multistep" (multi-step), a
        tuple (n_a, n_b, median): the runs where A, resp. B, has the lower
        RMSE and the median of RMSE_A - RMSE_B (% of level). If the pair is
        stored as (B, A), the counts are swapped and the medians change sign.

    Raises
    ------
    ValueError
        If a row is not found exactly once, in either orientation, or if the
        pair has ties.
    """
    flipped = not ((counts["a"] == a) & (counts["b"] == b)).any()
    first, second = (b, a) if flipped else (a, b)
    pair = counts[(counts["a"] == first) & (counts["b"] == second)]
    whole = _single_row(pair, pair["segment"] == "eval_all", f"{a} vs {b}, eval_all")
    window = _single_row(
        pair,
        (pair["segment"] == "window") & (pair["horizon_s"] == ONE_STEP_HORIZON_S),
        f"{a} vs {b}, window at {ONE_STEP_HORIZON_S:g} s",
    )
    for r in (whole, window):
        if r["n_ties"]:
            raise ValueError(f"{a} vs {b} ({r['segment']}) has ties, which n_X/n_Y would not show.")
    values = {
        "eval_all": (int(whole["n_a_lower"]), int(whole["n_b_lower"]), whole["median_diff"]),
        "window": (int(window["n_a_lower"]), int(window["n_b_lower"]), window["median_diff"]),
        "multistep": (
            int(window["multistep_n_a_lower"]),
            int(window["multistep_n_b_lower"]),
            window["multistep_median_diff"],
        ),
    }
    if values["multistep"][0] + values["multistep"][1] != window["n_runs"]:
        raise ValueError(f"{a} vs {b}: the multi-step counts do not add up to the runs (ties).")
    if flipped:
        values = {key: (n_b, n_a, -median) for key, (n_a, n_b, median) in values.items()}
    return values


def table_one_step(counts: pd.DataFrame, pairs=ONE_STEP_PAIRS) -> str:
    """Table of one-step against multi-step pair counts and median differences at ONE_STEP_HORIZON_S.

    Parameters
    ----------
    counts : pandas.DataFrame
        one_step_pair_counts.csv of E2_one_step.
    pairs : sequence of (str, str), default ONE_STEP_PAIRS
        Pairs (A, B) in the order and orientation wanted.

    Returns
    -------
    str
        The tabular, one row per pair.

    Raises
    ------
    ValueError
        As :func:`one_step_values`.
    """
    body = []
    for a, b in pairs:
        values = one_step_values(counts, a, b)
        cells = [_comparison(a, b)]
        for key in ("window", "multistep", "eval_all"):
            n_a, n_b, median = values[key]
            cells += [f"{n_a}/{n_b}", latex_number(median, ONE_STEP_DECIMALS)]
        body.append(_row(cells))
    header = [
        _row([
            "",
            r"\multicolumn{2}{c}{One-step}",
            r"\multicolumn{2}{c}{Multi-step}",
            r"\multicolumn{2}{c}{One-step}",
        ]),
        _row([
            "",
            r"\multicolumn{2}{c}{windows}",
            r"\multicolumn{2}{c}{windows}",
            r"\multicolumn{2}{c}{evaluation segment}",
        ]),
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        _row(["Comparison", *(["$n_X/n_Y$", r"Median $\delta$"] * 3)]),
    ]
    # Between groups, 3 column separations instead of the usual 2, so the group headers do not touch.
    group_gap = r"@{\hspace{3\tabcolsep}}"
    return _tabular(f"lcc{group_gap}cc{group_gap}cc", header, body)


# --- Figure --------------------------------------------------------------------


def window_start_times(windows: pd.DataFrame, run: str, horizon_s: float) -> list:
    """Absolute start times of the evaluation windows of one run at one horizon.

    Parameters
    ----------
    windows : pandas.DataFrame
        windows.csv of E2, with columns run, horizon_s and start_time_s (times
        separated by ";").
    run : str
        Name of the run, for example "Baseline_50pct".
    horizon_s : float
        Horizon, in s.

    Returns
    -------
    list of float
        Start times, in s from the start of the run.

    Raises
    ------
    ValueError
        If the run and horizon are not found exactly once.
    """
    r = _single_row(windows, (windows["run"] == run) & (windows["horizon_s"] == horizon_s), f"{run} at {horizon_s:g} s")
    return [float(x) for x in str(r["start_time_s"]).split(";")]


def largest_free_interval(start_s: float, end_s: float, band_starts_s, band_length_s: float) -> tuple:
    """Longest interval of [start_s, end_s] not covered by any band [b, b + band_length_s].

    Parameters
    ----------
    start_s, end_s : float
        Ends of the interval searched, in s.
    band_starts_s : sequence of float
        Start of each band, in s.
    band_length_s : float
        Length of every band, in s.

    Returns
    -------
    tuple of float
        (left, right) of the longest band-free interval; the first one if
        several have the same length.
    """
    free, left = [], start_s
    for b in sorted(band_starts_s):
        free.append((left, min(b, end_s)))
        left = max(left, b + band_length_s)
    free.append((left, end_s))
    return max(free, key=lambda interval: interval[1] - interval[0])


def figure_run_windows(t, level, setpoint, split_s: float, starts_s, horizons_s) -> plt.Figure:
    """Level and setpoint around the split, with the first N_WINDOWS_SHOWN windows shaded at every horizon.

    Parameters
    ----------
    t : numpy.ndarray
        Time of each sample, in s.
    level, setpoint : numpy.ndarray
        Level and setpoint at each sample, in % of level.
    split_s : float
        Time of the first sample of the evaluation part, in s.
    starts_s : sequence of float
        Start time of each evaluation window, in s, in time order; the first
        N_WINDOWS_SHOWN are drawn.
    horizons_s : sequence of float
        Horizons, in s; each window shown gets one band per horizon, from its
        start to start + horizon.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, FIGURE_SIZE_CM, with no title. The style of set_style and
        the fonts of FIGURE_FONT_RC are set in the global matplotlib settings,
        so that they also apply when the figure is saved. The time axis goes from
        split_s - MARGIN_BEFORE_SPLIT_S to the start of the last window shown
        plus the longest horizon plus MARGIN_AFTER_LAST_WINDOW_S; the level
        axis is scaled to the samples in that range. "Fit" is centred between
        the left end of the axis and the split line, and "Evaluation" in the
        longest interval of the evaluation part shown that no band covers
        (:func:`largest_free_interval` with the longest horizon), both in
        italics near the top.

    Raises
    ------
    ValueError
        If there are fewer than N_WINDOWS_SHOWN windows or more horizons than
        BAND_GREYS, or if a segment label, once drawn, does not fit inside its
        interval.
    """
    if len(starts_s) < N_WINDOWS_SHOWN:
        raise ValueError(f"{len(starts_s)} window(s) given, {N_WINDOWS_SHOWN} to be shown.")
    if len(horizons_s) > len(BAND_GREYS):
        raise ValueError(f"{len(horizons_s)} horizons but only {len(BAND_GREYS)} greys in BAND_GREYS.")
    shown = list(starts_s)[:N_WINDOWS_SHOWN]
    horizons = sorted(horizons_s)
    x_min = split_s - MARGIN_BEFORE_SPLIT_S
    x_max = shown[-1] + horizons[-1] + MARGIN_AFTER_LAST_WINDOW_S
    in_range = (t >= x_min) & (t <= x_max)

    set_style()
    plt.rcParams.update(FIGURE_FONT_RC)
    fig, ax = plt.subplots(figsize=(FIGURE_SIZE_CM[0] / 2.54, FIGURE_SIZE_CM[1] / 2.54))
    # Longest horizon first, so the shorter, darker bands are drawn on top.
    bands = {}
    for i in reversed(range(len(horizons))):
        for start in shown:
            bands[horizons[i]] = ax.axvspan(
                start, start + horizons[i], color=BAND_GREYS[i], linewidth=0, zorder=1 + len(horizons) - i
            )
    setpoint_line, = ax.plot(t[in_range], setpoint[in_range], color="0.3", linewidth=0.9, linestyle="--", zorder=10)
    level_line, = ax.plot(t[in_range], level[in_range], color="black", linewidth=0.8, zorder=11)
    ax.axvline(split_s, color="black", linewidth=0.9, linestyle=":", zorder=12)
    tick_pt = 7
    intervals = {
        "Fit": (x_min, split_s),
        "Evaluation": largest_free_interval(split_s, x_max, shown, horizons[-1]),
    }
    segment_labels = {
        text: ax.text(
            (left + right) / 2, 0.97, text, transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=tick_pt, style="italic",
        )
        for text, (left, right) in intervals.items()
    }
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Level (%)", fontsize=8)
    ax.tick_params(labelsize=tick_pt)
    handles = [level_line, setpoint_line, *(bands[h] for h in horizons)]
    labels = ["Level", "Setpoint", *(f"H = {h:g} s" for h in horizons)]
    # One row above the axes, so that no entry lies on the data or on a band of its own grey.
    ax.legend(
        handles, labels, fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=len(labels)
    )
    fig.tight_layout()
    renderer = fig.canvas.get_renderer()
    for text, (left, right) in intervals.items():
        box = segment_labels[text].get_window_extent(renderer)
        left_px, right_px = (ax.transData.transform((x, 0))[0] for x in (left, right))
        if box.x0 < left_px or box.x1 > right_px:
            raise ValueError(f"The label {text!r} does not fit inside its interval ({left:g}, {right:g}) s.")
    return fig


def run_figure(e2_dir=E2_DIR, raw_dir=DEFAULT_RAW_DIR, out_dir=FIGURE_DIR) -> Path:
    """Plot FIGURE_RUN around its split with its first evaluation windows and write the PDF and a metadata.json.

    Parameters
    ----------
    e2_dir : pathlib.Path or str, default E2_DIR
        Result folder of E2_dictionaries, with tables/windows.csv and
        metadata.json (config.ts_s, config.train_frac and config.horizons_s).
    raw_dir : pathlib.Path or str, default DEFAULT_RAW_DIR
        Folder that contains the Baseline subfolder of the raw data.
    out_dir : pathlib.Path or str, default FIGURE_DIR
        Folder for the PDF and the metadata; created if missing.

    Returns
    -------
    pathlib.Path
        The PDF written.

    Raises
    ------
    FileNotFoundError
        If the E2 metadata, tables/windows.csv or the raw file of FIGURE_RUN
        does not exist.
    ValueError
        If the E2 metadata does not come from a clean commit, if the run has
        gaps at the Ts of E2, if the window start times of windows.csv differ
        between horizons or are not the times of their start samples in the
        evaluation part, or as :func:`figure_run_windows`.
    """
    e2_dir, raw_dir, out_dir = map(Path, (e2_dir, raw_dir, out_dir))
    metadata = read_results_metadata(e2_dir / "metadata.json")
    config = metadata["config"]
    ts_s, train_frac, horizons_s = config["ts_s"], config["train_frac"], [float(h) for h in config["horizons_s"]]
    windows_path = e2_dir / "tables" / "windows.csv"
    raw_path = raw_dir / "Baseline" / f"{FIGURE_RUN}.csv"

    dataset = prepare(load_baseline_file(raw_path), ts_s)
    _, test = temporal_split(dataset, train_frac)
    windows = pd.read_csv(windows_path)
    starts_by_horizon = {h: window_start_times(windows, FIGURE_RUN, h) for h in horizons_s}
    starts_s = starts_by_horizon[horizons_s[0]]
    if any(s != starts_s for s in starts_by_horizon.values()):
        raise ValueError(f"{FIGURE_RUN}: the window start times of windows.csv differ between horizons.")
    w = _single_row(windows, (windows["run"] == FIGURE_RUN) & (windows["horizon_s"] == horizons_s[0]), FIGURE_RUN)
    starts = [int(x) for x in str(w["starts"]).split(";")]
    if not np.array_equal(test.t[starts], starts_s):
        raise ValueError(f"{FIGURE_RUN}: the window start times of windows.csv are not those of the evaluation part.")
    split_s = float(test.t[0])
    print(f"{FIGURE_RUN}: Ts = {ts_s:g} s, split at {split_s:g} s, window starts {starts_s} s")

    level, setpoint = dataset.denormalize_h(dataset.h), dataset.denormalize_h(dataset.sp)
    fig = figure_run_windows(dataset.t, level, setpoint, split_s, starts_s, horizons_s)
    git = git_state()
    script_commit = git["commit"] if git["clean"] is True else f"{git['commit']}-dirty"
    sources = [f"raw data Baseline/{FIGURE_RUN}.csv", _shown_path(windows_path)]
    subject = provenance_comment(sources, metadata["git"]["commit"], script_commit).removeprefix("% ")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "figure_run_windows.pdf"
    # No creation date, so that rerunning from the same commit gives the same file.
    fig.savefig(path, metadata={"Subject": subject, "CreationDate": None})
    plt.close(fig)
    print(f"Written {path}")
    write_metadata(
        out_dir,
        config={
            "sources": sources,
            "results_commit": metadata["git"]["commit"],
            "run": FIGURE_RUN,
            "ts_s": ts_s,
            "train_frac": train_frac,
            "split_s": split_s,
            "horizons_s": horizons_s,
            "window_start_times_s": starts_s[:N_WINDOWS_SHOWN],
            "n_windows_shown": N_WINDOWS_SHOWN,
            "margin_before_split_s": MARGIN_BEFORE_SPLIT_S,
            "margin_after_last_window_s": MARGIN_AFTER_LAST_WINDOW_S,
            "band_greys": list(BAND_GREYS),
            "size_cm": list(FIGURE_SIZE_CM),
        },
        summary={"files": [path.name]},
    )
    return path


# --- Run -----------------------------------------------------------------------


def _shown_path(path: Path) -> str:
    """Path relative to the repository root when it is inside it, otherwise as given."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def run_tables(e2_dir=E2_DIR, cmp_dir=CMP_DIR, one_step_dir=ONE_STEP_DIR, out_dir=OUT_DIR) -> list:
    """Read the result tables, write the three .tex tables and a metadata.json.

    Parameters
    ----------
    e2_dir, cmp_dir, one_step_dir : pathlib.Path or str, default E2_DIR, CMP_DIR and ONE_STEP_DIR
        Result folders of E2_dictionaries, E2_comparisons and E2_one_step.
    out_dir : pathlib.Path or str, default OUT_DIR
        Folder for the .tex files and the metadata; created if missing.

    Returns
    -------
    list of pathlib.Path
        The .tex files written.

    Raises
    ------
    FileNotFoundError
        If a source metadata.json or a source table does not exist.
    ValueError
        If a source metadata does not come from a clean commit, if
        ONE_STEP_PAIRS are not the pairs of the E2_one_step metadata (in either
        orientation), or as the table functions.
    """
    e2_dir, cmp_dir, one_step_dir, out_dir = map(Path, (e2_dir, cmp_dir, one_step_dir, out_dir))
    metadata = {name: read_results_metadata(d / "metadata.json") for name, d in
                (("e2", e2_dir), ("cmp", cmp_dir), ("one_step", one_step_dir))}
    stored_pairs = {frozenset(p) for p in metadata["one_step"]["config"]["pairs"]}
    if stored_pairs != {frozenset(p) for p in ONE_STEP_PAIRS}:
        raise ValueError("ONE_STEP_PAIRS are not the pairs of the E2_one_step metadata.")

    fit_path = e2_dir / "tables" / "fit_indicators.csv"
    summary_path = e2_dir / "tables" / "error_summary.csv"
    tests_path = cmp_dir / "paired_tests.csv"
    counts_path = one_step_dir / "one_step_pair_counts.csv"
    specs = (
        ("table_errors.tex", "e2", [fit_path, summary_path],
         lambda: table_errors(pd.read_csv(fit_path), pd.read_csv(summary_path))),
        ("table_paired_120s.tex", "cmp", [tests_path], lambda: table_paired(pd.read_csv(tests_path))),
        ("table_one_step.tex", "one_step", [counts_path], lambda: table_one_step(pd.read_csv(counts_path))),
    )
    # Build every table before writing any, so that a failure leaves nothing behind.
    git = git_state()
    script_commit = git["commit"] if git["clean"] is True else f"{git['commit']}-dirty"
    contents = {}
    for name, source, paths, build in specs:
        comment = provenance_comment([_shown_path(p) for p in paths], metadata[source]["git"]["commit"], script_commit)
        contents[name] = comment + "\n" + build()

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in contents.items():
        path = out_dir / name
        path.write_text(text, encoding="utf-8")
        written.append(path)
        print(f"Written {path}")
    write_metadata(
        out_dir,
        config={
            "sources": {name: [_shown_path(p) for p in paths] for name, _, paths, _ in specs},
            "results_commits": {name: metadata[source]["git"]["commit"] for name, source, _, _ in specs},
            "dictionaries": list(DICTIONARIES),
            "macros": MACROS,
            "error_horizons_s": list(ERROR_HORIZONS_S),
            "paired_horizon_s": PAIRED_HORIZON_S,
            "paired_groups": [
                {"label": label, "holm_family": family, "pairs": [list(p) for p in group_pairs]}
                for label, family, group_pairs in PAIRED_GROUPS
            ],
            "one_step_horizon_s": ONE_STEP_HORIZON_S,
            "one_step_pairs": [list(p) for p in ONE_STEP_PAIRS],
            "decimals": {
                "rmse": RMSE_DECIMALS,
                "fit_rmse": FIT_DECIMALS,
                "one_step_differences": ONE_STEP_DECIMALS,
                "p": P_DECIMALS,
            },
            "p_floor": P_FLOOR,
        },
        summary={"files": [p.name for p in written]},
    )
    return written


def main() -> None:
    """Write the paper tables and the figure from the results in the default folders."""
    run_tables()
    run_figure()


if __name__ == "__main__":
    main()
