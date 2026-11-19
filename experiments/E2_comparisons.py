"""E2 comparisons - The comparison tables of E2 that are cited, computed from its saved errors.

The script does not fit or score anything: it reads the errors table written
by E2 (one RMSE per run, series and horizon) and derives from it the
comparisons that are quoted, so that every quoted figure comes from a
versioned script run on the outputs of a clean E2 run.

Tables
------
summary
    Per series and horizon: median, minimum and maximum across runs, as in the
    summary of E2, and the mean next to the median.
pair_counts
    For each pair (A, B) in PAIRS and each horizon: in how many runs A has a
    strictly lower RMSE than B, in how many B has, and in how many they tie
    exactly. A count, with no statistical test.
median_ratios
    Median of each series divided by the median of D0, at the same horizon.
differences (file differences_<A>_<B>.csv)
    RMSE of A minus RMSE of B run by run, for DIFFERENCE_PAIR.
differences_summary
    Mean, median, minimum and maximum of those differences, per horizon. The
    median of the differences is not the difference of the medians, so a pair
    can have the lower median on one side and most of the run-by-run wins on
    the other.
paired_tests
    For each pair in PAIRS and each horizon, with d = RMSE of A minus RMSE of
    B over the runs: the counts of pair_counts, the exact two-sided sign test
    (ties left out), the exact two-sided Wilcoxon signed-rank test, the
    median, mean, minimum and maximum of d, and percentile bootstrap 95 %
    intervals of the median and the mean of d, resampling runs with
    replacement. p_sign and p_wilcoxon are unadjusted. Both are also adjusted
    by Holm within each family of HOLM_FAMILIES, at COMPARISON_HORIZON_S only,
    one column per test and family (empty outside the family). The column
    single_test names the single comparisons of SINGLE_TESTS at that horizon,
    which get no adjustment.
conditioning_sqrt_h
    For each pair (base, base + sqrt(h)) in SQRT_H_ADDITION, read from the
    models table of E2: median, minimum and maximum across runs of the
    condition number of the regression matrix without and with sqrt(h), and of
    their ratio run by run. Descriptive.

The runs were recorded on different days, with the same setpoint profile and
a different electrovalve opening. The summaries are descriptive, and the
p-values are descriptive too because the independence between runs cannot be
verified.

Invariants: E2 must have run from a clean commit, the median, minimum and
maximum recomputed here must match those in the summary table written by E2,
the win and tie counts of the paired tests must match pair_counts, and the
median condition number of each dictionary must match its fit indicators
table.

Outputs in results/E2_comparisons/: one CSV per table above (seven) and
metadata.json, which records the commit of this run and the E2 commit whose
tables were read.
"""

import json
from importlib.metadata import version
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from edmdc_tank_observables.paths import PROJECT_ROOT
from edmdc_tank_observables.provenance import write_metadata

# --- Configuration -------------------------------------------------------------

E2_DIR = PROJECT_ROOT / "results" / "E2_dictionaries"
OUT_DIR = PROJECT_ROOT / "results" / "E2_comparisons"

HORIZONS_S = (30.0, 120.0, 600.0)
E2_SERIES = ("D0", "D1a", "D1b", "D1c", "D2", "D3a", "D3b", "D3c")
REFERENCES = ("D0",)

# Pairs (A, B) whose run-by-run counts are reported.
PAIRS = (
    ("D1a", "D1b"),
    ("D1a", "D2"),
    ("D1a", "D3a"),
    ("D3c", "D2"),
    ("D1b", "D2"),
    ("D1b", "D3b"),
    ("D1c", "D0"),
    ("D3c", "D1c"),
    # Every other dictionary against D0.
    ("D1a", "D0"),
    ("D1b", "D0"),
    ("D2", "D0"),
    ("D3a", "D0"),
    ("D3b", "D0"),
    ("D3c", "D0"),
)

# Comparisons at one horizon only. The families are defined by the nested
# design of the dictionaries; within each family, Holm adjusts the p-values of
# the sign test and of the Wilcoxon test, separately for each test and family.
# The single comparisons get no adjustment. Pairs are unordered, matched to the
# pair in PAIRS whichever its orientation; a pair may belong to more than one
# family.
COMPARISON_HORIZON_S = 120.0
HOLM_FAMILIES = {
    # Every dictionary that adds a nonlinear function of h to D0, against D0.
    "curvature_vs_D0": (
        ("D1a", "D0"),
        ("D1b", "D0"),
        ("D2", "D0"),
        ("D3a", "D0"),
        ("D3b", "D0"),
        ("D3c", "D0"),
    ),
    # Pairs that differ only by the sqrt(h) observable.
    "sqrt_h": (
        ("D2", "D0"),
        ("D3c", "D1c"),
        ("D1a", "D3a"),
        ("D1b", "D3b"),
    ),
}
SINGLE_TESTS = {
    # Delayed levels against the state alone.
    "delay": ("D1c", "D0"),
    # Polynomial terms against sqrt(h) as the curvature term.
    "curvature_choice": ("D1a", "D2"),
}

# The pairs of the "sqrt_h" Holm family, written as (base, base + sqrt(h)),
# whose condition numbers are reported.
SQRT_H_ADDITION = (("D0", "D2"), ("D1c", "D3c"), ("D1a", "D3a"), ("D1b", "D3b"))

# Pair whose run-by-run differences are reported.
DIFFERENCE_PAIR = ("D1a", "D1b")

# Bootstrap of the paired differences: number of resamples of the runs, seed
# and coverage of the percentile interval.
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20261024
CONFIDENCE = 0.95


# --- Inputs and invariants -----------------------------------------------------


def read_e2_commit(metadata_path) -> str:
    """Commit of the E2 run whose tables are read, which must be clean.

    Parameters
    ----------
    metadata_path : pathlib.Path or str
        The metadata.json written by E2.

    Returns
    -------
    str
        The commit recorded in it.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If git.clean is not true.
    """
    metadata_path = Path(metadata_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"The E2 metadata {metadata_path} does not exist: run E2 first.")
    git = json.loads(metadata_path.read_text(encoding="utf-8")).get("git", {})
    if git.get("clean") is not True:
        raise ValueError(f"{metadata_path}: E2 was not run from a clean commit (git.clean = {git.get('clean')!r}).")
    return str(git.get("commit"))


def check_against_summary(errors: pd.DataFrame, e2_summary: pd.DataFrame) -> None:
    """Check that the median, minimum and maximum recomputed from the errors match the summary of E2.

    Parameters
    ----------
    errors : pandas.DataFrame
        Errors table of E2.
    e2_summary : pandas.DataFrame
        Summary table of E2, with the columns series, horizon_s, median, min
        and max.

    Raises
    ------
    ValueError
        If a series and horizon is missing from either table, or if any of the
        three values differs beyond a relative tolerance of 1e-12.
    """
    keys = ["series", "horizon_s"]
    joined = summary_with_mean(errors).merge(e2_summary, on=keys, how="outer", suffixes=("", "_e2"))
    for column in ("median", "min", "max"):
        if not np.allclose(joined[column], joined[f"{column}_e2"], rtol=1e-12, atol=0.0):
            raise ValueError(f"The {column} recomputed from errors.csv does not match error_summary.csv.")


# --- Tables --------------------------------------------------------------------


def summary_with_mean(errors: pd.DataFrame) -> pd.DataFrame:
    """Median, minimum, maximum and mean RMSE across runs per series and horizon.

    Parameters
    ----------
    errors : pandas.DataFrame
        Errors table with the columns series, horizon_s and rmse, one row per
        run, series and horizon.

    Returns
    -------
    pandas.DataFrame
        One row per (series, horizon_s) with median, min, max, mean and
        n_runs.
    """
    return (
        errors.groupby(["series", "horizon_s"])["rmse"]
        .agg(median="median", min="min", max="max", mean="mean", n_runs="count")
        .reset_index()
    )


def _by_run(errors: pd.DataFrame, series) -> pd.DataFrame:
    """RMSE with one row per (run, horizon_s) and one column per series."""
    missing = sorted(set(series) - set(errors["series"]))
    if missing:
        raise ValueError(f"Series not in the errors table: {missing}.")
    return errors.pivot_table(index=["run", "horizon_s"], columns="series", values="rmse")


def pair_counts(errors: pd.DataFrame, pairs) -> pd.DataFrame:
    """Count, run by run, which series of each pair has the lower RMSE.

    Parameters
    ----------
    errors : pandas.DataFrame
        Errors table with the columns run, series, horizon_s and rmse.
    pairs : iterable of tuple
        Pairs (a, b) of series ids.

    Returns
    -------
    pandas.DataFrame
        One row per pair and horizon with a, b, horizon_s, n_a_lower (runs in
        which a has a strictly lower RMSE), n_b_lower, n_ties (exactly equal)
        and n_runs.

    Raises
    ------
    ValueError
        If a series of a pair is not in the table.
    """
    pairs = list(pairs)
    table = _by_run(errors, {s for pair in pairs for s in pair})
    rows = []
    for a, b in pairs:
        for horizon_s, block in table.groupby(level="horizon_s"):
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "horizon_s": horizon_s,
                    "n_a_lower": int((block[a] < block[b]).sum()),
                    "n_b_lower": int((block[b] < block[a]).sum()),
                    "n_ties": int((block[a] == block[b]).sum()),
                    "n_runs": len(block),
                }
            )
    return pd.DataFrame(rows)


def median_ratios(summary: pd.DataFrame, references=REFERENCES) -> pd.DataFrame:
    """Median of each series divided by the median of each reference series at the same horizon.

    Parameters
    ----------
    summary : pandas.DataFrame
        Table returned by summary_with_mean.
    references : iterable of str, default REFERENCES
        Series to divide by.

    Returns
    -------
    pandas.DataFrame
        One row per (series, horizon_s, reference) with median, reference_median
        and ratio (median / reference_median).
    """
    medians = summary.set_index(["series", "horizon_s"])["median"]
    tables = []
    for reference in references:
        table = summary[["series", "horizon_s", "median"]].copy()
        table["reference"] = reference
        table["reference_median"] = [medians[(reference, h)] for h in table["horizon_s"]]
        table["ratio"] = table["median"] / table["reference_median"]
        tables.append(table)
    return pd.concat(tables, ignore_index=True)[
        ["series", "horizon_s", "reference", "median", "reference_median", "ratio"]
    ]


def run_differences(errors: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """RMSE of a minus RMSE of b, run by run, at each horizon.

    Parameters
    ----------
    errors : pandas.DataFrame
        Errors table with the columns run, series, horizon_s and rmse.
    a, b : str
        Series ids.

    Returns
    -------
    pandas.DataFrame
        One row per (run, horizon_s) with rmse_a, rmse_b and difference
        (rmse_a - rmse_b; negative when a is lower).

    Raises
    ------
    ValueError
        If a or b is not in the table.
    """
    table = _by_run(errors, (a, b))
    return pd.DataFrame(
        {"a": a, "b": b, "rmse_a": table[a], "rmse_b": table[b], "difference": table[a] - table[b]}
    ).reset_index()


def difference_summary(differences: pd.DataFrame) -> pd.DataFrame:
    """Mean, median, minimum and maximum of the run-by-run differences, per horizon.

    Parameters
    ----------
    differences : pandas.DataFrame
        Table returned by run_differences.

    Returns
    -------
    pandas.DataFrame
        One row per horizon_s with a, b, mean, median, min, max and n_runs.
    """
    summary = (
        differences.groupby("horizon_s")["difference"]
        .agg(mean="mean", median="median", min="min", max="max", n_runs="count")
        .reset_index()
    )
    summary.insert(0, "a", differences["a"].iloc[0])
    summary.insert(1, "b", differences["b"].iloc[0])
    return summary


# --- Paired tests --------------------------------------------------------------


def sign_test_p(n_a: int, n_b: int) -> float:
    """Exact two-sided sign test p-value for n_a runs won by A and n_b won by B, ties already left out.

    Under the null hypothesis each untied run is won by A with probability 1/2,
    so the p-value is twice the binomial tail of the smaller count, capped at 1.

    Parameters
    ----------
    n_a, n_b : int
        Runs in which A, and B, has the strictly lower RMSE.

    Returns
    -------
    float
        min(1, 2 * sum over k <= min(n_a, n_b) of C(n, k) / 2**n), with
        n = n_a + n_b.

    Raises
    ------
    ValueError
        If n_a + n_b is 0 (every run tied).
    """
    n = n_a + n_b
    if n == 0:
        raise ValueError("The sign test needs at least one untied run.")
    tail = sum(comb(n, k) for k in range(min(n_a, n_b) + 1))
    return min(1.0, 2 * tail / 2**n)


def wilcoxon_p(d) -> float:
    """Exact two-sided Wilcoxon signed-rank p-value of the paired differences d.

    Parameters
    ----------
    d : array_like
        Paired differences, one per run.

    Returns
    -------
    float
        The p-value from the exact null distribution of the signed-rank
        statistic (scipy.stats.wilcoxon with method="exact").

    Raises
    ------
    ValueError
        If any difference is zero or two absolute differences are equal, since
        the exact distribution assumes neither.
    """
    d = np.asarray(d, dtype=float)
    if np.any(d == 0.0) or len(np.unique(np.abs(d))) < len(d):
        raise ValueError("The exact Wilcoxon test needs nonzero differences with distinct absolute values.")
    return float(stats.wilcoxon(d, method="exact").pvalue)


def holm(p_values) -> list:
    """Holm step-down adjustment of a family of p-values.

    The k-th smallest of m p-values (k = 1, ..., m) is multiplied by m - k + 1,
    capped at 1, and made non-decreasing in that order.

    Parameters
    ----------
    p_values : sequence of float
        The p-values of the family.

    Returns
    -------
    list of float
        The adjusted p-values, in the order of the input.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    scaled = np.minimum(1.0, (m - np.arange(m)) * p[order])
    adjusted = np.empty(m)
    adjusted[order] = np.maximum.accumulate(scaled)
    return adjusted.tolist()


def bootstrap_ci(d, statistic, n_resamples: int, seed: int, confidence: float = CONFIDENCE) -> tuple:
    """Percentile bootstrap interval of a statistic of the paired differences, resampling runs with replacement.

    Parameters
    ----------
    d : array_like
        Paired differences, one per run.
    statistic : callable
        Function such as numpy.median or numpy.mean that takes an axis
        argument.
    n_resamples : int
        Number of bootstrap resamples.
    seed : int
        Seed of the random generator; the same seed gives the same resamples.
    confidence : float, default CONFIDENCE
        Coverage of the interval.

    Returns
    -------
    tuple of float
        Lower and upper percentiles of the statistic over the resamples.
    """
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    values = statistic(d[rng.integers(0, len(d), size=(n_resamples, len(d)))], axis=1)
    alpha = 1.0 - confidence
    low, high = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def paired_tests(
    errors: pd.DataFrame, pairs, n_resamples: int, seed: int, families, single_tests, comparison_horizon_s: float
) -> pd.DataFrame:
    """Paired tests and summaries of d = RMSE of a minus RMSE of b over the runs, per pair and horizon.

    Every pair and horizon uses a generator started from the same seed, so all
    of them see the same resampled runs. The resampled indices are applied to
    d with the runs in alphabetical order of their labels, so both bootstrap
    intervals depend on that order: the same differences in another order give
    slightly different intervals.

    Parameters
    ----------
    errors : pandas.DataFrame
        Errors table with the columns run, series, horizon_s and rmse.
    pairs : iterable of tuple
        Pairs (a, b) of series ids.
    n_resamples : int
        Number of bootstrap resamples.
    seed : int
        Seed of the bootstrap.
    families : dict
        Family name -> pairs whose p-values are adjusted together by Holm, the
        sign test and the Wilcoxon p-values each on their own. A member
        matches the pair of `pairs` with the same two series in either order.
    single_tests : dict
        Test name -> pair tested on its own, with no adjustment; matched as
        the family members.
    comparison_horizon_s : float
        The only horizon, in s, at which the families are adjusted and the
        single tests are named.

    Returns
    -------
    pandas.DataFrame
        One row per pair and horizon with a, b, horizon_s, n_runs, n_wins_a
        (runs with d < 0), n_wins_b (d > 0), n_ties (d == 0), p_sign and
        p_wilcoxon (both unadjusted), one column p_holm_<family> per family
        (Holm of p_sign within the family; NaN for rows outside it), then one
        column p_wilcoxon_holm_<family> per family (the same for p_wilcoxon),
        single_test (the
        name of the single test; empty elsewhere), diff_median,
        diff_median_ci_low, diff_median_ci_high, diff_mean, diff_mean_ci_low,
        diff_mean_ci_high, diff_min and diff_max.

    Raises
    ------
    ValueError
        If a series of a pair is not in the table, if every run of a pair and
        horizon ties, if the differences contain a zero or tied absolute
        values (exact Wilcoxon), if a family member or single test does not
        match exactly one row at comparison_horizon_s, or if a row is named by
        two single tests.
    """
    pairs = list(pairs)
    table = _by_run(errors, {s for pair in pairs for s in pair})
    rows = []
    for a, b in pairs:
        for horizon_s, block in table.groupby(level="horizon_s"):
            d = (block[a] - block[b]).to_numpy()
            n_wins_a, n_wins_b = int((d < 0).sum()), int((d > 0).sum())
            median_ci = bootstrap_ci(d, np.median, n_resamples, seed)
            mean_ci = bootstrap_ci(d, np.mean, n_resamples, seed)
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "horizon_s": horizon_s,
                    "n_runs": len(d),
                    "n_wins_a": n_wins_a,
                    "n_wins_b": n_wins_b,
                    "n_ties": int((d == 0).sum()),
                    "p_sign": sign_test_p(n_wins_a, n_wins_b),
                    "p_wilcoxon": wilcoxon_p(d),
                    "diff_median": float(np.median(d)),
                    "diff_median_ci_low": median_ci[0],
                    "diff_median_ci_high": median_ci[1],
                    "diff_mean": float(np.mean(d)),
                    "diff_mean_ci_low": mean_ci[0],
                    "diff_mean_ci_high": mean_ci[1],
                    "diff_min": float(np.min(d)),
                    "diff_max": float(np.max(d)),
                }
            )
    result = pd.DataFrame(rows)
    unordered = [frozenset(pair) for pair in zip(result["a"], result["b"])]

    def row_of(pair, name):
        match = [
            i for i, other in enumerate(unordered)
            if other == frozenset(pair) and result.at[i, "horizon_s"] == comparison_horizon_s
        ]
        if len(match) != 1:
            raise ValueError(f"{name}: {pair} matches {len(match)} rows at {comparison_horizon_s} s, not 1.")
        return match[0]

    position = result.columns.get_loc("p_wilcoxon") + 1
    family_rows = {name: [row_of(member, name) for member in members] for name, members in families.items()}
    for test, prefix in (("p_sign", "p_holm"), ("p_wilcoxon", "p_wilcoxon_holm")):
        for name, index in family_rows.items():
            column = np.full(len(result), np.nan)
            column[index] = holm(result.loc[index, test])
            result.insert(position, f"{prefix}_{name}", column)
            position += 1
    names = pd.Series(None, index=result.index, dtype=object)
    for name, pair in single_tests.items():
        i = row_of(pair, name)
        if pd.notna(names[i]):
            raise ValueError(f"{name}: row {pair} at {comparison_horizon_s} s is already the single test {names[i]}.")
        names[i] = name
    result.insert(position, "single_test", names)
    return result


def check_paired_counts(tests: pd.DataFrame, counts: pd.DataFrame) -> None:
    """Check that the win and tie counts of the paired tests match those of pair_counts.

    Parameters
    ----------
    tests : pandas.DataFrame
        Table returned by paired_tests.
    counts : pandas.DataFrame
        Table returned by pair_counts for the same pairs.

    Raises
    ------
    ValueError
        If the two tables do not have the same pairs and horizons or any count
        differs.
    """
    keys = ["a", "b", "horizon_s"]
    joined = tests[[*keys, "n_wins_a", "n_wins_b", "n_ties"]].merge(
        counts[[*keys, "n_a_lower", "n_b_lower", "n_ties"]], on=keys, how="outer", suffixes=("", "_counts")
    )
    same = (
        (joined["n_wins_a"] == joined["n_a_lower"])
        & (joined["n_wins_b"] == joined["n_b_lower"])
        & (joined["n_ties"] == joined["n_ties_counts"])
    )
    if not same.all():
        raise ValueError("The counts of the paired tests do not match pair_counts.")


# --- Conditioning --------------------------------------------------------------


def conditioning_table(models: pd.DataFrame, fit_indicators: pd.DataFrame, pairs) -> pd.DataFrame:
    """Condition number of the regression matrix without and with sqrt(h), for each pair.

    cond is the 2-norm condition number of the regression matrix
    Z = [Psi(h_k), u_k, 1] on the fit part, in normalized units and with the
    columns not standardized, as computed by fit_edmdc, so it reflects both the
    scale of the columns and how close they are to collinear. The table is
    descriptive.

    Parameters
    ----------
    models : pandas.DataFrame
        Models table written by E2, with at least the columns run, dictionary
        and cond, one row per run and dictionary.
    fit_indicators : pandas.DataFrame
        Fit indicators table written by E2, with at least the columns
        dictionary and cond (median of cond across runs).
    pairs : iterable of tuple of str
        Pairs (base, with_sqrt), where with_sqrt is base plus the sqrt(h)
        observable.

    Returns
    -------
    pandas.DataFrame
        One row per pair with the columns base, with_sqrt, n_runs,
        cond_base_median, cond_base_min, cond_base_max, cond_with_sqrt_median,
        cond_with_sqrt_min, cond_with_sqrt_max, ratio_median, ratio_min,
        ratio_max and n_runs_ratio_above_1, where ratio = cond(with_sqrt) /
        cond(base) in each run.

    Raises
    ------
    ValueError
        If the two dictionaries of a pair do not have the same runs (including
        a run that lacks one of them, or a dictionary with no runs), if a
        dictionary has more than one row for the same run, if some
        cond is not finite and positive, or if, for some dictionary of the
        pairs, the median of cond across runs differs from its cond in
        fit_indicators (relative tolerance 1e-12).
    """
    expected = fit_indicators.set_index("dictionary")["cond"]
    rows = []
    for base, with_sqrt in pairs:
        cond = {}
        for dictionary in (base, with_sqrt):
            values = models.loc[models["dictionary"] == dictionary].set_index("run")["cond"].astype(float)
            if len(values) == 0:
                raise ValueError(f"{dictionary} has no runs in the models table.")
            if values.index.duplicated().any():
                raise ValueError(f"{dictionary} has more than one row for some run in the models table.")
            if not np.all(np.isfinite(values) & (values > 0)):
                raise ValueError(f"{dictionary}: some cond is not finite and positive.")
            if dictionary not in expected.index or not np.isclose(
                values.median(), expected[dictionary], rtol=1e-12, atol=0.0
            ):
                raise ValueError(
                    f"{dictionary}: median cond across runs {values.median()!r} does not match "
                    f"fit_indicators ({expected.get(dictionary)!r})."
                )
            cond[dictionary] = values
        if set(cond[base].index) != set(cond[with_sqrt].index):
            only = sorted(set(cond[base].index) ^ set(cond[with_sqrt].index))
            raise ValueError(f"{base} and {with_sqrt} do not have the same runs; in only one of them: {only}.")
        ratio = cond[with_sqrt] / cond[base].loc[cond[with_sqrt].index]
        rows.append(
            {
                "base": base,
                "with_sqrt": with_sqrt,
                "n_runs": len(ratio),
                "cond_base_median": cond[base].median(),
                "cond_base_min": cond[base].min(),
                "cond_base_max": cond[base].max(),
                "cond_with_sqrt_median": cond[with_sqrt].median(),
                "cond_with_sqrt_min": cond[with_sqrt].min(),
                "cond_with_sqrt_max": cond[with_sqrt].max(),
                "ratio_median": ratio.median(),
                "ratio_min": ratio.min(),
                "ratio_max": ratio.max(),
                "n_runs_ratio_above_1": int((ratio > 1).sum()),
            }
        )
    return pd.DataFrame(rows)


# --- Console output ------------------------------------------------------------


def print_tables(summary, counts, ratios, differences, diff_summary, tests, e2_commit) -> None:
    """Print the comparison tables to the console."""
    order = list(E2_SERIES)
    print("=" * 78)
    print(f"E2 comparisons (tables of the clean E2 run {e2_commit})")
    print("=" * 78)

    print("\nRMSE (% of level): median [min, max] and mean across runs")
    cells = summary.assign(
        cell=summary.apply(lambda r: f"{r['median']:.3f} [{r['min']:.3f}, {r['max']:.3f}] mean {r['mean']:.3f}", axis=1)
    )
    table = cells.pivot(index="series", columns="horizon_s", values="cell")
    print(table.loc[[s for s in order if s in table.index]].to_string())

    print("\nRuns with the lower RMSE in each pair (a / b / exact ties, over n_runs)")
    cells = counts.assign(
        pair=counts["a"] + " vs " + counts["b"],
        cell=counts.apply(lambda r: f"{r['n_a_lower']} / {r['n_b_lower']} / {r['n_ties']} (n={r['n_runs']})", axis=1),
    )
    table = cells.pivot(index="pair", columns="horizon_s", values="cell")
    print(table.loc[cells["pair"].drop_duplicates()].to_string())

    print("\nRatio of medians to D0")
    table = ratios.pivot_table(index="series", columns=["reference", "horizon_s"], values="ratio")
    print(table.loc[[s for s in order if s in table.index]].to_string(float_format=lambda v: f"{v:.3f}"))

    a, b = DIFFERENCE_PAIR
    print(f"\nRun-by-run difference {a} - {b} (% of level; negative when {a} is lower)")
    table = differences.pivot(index="run", columns="horizon_s", values="difference")
    print(table.to_string(float_format=lambda v: f"{v:+.3f}"))
    print(diff_summary.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    print(f"\nPaired tests over runs (d = a - b, % of level; bootstrap B = {BOOTSTRAP_RESAMPLES}, seed {BOOTSTRAP_SEED})")
    cells = tests.assign(
        pair=tests["a"] + " vs " + tests["b"],
        wins=tests.apply(lambda r: f"{r['n_wins_a']}/{r['n_wins_b']}/{r['n_ties']}", axis=1),
        median_ci=tests.apply(
            lambda r: f"{r['diff_median']:+.3f} [{r['diff_median_ci_low']:+.3f}, {r['diff_median_ci_high']:+.3f}]",
            axis=1,
        ),
    )
    holm_columns = [f"{prefix}_{name}" for prefix in ("p_holm", "p_wilcoxon_holm") for name in HOLM_FAMILIES]
    columns = ["pair", "horizon_s", "wins", "p_sign", "p_wilcoxon", *holm_columns, "single_test", "median_ci"]
    print(cells[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}", na_rep="-"))
    print(
        f"Holm within each family and single tests (no adjustment) at {COMPARISON_HORIZON_S:g} s only;"
        " '-' outside them."
    )
    print()


def print_conditioning(conditioning: pd.DataFrame) -> None:
    """Print the condition numbers without and with sqrt(h) and their run-by-run ratio."""

    def cell(row, name, fmt):
        return f"{row[f'{name}_median']:{fmt}} [{row[f'{name}_min']:{fmt}}, {row[f'{name}_max']:{fmt}}]"

    print("Condition number of Z = [Psi, u, 1] on the fit part: median [min, max] across runs")
    for _, row in conditioning.iterrows():
        print(
            f"  {row['base']:<4} -> {row['with_sqrt']:<4} cond {cell(row, 'cond_base', '.0f')} -> "
            f"{cell(row, 'cond_with_sqrt', '.0f')}; ratio {cell(row, 'ratio', '.2f')}, "
            f"above 1 in {row['n_runs_ratio_above_1']}/{row['n_runs']} runs"
        )
    print()


# --- Run -----------------------------------------------------------------------


def run(e2_dir=E2_DIR, out_dir=OUT_DIR) -> dict:
    """Read the E2 outputs, compute the comparison tables and write them with a metadata.json.

    Parameters
    ----------
    e2_dir : pathlib.Path or str, default E2_DIR
        Output folder of E2, with tables/errors.csv, tables/error_summary.csv,
        tables/models.csv, tables/fit_indicators.csv and metadata.json.
    out_dir : pathlib.Path or str, default OUT_DIR
        Folder for the CSV tables and the metadata.

    Returns
    -------
    dict
        The tables, keyed summary, pair_counts, median_ratios,
        differences, differences_summary, paired_tests and
        conditioning_sqrt_h.

    Raises
    ------
    FileNotFoundError
        If the E2 metadata or any of the four E2 tables does not exist.
    ValueError
        If E2 did not run from a clean commit, if the recomputed summary does
        not match that of E2, if a series is missing from the errors table, if
        a paired test cannot be computed (see paired_tests), if its counts do
        not match pair_counts, or if the conditioning table cannot be built
        or does not match fit_indicators (see conditioning_table).
    """
    e2_dir, out_dir = Path(e2_dir), Path(out_dir)
    e2_commit = read_e2_commit(e2_dir / "metadata.json")
    errors = pd.read_csv(e2_dir / "tables" / "errors.csv")
    errors = errors[errors["horizon_s"].isin(HORIZONS_S)]
    check_against_summary(errors, pd.read_csv(e2_dir / "tables" / "error_summary.csv"))

    summary = summary_with_mean(errors)
    counts = pair_counts(errors, PAIRS)
    ratios = median_ratios(summary[summary["series"].isin(E2_SERIES)])
    differences = run_differences(errors, *DIFFERENCE_PAIR)
    diff_summary = difference_summary(differences)
    tests = paired_tests(
        errors, PAIRS, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, HOLM_FAMILIES, SINGLE_TESTS, COMPARISON_HORIZON_S
    )
    check_paired_counts(tests, counts)
    conditioning = conditioning_table(
        pd.read_csv(e2_dir / "tables" / "models.csv"),
        pd.read_csv(e2_dir / "tables" / "fit_indicators.csv"),
        SQRT_H_ADDITION,
    )
    print_tables(summary, counts, ratios, differences, diff_summary, tests, e2_commit)
    print_conditioning(conditioning)

    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "summary": summary,
        "pair_counts": counts,
        "median_ratios": ratios,
        "differences": differences,
        "differences_summary": diff_summary,
        "paired_tests": tests,
        "conditioning_sqrt_h": conditioning,
    }
    a, b = DIFFERENCE_PAIR
    names = {"differences": f"differences_{a}_{b}"}
    for key, table in tables.items():
        table.to_csv(out_dir / f"{names.get(key, key)}.csv", index=False)

    try:
        e2_source = e2_dir.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        e2_source = str(e2_dir)
    write_metadata(
        out_dir,
        config={
            "e2_results": e2_source,
            "e2_commit": e2_commit,
            "horizons_s": list(HORIZONS_S),
            "series": list(E2_SERIES),
            "references": list(REFERENCES),
            "pairs": [list(pair) for pair in PAIRS],
            "difference_pair": list(DIFFERENCE_PAIR),
            "ties": "exactly equal RMSE, counted apart from both sides",
            "paired_difference": "rmse_a - rmse_b per run; negative when a is lower",
            "sign_test": "exact two-sided binomial with p = 0.5, ties left out",
            "wilcoxon": "two-sided signed-rank, exact distribution (scipy.stats.wilcoxon)",
            "multiple_comparisons": {
                "method": (
                    "Holm within each family separately, applied to p_sign (p_holm_<family>) and to p_wilcoxon"
                    " (p_wilcoxon_holm_<family>); p_sign and p_wilcoxon are kept unadjusted"
                ),
                "adjusted_tests": ["p_sign", "p_wilcoxon"],
                "horizon_s": COMPARISON_HORIZON_S,
                "members": "unordered pairs, matched to the pairs above in either order",
                "families": {name: [list(pair) for pair in members] for name, members in HOLM_FAMILIES.items()},
            },
            "single_tests": {
                "method": "one comparison each, no adjustment; named in the single_test column",
                "horizon_s": COMPARISON_HORIZON_S,
                "tests": {name: list(pair) for name, pair in SINGLE_TESTS.items()},
            },
            "bootstrap": "percentile interval of the median and of the mean of d, runs resampled with replacement",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "confidence": CONFIDENCE,
            "scipy": version("scipy"),
            "conditioning": {
                "source": "tables/models.csv of the E2 results",
                "matrix": "Z = [Psi(h_k), u_k, 1], fit part, normalized units, columns not standardized",
                "pairs": [list(pair) for pair in SQRT_H_ADDITION],
                "ratio": "cond(with_sqrt) / cond(base) per run",
                "check": "median cond per dictionary equals fit_indicators.csv",
            },
        },
    )
    print(f"Results in {out_dir}")
    return tables


def main() -> None:
    """Compute the comparison tables from the outputs of E2 in E2_DIR and write them to OUT_DIR."""
    run()


if __name__ == "__main__":
    main()
