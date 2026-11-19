"""Tests of the E2 comparison tables on a synthetic errors table whose answers can be worked out by hand, plus regression checks against the saved E2 results when present."""

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from edmdc_tank_observables.paths import PROJECT_ROOT


def load_script():
    """Import experiments/E2_comparisons.py as a module, without running it."""
    path = PROJECT_ROOT / "experiments" / "E2_comparisons.py"
    spec = importlib.util.spec_from_file_location("e2_comparisons", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cmp = load_script()

# RMSE of each series in three runs, the same at every horizon. A and B tie
# exactly in run r3.
RMSE = {
    "D0": [2.0, 3.0, 4.0],
    "A": [1.0, 2.0, 6.0],
    "B": [1.5, 1.0, 6.0],
}


def errors_table(rmse=RMSE, horizons=(30.0, 120.0)):
    """Long errors table with one row per run, series and horizon, as written by E2."""
    rows = []
    for series, values in rmse.items():
        for run, value in zip(("r1", "r2", "r3"), values):
            for horizon_s in horizons:
                rows.append({"run": run, "series": series, "horizon_s": horizon_s, "rmse": value})
    return pd.DataFrame(rows)


def test_summary_gives_median_minimum_maximum_and_mean_per_series_and_horizon():
    """For A with RMSE 1, 2 and 6 in the three runs: median 2, minimum 1, maximum 6 and mean 3, at each horizon."""
    summary = cmp.summary_with_mean(errors_table())
    row = summary[(summary["series"] == "A") & (summary["horizon_s"] == 120.0)].iloc[0]
    assert (row["median"], row["min"], row["max"], row["mean"], row["n_runs"]) == (2.0, 1.0, 6.0, 3.0, 3)
    assert len(summary) == 3 * 2


def test_pair_counts_are_strict_and_report_exact_ties_apart():
    """A beats B in run 1, B beats A in run 2 and they tie exactly in run 3: counts 1 and 1, with 1 tie."""
    counts = cmp.pair_counts(errors_table(), [("A", "B")])
    row = counts[counts["horizon_s"] == 30.0].iloc[0]
    assert (row["a"], row["b"]) == ("A", "B")
    assert (row["n_a_lower"], row["n_b_lower"], row["n_ties"], row["n_runs"]) == (1, 1, 1, 3)
    assert len(counts) == 2


def test_pair_counts_fail_for_a_series_not_in_the_table():
    """Asking for a pair with an unknown series is an error, not a count of zero."""
    with pytest.raises(ValueError, match="D9"):
        cmp.pair_counts(errors_table(), [("A", "D9")])


def test_median_ratios_divide_each_median_by_that_of_each_reference():
    """A has median 2, D0 median 3 and B median 1.5: ratios 2/3 and 4/3."""
    ratios = cmp.median_ratios(cmp.summary_with_mean(errors_table()), references=("D0", "B"))
    a = ratios[(ratios["series"] == "A") & (ratios["horizon_s"] == 30.0)].set_index("reference")["ratio"]
    assert a["D0"] == pytest.approx(2.0 / 3.0)
    assert a["B"] == pytest.approx(4.0 / 3.0)
    d0 = ratios[(ratios["series"] == "D0") & (ratios["reference"] == "D0")]
    assert (d0["ratio"] == 1.0).all()


def test_run_differences_give_a_minus_b_per_run_and_their_summary():
    """A - B is -0.5, 1 and 0 in the three runs: mean 1/6, median 0, minimum -0.5, maximum 1."""
    differences = cmp.run_differences(errors_table(), "A", "B")
    at_30 = differences[differences["horizon_s"] == 30.0].set_index("run")["difference"]
    assert at_30.to_dict() == {"r1": -0.5, "r2": 1.0, "r3": 0.0}
    summary = cmp.difference_summary(differences)
    row = summary[summary["horizon_s"] == 30.0].iloc[0]
    assert row["mean"] == pytest.approx(1.0 / 6.0)
    assert (row["median"], row["min"], row["max"], row["n_runs"]) == (0.0, -0.5, 1.0, 3)


def test_median_of_differences_need_not_match_difference_of_medians():
    """With A = 1, 2, 6 and B = 1.5, 1, 6 the difference of medians is 0.5 but the median of differences is 0."""
    errors = errors_table()
    summary = cmp.summary_with_mean(errors).set_index(["series", "horizon_s"])["median"]
    differences = cmp.difference_summary(cmp.run_differences(errors, "A", "B")).set_index("horizon_s")["median"]
    assert summary[("A", 30.0)] - summary[("B", 30.0)] == pytest.approx(0.5)
    assert differences[30.0] == 0.0


def test_check_against_summary_accepts_the_e2_summary_and_rejects_a_different_one():
    """The recomputed median, minimum and maximum must match the summary written by E2."""
    errors = errors_table()
    e2_summary = cmp.summary_with_mean(errors).drop(columns="mean")
    cmp.check_against_summary(errors, e2_summary)
    broken = e2_summary.copy()
    broken.loc[0, "median"] += 0.01
    with pytest.raises(ValueError):
        cmp.check_against_summary(errors, broken)


def test_read_e2_commit_requires_a_clean_e2_run(tmp_path):
    """The E2 tables are used only if E2 ran from a clean commit; the commit is returned."""
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"git": {"commit": "abc1234", "clean": True}}), encoding="utf-8")
    assert cmp.read_e2_commit(path) == "abc1234"
    path.write_text(json.dumps({"git": {"commit": "abc1234", "clean": False}}), encoding="utf-8")
    with pytest.raises(ValueError, match="clean"):
        cmp.read_e2_commit(path)


def test_two_pairs_with_the_same_counts_are_still_computed_from_different_columns():
    """Two pairs that share A get the same counts only if B's values say so: changing one B changes only its pair."""
    rmse = dict(RMSE, C=[1.5, 1.0, 6.0])
    errors = errors_table(rmse)
    counts = cmp.pair_counts(errors, [("A", "B"), ("A", "C")]).set_index(["b", "horizon_s"])
    assert counts.loc[("B", 30.0), "n_a_lower"] == counts.loc[("C", 30.0), "n_a_lower"] == 1
    rmse["C"] = [0.5, 1.0, 6.0]
    counts = cmp.pair_counts(errors_table(rmse), [("A", "B"), ("A", "C")]).set_index(["b", "horizon_s"])
    assert counts.loc[("B", 30.0), "n_a_lower"] == 1 and counts.loc[("C", 30.0), "n_a_lower"] == 0


# --- Conditioning ----------------------------------------------------------------

# Condition numbers of a base dictionary B and of B plus sqrt(h), W, in three
# runs: ratios W / B of 1.5, 0.9 and 2.5.
COND = {"B": [10.0, 20.0, 40.0], "W": [15.0, 18.0, 100.0]}


def models_table(cond=COND):
    """Models table with one cond per run and dictionary, as written by E2."""
    return pd.DataFrame(
        [{"run": run, "dictionary": d, "cond": c} for d, values in cond.items() for run, c in zip(("r1", "r2", "r3"), values)]
    )


def fit_indicators_table(cond=COND):
    """Fit indicators table with the median cond of each dictionary, as written by E2."""
    return pd.DataFrame({"dictionary": list(cond), "cond": [float(np.median(v)) for v in cond.values()]})


def test_conditioning_table_by_hand():
    """B has cond 10, 20, 40 and W 15, 18, 100: medians 20 and 18, ratios 1.5, 0.9, 2.5 (median 1.5), two above 1."""
    table = cmp.conditioning_table(models_table(), fit_indicators_table(), [("B", "W")])
    assert len(table) == 1
    row = table.iloc[0]
    assert (row["base"], row["with_sqrt"], row["n_runs"]) == ("B", "W", 3)
    assert (row["cond_base_median"], row["cond_base_min"], row["cond_base_max"]) == (20.0, 10.0, 40.0)
    assert (row["cond_with_sqrt_median"], row["cond_with_sqrt_min"], row["cond_with_sqrt_max"]) == (18.0, 15.0, 100.0)
    assert row["ratio_median"] == pytest.approx(1.5)
    assert row["ratio_min"] == pytest.approx(0.9)
    assert row["ratio_max"] == pytest.approx(2.5)
    assert row["n_runs_ratio_above_1"] == 2


def test_conditioning_table_fails_when_a_run_lacks_one_dictionary():
    """If one dictionary of a pair is missing in a run, the table is not built."""
    models = models_table()
    models = models[~((models["run"] == "r2") & (models["dictionary"] == "W"))]
    indicators = fit_indicators_table({"B": COND["B"], "W": [15.0, 100.0]})
    with pytest.raises(ValueError, match="same runs"):
        cmp.conditioning_table(models, indicators, [("B", "W")])


def test_conditioning_table_fails_when_the_median_does_not_match_fit_indicators():
    """If the median cond of a dictionary differs from its fit indicators, the table is not built."""
    indicators = fit_indicators_table()
    indicators.loc[indicators["dictionary"] == "W", "cond"] = 18.0 * (1 + 1e-9)
    with pytest.raises(ValueError, match="W: median cond"):
        cmp.conditioning_table(models_table(), indicators, [("B", "W")])


def test_conditioning_table_fails_on_a_non_positive_cond():
    """A condition number of zero, negative or not finite is rejected."""
    with pytest.raises(ValueError, match="not finite and positive"):
        cmp.conditioning_table(models_table({"B": [10.0, 0.0, 40.0], "W": COND["W"]}), fit_indicators_table(), [("B", "W")])


def test_sqrt_h_addition_pairs_are_the_sqrt_h_family():
    """The (base, base + sqrt(h)) pairs are the four pairs of the sqrt_h Holm family, as unordered pairs."""
    assert {frozenset(p) for p in cmp.SQRT_H_ADDITION} == {frozenset(p) for p in cmp.HOLM_FAMILIES["sqrt_h"]}
    assert len(cmp.SQRT_H_ADDITION) == len(cmp.HOLM_FAMILIES["sqrt_h"])


def test_run_writes_the_tables_and_a_metadata_with_the_e2_commit(tmp_path):
    """A full run on synthetic E2 outputs writes one CSV per table and a metadata.json naming the E2 commit read."""
    e2_dir = tmp_path / "E2_dictionaries"
    (e2_dir / "tables").mkdir(parents=True)
    # Differences between two series change from run to run, as the exact Wilcoxon test needs.
    rmse = {s: [1.0 + i + 0.1 * k * (i + 1) for k in range(3)] for i, s in enumerate(cmp.E2_SERIES)}
    errors = errors_table(rmse, horizons=cmp.HORIZONS_S)
    errors.to_csv(e2_dir / "tables" / "errors.csv", index=False)
    cmp.summary_with_mean(errors).drop(columns="mean").to_csv(e2_dir / "tables" / "error_summary.csv", index=False)
    cond = {s: [10.0 * (i + 1) + k for k in range(3)] for i, s in enumerate(cmp.E2_SERIES)}
    models_table(cond).to_csv(e2_dir / "tables" / "models.csv", index=False)
    fit_indicators_table(cond).to_csv(e2_dir / "tables" / "fit_indicators.csv", index=False)
    (e2_dir / "metadata.json").write_text(json.dumps({"git": {"commit": "abc1234", "clean": True}}), encoding="utf-8")

    out_dir = tmp_path / "E2_comparisons"
    tables = cmp.run(e2_dir, out_dir)
    assert set(tables) == {
        "summary", "pair_counts", "median_ratios", "differences", "differences_summary", "paired_tests",
        "conditioning_sqrt_h",
    }
    for name in (
        "summary", "pair_counts", "median_ratios", "differences_D1a_D1b", "differences_summary", "paired_tests",
        "conditioning_sqrt_h",
    ):
        assert (out_dir / f"{name}.csv").is_file()
    conditioning = pd.read_csv(out_dir / "conditioning_sqrt_h.csv")
    assert list(zip(conditioning["base"], conditioning["with_sqrt"])) == list(cmp.SQRT_H_ADDITION)
    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["config"]["e2_commit"] == "abc1234"
    assert metadata["config"]["pairs"][0] == ["D1a", "D1b"]
    assert metadata["config"]["conditioning"]["pairs"] == [list(p) for p in cmp.SQRT_H_ADDITION]
    counts = pd.read_csv(out_dir / "pair_counts.csv")
    assert len(counts) == len(cmp.PAIRS) * len(cmp.HORIZONS_S)
    assert np.all(counts["n_a_lower"] + counts["n_b_lower"] + counts["n_ties"] == 3)
    tests = pd.read_csv(out_dir / "paired_tests.csv")
    assert len(tests) == len(cmp.PAIRS) * len(cmp.HORIZONS_S)
    assert (metadata["config"]["bootstrap_seed"], metadata["config"]["bootstrap_resamples"]) == (
        cmp.BOOTSTRAP_SEED,
        cmp.BOOTSTRAP_RESAMPLES,
    )


# --- Paired tests ----------------------------------------------------------------


@pytest.mark.parametrize(
    "n_a, n_b, expected",
    [(10, 0, 2 / 1024), (9, 1, 22 / 1024), (8, 2, 112 / 1024), (7, 3, 352 / 1024), (3, 7, 352 / 1024)],
)
def test_sign_test_gives_the_exact_two_sided_binomial_p_value(n_a, n_b, expected):
    """Over 10 runs, 10/0, 9/1, 8/2 and 7/3 wins give p = 2, 22, 112 and 352 out of 1024, on either side."""
    assert cmp.sign_test_p(n_a, n_b) == pytest.approx(expected, rel=1e-12)


def test_sign_test_is_capped_at_one_and_fails_without_untied_runs():
    """A 5/5 split gives p = 1, and with every run tied there is nothing to test."""
    assert cmp.sign_test_p(5, 5) == 1.0
    with pytest.raises(ValueError):
        cmp.sign_test_p(0, 0)


def test_holm_adjustment_on_a_known_vector():
    """Holm on 0.01, 0.04, 0.03, 0.005 gives 0.03, 0.06, 0.06, 0.02: sorted p times 4, 3, 2, 1, kept non-decreasing."""
    adjusted = cmp.holm([0.01, 0.04, 0.03, 0.005])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02], rel=1e-12)


def test_holm_adjustment_is_capped_at_one():
    """An adjusted p-value never exceeds 1."""
    assert cmp.holm([0.5, 0.6]) == pytest.approx([1.0, 1.0])


def test_wilcoxon_with_every_difference_on_one_side_gives_two_over_1024():
    """With all ten differences negative the exact two-sided Wilcoxon p-value is 2/1024, as for the sign test."""
    d = -np.arange(1.0, 11.0)
    assert cmp.wilcoxon_p(d) == pytest.approx(2 / 1024, rel=1e-12)


def test_wilcoxon_fails_on_zero_or_tied_absolute_differences():
    """The exact distribution does not hold with zero or tied absolute differences, so they are an error."""
    with pytest.raises(ValueError):
        cmp.wilcoxon_p(np.array([0.0, 1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        cmp.wilcoxon_p(np.array([-1.0, 1.0, 2.0, 3.0]))


def test_bootstrap_is_deterministic_with_the_seed():
    """The same seed gives the same interval, a different seed a different one."""
    d = np.array([-0.3, 0.1, -0.5, 0.2, -0.1, -0.4, 0.6, -0.2, -0.05, 0.15])
    first = cmp.bootstrap_ci(d, np.median, n_resamples=2000, seed=7)
    assert cmp.bootstrap_ci(d, np.median, n_resamples=2000, seed=7) == first
    assert cmp.bootstrap_ci(d, np.mean, n_resamples=2000, seed=8) != cmp.bootstrap_ci(d, np.mean, n_resamples=2000, seed=7)


def test_bootstrap_of_a_constant_difference_is_that_constant():
    """If every run has the same difference, every resample does too, so both ends of the interval equal it."""
    d = np.full(10, -0.25)
    for statistic in (np.median, np.mean):
        assert cmp.bootstrap_ci(d, statistic, n_resamples=500, seed=0) == pytest.approx((-0.25, -0.25), abs=1e-15)


# As RMSE but with B changed in run r3 so that A - B is -0.5, 1 and -0.25: no zero
# and no tied absolute difference, as the exact Wilcoxon test requires.
RMSE_NO_TIES = dict(RMSE, B=[1.5, 1.0, 6.25])


def test_paired_tests_report_counts_statistics_and_p_values_per_pair_and_horizon():
    """A - B is -0.5, 1 and -0.25: A wins twice, sign and Wilcoxon p = 1 with three runs, and the summaries of d."""
    tests = cmp.paired_tests(
        errors_table(RMSE_NO_TIES), [("A", "B")], n_resamples=200, seed=0,
        families={}, single_tests={}, comparison_horizon_s=120.0,
    )
    row = tests[tests["horizon_s"] == 30.0].iloc[0]
    assert (row["a"], row["b"], row["n_runs"]) == ("A", "B", 3)
    assert (row["n_wins_a"], row["n_wins_b"], row["n_ties"]) == (2, 1, 0)
    assert row["p_sign"] == 1.0
    assert row["p_wilcoxon"] == pytest.approx(1.0)
    assert (row["diff_median"], row["diff_min"], row["diff_max"]) == (-0.25, -0.5, 1.0)
    assert row["diff_mean"] == pytest.approx(0.25 / 3.0)
    assert row["diff_median_ci_low"] <= row["diff_median"] <= row["diff_median_ci_high"]
    assert row["diff_mean_ci_low"] <= row["diff_mean"] <= row["diff_mean_ci_high"]
    assert len(tests) == 2


# D0 = 2, 3, 4 as in RMSE. A - D0 is -1, -0.5, 0.75 (2 wins to 1, sign p = 1); B - D0 is
# -0.25, -1.1, -0.4 (3 to 0, sign p = 2/8); B - A is 0.75, -0.6, -1.15 (2 to 1, sign p = 1).
RMSE_FAMILIES = dict(RMSE, A=[1.0, 2.5, 4.75], B=[1.75, 1.9, 3.6])
FAMILY_PAIRS = [("A", "D0"), ("B", "D0"), ("B", "A")]


def test_holm_families_are_adjusted_separately_at_one_horizon_with_a_row_in_both():
    """Family f1 = {A-D0, B-D0} gives 1 and 0.5; f2 = {B-D0, A-B} gives 0.5 and 1; B-D0 is in both; other rows are NaN."""
    families = {"f1": [("A", "D0"), ("B", "D0")], "f2": [("D0", "B"), ("A", "B")]}
    tests = cmp.paired_tests(
        errors_table(RMSE_FAMILIES), FAMILY_PAIRS, n_resamples=200, seed=0,
        families=families, single_tests={}, comparison_horizon_s=120.0,
    )
    at_120 = tests[tests["horizon_s"] == 120.0].set_index(["a", "b"])
    assert at_120["p_sign"].tolist() == pytest.approx([1.0, 0.25, 1.0])
    assert at_120.loc[("A", "D0"), "p_holm_f1"] == pytest.approx(1.0)
    assert at_120.loc[("B", "D0"), "p_holm_f1"] == pytest.approx(0.5)
    assert at_120.loc[("B", "D0"), "p_holm_f2"] == pytest.approx(0.5)
    assert at_120.loc[("B", "A"), "p_holm_f2"] == pytest.approx(1.0)
    assert np.isnan(at_120.loc[("B", "A"), "p_holm_f1"]) and np.isnan(at_120.loc[("A", "D0"), "p_holm_f2"])
    at_30 = tests[tests["horizon_s"] == 30.0]
    assert at_30[["p_holm_f1", "p_holm_f2"]].isna().all().all()
    assert at_30["p_sign"].notna().all() and at_30["p_wilcoxon"].notna().all()


def test_wilcoxon_holm_families_are_adjusted_separately_at_one_horizon_with_a_row_in_both():
    """Exact Wilcoxon p is 0.75 (A-D0), 0.25 (B-D0), 0.75 (B-A); Holm gives f1 = 0.75, 0.5 and f2 = 0.5, 0.75; NaN elsewhere."""
    families = {"f1": [("A", "D0"), ("B", "D0")], "f2": [("D0", "B"), ("A", "B")]}
    tests = cmp.paired_tests(
        errors_table(RMSE_FAMILIES), FAMILY_PAIRS, n_resamples=200, seed=0,
        families=families, single_tests={}, comparison_horizon_s=120.0,
    )
    columns = list(tests.columns)
    assert columns[columns.index("p_wilcoxon") + 1 : columns.index("single_test")] == [
        "p_holm_f1", "p_holm_f2", "p_wilcoxon_holm_f1", "p_wilcoxon_holm_f2",
    ]
    at_120 = tests[tests["horizon_s"] == 120.0].set_index(["a", "b"])
    assert at_120["p_wilcoxon"].tolist() == pytest.approx([0.75, 0.25, 0.75])
    assert at_120.loc[("A", "D0"), "p_wilcoxon_holm_f1"] == pytest.approx(0.75)
    assert at_120.loc[("B", "D0"), "p_wilcoxon_holm_f1"] == pytest.approx(0.5)
    assert at_120.loc[("B", "D0"), "p_wilcoxon_holm_f2"] == pytest.approx(0.5)
    assert at_120.loc[("B", "A"), "p_wilcoxon_holm_f2"] == pytest.approx(0.75)
    assert np.isnan(at_120.loc[("B", "A"), "p_wilcoxon_holm_f1"])
    assert np.isnan(at_120.loc[("A", "D0"), "p_wilcoxon_holm_f2"])
    at_30 = tests[tests["horizon_s"] == 30.0]
    assert at_30[["p_wilcoxon_holm_f1", "p_wilcoxon_holm_f2"]].isna().all().all()


def test_single_tests_are_named_at_one_horizon_and_keep_their_p_values():
    """The single test s1 = A-B names row B-A at 120 s only; its p-values are those without it, and other rows are empty."""
    kwargs = dict(n_resamples=200, seed=0, families={}, comparison_horizon_s=120.0)
    tests = cmp.paired_tests(errors_table(RMSE_FAMILIES), FAMILY_PAIRS, single_tests={"s1": ("A", "B")}, **kwargs)
    plain = cmp.paired_tests(errors_table(RMSE_FAMILIES), FAMILY_PAIRS, single_tests={}, **kwargs)
    named = tests[tests["single_test"].notna()]
    assert named[["a", "b", "horizon_s", "single_test"]].values.tolist() == [["B", "A", 120.0, "s1"]]
    assert plain["single_test"].isna().all()
    for column in ("p_sign", "p_wilcoxon"):
        assert tests[column].tolist() == plain[column].tolist()


def test_holm_family_member_or_single_test_must_match_a_reported_pair():
    """A family member or single test that is not among the pairs is an error, not silently left out."""
    kwargs = dict(n_resamples=200, seed=0, comparison_horizon_s=120.0)
    with pytest.raises(ValueError, match="f1"):
        cmp.paired_tests(
            errors_table(RMSE_FAMILIES), FAMILY_PAIRS, families={"f1": [("A", "D9")]}, single_tests={}, **kwargs
        )
    with pytest.raises(ValueError, match="s1"):
        cmp.paired_tests(
            errors_table(RMSE_FAMILIES), FAMILY_PAIRS, families={}, single_tests={"s1": ("A", "D9")}, **kwargs
        )


def test_script_families_and_single_tests_have_the_declared_members_and_are_all_in_pairs():
    """curvature_vs_D0 has 6 pairs, sqrt_h 4, plus the single tests delay and curvature_choice, all among PAIRS at 120 s."""
    assert cmp.COMPARISON_HORIZON_S == 120.0
    assert set(cmp.HOLM_FAMILIES) == {"curvature_vs_D0", "sqrt_h"}
    assert {frozenset(p) for p in cmp.HOLM_FAMILIES["curvature_vs_D0"]} == {
        frozenset((s, "D0")) for s in ("D1a", "D1b", "D2", "D3a", "D3b", "D3c")
    }
    assert len(cmp.HOLM_FAMILIES["curvature_vs_D0"]) == 6
    assert {frozenset(p) for p in cmp.HOLM_FAMILIES["sqrt_h"]} == {
        frozenset(p) for p in [("D2", "D0"), ("D3c", "D1c"), ("D1a", "D3a"), ("D1b", "D3b")]
    }
    assert len(cmp.HOLM_FAMILIES["sqrt_h"]) == 4
    assert {name: frozenset(p) for name, p in cmp.SINGLE_TESTS.items()} == {
        "delay": frozenset(("D1c", "D0")),
        "curvature_choice": frozenset(("D1a", "D2")),
    }
    pairs = {frozenset(p) for p in cmp.PAIRS}
    assert all(frozenset(p) in pairs for members in cmp.HOLM_FAMILIES.values() for p in members)
    assert all(frozenset(p) in pairs for p in cmp.SINGLE_TESTS.values())


# --- Regression against the saved E2 comparison tables ---------------------------

RESULTS_DIR = PROJECT_ROOT / "results" / "E2_comparisons"
E2_RESULTS_DIR = PROJECT_ROOT / "results" / "E2_dictionaries"
# Files of the E2 results that the comparison script reads.
E2_INPUTS = [
    E2_RESULTS_DIR / "metadata.json",
    *(E2_RESULTS_DIR / "tables" / f"{name}.csv" for name in ("errors", "error_summary", "models", "fit_indicators")),
]


def skip_without(*paths):
    """Skip the test if any of the given result files does not exist."""
    missing = [p.relative_to(PROJECT_ROOT).as_posix() for p in paths if not p.is_file()]
    if missing:
        pytest.skip(f"results not available: {missing}")


# The eight pairs listed first in PAIRS, before the pairs of each dictionary against D0.
LEADING_PAIRS = [
    ("D1a", "D1b"),
    ("D1a", "D2"),
    ("D1a", "D3a"),
    ("D3c", "D2"),
    ("D1b", "D2"),
    ("D1b", "D3b"),
    ("D1c", "D0"),
    ("D3c", "D1c"),
]


def test_pairs_list_the_eight_specific_pairs_then_each_dictionary_against_d0():
    """PAIRS starts with the eight specific pairs, then every dictionary except D1c (already among them) against D0."""
    assert list(cmp.PAIRS[: len(LEADING_PAIRS)]) == LEADING_PAIRS
    against_d0 = list(cmp.PAIRS[len(LEADING_PAIRS) :])
    assert against_d0 == [("D1a", "D0"), ("D1b", "D0"), ("D2", "D0"), ("D3a", "D0"), ("D3b", "D0"), ("D3c", "D0")]
    assert all(s in cmp.E2_SERIES for pair in cmp.PAIRS for s in pair)


def test_saved_e2_errors_have_exactly_the_compared_series():
    """The series of the saved E2 errors table are exactly E2_SERIES and REFERENCES, so no other series appears in the comparisons."""
    skip_without(E2_RESULTS_DIR / "tables" / "errors.csv")
    errors = pd.read_csv(E2_RESULTS_DIR / "tables" / "errors.csv")
    assert set(errors["series"]) == set(cmp.E2_SERIES) | set(cmp.REFERENCES)


def test_rerun_on_the_e2_results_reproduces_the_saved_tables(tmp_path):
    """Run on the saved E2 results, the script reproduces the saved summary, ratio and difference tables and the counts of the eight leading pairs."""
    saved = ("summary", "median_ratios", "differences_D1a_D1b", "differences_summary", "pair_counts")
    skip_without(*E2_INPUTS, *(RESULTS_DIR / f"{name}.csv" for name in saved))
    cmp.run(E2_RESULTS_DIR, tmp_path)
    for name in ("summary", "median_ratios", "differences_D1a_D1b", "differences_summary"):
        pd.testing.assert_frame_equal(
            pd.read_csv(tmp_path / f"{name}.csv"), pd.read_csv(RESULTS_DIR / f"{name}.csv"), rtol=1e-12, atol=0.0
        )
    keys = pd.DataFrame(LEADING_PAIRS, columns=["a", "b"])
    rerun = keys.merge(pd.read_csv(tmp_path / "pair_counts.csv"), on=["a", "b"], how="left")
    saved_counts = keys.merge(pd.read_csv(RESULTS_DIR / "pair_counts.csv"), on=["a", "b"], how="left")
    pd.testing.assert_frame_equal(rerun, saved_counts)
    assert len(rerun) == len(LEADING_PAIRS) * len(cmp.HORIZONS_S)


# Columns of paired_tests.csv other than the Wilcoxon Holm columns, in order.
PAIRED_TESTS_COLUMNS = [
    "a", "b", "horizon_s", "n_runs", "n_wins_a", "n_wins_b", "n_ties", "p_sign", "p_wilcoxon",
    "p_holm_curvature_vs_D0", "p_holm_sqrt_h", "single_test", "diff_median", "diff_median_ci_low",
    "diff_median_ci_high", "diff_mean", "diff_mean_ci_low", "diff_mean_ci_high", "diff_min", "diff_max",
]
WILCOXON_HOLM_COLUMNS = ["p_wilcoxon_holm_curvature_vs_D0", "p_wilcoxon_holm_sqrt_h"]


def test_paired_tests_keeps_the_saved_columns_and_places_the_wilcoxon_holm_columns_after_the_sign_ones(tmp_path):
    """Rerun on the saved E2 results: the columns of the saved paired_tests are identical, and the two Wilcoxon Holm columns follow the sign-test Holm columns."""
    skip_without(*E2_INPUTS, RESULTS_DIR / "paired_tests.csv")
    cmp.run(E2_RESULTS_DIR, tmp_path)
    new = pd.read_csv(tmp_path / "paired_tests.csv")
    saved = pd.read_csv(RESULTS_DIR / "paired_tests.csv")
    pd.testing.assert_frame_equal(new[PAIRED_TESTS_COLUMNS], saved[PAIRED_TESTS_COLUMNS], check_exact=True)
    expected = PAIRED_TESTS_COLUMNS[:11] + WILCOXON_HOLM_COLUMNS + PAIRED_TESTS_COLUMNS[11:]
    assert list(new.columns) == expected
    single = new[new["single_test"].notna()]
    assert single[[*WILCOXON_HOLM_COLUMNS, "p_holm_curvature_vs_D0", "p_holm_sqrt_h"]].isna().all().all()
    for name, members in cmp.HOLM_FAMILIES.items():
        assert new[f"p_wilcoxon_holm_{name}"].notna().sum() == len(members)
