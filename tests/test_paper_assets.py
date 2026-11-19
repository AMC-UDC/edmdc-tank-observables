"""Tests of the paper tables and figure on small synthetic results, plus one check on the real results."""

import importlib.util
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from edmdc_tank_observables.paths import PROJECT_ROOT


def load_script():
    """Import experiments/paper_assets.py as a module, without running it."""
    path = PROJECT_ROOT / "experiments" / "paper_assets.py"
    spec = importlib.util.spec_from_file_location("paper_assets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pa = load_script()

ORDER = ("D0", "D1a", "D1b", "D1c", "D2", "D3a", "D3b", "D3c")


# --- Synthetic tables ------------------------------------------------------------


def fit_indicators():
    """Fit indicators with fit_rmse_pct = 0.1 * (position + 1), rows in reverse order."""
    return pd.DataFrame(
        {"dictionary": list(reversed(ORDER)), "fit_rmse_pct": [0.1 * (ORDER.index(d) + 1) for d in reversed(ORDER)]}
    )


def error_summary():
    """Error summary where the median at H seconds is H / 100 + position, min = median - 1, max = median + 1."""
    rows = []
    for d in ORDER:
        for h in (30.0, 120.0, 600.0):
            median = h / 100 + ORDER.index(d)
            rows.append({"series": d, "horizon_s": h, "median": median, "min": median - 1, "max": median + 1})
    return pd.DataFrame(rows)


def paired_row(a, b, **values):
    """One row of paired_tests at 120 s with neutral defaults, overridden by values."""
    row = {
        "a": a,
        "b": b,
        "horizon_s": 120.0,
        "n_runs": 10,
        "n_wins_a": 6,
        "n_wins_b": 4,
        "n_ties": 0,
        "p_sign": 0.5,
        "p_wilcoxon": 0.5,
        "p_holm_curvature_vs_D0": None,
        "p_holm_sqrt_h": None,
        "p_wilcoxon_holm_curvature_vs_D0": None,
        "p_wilcoxon_holm_sqrt_h": None,
        "single_test": None,
        "diff_median": -0.1,
        "diff_median_ci_low": -0.2,
        "diff_median_ci_high": 0.05,
    }
    row.update(values)
    return row


def paired_tests():
    """Every pair the table needs, with D3a/D1a and D3b/D1b stored the other way round as (D1a, D3a) and (D1b, D3b)."""
    curvature = dict(p_holm_curvature_vs_D0=0.2, p_wilcoxon_holm_curvature_vs_D0=0.3)
    sqrt_h = dict(p_holm_sqrt_h=0.4, p_wilcoxon_holm_sqrt_h=0.6)
    rows = [paired_row(d, "D0", **curvature) for d in ("D1a", "D1b", "D3a", "D3b", "D3c")]
    rows += [
        paired_row("D2", "D0", **curvature, **sqrt_h),
        paired_row("D3c", "D1c", **sqrt_h),
        paired_row("D1a", "D3a", **sqrt_h, n_wins_a=9, n_wins_b=1, diff_median=-0.7, diff_median_ci_low=-1.5, diff_median_ci_high=-0.5),
        paired_row("D1b", "D3b", **sqrt_h),
        paired_row("D1c", "D0", single_test="delay", p_sign=0.0004, p_wilcoxon=0.25),
        paired_row("D1a", "D2", single_test="curvature_choice"),
        paired_row("D3c", "D2", n_wins_a=4, n_wins_b=6, diff_median=0.05, diff_median_ci_low=-0.09, diff_median_ci_high=0.18, p_sign=0.754, p_wilcoxon=0.625),
        # The same pair at another horizon must be ignored.
        paired_row("D1c", "D0", horizon_s=30.0, p_sign=0.9),
    ]
    return pd.DataFrame(rows)


# The six pairs as E2_one_step stores them: (D1a, D3a) and (D1b, D3b) are the
# other way round from the one-step table.
STORED_PAIRS = (("D2", "D0"), ("D1a", "D3a"), ("D1b", "D3b"), ("D3c", "D1c"), ("D1c", "D0"), ("D1a", "D2"))


def one_step_counts():
    """One-step counts for STORED_PAIRS: eval_all, window rows at 30 and 120 s and a fit row that must be ignored."""
    rows = []
    for a, b in STORED_PAIRS:
        common = dict(a=a, b=b, n_ties=0, n_runs=10)
        rows += [
            dict(common, segment="fit", horizon_s=None, n_a_lower=1, n_b_lower=9, median_diff=9.0),
            dict(common, segment="eval_all", horizon_s=None, n_a_lower=7, n_b_lower=3, median_diff=-0.0234),
            dict(common, segment="window", horizon_s=30.0, n_a_lower=2, n_b_lower=8, median_diff=8.0,
                 multistep_n_a_lower=2.0, multistep_n_b_lower=8.0, multistep_median_diff=8.0),
            dict(common, segment="window", horizon_s=120.0, n_a_lower=8, n_b_lower=2, median_diff=0.0456,
                 multistep_n_a_lower=10.0, multistep_n_b_lower=0.0, multistep_median_diff=-1.2611),
        ]
    return pd.DataFrame(rows)


def body_lines(table):
    """Lines of a tabular between \\midrule (the first) and \\bottomrule."""
    lines = table.splitlines()
    return lines[lines.index(r"\midrule") + 1 : lines.index(r"\bottomrule")]


def header_lines(table):
    """Lines of a tabular between \\toprule and the first \\midrule."""
    lines = table.splitlines()
    return lines[lines.index(r"\toprule") + 1 : lines.index(r"\midrule")]


# --- Formatting ------------------------------------------------------------------


def test_numbers_use_a_decimal_point_and_a_math_minus():
    """1.234 with 2 decimals is 1.23 and -0.0456 with 3 decimals is $-$0.046."""
    assert pa.latex_number(1.234, 2) == "1.23"
    assert pa.latex_number(-0.0456, 3) == "$-$0.046"


def test_a_negative_number_that_rounds_to_zero_has_no_minus():
    """-0.001 with 2 decimals is written 0.00, not $-$0.00."""
    assert pa.latex_number(-0.001, 2) == "0.00"


def test_p_values_have_three_decimals_or_a_bound():
    """0.0429 is 0.043, 0.001 is 0.001 and anything below 0.001 is $<$0.001."""
    assert pa.latex_p(0.04296875) == "0.043"
    assert pa.latex_p(0.001) == "0.001"
    assert pa.latex_p(0.0004) == "$<$0.001"


def test_dictionaries_are_written_with_the_manuscript_macros():
    """Each dictionary ID maps to its macro, and an unknown ID is an error."""
    expected = (r"\dLin", r"\dPoly", r"\dRBF", r"\dDelay", r"\dSqrt", r"\dPolySqrt", r"\dRBFSqrt", r"\dDelaySqrt")
    assert tuple(pa.dictionary_macro(d) for d in ORDER) == expected
    with pytest.raises(KeyError):
        pa.dictionary_macro("D9")


def test_tables_have_booktabs_rules_and_no_vertical_lines_or_float():
    """Every table is a bare tabular with booktabs rules and no |, \\hline, colour, caption or label."""
    tables = (
        pa.table_errors(fit_indicators(), error_summary()),
        pa.table_paired(paired_tests()),
        pa.table_one_step(one_step_counts()),
    )
    for table in tables:
        assert table.startswith(r"\begin{tabular}") and table.rstrip().endswith(r"\end{tabular}")
        for rule in (r"\toprule", r"\midrule", r"\bottomrule"):
            assert rule in table
        for banned in ("|", r"\hline", "color", r"\caption", r"\label", r"\begin{table}"):
            assert banned not in table


def test_provenance_comment_names_sources_and_both_commits():
    """The first line is one comment with the source files, the results commit and the script commit."""
    line = pa.provenance_comment(["results/x.csv", "results/y.csv"], "aaa1111", "bbb2222")
    assert line == "% Sources: results/x.csv, results/y.csv; results commit aaa1111; script commit bbb2222"


# --- T2: errors ------------------------------------------------------------------


def test_errors_table_has_one_row_per_dictionary_in_the_fixed_order():
    """Rows come out D0 to D3c even though fit_indicators lists them in reverse."""
    rows = body_lines(pa.table_errors(fit_indicators(), error_summary()))
    assert [row.split(" & ")[0] for row in rows] == [pa.dictionary_macro(d) for d in ORDER]


def test_errors_table_row_by_hand():
    """D1a (position 1): one-step 0.200 (3 decimals); at 30, 120, 600 s medians 1.30, 2.20, 7.00, min and max one below and above."""
    rows = body_lines(pa.table_errors(fit_indicators(), error_summary()))
    assert rows[1] == r"\dPoly & 0.200 & 1.30 [0.30, 2.30] & 2.20 [1.20, 3.20] & 7.00 [6.00, 8.00] \\"


def test_errors_table_fails_when_a_dictionary_is_missing():
    """Without the D3c rows of the error summary the table is not written."""
    summary = error_summary()
    with pytest.raises(ValueError, match="D3c"):
        pa.table_errors(fit_indicators(), summary[summary["series"] != "D3c"])


# --- T3: paired tests ------------------------------------------------------------


def test_paired_values_in_the_stored_order_are_taken_as_they_are():
    """D1a vs D3a is stored as (D1a, D3a): counts 9/1, median -0.7 and CI [-1.5, -0.5] unchanged."""
    v = pa.paired_values(paired_tests(), "D1a", "D3a", None)
    assert (v["n_a"], v["n_b"]) == (9, 1)
    assert (v["median"], v["ci_low"], v["ci_high"]) == (-0.7, -1.5, -0.5)


def test_paired_values_of_a_reversed_pair_flip_signs_counts_and_bounds():
    """D3a vs D1a, stored as (D1a, D3a): counts 1/9, median 0.7 and CI [0.5, 1.5]; p-values unchanged."""
    v = pa.paired_values(paired_tests(), "D3a", "D1a", "sqrt_h")
    assert (v["n_a"], v["n_b"]) == (1, 9)
    assert (v["median"], v["ci_low"], v["ci_high"]) == (0.7, 0.5, 1.5)
    assert (v["p_sign"], v["p_wilcoxon"]) == (0.4, 0.6)


def test_paired_values_use_the_holm_columns_of_the_family_asked_for():
    """D2 vs D0 is in both families: curvature gives p 0.2 and 0.3, sqrt(h) gives 0.4 and 0.6, none gives 0.5 and 0.5."""
    tests = paired_tests()
    curvature = pa.paired_values(tests, "D2", "D0", "curvature_vs_D0")
    sqrt_h = pa.paired_values(tests, "D2", "D0", "sqrt_h")
    unadjusted = pa.paired_values(tests, "D2", "D0", None)
    assert (curvature["p_sign"], curvature["p_wilcoxon"]) == (0.2, 0.3)
    assert (sqrt_h["p_sign"], sqrt_h["p_wilcoxon"]) == (0.4, 0.6)
    assert (unadjusted["p_sign"], unadjusted["p_wilcoxon"]) == (0.5, 0.5)


def test_paired_values_fail_when_the_family_column_is_empty():
    """D1c vs D0 is not in the curvature family, so asking for its Holm p-value is an error."""
    with pytest.raises(ValueError, match="curvature_vs_D0"):
        pa.paired_values(paired_tests(), "D1c", "D0", "curvature_vs_D0")


def test_paired_values_fail_for_a_missing_pair():
    """A pair that is not in the table, in either order, is an error."""
    with pytest.raises(ValueError, match="D1c"):
        pa.paired_values(paired_tests(), "D1c", "D2", None)


def test_paired_table_has_three_labelled_groups_in_the_fixed_order():
    """Group 1 has the six curvature dictionaries vs D0, group 2 the four sqrt(h) pairs (with sqrt(h) first), group 3 three pairs."""
    rows = [row for row in body_lines(pa.table_paired(paired_tests())) if row != r"\midrule"]
    labels = [row for row in rows if row.startswith(r"\multicolumn")]
    comparisons = [row.split(" & ")[0] for row in rows if not row.startswith(r"\multicolumn")]
    assert len(labels) == 3
    m = pa.dictionary_macro
    expected = [(d, "D0") for d in ("D1a", "D1b", "D2", "D3a", "D3b", "D3c")]
    expected += [("D2", "D0"), ("D3c", "D1c"), ("D3a", "D1a"), ("D3b", "D1b"), ("D1c", "D0"), ("D1a", "D2"), ("D3c", "D2")]
    assert comparisons == [f"{m(a)}{{}} vs {m(b)}" for a, b in expected]
    assert rows.index(labels[1]) == 7 and rows.index(labels[2]) == 12


def test_paired_table_header_and_sqrt_label_use_the_manuscript_notation():
    """The header reads Comparison, n_X/n_Y, Median delta [95% CI] and the two p-values; group 2 is labelled with sqrt(h_k)."""
    table = pa.table_paired(paired_tests())
    assert header_lines(table) == [r"Comparison & $n_X/n_Y$ & Median $\delta$ [95\% CI] & $p$ sign & $p$ Wilcoxon \\"]
    assert r"\multicolumn{5}{l}{\emph{Adding $\sqrt{h_k}$}} \\" in body_lines(table)


def test_paired_table_rows_by_hand():
    """D3a vs D1a in group 2 is flipped and uses the sqrt(h) Holm p; D1c vs D0 and D3c vs D2 in group 3 use the unadjusted p."""
    rows = body_lines(pa.table_paired(paired_tests()))
    assert r"\dPolySqrt{} vs \dPoly & 1/9 & 0.70 [0.50, 1.50] & 0.400 & 0.600 \\" in rows
    assert r"\dDelay{} vs \dLin & 6/4 & $-$0.10 [$-$0.20, 0.05] & $<$0.001 & 0.250 \\" in rows
    assert r"\dDelaySqrt{} vs \dSqrt & 4/6 & 0.05 [$-$0.09, 0.18] & 0.754 & 0.625 \\" in rows


def test_paired_table_fails_when_there_are_ties():
    """A tie would not show in n_X/n_Y, so a pair with ties is an error."""
    tests = paired_tests()
    tests.loc[(tests["a"] == "D1c") & (tests["horizon_s"] == 120.0), ["n_wins_a", "n_ties"]] = (5, 1)
    with pytest.raises(ValueError, match="tie"):
        pa.table_paired(tests)


# --- T4: one step ----------------------------------------------------------------


def test_one_step_values_in_the_stored_order_are_taken_as_they_are():
    """D2 vs D0 is stored as (D2, D0): eval_all 7/3 and -0.0234, window 8/2 and 0.0456, multi-step 10/0 and -1.2611."""
    v = pa.one_step_values(one_step_counts(), "D2", "D0")
    assert v == {"eval_all": (7, 3, -0.0234), "window": (8, 2, 0.0456), "multistep": (10, 0, -1.2611)}


def test_one_step_values_of_a_reversed_pair_flip_counts_and_signs():
    """D3a vs D1a, stored as (D1a, D3a): eval_all 3/7 and 0.0234, window 2/8 and -0.0456, multi-step 0/10 and 1.2611."""
    v = pa.one_step_values(one_step_counts(), "D3a", "D1a")
    assert v == {"eval_all": (3, 7, 0.0234), "window": (2, 8, -0.0456), "multistep": (0, 10, 1.2611)}


def test_one_step_values_fail_when_the_multistep_counts_have_ties():
    """Multi-step counts 9/0 out of 10 runs hide a tie, so they are an error."""
    counts = one_step_counts()
    counts.loc[(counts["a"] == "D2") & (counts["horizon_s"] == 120.0), "multistep_n_a_lower"] = 9.0
    with pytest.raises(ValueError, match="ties"):
        pa.one_step_values(counts, "D2", "D0")


def test_one_step_table_header_has_three_groups_without_the_horizon_and_wider_gaps_between_them():
    """Groups One-step windows, Multi-step windows and One-step evaluation segment, on two lines, over Comparison and n_X/n_Y, Median delta; 3 column separations between groups."""
    table = pa.table_one_step(one_step_counts())
    groups, segments, rule, columns = header_lines(table)
    assert groups == (
        r" & \multicolumn{2}{c}{One-step} & \multicolumn{2}{c}{Multi-step} & \multicolumn{2}{c}{One-step} \\"
    )
    assert segments == (
        r" & \multicolumn{2}{c}{windows} & \multicolumn{2}{c}{windows} & \multicolumn{2}{c}{evaluation segment} \\"
    )
    assert "120" not in groups + segments
    assert rule == r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}"
    assert columns == "Comparison" + r" & $n_X/n_Y$ & Median $\delta$" * 3 + r" \\"
    gap = r"@{\hspace{3\tabcolsep}}"
    assert table.splitlines()[0] == rf"\begin{{tabular}}{{lcc{gap}cc{gap}cc}}"


def test_one_step_table_rows_follow_the_paired_table_orientation():
    """Rows are D2 vs D0, D3c vs D1c, D3a vs D1a, D3b vs D1b, D1c vs D0, D1a vs D2, with columns windows, multi-step, evaluation segment; a flipped row is written by hand."""
    rows = body_lines(pa.table_one_step(one_step_counts()))
    m = pa.dictionary_macro
    expected = [("D2", "D0"), ("D3c", "D1c"), ("D3a", "D1a"), ("D3b", "D1b"), ("D1c", "D0"), ("D1a", "D2")]
    assert [row.split(" & ")[0] for row in rows] == [f"{m(a)}{{}} vs {m(b)}" for a, b in expected]
    assert rows[0] == r"\dSqrt{} vs \dLin & 8/2 & 0.046 & 10/0 & $-$1.261 & 7/3 & $-$0.023 \\"
    assert rows[2] == r"\dPolySqrt{} vs \dPoly & 2/8 & $-$0.046 & 0/10 & 1.261 & 3/7 & 0.023 \\"


# --- Whole run -------------------------------------------------------------------


def write_results(root, clean=True):
    """Write the synthetic tables and metadata in the folder layout of the real results; return the three folders."""
    e2_dir, cmp_dir, os_dir = root / "E2_dictionaries", root / "E2_comparisons", root / "E2_one_step"
    (e2_dir / "tables").mkdir(parents=True)
    cmp_dir.mkdir()
    os_dir.mkdir()
    fit_indicators().to_csv(e2_dir / "tables" / "fit_indicators.csv", index=False)
    error_summary().to_csv(e2_dir / "tables" / "error_summary.csv", index=False)
    paired_tests().to_csv(cmp_dir / "paired_tests.csv", index=False)
    one_step_counts().to_csv(os_dir / "one_step_pair_counts.csv", index=False)
    for folder, commit in ((e2_dir, "e2e2e2e"), (cmp_dir, "cmpcmpc"), (os_dir, "ososos0")):
        metadata = {"git": {"commit": commit, "clean": clean}, "config": {"pairs": [list(p) for p in STORED_PAIRS]}}
        (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return e2_dir, cmp_dir, os_dir


def test_run_writes_three_tables_and_metadata_into_a_new_folder(tmp_path):
    """The output folder is created; each table starts with its comment and the results commit of its source."""
    e2_dir, cmp_dir, os_dir = write_results(tmp_path / "results")
    out_dir = tmp_path / "assets" / "tables"
    paths = pa.run_tables(e2_dir, cmp_dir, os_dir, out_dir)
    assert sorted(p.name for p in paths) == ["table_errors.tex", "table_one_step.tex", "table_paired_120s.tex"]
    first_lines = {p.name: p.read_text(encoding="utf-8").splitlines()[0] for p in paths}
    assert "results commit e2e2e2e" in first_lines["table_errors.tex"]
    assert "results commit cmpcmpc" in first_lines["table_paired_120s.tex"]
    assert "results commit ososos0" in first_lines["table_one_step.tex"]
    assert all(line.startswith("% Sources: ") for line in first_lines.values())
    assert (out_dir / "metadata.json").is_file()


def test_run_does_not_change_the_source_results(tmp_path):
    """Every file in the results folders is byte for byte the same after the run."""
    folders = write_results(tmp_path / "results")
    before = {p: p.read_bytes() for p in (tmp_path / "results").rglob("*") if p.is_file()}
    pa.run_tables(*folders, tmp_path / "out")
    after = {p: p.read_bytes() for p in (tmp_path / "results").rglob("*") if p.is_file()}
    assert after == before


def test_run_refuses_results_from_an_unclean_commit(tmp_path):
    """If a source metadata says the results came from uncommitted code, nothing is written."""
    folders = write_results(tmp_path / "results", clean=False)
    with pytest.raises(ValueError, match="clean"):
        pa.run_tables(*folders, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_run_refuses_one_step_pairs_other_than_the_stored_ones(tmp_path):
    """If the E2_one_step metadata lists a different set of pairs, nothing is written."""
    e2_dir, cmp_dir, os_dir = write_results(tmp_path / "results")
    metadata = json.loads((os_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata["config"]["pairs"] = metadata["config"]["pairs"][:5]
    (os_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="ONE_STEP_PAIRS"):
        pa.run_tables(e2_dir, cmp_dir, os_dir, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_default_outputs_are_in_the_paper_assets_folder_of_the_results():
    """By default the tables and the figure are written to results/paper_assets/, inside the repository."""
    assert pa.OUT_DIR == PROJECT_ROOT / "results" / "paper_assets" / "tables"
    assert pa.FIGURE_DIR == PROJECT_ROOT / "results" / "paper_assets" / "figures"


# --- Figure ----------------------------------------------------------------------


def test_window_start_times_are_read_for_one_run_and_horizon():
    """Start times "75;80" of run R at 120 s give [75.0, 80.0]; those of another run or horizon are ignored."""
    windows = pd.DataFrame(
        {"run": ["R", "R", "S"], "horizon_s": [30.0, 120.0, 120.0], "start_time_s": ["1;2", "75;80", "3;4"]}
    )
    assert pa.window_start_times(windows, "R", 120.0) == [75.0, 80.0]


def draw_figure(starts_s=(1750.0, 3550.0, 5350.0)):
    """Figure of a flat synthetic run sampled every 5 s up to 10 000 s, split at 1500 s, at horizons 30, 120 and 600 s."""
    t = np.arange(0.0, 10000.0, 5.0)
    return pa.figure_run_windows(t, 50 + 0 * t, 60 + 0 * t, 1500.0, list(starts_s), [600.0, 30.0, 120.0])


def test_figure_has_the_labels_split_line_and_no_title():
    """Axis labels Time (s) and Level (%), no title, a vertical line at the split and the texts Fit and Evaluation."""
    fig = draw_figure()
    ax = fig.axes[0]
    assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_title()) == ("Time (s)", "Level (%)", "")
    vertical = [line for line in ax.get_lines() if list(line.get_xdata()) == [1500.0, 1500.0]]
    assert len(vertical) == 1
    assert [text.get_text() for text in ax.texts] == ["Fit", "Evaluation"]
    width_cm, height_cm = fig.get_size_inches() * 2.54
    assert (round(width_cm, 6), round(height_cm, 6)) == pa.FIGURE_SIZE_CM
    plt.close(fig)


def test_largest_free_interval_by_hand():
    """Bands of 600 s at 1750 and 3550 in [1500, 4550] leave 250, 1200 and 400 s free: the longest is (2350, 3550)."""
    assert pa.largest_free_interval(1500.0, 4550.0, [3550.0, 1750.0], 600.0) == (2350.0, 3550.0)


def test_largest_free_interval_ignores_overlapping_bands_and_the_part_past_the_end():
    """Bands at 0 and 50 of length 100 overlap; past them (150, 400) is the longest free interval of [0, 400]."""
    assert pa.largest_free_interval(0.0, 400.0, [0.0, 50.0], 100.0) == (150.0, 400.0)


def test_segment_labels_lie_inside_their_segments_and_off_every_band():
    """Drawn, Fit is in italics between the left axis end and the split, Evaluation between 2350 and 3550 s, and neither touches a band."""
    fig = draw_figure()
    ax = fig.axes[0]
    renderer = fig.canvas.get_renderer()
    px = lambda x: ax.transData.transform((x, 0))[0]
    fit, evaluation = (text.get_window_extent(renderer) for text in ax.texts)
    assert px(ax.get_xlim()[0]) <= fit.x0 < fit.x1 <= px(1500.0)
    assert px(2350.0) <= evaluation.x0 < evaluation.x1 <= px(3550.0)
    assert fit.y0 == pytest.approx(evaluation.y0)
    assert all(text.get_style() == "italic" for text in ax.texts)
    for band in ax.patches:
        left, right = px(band.get_x()), px(band.get_x() + band.get_width())
        for box in (fit, evaluation):
            assert box.x1 <= left or box.x0 >= right
    plt.close(fig)


def test_segment_label_that_does_not_fit_its_segment_is_an_error(monkeypatch):
    """With only 5 s of fit part shown, Fit cannot fit there, which is an error rather than a label out of place."""
    monkeypatch.setattr(pa, "MARGIN_BEFORE_SPLIT_S", 5.0)
    with pytest.raises(ValueError, match="'Fit'"):
        draw_figure()
    plt.close("all")


def test_figure_time_axis_goes_from_the_margin_before_the_split_to_past_the_last_window_shown():
    """With two windows shown, the axis runs from 1500 - MARGIN_BEFORE_SPLIT_S to 3550 + 600 + MARGIN_AFTER_LAST_WINDOW_S."""
    fig = draw_figure()
    assert pa.N_WINDOWS_SHOWN == 2
    expected = (1500.0 - pa.MARGIN_BEFORE_SPLIT_S, 3550.0 + 600.0 + pa.MARGIN_AFTER_LAST_WINDOW_S)
    assert fig.axes[0].get_xlim() == pytest.approx(expected)
    plt.close(fig)


def test_figure_shades_three_nested_bands_per_window_shown_longest_below():
    """Each of the two windows shown has bands of 30, 120 and 600 s from its start, darker and higher the shorter they are."""
    fig = draw_figure()
    bands = sorted((p.get_x(), p.get_width(), p.get_zorder(), p.get_facecolor()[0]) for p in fig.axes[0].patches)
    assert [(x, w) for x, w, _, _ in bands] == [
        (1750.0, 30.0), (1750.0, 120.0), (1750.0, 600.0), (3550.0, 30.0), (3550.0, 120.0), (3550.0, 600.0)
    ]
    for window in (bands[:3], bands[3:]):
        zorders = [z for _, _, z, _ in window]
        greys = [g for _, _, _, g in window]
        assert zorders == sorted(zorders, reverse=True)
        assert greys == sorted(greys)
    plt.close(fig)


def test_figure_legend_names_level_setpoint_and_each_horizon():
    """The legend reads Level, Setpoint, H = 30 s, H = 120 s, H = 600 s."""
    fig = draw_figure()
    labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    assert labels == ["Level", "Setpoint", "H = 30 s", "H = 120 s", "H = 600 s"]
    plt.close(fig)


def test_figure_fails_with_fewer_windows_than_shown():
    """One window cannot fill a figure that shows two."""
    with pytest.raises(ValueError, match="window"):
        draw_figure(starts_s=(1750.0,))


def write_figure_inputs(root, start_times=("75;80", "75;80", "75;80")):
    """A 100 s raw run at 1 Hz and E2 results with Ts = 5 s, train_frac 0.7 and windows at evaluation samples 1 and 2."""
    baseline = root / "raw" / "Baseline"
    baseline.mkdir(parents=True)
    n = 100
    pd.DataFrame(
        {
            "Level_pct": np.linspace(40.0, 60.0, n),
            "Control_Signal_pct": np.full(n, 50.0),
            "Setpoint_pct": np.full(n, 55.0),
            "Valve_Closure_pct": np.full(n, 50.0),
        }
    ).to_csv(baseline / f"{pa.FIGURE_RUN}.csv", index=False)
    e2_dir = root / "E2_dictionaries"
    (e2_dir / "tables").mkdir(parents=True)
    horizons = [30.0, 120.0, 600.0]
    pd.DataFrame(
        {"run": [pa.FIGURE_RUN] * 3, "horizon_s": horizons, "starts": ["1;2"] * 3, "start_time_s": list(start_times)}
    ).to_csv(e2_dir / "tables" / "windows.csv", index=False)
    metadata = {
        "git": {"commit": "e2e2e2e", "clean": True},
        "config": {"ts_s": 5.0, "train_frac": 0.7, "horizons_s": horizons},
    }
    (e2_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return e2_dir, root / "raw"


def test_run_figure_writes_a_pdf_and_metadata_with_the_split_of_e2(tmp_path):
    """Samples at 0, 5, ..., 95 s cut at 70 % of 100 s: the split is at 70 s and the windows start at samples 1 and 2."""
    e2_dir, raw_dir = write_figure_inputs(tmp_path)
    out_dir = tmp_path / "assets" / "figures"
    path = pa.run_figure(e2_dir, raw_dir, out_dir)
    assert path == out_dir / "figure_run_windows.pdf"
    assert path.read_bytes().startswith(b"%PDF")
    config = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))["config"]
    assert (config["split_s"], config["window_start_times_s"], config["results_commit"]) == (70.0, [75.0, 80.0], "e2e2e2e")
    assert config["horizons_s"] == [30.0, 120.0, 600.0]


def test_run_figure_embeds_no_type3_fonts(tmp_path):
    """The PDF embeds its fonts as TrueType, plain or as CID fonts (Type 3 fonts are rejected by some publisher systems)."""
    e2_dir, raw_dir = write_figure_inputs(tmp_path)
    pdf = pa.run_figure(e2_dir, raw_dir, tmp_path / "out").read_bytes()
    assert b"/Type3" not in pdf
    assert b"/CIDFontType2" in pdf or b"/TrueType" in pdf


def test_run_figure_fails_when_a_window_time_is_not_that_of_its_start_sample(tmp_path):
    """Windows at samples 1 and 2 (75 and 80 s) listed at 80 and 85 s mean the split rebuilt here differs from E2's."""
    e2_dir, raw_dir = write_figure_inputs(tmp_path, start_times=("80;85",) * 3)
    with pytest.raises(ValueError, match="evaluation part"):
        pa.run_figure(e2_dir, raw_dir, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_run_figure_fails_when_the_windows_start_at_different_times_per_horizon(tmp_path):
    """Nested bands need one start per window, so start times that differ at 600 s are an error."""
    e2_dir, raw_dir = write_figure_inputs(tmp_path, start_times=("75;80", "75;80", "75;85"))
    with pytest.raises(ValueError, match="differ between horizons"):
        pa.run_figure(e2_dir, raw_dir, tmp_path / "out")


# --- Real results ----------------------------------------------------------------


def test_three_values_of_the_real_tables_match_the_result_csvs():
    """D0 at 120 s in T2 and D3a vs D1a in T3 and T4 (stored as D1a, D3a), recomputed here from the CSVs."""
    results = PROJECT_ROOT / "results"
    sources = [
        results / "E2_dictionaries" / "tables" / "fit_indicators.csv",
        results / "E2_dictionaries" / "tables" / "error_summary.csv",
        results / "E2_comparisons" / "paired_tests.csv",
        results / "E2_one_step" / "one_step_pair_counts.csv",
        results / "E2_one_step" / "metadata.json",
    ]
    missing = [p.relative_to(PROJECT_ROOT).as_posix() for p in sources if not p.is_file()]
    if missing:
        pytest.skip(f"results not available: {missing}")
    fit = pd.read_csv(results / "E2_dictionaries" / "tables" / "fit_indicators.csv")
    summary = pd.read_csv(results / "E2_dictionaries" / "tables" / "error_summary.csv")
    tests = pd.read_csv(results / "E2_comparisons" / "paired_tests.csv")
    counts = pd.read_csv(results / "E2_one_step" / "one_step_pair_counts.csv")
    pairs = json.loads((results / "E2_one_step" / "metadata.json").read_text(encoding="utf-8"))["config"]["pairs"]

    def text(x, decimals):
        return f"{x:.{decimals}f}".replace("-", "$-$")

    d0 = summary[(summary["series"] == "D0") & (summary["horizon_s"] == 120.0)].iloc[0]
    expected = f"{text(d0['median'], 2)} [{text(d0['min'], 2)}, {text(d0['max'], 2)}]"
    assert expected in body_lines(pa.table_errors(fit, summary))[0]

    row = tests[(tests["a"] == "D1a") & (tests["b"] == "D3a") & (tests["horizon_s"] == 120.0)].iloc[0]
    expected = (
        rf"\dPolySqrt{{}} vs \dPoly & {row['n_wins_b']}/{row['n_wins_a']} & {text(-row['diff_median'], 2)} "
        f"[{text(-row['diff_median_ci_high'], 2)}, {text(-row['diff_median_ci_low'], 2)}] "
        f"& {row['p_holm_sqrt_h']:.3f} & {row['p_wilcoxon_holm_sqrt_h']:.3f}"
    )
    assert any(line.startswith(expected) for line in body_lines(pa.table_paired(tests)))

    row = counts[(counts["a"] == "D1a") & (counts["b"] == "D3a") & (counts["segment"] == "eval_all")].iloc[0]
    expected = rf" & {row['n_b_lower']}/{row['n_a_lower']} & {text(-row['median_diff'], 3)} \\"
    assert {frozenset(p) for p in pairs} == {frozenset(p) for p in pa.ONE_STEP_PAIRS}
    line = body_lines(pa.table_one_step(counts))[2]
    assert line.startswith(r"\dPolySqrt{} vs \dPoly & ") and line.endswith(expected)
