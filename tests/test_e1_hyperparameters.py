"""Tests of the E1 script (hyperparameter selection): grids, selection rule, inner validation and a smoke run."""

import importlib.util
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from edmdc_tank_observables.data.preprocess import PreprocessedDataset
from edmdc_tank_observables.observables.dictionaries import make_dictionary
from edmdc_tank_observables.paths import PROJECT_ROOT


def load_script():
    """Import experiments/E1_hyperparameters.py as a module, without running it."""
    path = PROJECT_ROOT / "experiments" / "E1_hyperparameters.py"
    spec = importlib.util.spec_from_file_location("e1_hyperparameters", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e1 = load_script()

# Full synthetic run of 2000 samples: the fit part is samples 0-1399, the inner
# fit part 0-1049 and the inner validation part 1050-1399.
N_SAMPLES = 2000
# Setpoint steps: two in the inner fit part, two in the inner validation part
# (its samples 10 and 140) and one in the evaluation part.
STEPS = (300, 600, 1060, 1190, 1600)
VALIDATION_STARTS = [9, 139]


def make_full_run(seed, name):
    """A synthetic full run at Ts = 5 s: a tank-like level driven by a varying control, with a setpoint staircase at STEPS."""
    rng = np.random.default_rng(seed)
    n = N_SAMPLES
    u = 0.5 + 0.3 * np.sin(0.05 * np.arange(n) + seed) + 0.02 * rng.standard_normal(n)
    h = [0.4]
    for k in range(n - 1):
        h.append(h[-1] + 0.05 * (u[k] - 0.8 * np.sqrt(h[-1])))
    sp = np.full(n, 0.5)
    for p in STEPS:
        sp[p:] += 0.05
    return PreprocessedDataset(
        h=np.array(h), u=u, sp=sp, t=5.0 * np.arange(n), ts=5.0,
        h_scale=(0.0, 100.0), u_scale=(0.0, 100.0), name=name,
    )


def make_runs():
    """Two synthetic runs as (name, opening, inner fit part, inner validation part)."""
    runs = []
    for name, opening, seed in (("Baseline_30pct", 30.0, 1), ("Baseline_50pct", 50.0, 2)):
        fit, validation = e1.split_run(make_full_run(seed, name))
        runs.append((name, opening, fit, validation))
    return runs


def family_table(values, medians, maxima):
    """Summary rows of one family at the selection horizon, with p growing with the value."""
    return pd.DataFrame(
        {
            "value": values,
            "p": [v + 1 for v in values],
            "median": medians,
            "min": [m - 0.1 for m in medians],
            "max": maxima,
        }
    )


# --- Grids ---------------------------------------------------------------------


def test_grids_cover_every_integer_of_each_range_for_the_six_families():
    """The grids are order 2-6, n_centers 1-9 and n_delays 1-8, every integer, for D1a/b/c and D3a/b/c."""
    expected = {
        "D1a": ("order", [2, 3, 4, 5, 6]),
        "D3a": ("order", [2, 3, 4, 5, 6]),
        "D1b": ("n_centers", list(range(1, 10))),
        "D3b": ("n_centers", list(range(1, 10))),
        "D1c": ("n_delays", list(range(1, 9))),
        "D3c": ("n_delays", list(range(1, 9))),
    }
    assert {f: (name, list(values)) for f, (name, values) in e1.GRIDS.items()} == expected


def test_grid_candidates_lists_one_buildable_candidate_per_value():
    """Every (family, hyperparameter, value) of the grids is listed once, 44 in all, and each builds a dictionary."""
    candidates = e1.grid_candidates(e1.GRIDS)
    assert len(candidates) == 2 * (5 + 9 + 8)
    assert len(set(candidates)) == len(candidates)
    assert ("D1b", "n_centers", 1) in candidates and ("D3c", "n_delays", 8) in candidates
    for family, name, value in candidates:
        make_dictionary(family, **{name: value})


def test_rbf_width_is_the_spacing_between_centres():
    """The RBF width reported is 1/(M-1) for M centres in [0, 1] and 1 for a single centre, as the dictionary uses it."""
    assert e1.rbf_width(1) == 1.0
    assert e1.rbf_width(2) == 1.0
    assert e1.rbf_width(5) == pytest.approx(0.25)
    # At one width from the centre 0.5 the Gaussian is exp(-1/2).
    psi = make_dictionary("D1b", n_centers=3).transform(np.array([0.5 + e1.rbf_width(3)]))
    assert psi[0, 2] == pytest.approx(np.exp(-0.5))


# --- Selection rule ------------------------------------------------------------


def test_selection_keeps_the_lowest_median_even_by_a_small_margin():
    """The selected value is the candidate with the lowest median, even if the next one is only slightly worse."""
    table = family_table([2, 3, 4, 5, 6], medians=[5.0, 3.0, 2.99, 3.5, 4.0], maxima=[5.5, 3.4, 3.2, 3.9, 4.4])
    result = e1.select(table)
    assert result["selected_value"] == 4
    assert result["selected_median"] == pytest.approx(2.99)
    assert result["selected_at_upper_edge"] is False


def test_selection_breaks_an_exact_tie_of_medians_with_the_smaller_value():
    """If two candidates have exactly the same median, the one with fewer observables is selected."""
    table = family_table([1, 2, 3], medians=[2.0, 1.0, 1.0], maxima=[2.1, 1.05, 1.05])
    assert e1.select(table)["selected_value"] == 2


def test_selection_flags_a_choice_at_the_upper_edge_of_the_grid():
    """When the error keeps falling up to the last value, that value is selected and flagged as upper edge."""
    table = family_table([1, 2, 3, 4], medians=[4.0, 3.0, 2.0, 1.0], maxima=[4.1, 3.1, 2.1, 1.1])
    result = e1.select(table)
    assert result["selected_value"] == 4
    assert result["selected_at_upper_edge"] is True


def test_selection_does_not_flag_the_lower_edge():
    """When the first value of the grid is the best, it is selected and not flagged: it is the limit of the family."""
    table = family_table([2, 3, 4, 5, 6], medians=[1.0, 2.0, 3.0, 4.0, 5.0], maxima=[1.5, 2.5, 3.5, 4.5, 5.5])
    result = e1.select(table)
    assert result["selected_value"] == 2
    assert result["selected_at_upper_edge"] is False


def test_parsimony_sensitivity_reports_the_smallest_candidate_within_the_threshold_without_changing_the_selection():
    """Parsimony takes the fewest observables among medians within the selected one's maximum across runs; the selection stays the minimum."""
    table = family_table([2, 3, 4, 5, 6], medians=[5.0, 3.0, 2.9, 3.5, 4.0], maxima=[5.5, 3.4, 3.2, 3.9, 4.4])
    result = e1.select(table)
    assert result["parsimony_threshold"] == pytest.approx(3.2)
    assert result["parsimony_tied_values"] == [3, 4]
    assert result["parsimony_value"] == 3
    assert result["selected_value"] == 4


def test_upper_edge_warning_is_given_only_for_the_selected_value():
    """A family whose selected value is the last of its grid gets one warning; a family selected inside its grid gets none."""
    grids = {"D1a": ("order", (2, 3, 4)), "D1b": ("n_centers", (1, 2, 3))}
    selection = pd.DataFrame(
        {"family": ["D1a", "D1b"], "selected_value": [4, 2], "selected_at_upper_edge": [True, False]}
    )
    warnings = e1.edge_warnings(selection, grids)
    assert len(warnings) == 1 and warnings[0].startswith("D1a")


# --- Sensitivities S3 and S4 ---------------------------------------------------


def errors_table(rmse_by_candidate):
    """Errors rows at every horizon for three runs, from {(series, value): [rmse run 1, run 2, run 3]}."""
    rows = []
    for (series, value), rmses in rmse_by_candidate.items():
        for run, rmse in zip(("r1", "r2", "r3"), rmses):
            for horizon_s in e1.HORIZONS_S:
                rows.append({"run": run, "series": series, "value": value, "horizon_s": horizon_s, "rmse": rmse})
    return pd.DataFrame(rows)


def summary_from(errors):
    """Median, minimum and maximum across runs per candidate and horizon, with p = value + 1."""
    summary = (
        errors.groupby(["series", "value", "horizon_s"])["rmse"]
        .agg(median="median", min="min", max="max")
        .reset_index()
    )
    summary["p"] = summary["value"] + 1
    return summary


def test_s4_counts_the_runs_in_which_a_candidate_is_the_best_of_its_family():
    """Each candidate is counted in the runs where it has the lowest error of its family; a tie counts for both."""
    errors = errors_table({("D1a", 2): [1.0, 1.0, 2.0], ("D1a", 3): [2.0, 1.0, 1.0]})
    assert e1.runs_where_best(errors, "D1a", 2, 120.0) == (2, 3)
    assert e1.runs_where_best(errors, "D1a", 3, 120.0) == (2, 3)
    errors = errors_table({("D1a", 2): [1.0, 1.0, 2.0], ("D1a", 3): [2.0, 1.5, 1.0]})
    assert e1.runs_where_best(errors, "D1a", 2, 120.0) == (2, 3)
    assert e1.runs_where_best(errors, "D1a", 3, 120.0) == (1, 3)


def test_s3_combined_family_inherits_the_generic_value_and_reports_whether_its_own_differs():
    """D3a uses the value selected for D1a in E2; its own selection is kept apart and flagged when it differs."""
    errors = errors_table(
        {
            ("D1a", 2): [3.0, 3.0, 3.0], ("D1a", 3): [1.0, 1.0, 1.0],
            ("D3a", 2): [1.0, 1.0, 1.0], ("D3a", 3): [2.0, 2.0, 2.0],
        }
    )
    grids = {"D1a": ("order", (2, 3)), "D3a": ("order", (2, 3))}
    selection = e1.selection_table(summary_from(errors), errors, grids).set_index("family")
    assert selection.loc["D1a", "selected_value"] == 3 and selection.loc["D1a", "e2_value"] == 3
    assert selection.loc["D3a", "selected_value"] == 2
    assert selection.loc["D3a", "inherited_value"] == 3 and selection.loc["D3a", "e2_value"] == 3
    assert selection.loc["D3a", "inherited_median"] == pytest.approx(2.0)
    assert selection.loc["D3a", "own_differs"] == True  # noqa: E712 (numpy bool)
    assert selection.loc["D3a", "n_runs_inherited_best"] == 0


# --- Inner validation ----------------------------------------------------------


def test_split_run_takes_the_inner_parts_from_the_fit_part_only():
    """The inner fit part is the first 75 % of the first 70 % of the run and the validation part the rest of that 70 %."""
    fit, validation = e1.split_run(make_full_run(1, "Baseline_30pct"))
    assert len(fit) == 1050 and len(validation) == 350
    assert fit.t[0] == 0.0 and validation.t[0] == 5.0 * 1050 and validation.t[-1] == 5.0 * 1399


def test_inner_validation_has_one_window_per_step_and_the_same_windows_for_every_series():
    """With two setpoint steps in the inner validation part, every series is scored on the same two windows at each horizon."""
    runs = make_runs()
    grids = {"D1c": ("n_delays", (1, 8)), "D3b": ("n_centers", (1, 3))}
    models, errors = e1.score_candidates(runs, grids)
    windows = e1.window_table(runs, e1.HORIZONS_S, max_delays=8)

    assert set(errors["series"]) == {"D0", "D1c", "D3b"}
    assert (errors["n_windows"] == 2).all()
    assert len(windows) == 2 * len(e1.HORIZONS_S)
    assert (windows["starts"] == ";".join(str(s) for s in VALIDATION_STARTS)).all()
    assert (windows["start_time_s"] == "5295;5945").all()
    e1.check_same_windows(errors, windows)

    broken = errors.copy()
    broken.loc[broken.index[0], "n_windows"] = 1
    with pytest.raises(ValueError):
        e1.check_same_windows(broken, windows)


def test_split_run_cuts_at_the_same_instants_when_a_run_starts_one_sample_later():
    """A run that lacks its first sample is cut at the same instants: its inner fit part ends at 5245 s and its validation part is identical."""
    full = make_full_run(1, "Baseline_30pct")
    later = replace(full, h=full.h[1:], u=full.u[1:], sp=full.sp[1:], t=full.t[1:])
    fit, validation = e1.split_run(full)
    fit_later, validation_later = e1.split_run(later)
    assert fit.t[-1] == fit_later.t[-1] == 5245.0
    np.testing.assert_array_equal(validation.t, validation_later.t)
    np.testing.assert_array_equal(validation.h, validation_later.h)


# --- Smoke run -----------------------------------------------------------------


@pytest.fixture(scope="module")
def e1_output(tmp_path_factory):
    """Run the E1 script with its full grids on two synthetic runs, writing into a temporary folder."""
    out_dir = tmp_path_factory.mktemp("E1_hyperparameters")
    result = e1.run_experiment(make_runs(), out_dir)
    return out_dir, result


def test_e1_writes_the_tables_the_figures_and_the_metadata(e1_output):
    """The script leaves its CSV tables, one figure per family in PDF and PNG and a metadata.json."""
    out_dir, _ = e1_output
    for name in ("models", "errors", "error_summary", "selection", "windows", "fit_indicators"):
        assert (out_dir / "tables" / f"{name}.csv").is_file()
    for family in e1.GRIDS:
        for extension in ("pdf", "png"):
            assert (out_dir / "figures" / f"error_vs_{family}.{extension}").is_file()
    assert (out_dir / "metadata.json").is_file()


def test_e1_scores_every_candidate_and_the_references_on_every_run_and_horizon(e1_output):
    """The errors table has one row per run, candidate (44 plus D0) and horizon, never the evaluation part."""
    out_dir, _ = e1_output
    errors = pd.read_csv(out_dir / "tables" / "errors.csv")
    assert len(errors) == 2 * (44 + 1) * 3
    assert (errors["n_windows"] == 2).all()
    summary = pd.read_csv(out_dir / "tables" / "error_summary.csv")
    assert len(summary) == (44 + 1) * 3
    assert (summary["n_runs"] == 2).all()


def test_e1_selection_table_has_one_row_per_family_with_the_sensitivities(e1_output):
    """The selection table has one row per family with the selected value, the parsimony sensitivity, the winners at the three horizons and the run counts."""
    out_dir, _ = e1_output
    selection = pd.read_csv(out_dir / "tables" / "selection.csv")
    assert list(selection["family"]) == list(e1.GRIDS)
    for column in (
        "selected_value", "parsimony_value", "parsimony_tied_values", "winner_30s", "winner_120s",
        "winner_600s", "winner_differs", "n_runs_selected_best", "inherited_value", "e2_value",
    ):
        assert column in selection.columns
    d1a = selection.set_index("family").loc["D1a"]
    d3a = selection.set_index("family").loc["D3a"]
    assert d3a["inherited_value"] == d1a["selected_value"] == d3a["e2_value"] == d1a["e2_value"]


def test_e1_metadata_records_the_configuration_and_the_selected_hyperparameters(e1_output):
    """The metadata keeps the split fractions, grids, horizons, rule and references, and the values E2 will use."""
    out_dir, result = e1_output
    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    config, summary = metadata["config"], metadata["summary"]
    assert config["train_frac"] == 0.7 and config["inner_fit_frac"] == 0.75
    assert config["grids"]["D1b"] == {"hyperparameter": "n_centers", "values": list(range(1, 10))}
    assert config["selection_horizon_s"] == 120.0 and config["horizons_s"] == [30.0, 120.0, 600.0]
    assert "stability_horizons_s" not in config
    assert config["references"] == ["D0"]
    assert "lowest median" in config["selection_rule"]
    selected = summary["selected_hyperparameters"]
    assert set(selected) == set(e1.GRIDS)
    assert selected["D3c"] == selected["D1c"]
    assert set(summary["parsimony_sensitivity"]) == set(e1.GRIDS)
    assert summary["upper_edge_warnings"] == result["warnings"]
