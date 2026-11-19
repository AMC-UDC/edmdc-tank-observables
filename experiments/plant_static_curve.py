"""Static curve of the plant, measured from the raw data.

At the end of each setpoint plateau the closed loop holds the level still, so
inflow equals outflow and the pair (u_ss, h_ss) is one point of the static
curve of the plant: the control the pump needs to hold that level. The pairs
are taken from the raw 1 Hz data, with no model and no fit/evaluation split.

Plateaus
    A plateau is a stretch of constant Setpoint_pct. Its change instant is the
    time of the first sample with the next setpoint (for the last plateau, one
    second after the last sample). h_ss and u_ss are the means of the valid
    samples in the WINDOW_S seconds before the change instant. A plateau is
    stationary if, in that window, both the drift of the level (slope of a
    least-squares line times WINDOW_S) and its offset from the setpoint are
    smaller than one recording step of the level, STEP = 100/255 %.

Shape of the curve
    For each run, u_ss is fitted against x = h_ss / 100 with three shapes by
    least squares: a straight line u = a + b x, and the line plus a curvature
    term, c x^2 or c sqrt(x). The shapes are compared by the leave-one-out
    RMSE (each plateau predicted by the fit on the others), which compares
    shapes with different numbers of parameters. Main variant: stationary
    plateaus only; sensitivity: every plateau. The comparison across runs is
    descriptive (median, range and counts), with no statistical test.

Time to 95 % of the step
    For each setpoint step, the time from the change instant to the first
    sample, within the new plateau, at which the level reaches 95 % of the
    step. Steps after the fit/evaluation cut used by the model comparison are
    flagged; the cut is computed with the split of the preprocessing module at
    the period and fraction of that comparison, and only the cut is taken from
    it: everything else is computed on the 1 Hz data.

Outputs in results/plant_static_curve/: CSV tables in tables/, a figure in
figures/ and metadata.json.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window: the script only writes files

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from edmdc_tank_observables.data.loader import TS_NOMINAL, load_baseline
from edmdc_tank_observables.data.preprocess import prepare, temporal_split
from edmdc_tank_observables.paths import PROJECT_ROOT
from edmdc_tank_observables.provenance import write_metadata
from edmdc_tank_observables.reporting import save_figure, set_style

# --- Configuration -------------------------------------------------------------

# Recording step of the level: Level_pct is an 8-bit reading, so it moves in
# steps of 100/255 % of level.
STEP = 100.0 / 255.0

# Length of the window at the end of each plateau, in seconds.
WINDOW_S = 300

# Fewest plateaus a run must have to fit a curve with three parameters and
# still leave one out.
MIN_PLATEAUS = 5

SHAPES = ("linear", "quadratic", "sqrt")
VARIANTS = ("stationary_only", "all_plateaus")
MAIN_VARIANT = "stationary_only"

# Pairs of shapes compared run by run on the leave-one-out RMSE.
PAIRS = (("quadratic", "linear"), ("sqrt", "linear"), ("quadratic", "sqrt"))

# Sampling period and fit fraction of the model comparison. They are used only
# to place its fit/evaluation cut and must match the dictionary comparison.
SPLIT_TS_S = 5.0
TRAIN_FRAC = 0.7

# Plateaus beyond the range of the fit part, reported apart.
HIGH_SETPOINTS_PCT = (70.0, 90.0)

SHAPE_NAMES = {"linear": "a + b x", "quadratic": "a + b x + c x²", "sqrt": "a + b x + c √x"}
SHAPE_COLORS = {"linear": "#0072B2", "quadratic": "#009E73", "sqrt": "#D55E00"}

# Rows and columns of panels in the figure, one panel per Baseline run (10).
FIGURE_GRID = (2, 5)

OUT_DIR = PROJECT_ROOT / "results" / "plant_static_curve"

PLATEAU_COLUMNS = (
    "run", "valve_opening_pct", "setpoint_pct", "window_start_s", "window_end_s", "n_samples",
    "h_ss_pct", "h_std_pct", "u_ss_pct", "u_std_pct", "drift_pct", "offset_pct", "stationary",
)
STEP_COLUMNS = (
    "run", "valve_opening_pct", "t_change_s", "sp_before_pct", "sp_after_pct", "t95_s",
    "is_evaluation_step",
)


# --- Plateaus ------------------------------------------------------------------


def plateau_changes(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Setpoint and change instant of each plateau of a run.

    Parameters
    ----------
    df : pandas.DataFrame
        Run data with the columns t_s (s) and Setpoint_pct (%), in time order.

    Returns
    -------
    setpoints : numpy.ndarray
        Setpoint of each plateau, in %.
    starts_s : numpy.ndarray
        Time of the first sample of each plateau, in s.
    changes_s : numpy.ndarray
        Change instant of each plateau, in s: the time of the first sample
        with the next setpoint, or, for the last plateau, the time of the last
        sample plus TS_NOMINAL.
    """
    t = df["t_s"].to_numpy(dtype=float)
    sp = df["Setpoint_pct"].to_numpy(dtype=float)
    first = np.r_[0, np.flatnonzero(np.diff(sp) != 0) + 1]
    starts = t[first]
    changes = np.r_[starts[1:], t[-1] + TS_NOMINAL]
    return sp[first], starts, changes


def plateau_table(df: pd.DataFrame, run: str, opening_pct: float) -> pd.DataFrame:
    """Steady-state level and control at the end of each plateau of a run.

    Parameters
    ----------
    df : pandas.DataFrame
        Valid samples of the run (sensor faults removed), with the columns
        t_s, Setpoint_pct, Level_pct and Control_Signal_pct, in time order.
    run : str
        Name of the run.
    opening_pct : float
        Electrovalve opening of the run, in %.

    Returns
    -------
    pandas.DataFrame
        One row per plateau with the columns PLATEAU_COLUMNS. The window holds
        the samples with window_start_s <= t_s < window_end_s, where
        window_end_s is the change instant and window_start_s = window_end_s -
        WINDOW_S. h_ss_pct and u_ss_pct are window means, h_std_pct and
        u_std_pct population standard deviations (ddof = 0), drift_pct the
        slope of the least-squares line of level against time times WINDOW_S,
        offset_pct = h_ss_pct - setpoint_pct, and stationary is True when both
        |drift_pct| and |offset_pct| are below STEP.

    Raises
    ------
    ValueError
        If a window has fewer than two samples, so no drift can be fitted.
    """
    t = df["t_s"].to_numpy(dtype=float)
    h = df["Level_pct"].to_numpy(dtype=float)
    u = df["Control_Signal_pct"].to_numpy(dtype=float)
    rows = []
    for setpoint, _, change in zip(*plateau_changes(df)):
        start = change - WINDOW_S
        in_window = (t >= start) & (t < change)
        n = int(in_window.sum())
        if n < 2:
            raise ValueError(f"{run}: the window [{start:g}, {change:g}) s of the {setpoint:g} % plateau has {n} sample(s).")
        h_w = h[in_window]
        drift = np.polyfit(t[in_window], h_w, 1)[0] * WINDOW_S
        offset = h_w.mean() - setpoint
        rows.append(
            {
                "run": run,
                "valve_opening_pct": opening_pct,
                "setpoint_pct": setpoint,
                "window_start_s": start,
                "window_end_s": change,
                "n_samples": n,
                "h_ss_pct": h_w.mean(),
                "h_std_pct": h_w.std(),
                "u_ss_pct": u[in_window].mean(),
                "u_std_pct": u[in_window].std(),
                "drift_pct": drift,
                "offset_pct": offset,
                "stationary": bool(abs(drift) < STEP and abs(offset) < STEP),
            }
        )
    return pd.DataFrame(rows, columns=list(PLATEAU_COLUMNS))


# --- Shape of the curve --------------------------------------------------------


def design_matrix(x: np.ndarray, shape: str) -> np.ndarray:
    """Regressors of a shape of the static curve.

    Parameters
    ----------
    x : numpy.ndarray
        Level as a fraction, h / 100.
    shape : str
        "linear" (columns 1, x), "quadratic" (1, x, x^2) or "sqrt" (1, x, sqrt(x)).

    Returns
    -------
    numpy.ndarray, shape (len(x), 2 or 3)
        One column per coefficient, in the order a, b, c.

    Raises
    ------
    ValueError
        If the shape is not one of SHAPES.
    """
    x = np.asarray(x, dtype=float)
    columns = [np.ones_like(x), x]
    if shape == "quadratic":
        columns.append(x**2)
    elif shape == "sqrt":
        columns.append(np.sqrt(x))
    elif shape != "linear":
        raise ValueError(f"Unknown shape {shape!r}; expected one of {SHAPES}.")
    return np.column_stack(columns)


def fit_shape(x: np.ndarray, u: np.ndarray, shape: str) -> tuple[np.ndarray, float, float]:
    """Fit one shape of the static curve by least squares and score it in sample and by leave-one-out.

    Parameters
    ----------
    x : numpy.ndarray
        Level of each plateau as a fraction, h_ss / 100.
    u : numpy.ndarray
        Steady-state control of each plateau, in % of actuation.
    shape : str
        One of SHAPES; see design_matrix.

    Returns
    -------
    coefs : numpy.ndarray
        Coefficients a, b (and c), fitted on every point.
    rmse_in_sample : float
        RMSE of that fit on the same points, in % of actuation.
    rmse_loo : float
        RMSE of the leave-one-out predictions, in % of actuation: each point
        is predicted by the fit on all the others.

    Raises
    ------
    ValueError
        If the shape is not one of SHAPES.
    """
    u = np.asarray(u, dtype=float)
    X = design_matrix(x, shape)
    coefs = np.linalg.lstsq(X, u, rcond=None)[0]
    rmse_in = float(np.sqrt(np.mean((u - X @ coefs) ** 2)))
    misses = []
    for i in range(len(u)):
        keep = np.arange(len(u)) != i
        coefs_i = np.linalg.lstsq(X[keep], u[keep], rcond=None)[0]
        misses.append(u[i] - X[i] @ coefs_i)
    rmse_loo = float(np.sqrt(np.mean(np.square(misses))))
    return coefs, rmse_in, rmse_loo


def curve_fits(plateaus: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Fit every shape to the plateaus of each run.

    Parameters
    ----------
    plateaus : pandas.DataFrame
        Plateaus table with at least the columns run, valve_opening_pct,
        h_ss_pct, u_ss_pct and stationary.
    variant : str
        "stationary_only" (only the stationary plateaus) or "all_plateaus".

    Returns
    -------
    pandas.DataFrame
        One row per (run, shape) with the columns variant, run,
        valve_opening_pct, shape, n_plateaus, a, b, c (NaN for the line),
        rmse_in_sample and rmse_loo (both in % of actuation).

    Raises
    ------
    ValueError
        If the variant is unknown, if a run has fewer than MIN_PLATEAUS
        plateaus in the variant, or if some h_ss_pct is zero or negative.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}.")
    used = plateaus[plateaus["stationary"]] if variant == "stationary_only" else plateaus
    rows = []
    for run in plateaus["run"].unique():
        block = used[used["run"] == run]
        if len(block) < MIN_PLATEAUS:
            raise ValueError(f"{run} ({variant}): {len(block)} plateaus, fewer than {MIN_PLATEAUS}.")
        if (block["h_ss_pct"] <= 0).any():
            raise ValueError(f"{run} ({variant}): some h_ss_pct is zero or negative, so sqrt(x) is undefined.")
        x = block["h_ss_pct"].to_numpy() / 100.0
        for shape in SHAPES:
            coefs, rmse_in, rmse_loo = fit_shape(x, block["u_ss_pct"].to_numpy(), shape)
            rows.append(
                {
                    "variant": variant,
                    "run": run,
                    "valve_opening_pct": block["valve_opening_pct"].iloc[0],
                    "shape": shape,
                    "n_plateaus": len(block),
                    "a": coefs[0],
                    "b": coefs[1],
                    "c": coefs[2] if len(coefs) > 2 else np.nan,
                    "rmse_in_sample": rmse_in,
                    "rmse_loo": rmse_loo,
                }
            )
    return pd.DataFrame(rows)


def fit_summary(fits: pd.DataFrame) -> pd.DataFrame:
    """Median, minimum and maximum across runs of the leave-one-out RMSE, per variant and shape.

    Parameters
    ----------
    fits : pandas.DataFrame
        Table returned by curve_fits (one or several variants).

    Returns
    -------
    pandas.DataFrame
        One row per (variant, shape) with the columns rmse_loo_median,
        rmse_loo_min, rmse_loo_max and n_runs.
    """
    return (
        fits.groupby(["variant", "shape"], sort=False)["rmse_loo"]
        .agg(rmse_loo_median="median", rmse_loo_min="min", rmse_loo_max="max", n_runs="count")
        .reset_index()
    )


def pair_counts(fits: pd.DataFrame, pairs=PAIRS) -> pd.DataFrame:
    """Count in how many runs each shape of a pair has the lower leave-one-out RMSE.

    Parameters
    ----------
    fits : pandas.DataFrame
        Table returned by curve_fits (one or several variants).
    pairs : iterable of tuple of str, default PAIRS
        Pairs (a, b) of shapes to compare.

    Returns
    -------
    pandas.DataFrame
        One row per (variant, pair) with the columns variant, a, b, n_a_lower,
        n_b_lower, n_ties (exactly equal RMSE) and n_runs.
    """
    rows = []
    for variant, block in fits.groupby("variant", sort=False):
        by_run = block.pivot(index="run", columns="shape", values="rmse_loo")
        for a, b in pairs:
            rows.append(
                {
                    "variant": variant,
                    "a": a,
                    "b": b,
                    "n_a_lower": int((by_run[a] < by_run[b]).sum()),
                    "n_b_lower": int((by_run[b] < by_run[a]).sum()),
                    "n_ties": int((by_run[a] == by_run[b]).sum()),
                    "n_runs": len(by_run),
                }
            )
    return pd.DataFrame(rows)


# --- Time to 95 % of the step --------------------------------------------------


def step_table(df: pd.DataFrame, run: str, opening_pct: float, cut_time_s: float) -> pd.DataFrame:
    """Time for the level to reach 95 % of each setpoint step of a run.

    Parameters
    ----------
    df : pandas.DataFrame
        Valid samples of the run, with the columns t_s, Setpoint_pct and
        Level_pct, in time order.
    run : str
        Name of the run.
    opening_pct : float
        Electrovalve opening of the run, in %.
    cut_time_s : float
        Fit/evaluation cut of the model comparison, in s.

    Returns
    -------
    pandas.DataFrame
        One row per step with the columns STEP_COLUMNS. t_change_s is the
        change instant; t95_s is the time from it to the first sample of the
        new plateau with level >= sp_before + 0.95 (sp_after - sp_before), or
        NaN if no sample of the new plateau reaches it; is_evaluation_step is
        True when t_change_s > cut_time_s.

    Raises
    ------
    ValueError
        If some step is not ascending (sp_after <= sp_before).
    """
    t = df["t_s"].to_numpy(dtype=float)
    h = df["Level_pct"].to_numpy(dtype=float)
    setpoints, _, changes = plateau_changes(df)
    rows = []
    for k in range(1, len(setpoints)):
        sp_before, sp_after = setpoints[k - 1], setpoints[k]
        if sp_after <= sp_before:
            raise ValueError(f"{run}: the step at t = {changes[k - 1]:g} s ({sp_before:g} -> {sp_after:g} %) is not ascending.")
        t_change = changes[k - 1]
        in_plateau = (t >= t_change) & (t < changes[k])
        target = sp_before + 0.95 * (sp_after - sp_before)
        reached = t[in_plateau][h[in_plateau] >= target]
        rows.append(
            {
                "run": run,
                "valve_opening_pct": opening_pct,
                "t_change_s": t_change,
                "sp_before_pct": sp_before,
                "sp_after_pct": sp_after,
                "t95_s": reached[0] - t_change if len(reached) > 0 else np.nan,
                "is_evaluation_step": bool(t_change > cut_time_s),
            }
        )
    return pd.DataFrame(rows, columns=list(STEP_COLUMNS))


def t95_summary(steps: pd.DataFrame) -> dict:
    """Median, minimum, maximum and counts of t95_s, for the evaluation steps and for all.

    Parameters
    ----------
    steps : pandas.DataFrame
        Table returned by step_table, for one or several runs.

    Returns
    -------
    dict
        {"evaluation_steps": {...}, "all_steps": {...}}, each with median_s,
        min_s and max_s (over the steps that reach 95 %), n (steps) and n_nan
        (steps that do not reach it).
    """

    def describe(t95):
        return {
            "median_s": float(t95.median()),
            "min_s": float(t95.min()),
            "max_s": float(t95.max()),
            "n": int(len(t95)),
            "n_nan": int(t95.isna().sum()),
        }

    return {
        "evaluation_steps": describe(steps.loc[steps["is_evaluation_step"], "t95_s"]),
        "all_steps": describe(steps["t95_s"]),
    }


def evaluation_cut_time(raw) -> float:
    """Fit/evaluation cut of the model comparison for one run, in s.

    The run is resampled and split exactly as in the model comparison
    (SPLIT_TS_S, TRAIN_FRAC); the cut is the time of the first sample of the
    evaluation part. Only that instant is used.

    Parameters
    ----------
    raw : RawDataset
        Run returned by the loader.

    Returns
    -------
    float
        Time of the first sample of the evaluation part, in s.

    Raises
    ------
    ValueError
        If the run cannot be resampled or split; see prepare and temporal_split.
    """
    _, test = temporal_split(prepare(raw, SPLIT_TS_S), TRAIN_FRAC)
    return float(test.t[0])


# --- Figure --------------------------------------------------------------------


def plot_curves(plateaus: pd.DataFrame, fits: pd.DataFrame) -> plt.Figure:
    """u_ss against h_ss, one panel per run, with the shapes fitted on the stationary plateaus.

    Stationary plateaus are filled markers, the others hollow. Each fitted
    shape is drawn over the level range of the stationary plateaus. The panels
    form a grid of FIGURE_GRID rows and columns, one per Baseline run.

    Raises
    ------
    ValueError
        If the number of runs in plateaus is not the number of panels of
        FIGURE_GRID.
    """
    runs = list(dict.fromkeys(plateaus["run"]))
    n_rows, n_cols = FIGURE_GRID
    if len(runs) != n_rows * n_cols:
        raise ValueError(
            f"The figure has a {n_rows} x {n_cols} grid, one panel per run, but there are {len(runs)} runs."
        )
    set_style()
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12.0, 5.2), sharex=True, sharey=True)
    for ax, run in zip(axes.flat, runs):
        block = plateaus[plateaus["run"] == run]
        stationary = block[block["stationary"]]
        other = block[~block["stationary"]]
        ax.plot(stationary["h_ss_pct"], stationary["u_ss_pct"], "o", color="black", markersize=4, label="Stationary plateau")
        ax.plot(
            other["h_ss_pct"], other["u_ss_pct"], "o", markerfacecolor="none", markeredgecolor="black",
            markersize=4, label="Non-stationary plateau",
        )
        h_grid = np.linspace(stationary["h_ss_pct"].min(), stationary["h_ss_pct"].max(), 200)
        run_fits = fits[fits["run"] == run].set_index("shape")
        for shape in SHAPES:
            coefs = run_fits.loc[shape, ["a", "b", "c"]].dropna().to_numpy(dtype=float)
            ax.plot(h_grid, design_matrix(h_grid / 100.0, shape) @ coefs, color=SHAPE_COLORS[shape], linewidth=1.2, label=SHAPE_NAMES[shape])
        ax.set_title(f"Opening {block['valve_opening_pct'].iloc[0]:g} %")
    for ax in axes[1]:
        ax.set_xlabel("h_ss (% of level)")
    for ax in axes[:, 0]:
        ax.set_ylabel("u_ss (% of actuation)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    return fig


# --- Run -----------------------------------------------------------------------


def print_results(plateaus, fits, pairs, t95) -> None:
    """Print the stationary plateaus per run, the leave-one-out RMSE per run and shape, the counts and t95."""
    high = plateaus["setpoint_pct"].between(*HIGH_SETPOINTS_PCT)
    counts = plateaus.groupby("run", sort=False).agg(
        stationary=("stationary", "sum"), plateaus=("stationary", "size")
    )
    counts["stationary_70_90"] = plateaus[high].groupby("run", sort=False)["stationary"].sum()
    print("\nStationary plateaus per run")
    print(counts.to_string())
    for variant in VARIANTS:
        print(f"\nLeave-one-out RMSE (% of actuation), {variant}")
        print(fits[fits["variant"] == variant].pivot(index="run", columns="shape", values="rmse_loo").loc[counts.index].round(3).to_string())
    print("\nPair counts on the leave-one-out RMSE")
    print(pairs.to_string(index=False))
    print("\nTime to 95 % of the step (s)")
    for name, values in t95.items():
        print(f"  {name}: {values}")


def run(out_dir=OUT_DIR) -> dict:
    """Compute the plateaus, curve fits and step times of every Baseline run and write them to out_dir.

    Parameters
    ----------
    out_dir : pathlib.Path or str, default OUT_DIR
        Output folder.

    Returns
    -------
    dict
        The summary written to metadata.json.

    Raises
    ------
    ValueError
        If the runs do not share the same setpoint profile or the same
        fit/evaluation cut, or if a check of plateau_table, curve_fits or
        step_table fails.
    """
    raws = load_baseline()

    profiles = {tuple(plateau_changes(r.df)[0]) for r in raws}
    if len(profiles) != 1:
        raise ValueError(f"The runs do not share the same setpoint profile: {profiles}.")
    cuts = {r.name: evaluation_cut_time(r) for r in raws}
    if len(set(cuts.values())) != 1:
        raise ValueError(f"The runs are not cut at the same instant: {cuts}.")
    cut_time_s = next(iter(cuts.values()))

    plateaus = pd.concat([plateau_table(r.df, r.name, r.valve_opening_pct) for r in raws], ignore_index=True)
    steps = pd.concat([step_table(r.df, r.name, r.valve_opening_pct, cut_time_s) for r in raws], ignore_index=True)
    fits = pd.concat([curve_fits(plateaus, variant) for variant in VARIANTS], ignore_index=True)
    summary_table = fit_summary(fits)
    pairs = pair_counts(fits)
    t95 = t95_summary(steps)

    print(f"{len(raws)} runs, {len(profiles.pop())} plateaus each; fit/evaluation cut at t = {cut_time_s:g} s")
    print_results(plateaus, fits, pairs, t95)

    out_dir = Path(out_dir)
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    plateaus.to_csv(tables / "plateaus.csv", index=False)
    steps.to_csv(tables / "steps.csv", index=False)
    fits.to_csv(tables / "curve_fits.csv", index=False)
    summary_table.to_csv(tables / "curve_summary.csv", index=False)
    pairs.to_csv(tables / "curve_pair_counts.csv", index=False)
    save_figure(plot_curves(plateaus, fits[fits["variant"] == MAIN_VARIANT]), out_dir / "figures", "static_curve_by_run")

    high = plateaus["setpoint_pct"].between(*HIGH_SETPOINTS_PCT)
    summary = {
        "n_stationary_plateaus_by_run": {k: int(v) for k, v in plateaus.groupby("run", sort=False)["stationary"].sum().items()},
        "n_stationary_plateaus_70_90_by_run": {
            k: int(v) for k, v in plateaus[high].groupby("run", sort=False)["stationary"].sum().items()
        },
        "n_stationary_plateaus": int(plateaus["stationary"].sum()),
        "n_plateaus": len(plateaus),
        "pair_counts": pairs.to_dict("records"),
        "t95": t95,
    }
    write_metadata(
        out_dir,
        config={
            "subset": "Baseline",
            "sampling": "1 Hz raw data, sensor faults removed",
            "level_recording_step_pct": STEP,
            "window_s": WINDOW_S,
            "stationarity_criterion": "|drift_pct| < level_recording_step_pct and |offset_pct| < level_recording_step_pct, "
            "with drift_pct = least-squares slope of level against time over the window x window_s",
            "min_plateaus": MIN_PLATEAUS,
            "shapes": SHAPE_NAMES,
            "variants": VARIANTS,
            "main_variant": MAIN_VARIANT,
            "t95_fraction": 0.95,
            "split_ts_s": SPLIT_TS_S,
            "train_frac": TRAIN_FRAC,
            "cut_time_s": cut_time_s,
        },
        summary=summary,
    )
    return summary


def main() -> None:
    """Compute the static curve of the plant from the Baseline runs and write it to OUT_DIR."""
    run()


if __name__ == "__main__":
    main()
