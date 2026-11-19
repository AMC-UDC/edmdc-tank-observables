"""Summary of the raw Baseline data: runs, sensor faults, quantization and bounds on the sampling period.

Reads only the raw 1 Hz data, with the loader, the preprocessing functions and
helpers of plant_static_curve and E2_dictionaries; no model is fitted and no
figure is made.

Runs
    For each run: samples, duration, date of recording, sensor faults (count
    and rows) and the samples and interior gaps left after decimating to the
    sampling period of the model comparison, TS_MODEL_S. Sensor faults are
    detected by the loader from their signature, level 0 with a positive
    setpoint.

Quantization
    The level is an 8-bit reading, so it moves in steps of 100/255 % of level.
    For each run, whether every level value is a whole multiple of that step.

Bounds on the sampling period
    The steps are the setpoint changes of each run, with the time to 95 % of
    the step (t95) computed as in the static curve of the plant. The transient
    of a step is the interval [change instant, change instant + t95]. For each
    period in TS_VALUES the valid samples are decimated as in the
    preprocessing (samples whose time is a multiple of the period, no
    averaging), and the pairs (k, k+1) exactly one period apart with both
    times inside a transient are taken. For each pair |dh| is measured in
    recording steps; "below one step" means |dh| < 1 step, which with
    quantized data means dh = 0. Lower bound: the fraction of pairs below one
    step. Upper bound: the samples per rise, t95 / Ts. Steps with t95 = 0 or
    undefined have no transient: they are counted as excluded and left out of
    the samples per rise.

Outputs in results/data_summary/: runs.csv, quantization.csv,
sampling_bounds.csv and metadata.json.
"""

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from edmdc_tank_observables.data.loader import TS_NOMINAL, RawDataset, load_baseline
from edmdc_tank_observables.data.preprocess import find_gaps, resample
from edmdc_tank_observables.paths import PROJECT_ROOT
from edmdc_tank_observables.provenance import write_metadata


def _load_script(name: str):
    """Import a script of this folder as a module, without running it, to reuse its functions and settings."""
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Steps, t95 and the recording step come from the static curve of the plant,
# the acquisition date and the sampling period of the models from E2.
psc = _load_script("plant_static_curve")
e2 = _load_script("E2_dictionaries")

# --- Configuration -------------------------------------------------------------

# Recording step of the level, in % of level (100/255).
STEP = psc.STEP

# Sampling periods compared for the bounds, in seconds.
TS_VALUES = (1.0, 2.0, 5.0, 10.0)

# Sampling period of the model comparison, in seconds; the gaps of each run are
# counted at this period.
TS_MODEL_S = 5.0

# Setpoint changes expected in each run.
STEPS_PER_RUN = 16

# Tolerance, in recording steps, for comparing a quantized |dh| with one step.
STEP_TOL = 1e-6

OUT_DIR = PROJECT_ROOT / "results" / "data_summary"

RUN_COLUMNS = (
    "run", "opening_pct", "acquisition_date", "n_samples_raw", "duration_s", "n_sensor_faults",
    "sensor_fault_rows", "n_samples_ts5", "n_interior_gaps_ts5",
)
QUANTIZATION_COLUMNS = ("run", "level_step_pct", "all_levels_on_grid")
BOUND_COLUMNS = (
    "ts_s", "n_pairs", "n_pairs_below_one_step", "frac_below_one_step", "median_abs_dh_steps",
    "median_samples_per_rise", "min_samples_per_rise", "n_steps_used", "n_steps_excluded",
)


# --- Runs and quantization -----------------------------------------------------


def valid_samples(raw: RawDataset) -> RawDataset:
    """The run without the samples flagged as sensor faults, as the loader returns it by default.

    Parameters
    ----------
    raw : RawDataset
        Run loaded with drop_faults=False.

    Returns
    -------
    RawDataset
        The same run with only the rows where sensor_ok is True.
    """
    return replace(raw, df=raw.df[raw.df["sensor_ok"]].reset_index(drop=True))


def run_row(raw: RawDataset) -> dict:
    """Samples, duration, date, sensor faults and gaps at TS_MODEL_S of one run.

    Parameters
    ----------
    raw : RawDataset
        Run loaded with drop_faults=False, with a Timestamp column.

    Returns
    -------
    dict
        Keys RUN_COLUMNS. duration_s is n_samples_raw x TS_NOMINAL;
        sensor_fault_rows lists the rows of the CSV (0-based, header not
        counted) flagged as faults, separated by ";" (empty if none);
        n_samples_ts5 and n_interior_gaps_ts5 are the samples kept and the
        pairs not TS_MODEL_S apart after decimating the valid samples to
        TS_MODEL_S.

    Raises
    ------
    ValueError
        If the date cannot be read; see acquisition_date in E2_dictionaries.
    """
    fault_rows = np.flatnonzero(~raw.df["sensor_ok"].to_numpy())
    _, _, _, t = resample(valid_samples(raw), TS_MODEL_S)
    return {
        "run": raw.name,
        "opening_pct": raw.valve_opening_pct,
        "acquisition_date": e2.acquisition_date(raw),
        "n_samples_raw": raw.n_raw_rows,
        "duration_s": raw.n_raw_rows * TS_NOMINAL,
        "n_sensor_faults": raw.n_sensor_faults,
        "sensor_fault_rows": ";".join(str(r) for r in fault_rows),
        "n_samples_ts5": len(t),
        "n_interior_gaps_ts5": len(find_gaps(t, TS_MODEL_S)),
    }


def quantization_row(raw: RawDataset) -> dict:
    """Whether every level value of a run is a whole multiple of the recording step.

    Parameters
    ----------
    raw : RawDataset
        Run loaded with drop_faults=False.

    Returns
    -------
    dict
        Keys QUANTIZATION_COLUMNS. all_levels_on_grid is True if every
        Level_pct x 255 / 100 is within 1e-6 of a whole number.
    """
    counts = raw.df["Level_pct"].to_numpy(dtype=float) * 255 / 100
    return {
        "run": raw.name,
        "level_step_pct": STEP,
        "all_levels_on_grid": bool(np.allclose(counts, np.round(counts), rtol=0, atol=1e-6)),
    }


# --- Bounds on the sampling period ---------------------------------------------


def transient_increments(raw: RawDataset, steps: pd.DataFrame, ts: float) -> tuple[np.ndarray, int]:
    """|dh| between consecutive samples at period ts inside the transients of a run, in recording steps.

    Parameters
    ----------
    raw : RawDataset
        Valid samples of the run (sensor faults removed).
    steps : pandas.DataFrame
        Steps of the same run, with the columns t_change_s and t95_s (s).
        A step with t95_s NaN has no transient.
    ts : float
        Sampling period, in seconds.

    Returns
    -------
    increments : numpy.ndarray
        |h[k+1] - h[k]| / STEP for every pair of the decimated series exactly
        ts apart with both times in some [t_change_s, t_change_s + t95_s].
    n_across_gaps : int
        Pairs with both times in a transient that are not ts apart (they cross
        a gap left by a sensor fault) and are therefore not counted.

    Raises
    ------
    ValueError
        If ts is not a positive whole multiple of TS_NOMINAL; see resample.
    """
    h, _, _, t = resample(raw, ts)
    in_transient = np.zeros(len(t), dtype=bool)
    for t_change, t95 in zip(steps["t_change_s"], steps["t95_s"]):
        if np.isfinite(t95):
            in_transient |= (t >= t_change) & (t <= t_change + t95)
    both = in_transient[:-1] & in_transient[1:]
    across_gap = np.zeros(len(both), dtype=bool)
    across_gap[find_gaps(t, ts)] = True
    increments = np.abs(np.diff(h)) / STEP
    return increments[both & ~across_gap], int((both & across_gap).sum())


def sampling_bounds_row(ts: float, increments: np.ndarray, t95: np.ndarray) -> dict:
    """Fraction of transient pairs below one recording step and samples per rise at one period.

    Parameters
    ----------
    ts : float
        Sampling period, in seconds.
    increments : numpy.ndarray
        |dh| of the transient pairs at ts, in recording steps, pooled over runs.
    t95 : numpy.ndarray
        t95 of every step, in s, pooled over runs; NaN where undefined.

    Returns
    -------
    dict
        Keys BOUND_COLUMNS. A pair is below one step if |dh| < 1 - STEP_TOL.
        Samples per rise are t95 / ts over the steps with finite t95 > 0
        (n_steps_used); the others are n_steps_excluded. Fraction and medians
        are NaN when there are no pairs or no steps used.
    """
    increments = np.asarray(increments, dtype=float)
    t95 = np.asarray(t95, dtype=float)
    used = np.isfinite(t95) & (t95 > 0)
    below = int((increments < 1 - STEP_TOL).sum())
    per_rise = t95[used] / ts
    return {
        "ts_s": ts,
        "n_pairs": len(increments),
        "n_pairs_below_one_step": below,
        "frac_below_one_step": below / len(increments) if len(increments) else np.nan,
        "median_abs_dh_steps": float(np.median(increments)) if len(increments) else np.nan,
        "median_samples_per_rise": float(np.median(per_rise)) if len(per_rise) else np.nan,
        "min_samples_per_rise": float(per_rise.min()) if len(per_rise) else np.nan,
        "n_steps_used": int(used.sum()),
        "n_steps_excluded": int((~used).sum()),
    }


def all_steps(raws) -> pd.DataFrame:
    """Setpoint steps and t95 of every run, computed as in the static curve of the plant.

    Parameters
    ----------
    raws : list of RawDataset
        Runs loaded with drop_faults=False.

    Returns
    -------
    pandas.DataFrame
        One row per step with the columns of step_table in plant_static_curve
        (is_evaluation_step is not used here and is always False).

    Raises
    ------
    ValueError
        If a run does not have STEPS_PER_RUN steps, or if a step is not
        ascending (see step_table).
    """
    tables = []
    for raw in raws:
        # The fit/evaluation cut is not used here: an infinite cut leaves every
        # step flagged False.
        steps = psc.step_table(valid_samples(raw).df, raw.name, raw.valve_opening_pct, np.inf)
        if len(steps) != STEPS_PER_RUN:
            raise ValueError(f"{raw.name}: {len(steps)} steps, expected {STEPS_PER_RUN}.")
        tables.append(steps)
    return pd.concat(tables, ignore_index=True)


def sampling_bounds_table(raws, steps: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Bounds on the sampling period for every period in TS_VALUES.

    Parameters
    ----------
    raws : list of RawDataset
        Runs loaded with drop_faults=False.
    steps : pandas.DataFrame
        Table returned by all_steps for the same runs.

    Returns
    -------
    table : pandas.DataFrame
        One row per period with the columns BOUND_COLUMNS.
    n_across_gaps : dict
        Period (s) -> transient pairs left out because they cross a gap.
    """
    rows, n_across_gaps = [], {}
    for ts in TS_VALUES:
        increments, n_gaps = [], 0
        for raw in raws:
            inc, n = transient_increments(valid_samples(raw), steps[steps["run"] == raw.name], ts)
            increments.append(inc)
            n_gaps += n
        rows.append(sampling_bounds_row(ts, np.concatenate(increments), steps["t95_s"].to_numpy()))
        n_across_gaps[ts] = n_gaps
    return pd.DataFrame(rows, columns=list(BOUND_COLUMNS)), n_across_gaps


# --- Run -----------------------------------------------------------------------


def run(out_dir=OUT_DIR) -> dict:
    """Summarize the Baseline runs and write the tables and metadata to out_dir.

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
        If TS_MODEL_S differs from the sampling period of E2_dictionaries, or
        if a check of run_row or all_steps fails.
    """
    if TS_MODEL_S != e2.TS_S:
        raise ValueError(f"TS_MODEL_S = {TS_MODEL_S} s but E2_dictionaries uses {e2.TS_S} s.")
    raws = load_baseline(drop_faults=False)

    runs = pd.DataFrame([run_row(r) for r in raws], columns=list(RUN_COLUMNS))
    quantization = pd.DataFrame([quantization_row(r) for r in raws], columns=list(QUANTIZATION_COLUMNS))
    steps = all_steps(raws)
    bounds, n_across_gaps = sampling_bounds_table(raws, steps)

    print(f"{len(raws)} runs, {len(steps)} steps")
    print("\nRuns")
    print(runs.to_string(index=False))
    print("\nQuantization")
    print(quantization.to_string(index=False))
    print("\nBounds on the sampling period")
    print(bounds.to_string(index=False))
    print(f"\nTransient pairs left out because they cross a gap, by Ts: {n_across_gaps}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out_dir / "runs.csv", index=False)
    quantization.to_csv(out_dir / "quantization.csv", index=False)
    bounds.to_csv(out_dir / "sampling_bounds.csv", index=False)

    summary = {
        "n_runs": len(runs),
        "n_steps": len(steps),
        "n_sensor_faults": int(runs["n_sensor_faults"].sum()),
        "n_interior_gaps_ts5": int(runs["n_interior_gaps_ts5"].sum()),
        "all_levels_on_grid": bool(quantization["all_levels_on_grid"].all()),
        "n_transient_pairs_across_gaps_by_ts": {f"{ts:g}": n for ts, n in n_across_gaps.items()},
    }
    write_metadata(
        out_dir,
        config={
            "subset": "Baseline",
            "sampling": "1 Hz raw data; sensor faults removed except for counting them and checking quantization",
            "sensor_fault_signature": "Level_pct == 0 and Setpoint_pct > 0",
            "level_recording_step_pct": STEP,
            "ts_values_s": TS_VALUES,
            "ts_model_s": TS_MODEL_S,
            "steps_per_run": STEPS_PER_RUN,
            "t95_fraction": 0.95,
            "transient": "[t_change_s, t_change_s + t95_s]",
            "below_one_step": f"|dh| < 1 - {STEP_TOL} recording steps",
            "samples_per_rise": "t95_s / ts_s, over steps with finite t95_s > 0",
        },
        summary=summary,
    )
    return summary


def main() -> None:
    """Summarize the raw Baseline data and write it to OUT_DIR."""
    run()


if __name__ == "__main__":
    main()
