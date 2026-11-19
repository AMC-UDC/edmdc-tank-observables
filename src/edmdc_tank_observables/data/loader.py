"""Loader for the Baseline CSV files of the tank test bench.

The dataset has 10 runs of about 8.5 h sampled at 1 Hz in the Baseline folder.
Two particularities of the files are handled here:

1. The Timestamp column is not reliable in Baseline (it refreshes once per
   minute), so time is always rebuilt as row index x 1 s.
2. Ultrasonic sensor faults are detected by their signature: Level_pct == 0
   with a strictly positive setpoint.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from edmdc_tank_observables.paths import DEFAULT_RAW_DIR

# Sampling period of the acquisition, in seconds.
TS_NOMINAL = 1.0

# Columns every file must have.
REQUIRED_COLUMNS = (
    "Level_pct",
    "Control_Signal_pct",
    "Setpoint_pct",
    "Valve_Closure_pct",
)


@dataclass
class RawDataset:
    """One run, loaded from its CSV.

    Attributes
    ----------
    name : str
        File name without extension, e.g. "Baseline_50pct".
    valve_opening_pct : float
        Electrovalve opening of the run, read from the file name.
        Constant within a run.
    df : pandas.DataFrame
        The CSV data plus two columns added here: t_s (rebuilt time, in s)
        and sensor_ok (False on a sensor fault).
    n_raw_rows : int
        Rows read from the CSV.
    n_sensor_faults : int
        Samples flagged as ultrasonic sensor faults.
    """

    name: str
    valve_opening_pct: float
    df: pd.DataFrame
    n_raw_rows: int
    n_sensor_faults: int


def valve_opening_pct(name: str) -> float:
    """Read the electrovalve opening from a file name such as "Baseline_50pct".

    Parameters
    ----------
    name : str
        File name, with or without extension. The first integer immediately
        followed by "pct" is taken as the opening.

    Returns
    -------
    float
        Electrovalve opening, in %.

    Raises
    ------
    ValueError
        If the name does not contain an integer followed by "pct".
    """
    match = re.search(r"(\d+)pct", name)
    if match is None:
        raise ValueError(f"No valve opening in file name {name!r}.")
    return float(match.group(1))


def load_baseline_file(path: Path | str, drop_faults: bool = True) -> RawDataset:
    """Load one Baseline CSV.

    Time is rebuilt as row index x TS_NOMINAL. Samples with Level_pct == 0 and
    Setpoint_pct > 0 are flagged in sensor_ok; if drop_faults is True they are
    removed from df. Dropping rows breaks the uniform spacing of the series,
    which is why t_s keeps the original time.

    Parameters
    ----------
    path : pathlib.Path or str
        Path to the CSV file. Its name must contain the opening, e.g.
        "Baseline_50pct.csv".
    drop_faults : bool, default True
        If True, samples flagged as sensor faults are removed from the returned
        DataFrame. If False they are kept, flagged by sensor_ok == False.

    Returns
    -------
    RawDataset
        The run. n_raw_rows counts the rows of the CSV before any removal and
        n_sensor_faults the flagged samples, whether or not they were dropped.

    Raises
    ------
    FileNotFoundError
        If path does not exist (raised by pandas when reading the CSV).
    ValueError
        If the CSV lacks any of REQUIRED_COLUMNS, or if the file name does not
        contain an opening.
    """
    path = Path(path)
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing required columns {missing}.")

    df["t_s"] = np.arange(len(df), dtype=float) * TS_NOMINAL
    df["sensor_ok"] = ~((df["Level_pct"] == 0) & (df["Setpoint_pct"] > 0))
    n_faults = int((~df["sensor_ok"]).sum())

    n_raw = len(df)
    if drop_faults:
        df = df[df["sensor_ok"]].reset_index(drop=True)

    return RawDataset(
        name=path.stem,
        valve_opening_pct=valve_opening_pct(path.stem),
        df=df,
        n_raw_rows=n_raw,
        n_sensor_faults=n_faults,
    )


def load_baseline(data_dir: Path | str = DEFAULT_RAW_DIR, drop_faults: bool = True) -> list[RawDataset]:
    """Load every CSV in <data_dir>/Baseline, from lowest to highest opening.

    data_dir is the only way to change the data location.

    Parameters
    ----------
    data_dir : pathlib.Path or str, default DEFAULT_RAW_DIR
        Folder that contains the Baseline subfolder.
    drop_faults : bool, default True
        Passed to load_baseline_file: if True, samples flagged as sensor faults
        are removed from each DataFrame.

    Returns
    -------
    list of RawDataset
        One per CSV, sorted by increasing electrovalve opening.

    Raises
    ------
    FileNotFoundError
        If <data_dir>/Baseline does not exist, or contains no CSV files.
    ValueError
        If a CSV lacks any of REQUIRED_COLUMNS, or if its file name does not
        contain an opening.
    """
    folder = Path(data_dir) / "Baseline"
    if not folder.is_dir():
        raise FileNotFoundError(
            f"{folder} not found. Raw data are not versioned; see data/README.md."
        )
    files = sorted(folder.glob("*.csv"), key=lambda p: valve_opening_pct(p.stem))
    if not files:
        raise FileNotFoundError(f"No CSV files in {folder}.")
    return [load_baseline_file(p, drop_faults) for p in files]
