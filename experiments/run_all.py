"""Run every experiment script, one after another, in a fixed order.

Each script runs in its own Python process with the interpreter that runs this
file, so the scripts do not share state. The first script that fails stops the
run: the scripts after it are not started and this file exits with an error.

The scripts take no arguments; their settings are the constants at the top of
each script.
"""

import subprocess
import sys
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent

# Experiment scripts in execution order. Each script comes after the ones whose
# outputs it reads: E2_dictionaries reads the hyperparameters selected by
# E1_hyperparameters, and E2_comparisons and E2_one_step read the tables of
# E2_dictionaries. data_summary and plant_static_curve read only the raw data;
# data_summary imports functions and settings from plant_static_curve and
# E2_dictionaries, but does not read their outputs. paper_assets comes last: it
# reads the results of E2_dictionaries, E2_comparisons and E2_one_step.
SCRIPTS = (
    "data_summary.py",
    "E1_hyperparameters.py",
    "E2_dictionaries.py",
    "E2_comparisons.py",
    "E2_one_step.py",
    "plant_static_curve.py",
    "paper_assets.py",
)


def run_all(scripts, folder) -> None:
    """Run the scripts in order and stop at the first one that fails.

    Parameters
    ----------
    scripts : iterable of str
        File names of the scripts, in execution order.
    folder : pathlib.Path
        Folder that contains the scripts.

    Raises
    ------
    subprocess.CalledProcessError
        If a script exits with a non-zero code. The scripts after it are not run.
    """
    for name in scripts:
        print(f"=== Running {name}", flush=True)
        subprocess.run([sys.executable, str(Path(folder) / name)], check=True)


def main() -> None:
    """Run SCRIPTS from EXPERIMENTS_DIR."""
    run_all(SCRIPTS, EXPERIMENTS_DIR)


if __name__ == "__main__":
    main()
