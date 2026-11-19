"""Location of the raw data: one default, overridable by a single argument.

The module defines no functions. It exposes two constants:

PROJECT_ROOT : pathlib.Path
    Root of the code repository, the folder that holds pyproject.toml,
    whatever its name.
DEFAULT_RAW_DIR : pathlib.Path
    PROJECT_ROOT / "data" / "raw", the default data folder. Loaders take it as
    the default value of their optional data_dir argument.
"""

from pathlib import Path

# Repository root: two levels above the package folder (this file is
# <repository root>/src/edmdc_tank_observables/paths.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Raw dataset, not versioned (see data/README.md).
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
