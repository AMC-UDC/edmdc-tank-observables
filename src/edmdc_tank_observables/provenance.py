"""Provenance of the runs.

Every script that produces results leaves a ``metadata.json`` next to them with
the state of the project at that moment: git commit, time, library versions and
the configuration used. It answers, for any figure, which version of the code
produced it and with which parameters.
"""

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from edmdc_tank_observables.paths import PROJECT_ROOT

_UNKNOWN_GIT = {"commit": "unknown", "branch": "unknown", "clean": "unknown"}

# Folder, relative to the repository root, that the scripts write their results
# in. It is left out of the clean check because a run writes into it.
RESULTS_DIR = "results"


def git_state(repo_root: Path | str = PROJECT_ROOT) -> dict:
    """Read the git state of a repository.

    Parameters
    ----------
    repo_root : pathlib.Path or str, default PROJECT_ROOT
        Folder inside the repository to inspect.

    Returns
    -------
    dict
        Keys ``commit`` (short hash of HEAD), ``branch`` (current branch name)
        and ``clean`` (True if there are no uncommitted or untracked changes
        outside RESULTS_DIR at the repository root, False otherwise). Changes
        inside RESULTS_DIR are ignored because the scripts write their results
        there while they run, so a later script in the same session would
        otherwise record an unclean tree. If git is not available or the
        folder is not in a repository, the three values are the string
        "unknown".
    """

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "--short", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "clean": git("status", "--porcelain", "--", ":(top)", f":(top,exclude){RESULTS_DIR}") == "",
        }
    except (subprocess.CalledProcessError, OSError):
        return dict(_UNKNOWN_GIT)


def _environment() -> dict:
    """Versions of Python, the platform and the main libraries."""
    environment = {"python": sys.version.split()[0], "platform": platform.platform()}
    for package in ("numpy", "pandas", "matplotlib"):
        try:
            environment[package] = version(package)
        except PackageNotFoundError:
            environment[package] = "not installed"
    return environment


def _script_name() -> str:
    """The script that is running, as started, relative to the repository root if it is inside it."""
    script = Path(sys.argv[0])
    try:
        return script.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return sys.argv[0]


def write_metadata(out_dir: Path | str, config: dict | None = None, summary: dict | None = None) -> Path:
    """Write the ``metadata.json`` of a run.

    Parameters
    ----------
    out_dir : pathlib.Path or str
        Folder to write the file in, typically the one with the results. It is
        created if it does not exist.
    config : dict, optional
        Parameters used: sampling period, dictionaries, horizons, etc.
        Values that JSON cannot represent are stored as text.
    summary : dict, optional
        Main figures obtained, to read them without opening the tables.

    Returns
    -------
    pathlib.Path
        Path of the file written.

    Notes
    -----
    The file has the keys ``script`` (the program that is running, taken from
    sys.argv[0] and stored relative to the repository root when it is inside
    it), ``utc_time``, ``git`` (see :func:`git_state`), ``environment``,
    ``config`` and ``summary``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "script": _script_name(),
        "utc_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "environment": _environment(),
        "config": config or {},
        "summary": summary or {},
    }
    path = out_dir / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path
