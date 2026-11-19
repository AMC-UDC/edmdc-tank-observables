"""Tests of experiments/run_all.py with fake experiment scripts."""

import importlib.util
import subprocess

import pytest

from edmdc_tank_observables.paths import PROJECT_ROOT


def load_script():
    """Import experiments/run_all.py as a module, without running it."""
    path = PROJECT_ROOT / "experiments" / "run_all.py"
    spec = importlib.util.spec_from_file_location("run_all", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_fake_script(folder, name, log, fail=False):
    """Write a script that appends its name to the log file and, if fail, exits with code 1."""
    code = f"with open({str(log)!r}, 'a') as f:\n    f.write({name!r} + '\\n')\n"
    if fail:
        code += "raise SystemExit(1)\n"
    (folder / name).write_text(code, encoding="utf-8")


def test_run_all_lists_existing_scripts():
    """Every script that run_all is set to run exists in the experiments folder."""
    run_all = load_script()
    for name in run_all.SCRIPTS:
        assert (run_all.EXPERIMENTS_DIR / name).is_file()


def test_run_all_runs_every_experiment_after_the_ones_it_reads():
    """SCRIPTS lists every experiment script once, in order: data summary, E1, E2, the comparison and one-step scripts that read E2's tables, the static curve, and the paper assets last."""
    run_all = load_script()
    assert run_all.SCRIPTS == (
        "data_summary.py",
        "E1_hyperparameters.py",
        "E2_dictionaries.py",
        "E2_comparisons.py",
        "E2_one_step.py",
        "plant_static_curve.py",
        "paper_assets.py",
    )
    scripts = {p.name for p in run_all.EXPERIMENTS_DIR.glob("*.py")} - {"run_all.py"}
    assert set(run_all.SCRIPTS) == scripts


def test_run_all_runs_the_scripts_in_order(tmp_path):
    """The scripts are run one after another, in the order given."""
    run_all = load_script()
    log = tmp_path / "log.txt"
    names = ("first.py", "second.py", "third.py")
    for name in names:
        write_fake_script(tmp_path, name, log)
    run_all.run_all(names, tmp_path)
    assert log.read_text().split() == list(names)


def test_run_all_stops_at_the_first_failing_script(tmp_path):
    """If a script fails, the error is raised and the scripts after it are not run."""
    run_all = load_script()
    log = tmp_path / "log.txt"
    write_fake_script(tmp_path, "first.py", log)
    write_fake_script(tmp_path, "second.py", log, fail=True)
    write_fake_script(tmp_path, "third.py", log)
    with pytest.raises(subprocess.CalledProcessError):
        run_all.run_all(("first.py", "second.py", "third.py"), tmp_path)
    assert log.read_text().split() == ["first.py", "second.py"]
