"""Tests of the run provenance: git state and metadata file."""

import json
import subprocess
import sys
from datetime import datetime, timedelta
from importlib.metadata import version

import pytest

from edmdc_tank_observables import provenance
from edmdc_tank_observables.paths import PROJECT_ROOT
from edmdc_tank_observables.provenance import git_state, write_metadata


def run_git(repo, *args):
    """Run a git command inside a repository and return its output."""
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A temporary git repository with one commit on the branch "main"."""
    run_git(tmp_path, "init", "-b", "main")
    run_git(tmp_path, "config", "user.email", "test@example.com")
    run_git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "file.txt").write_text("content\n")
    run_git(tmp_path, "add", "file.txt")
    run_git(tmp_path, "commit", "-m", "first")
    return tmp_path


def test_git_state_reports_commit_branch_and_clean_tree(repo):
    """On a committed repository the state gives its short commit, its branch and a clean tree."""
    state = git_state(repo)
    assert set(state) == {"commit", "branch", "clean"}
    assert state["commit"] == run_git(repo, "rev-parse", "--short", "HEAD")
    assert state["branch"] == "main"
    assert state["clean"] is True


def test_git_state_sees_a_modified_file_as_not_clean(repo):
    """Editing a tracked file makes the tree not clean."""
    (repo / "file.txt").write_text("changed\n")
    assert git_state(repo)["clean"] is False


def test_git_state_sees_an_untracked_file_as_not_clean(repo):
    """A new file that git does not track also makes the tree not clean."""
    (repo / "new.txt").write_text("new\n")
    assert git_state(repo)["clean"] is False


def test_git_state_ignores_a_new_file_in_results(repo):
    """A file written inside results/ does not make the tree not clean."""
    (repo / "results" / "experiment").mkdir(parents=True)
    (repo / "results" / "experiment" / "errors.csv").write_text("a,b\n")
    assert git_state(repo)["clean"] is True


def test_git_state_ignores_a_modified_file_in_results(repo):
    """Overwriting a committed result inside results/ does not make the tree not clean."""
    (repo / "results").mkdir()
    (repo / "results" / "errors.csv").write_text("a,b\n")
    run_git(repo, "add", "results/errors.csv")
    run_git(repo, "commit", "-m", "results")
    (repo / "results" / "errors.csv").write_text("a,b\n1,2\n")
    assert git_state(repo)["clean"] is True


def test_git_state_sees_a_new_file_outside_results_as_not_clean(repo):
    """A new file outside results/ still makes the tree not clean, even next to one in results/."""
    (repo / "results").mkdir()
    (repo / "results" / "errors.csv").write_text("a,b\n")
    (repo / "experiments").mkdir()
    (repo / "experiments" / "new.py").write_text("x = 1\n")
    assert git_state(repo)["clean"] is False


def test_git_state_only_ignores_results_at_the_repository_root(repo):
    """A folder called results/ below the root is not ignored, and the root results/ is ignored from a subfolder too."""
    (repo / "src" / "results").mkdir(parents=True)
    (repo / "src" / "results" / "note.txt").write_text("x\n")
    assert git_state(repo)["clean"] is False
    (repo / "src" / "results" / "note.txt").unlink()
    (repo / "results").mkdir()
    (repo / "results" / "errors.csv").write_text("a,b\n")
    assert git_state(repo / "src")["clean"] is True


def test_git_state_follows_a_new_commit(repo):
    """After committing a change the commit changes and the tree is clean again."""
    before = git_state(repo)["commit"]
    (repo / "file.txt").write_text("changed\n")
    run_git(repo, "commit", "-am", "second")
    after = git_state(repo)
    assert after["commit"] != before
    assert after["clean"] is True


def test_git_state_outside_a_repository_is_unknown(tmp_path):
    """A folder that is not a git repository gives "unknown" for every field."""
    assert git_state(tmp_path) == {"commit": "unknown", "branch": "unknown", "clean": "unknown"}


def test_git_state_without_git_installed_is_unknown(monkeypatch, repo):
    """If the git program cannot be run, every field is "unknown" instead of an error."""
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(provenance.subprocess, "run", no_git)
    assert git_state(repo) == {"commit": "unknown", "branch": "unknown", "clean": "unknown"}


def test_metadata_file_has_all_the_sections(tmp_path):
    """The metadata file is valid JSON with script, time, git, environment, config and summary."""
    path = write_metadata(tmp_path)
    assert path == tmp_path / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    assert set(metadata) == {"script", "utc_time", "git", "environment", "config", "summary"}
    assert set(metadata["git"]) == {"commit", "branch", "clean"}
    assert metadata["config"] == {} and metadata["summary"] == {}


def test_metadata_records_the_library_versions_of_the_environment(tmp_path):
    """The environment lists Python, the platform and the installed versions of the main libraries."""
    environment = json.loads(write_metadata(tmp_path).read_text(encoding="utf-8"))["environment"]
    assert set(environment) == {"python", "platform", "numpy", "pandas", "matplotlib"}
    assert environment["python"] == sys.version.split()[0]
    for package in ("numpy", "pandas", "matplotlib"):
        assert environment[package] == version(package)


def test_metadata_time_is_utc_and_current(tmp_path):
    """The recorded time is a UTC timestamp taken when the file is written."""
    before = datetime.now().astimezone()
    stamp = json.loads(write_metadata(tmp_path).read_text(encoding="utf-8"))["utc_time"]
    moment = datetime.fromisoformat(stamp)
    assert moment.utcoffset() == timedelta(0)
    assert before - timedelta(seconds=2) <= moment <= datetime.now().astimezone() + timedelta(seconds=2)


def test_metadata_keeps_config_and_summary(tmp_path):
    """The configuration and the summary given by the caller are stored as they are."""
    config = {"ts_s": 5.0, "dictionaries": ["D0", "D2"], "hyperparameters": {"D1a": {"order": 2}}}
    summary = {"n_models": 80}
    metadata = json.loads(write_metadata(tmp_path, config, summary).read_text(encoding="utf-8"))
    assert metadata["config"] == config
    assert metadata["summary"] == summary


def test_metadata_creates_the_output_folder(tmp_path):
    """The output folder is created if it does not exist."""
    out_dir = tmp_path / "results" / "experiment"
    assert write_metadata(out_dir).is_file()


def test_metadata_script_is_the_name_the_program_was_started_with(monkeypatch, tmp_path):
    """The script field is the path the program was started with."""
    monkeypatch.setattr(sys, "argv", ["experiments/run_all.py"])
    metadata = json.loads(write_metadata(tmp_path).read_text(encoding="utf-8"))
    assert metadata["script"] == "experiments/run_all.py"


def test_metadata_script_inside_the_repository_is_stored_without_the_absolute_path(monkeypatch, tmp_path):
    """A script started with an absolute path inside the repository is stored relative to its root."""
    monkeypatch.setattr(sys, "argv", [str(PROJECT_ROOT / "experiments" / "run_all.py")])
    metadata = json.loads(write_metadata(tmp_path).read_text(encoding="utf-8"))
    assert metadata["script"] == "experiments/run_all.py"
