"""The command-line scripts themselves: every file compiles, and the Kaggle launcher runs a real
(tiny, CPU) plan end to end, including the failure path."""
import py_compile
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path", sorted(str(p.relative_to(ROOT)) for p in ROOT.rglob("*.py")
                                        if ".venv" not in p.parts))
def test_every_python_file_compiles(path):
    py_compile.compile(str(ROOT / path), doraise=True)


def _launch(tmp_path, methods):
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        "runs:\n  - dataset: synthetic\n    methods: [" + ", ".join(methods) + "]\n"
        "    buffer_sizes: [40]\n    seeds: [0]\n"
    )
    cmd = [sys.executable, str(ROOT / "experiments" / "kaggle_launch.py"), str(plan),
           "--n-shards", "2", "--poll-seconds", "1",
           "--log-dir", str(tmp_path / "logs"), "--out-dir", str(tmp_path),
           "--set", f"results_dir={tmp_path / 'results'}", f"experiment_log={tmp_path / 'exp.md'}",
           f"data_root={tmp_path / 'data'}", "device=cpu", "n_epochs=1", "head_refit_steps=2"]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600)


def test_launcher_success(tmp_path):
    proc = _launch(tmp_path, ["sgd", "derpp"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "exit codes: [0, 0]" in proc.stdout
    with tarfile.open(tmp_path / "results_plan.tar.gz") as tar:
        names = tar.getnames()
    assert sum(n.endswith("summary.json") for n in names) == 2
    assert (tmp_path / "exp.md").read_text().count("complete") == 2


def test_launcher_failure_is_reported(tmp_path):
    proc = _launch(tmp_path, ["sgd", "no_such_method"])
    assert proc.returncode == 1
    assert "failed, last lines" in proc.stdout
    assert "no_such_method" in proc.stdout  # the actual error is shown, not just an exit code


def test_failing_run_does_not_stop_its_shard(tmp_path):
    # shard 0 gets [no_such_method, derpp]: derpp must still run after the first run fails
    proc = _launch(tmp_path, ["no_such_method", "sgd", "derpp"])
    assert proc.returncode == 1
    with tarfile.open(tmp_path / "results_plan.tar.gz") as tar:
        names = tar.getnames()
    assert sum(n.endswith("summary.json") for n in names) == 2
