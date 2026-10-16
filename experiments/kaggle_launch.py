"""Run a plan on every visible GPU in parallel (Kaggle 2xT4: two shards), then pack the results.

    python experiments/kaggle_launch.py experiments/plans/s1_h1_pilot.yaml [--set key=value ...]

Each GPU runs its own shard sequentially; finished runs are skipped and unfinished runs resume
from their last completed task, so the same command can be re-run in a new session.
"""
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    plan, extra = sys.argv[1], sys.argv[2:]
    import torch

    n_gpu = max(torch.cuda.device_count(), 1)
    procs = []
    for shard in range(n_gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard))
        log = open(ROOT / "logs" / f"shard{shard}_{Path(plan).stem}.log", "a")
        cmd = [sys.executable, str(ROOT / "experiments" / "run.py"), "--plan", plan,
               "--shard", str(shard), "--n-shards", str(n_gpu)] + (["--set", *extra[1:]] if extra[:1] == ["--set"] else [])
        procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT))
        print(f"shard {shard} -> GPU {shard}: {' '.join(cmd)}")
    t0 = time.time()
    while any(p.poll() is None for p in procs):
        time.sleep(60)
        print(f"{(time.time() - t0) / 60:.0f} min elapsed, running shards: "
              f"{[i for i, p in enumerate(procs) if p.poll() is None]}", flush=True)
    print("exit codes:", [p.returncode for p in procs])
    out = Path("/kaggle/working") if Path("/kaggle/working").exists() else ROOT
    archive = out / f"results_{Path(plan).stem}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(ROOT / "results", arcname="results")
        tar.add(ROOT / "logs", arcname="logs")
    print("packed", archive)


if __name__ == "__main__":
    main()
