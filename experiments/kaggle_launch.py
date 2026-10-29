"""Run a plan on every visible GPU in parallel (Kaggle 2xT4: two shards), then pack the results.

    python experiments/kaggle_launch.py experiments/plans/s1_h1_pilot.yaml [--set key=value ...]

Each GPU runs its own shard sequentially; finished runs are skipped and unfinished runs resume
from their last completed task, so the same command can be re-run in a new session.
Exits non-zero (after printing the end of the failing shard's log) if any shard fails.
"""
import argparse
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value for every run")
    ap.add_argument("--n-shards", type=int, default=None, help="default: visible GPUs x --per-gpu")
    ap.add_argument("--per-gpu", type=int, default=1,
                    help="runs sharing one GPU at a time (small-batch runs leave a T4 partly idle)")
    ap.add_argument("--log-dir", default=str(ROOT / "logs"))
    ap.add_argument("--out-dir", default="/kaggle/working" if Path("/kaggle/working").exists() else str(ROOT))
    ap.add_argument("--poll-seconds", type=float, default=60)
    args = ap.parse_args(argv)

    import torch

    n_gpus = max(torch.cuda.device_count(), 1)
    n_shards = args.n_shards or n_gpus * args.per_gpu
    overrides = dict(s.split("=", 1) for s in args.set)
    results_dir = Path(overrides.get("results_dir", ROOT / "results"))
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.plan).stem

    procs, log_paths = [], []
    for shard in range(n_shards):
        gpu = shard % n_gpus
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        log_path = log_dir / f"shard{shard}_{stem}.log"
        log_paths.append(log_path)
        cmd = [sys.executable, str(ROOT / "experiments" / "run.py"), "--plan", args.plan,
               "--shard", str(shard), "--n-shards", str(n_shards)]
        if args.set:
            cmd += ["--set", *args.set]
        with open(log_path, "a") as log:
            procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT))
        print(f"shard {shard} -> GPU {gpu}: {' '.join(cmd)}", flush=True)

    t0 = time.time()
    while any(p.poll() is None for p in procs):
        time.sleep(args.poll_seconds)
        running = [i for i, p in enumerate(procs) if p.poll() is None]
        print(f"{(time.time() - t0) / 60:.0f} min elapsed, running shards: {running}", flush=True)

    codes = [p.returncode for p in procs]
    print("exit codes:", codes)
    for shard, code in enumerate(codes):
        if code != 0:  # show the error instead of failing silently
            print(f"----- shard {shard} failed, last lines of {log_paths[shard]} -----")
            for line in log_paths[shard].read_text().splitlines()[-40:]:
                print(line)

    archive = Path(args.out_dir) / f"results_{stem}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        if results_dir.exists():
            tar.add(results_dir, arcname="results")
        tar.add(log_dir, arcname="logs")
    print("packed", archive)
    return 1 if any(codes) else 0


if __name__ == "__main__":
    sys.exit(main())
