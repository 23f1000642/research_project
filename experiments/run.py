"""Run one or many continual-learning experiments.

Examples
  python experiments/run.py --dataset seq-cifar100 --method derpp --set buffer_size=500 seed=0
  python experiments/run.py --dataset seq-cifar100 --method face --seeds 0 1 2 --set buffer_size=2000
  python experiments/run.py --plan experiments/plans/s1_h1_pilot.yaml --shard 0 --n-shards 2

A plan file lists runs (dataset, method, overrides); --shard/--n-shards splits it between GPUs.
Finished runs are skipped automatically, so a plan can be re-launched after a session ends.
"""
import argparse
import sys
import traceback
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facecl.config import load_config, parse_overrides  # noqa: E402
from facecl.data import build_benchmark  # noqa: E402
from facecl.evaluation import evaluate_run  # noqa: E402
from facecl.training import run  # noqa: E402


def expand_plan(path):
    plan = yaml.safe_load(Path(path).read_text())
    jobs = []
    for block in plan["runs"]:
        for seed in block.get("seeds", [0]):
            for buf in block.get("buffer_sizes", [None]):
                for method in block["methods"]:
                    ov = dict(block.get("set", {}) or {})
                    ov["seed"] = seed
                    if buf is not None:
                        ov["buffer_size"] = buf
                    jobs.append((block["dataset"], method, ov))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--method")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    ap.add_argument("--plan", help="YAML plan file with many runs")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--no-eval", action="store_true", help="skip offline metric computation")
    ap.add_argument("--prepare", action="store_true",
                    help="only download/cache the datasets of these runs (do this once before parallel shards)")
    args = ap.parse_args()

    common = parse_overrides(args.set)
    if args.plan:
        jobs = [(d, m, {**ov, **common}) for d, m, ov in expand_plan(args.plan)]
    else:
        seeds = args.seeds if args.seeds else [common.pop("seed", 0)]
        jobs = [(args.dataset, args.method, {**common, "seed": s}) for s in seeds]
    if args.prepare:
        for dataset in dict.fromkeys(d for d, _, _ in jobs):
            ov = next(o for d, _, o in jobs if d == dataset)
            build_benchmark(load_config(dataset, "sgd", ov))  # data settings do not depend on the method
            print(f"prepared {dataset}")
        return 0
    jobs = jobs[args.shard::args.n_shards]
    print(f"{len(jobs)} run(s) in this shard")
    failed = []
    for i, (dataset, method, ov) in enumerate(jobs):
        print(f"=== [{i + 1}/{len(jobs)}] {dataset} {method} {ov}", flush=True)
        try:  # one failing run must not take the rest of the shard down with it
            cfg = load_config(dataset, method, ov)
            run_dir = run(cfg)
            if not args.no_eval:
                evaluate_run(run_dir)
        except Exception:
            traceback.print_exc()
            failed.append(f"{dataset} {method} {ov}")
    for f in failed:
        print(f"FAILED: {f}")
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
