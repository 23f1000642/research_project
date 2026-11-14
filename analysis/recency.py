"""Exploratory (after RQ1): are old-class errors mostly predictions of current-task classes
(recency bias), and does that explain when task-wise temperature scaling helps failure detection?

    python analysis/recency.py results
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rq1_calibration as R  # noqa: E402
from summarize import variant  # noqa: E402


def recency_share(run_dir):
    """Share of old-class test errors (after the last task) predicted as a current-task class."""
    s = json.loads((run_dir / "summary.json").read_text())
    d = np.load(sorted((run_dir / "eval").glob("task_*.npz"))[-1])
    T = s["n_tasks"]
    k = len(s["class_order"]) // T
    logits = np.concatenate([d[f"test{j}_logits"] for j in range(T - 1)]).astype(float)
    y = np.concatenate([d[f"test{j}_y"] for j in range(T - 1)]).astype(int)
    pred = logits.argmax(1)
    wrong = pred != y
    return (pred[wrong] // k == T - 1).mean() * 100, k * 1.0 / (len(s["class_order"]) - 1) * 100


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="results")
    args = ap.parse_args(argv)
    rows = []
    for rd in sorted(Path(args.results).glob("seq-*/")):
        s = json.loads((rd / "summary.json").read_text())
        if s.get("status") != "complete" or s["method"] not in ("er", "derpp") or not list((rd / "eval").glob("*.npz")):
            continue
        share, chance = recency_share(rd)
        res = R.evaluate_run(rd)[1]
        key = (s["dataset"], s["method"], s["buffer_size"], variant(yaml.safe_load((rd / "config.yaml").read_text())))
        rows.append((key, share, chance, res["task_ts"]["auroc_all"] - res["none"]["auroc_all"]))

    groups = defaultdict(list)
    for key, share, chance, gain in rows:
        groups[key].append((share, chance, gain))
    print("| dataset | method | buffer | variant | n | old-class errors predicted as current task (%) | "
          "uniform-error baseline (%) | task-wise TS gain in AUROC_all |\n|---|---|---|---|---|---|---|---|")
    for key, v in sorted(groups.items()):
        v = np.array(v)
        print(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {len(v)} | {v[:, 0].mean():.1f} | "
              f"{v[0, 1]:.1f} | {v[:, 2].mean():+.2f} |")
    share, gain = np.array([r[1] for r in rows]), np.array([r[3] for r in rows])
    rho, p = stats.spearmanr(share, gain)
    print(f"\nSpearman(recency share, task-wise TS gain) over {len(rows)} runs: rho = {rho:+.2f}, p = {p:.4f}")


if __name__ == "__main__":
    main()
