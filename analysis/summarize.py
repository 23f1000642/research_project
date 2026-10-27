"""Summarise finished runs: accuracy vs published numbers, final old-class uncertainty quality, and
confidence/accuracy by task age (the first look at H1, "confident forgetting").

    python analysis/summarize.py results [--out tables/s0_summary.md]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# Class-IL final average accuracy, buffer 500 (Buzzega et al., NeurIPS 2020, Table 2; 10 runs)
PUBLISHED = {("seq-cifar10", "sgd", 500): (19.62, 0.05),
             ("seq-cifar10", "er", 500): (57.74, 0.27),
             ("seq-cifar10", "derpp", 500): (72.70, 1.36)}


def ms(xs, scale=1.0):
    xs = np.asarray(xs, dtype=float) * scale
    return f"{xs.mean():.2f} ± {xs.std(ddof=1):.2f}" if len(xs) > 1 else f"{xs.mean():.2f}"


def load_runs(root):
    runs = []
    for d in sorted(Path(root).glob("*/")):
        s, m = d / "summary.json", d / "metrics.json"
        if s.exists() and m.exists():
            summary = json.loads(s.read_text())
            if summary.get("status") == "complete":
                runs.append((summary, json.loads(m.read_text())))
    return runs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    groups = defaultdict(list)
    for s, m in load_runs(args.results):
        groups[(s["dataset"], s["method"], s["buffer_size"])].append((s, m))

    lines = ["## Accuracy (class-IL, %)", "",
             "| dataset | method | buffer | seeds | per seed | final acc | avg inc acc | forgetting | published | minutes/run |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for key, rs in sorted(groups.items()):
        acc = [m["accuracy"]["final_avg_acc"] for _, m in rs]
        pub = PUBLISHED.get(key)
        minutes = [sum(t["train_seconds"] for t in s["timings"]) / 60 for s, _ in rs]
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {len(rs)} | "
                     f"{' / '.join(f'{100 * a:.2f}' for a in acc)} | {ms(acc, 100)} | "
                     f"{ms([m['accuracy']['avg_incremental_acc'] for _, m in rs], 100)} | "
                     f"{ms([m['accuracy']['forgetting'] for _, m in rs], 100)} | "
                     f"{f'{pub[0]:.2f} ± {pub[1]:.2f}' if pub else '-'} | {ms(minutes)} |")

    lines += ["", "## Old-class uncertainty after the last task (MSP; TS = temperature-scaled MSP)", "",
              "| method | buffer | acc old | mean conf old | ECE old | ECE old (TS) | AUROC old | AUROC old (energy) "
              "| CER@0.9 old | CER@0.9 old (TS) | UF (MSP) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, rs in sorted(groups.items()):
        f = lambda score, k: [m["final"][score][k] for _, m in rs]
        acc_old = [np.mean(s["acc_class_il"][-1][:-1]) for s, _ in rs]
        lines.append(f"| {key[1]} | {key[2]} | {ms(acc_old, 100)} | {ms(f('msp', 'mean_conf_old'), 100)} | "
                     f"{ms(f('msp', 'ece_old'), 100)} | {ms(f('msp_ts', 'ece_old'), 100)} | "
                     f"{ms(f('msp', 'auroc_old'), 100)} | {ms(f('energy', 'auroc_old'), 100)} | "
                     f"{ms(f('msp', 'cer_0.9_old'), 100)} | {ms(f('msp_ts', 'cer_0.9_old'), 100)} | "
                     f"{ms([m['uncertainty_forgetting']['msp'] for _, m in rs], 100)} |")

    lines += ["", "## By task age after the last task (age 0 = current task; MSP, %, mean over seeds)", "",
              "| method | buffer | age | accuracy | mean confidence | gap (conf - acc) | ECE | AUROC |",
              "|---|---|---|---|---|---|---|---|"]
    for key, rs in sorted(groups.items()):
        n_tasks = len(rs[0][0]["acc_class_il"][-1])
        for age in range(n_tasks):
            j = n_tasks - 1 - age
            acc = [s["acc_class_il"][-1][j] for s, _ in rs]
            per = lambda k: [m["per_task"]["msp"][k][-1][j] for _, m in rs]
            conf = per("mean_conf")
            lines.append(f"| {key[1]} | {key[2]} | {age} | {ms(acc, 100)} | {ms(conf, 100)} | "
                         f"{ms(np.subtract(conf, acc), 100)} | {ms(per('ece'), 100)} | {ms(per('auroc'), 100)} |")

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
