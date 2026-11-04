"""Pre-registered test of H1 ("confident forgetting"), written before the S1 results were seen.

H1 (docs/phase2_method.md): as a task gets older, errors on it become more confident
(CER@0.9 = fraction of its errors with max-softmax > 0.9 rises) and errors become harder to
detect (misclassification-detection AUROC falls).

Statistic. For every run and every task j that is followed by at least one later task, follow
that task's MSP metric over the evaluations after tasks i = j, ..., T-1 (age a = i - j) and fit a
least-squares slope in a. Following one fixed task separates "older" from "a different task".
The run's slope is the mean over its tasks (tasks with NaN metrics are skipped).

Decision rule, per dataset, pooled over every ER / DER++ run found:
  H1 supported  <=>  mean CER slope > 0 and mean AUROC slope < 0, each one-sided p < 0.05 after
                     Holm correction over the two tests (one-sample t-test over runs), and both
                     signs agree in at least 3 of the 4 (method, buffer) groups.
Caveat: runs with the same seed share a class order, so pooled runs are not fully independent;
per-group results are reported next to the pooled ones.

    python analysis/h1_test.py results --dataset seq-cifar100 [--fig figures/h1_seq-cifar100.png]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

from summarize import variant

METRICS = ("cer_0.9", "auroc")
EXPECTED_SIGN = {"cer_0.9": +1, "auroc": -1}


def task_slopes(per_task_metric):
    """per_task_metric[i][j] for j <= i. Returns the mean within-task slope over age (NaN if none)."""
    T = len(per_task_metric)
    slopes = []
    for j in range(T - 1):
        ages = np.arange(T - j)
        vals = np.array([per_task_metric[j + a][j] for a in ages], float)
        ok = ~np.isnan(vals)
        if ok.sum() >= 2:
            slopes.append(np.polyfit(ages[ok], vals[ok], 1)[0])
    return float(np.mean(slopes)) if slopes else float("nan")


def by_age(per_task_metric):
    """Mean over tasks of the metric at each age (for the figure)."""
    T = len(per_task_metric)
    return [np.nanmean([per_task_metric[j + a][j] for j in range(T - a)]) for a in range(T)]


def holm(pvals):
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, k in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[k])
        adj[k] = min(running, 1.0)
    return adj


def load(root, dataset):
    runs = []
    for d in sorted(Path(root).glob(f"{dataset}-*/")):
        if not (d / "metrics.json").exists():
            continue
        s = json.loads((d / "summary.json").read_text())
        if s.get("status") != "complete" or s["method"] not in ("er", "derpp"):
            continue
        m = json.loads((d / "metrics.json").read_text())["per_task"]["msp"]
        group = (s["method"], s["buffer_size"], variant(yaml.safe_load((d / "config.yaml").read_text())))
        runs.append({"group": group, "seed": s["seed"],
                     **{k: task_slopes(m[k]) for k in METRICS},
                     **{f"{k}_by_age": by_age(m[k]) for k in METRICS}})
    return runs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="results")
    ap.add_argument("--dataset", default="seq-cifar100")
    ap.add_argument("--fig", default=None)
    args = ap.parse_args(argv)

    runs = load(args.results, args.dataset)
    if not runs:
        raise SystemExit(f"no complete ER/DER++ runs for {args.dataset} in {args.results}")
    groups = defaultdict(list)
    for r in runs:
        groups[r["group"]].append(r)

    print(f"H1 test on {args.dataset}: {len(runs)} runs, within-task slope per unit of task age (MSP)\n")
    print("| method | buffer | variant | runs | CER@0.9 slope | AUROC slope |\n|---|---|---|---|---|---|")
    agree = {k: 0 for k in METRICS}
    for g, rs in sorted(groups.items()):
        cells = []
        for k in METRICS:
            v = np.array([r[k] for r in rs])
            agree[k] += int(np.sign(np.nanmean(v)) == EXPECTED_SIGN[k])
            cells.append(f"{100 * np.nanmean(v):+.2f} ({', '.join(f'{100 * x:+.2f}' for x in v)})")
        print(f"| {g[0]} | {g[1]} | {g[2]} | {len(rs)} | {cells[0]} | {cells[1]} |")

    pvals, means = [], []
    for k in METRICS:
        v = np.array([r[k] for r in runs])
        v = v[~np.isnan(v)]
        alt = "greater" if EXPECTED_SIGN[k] > 0 else "less"
        pvals.append(stats.ttest_1samp(v, 0.0, alternative=alt).pvalue if len(v) > 1 else 1.0)
        means.append(v.mean())
    adj = holm(np.array(pvals))
    print("\nPooled (one-sample t-test over runs, one-sided, Holm-corrected):")
    ok = True
    for k, mu, p, pa in zip(METRICS, means, pvals, adj):
        sign_ok = np.sign(mu) == EXPECTED_SIGN[k]
        groups_ok = agree[k] >= min(3, len(groups))
        ok &= bool(sign_ok and pa < 0.05 and groups_ok)
        print(f"  {k}: mean slope {100 * mu:+.2f} points/task-age, p={p:.4f}, Holm p={pa:.4f}, "
              f"expected sign in {agree[k]}/{len(groups)} groups")
    if len(groups) < 4:
        print(f"  note: the rule asks for 4 (method, buffer) groups, only {len(groups)} found")
    print(f"\nH1 {'SUPPORTED' if ok else 'NOT SUPPORTED'} on {args.dataset} under the pre-registered rule")

    if args.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, k, label in zip(axes, METRICS, ("errors with confidence > 0.9 (%)", "error-detection AUROC (%)")):
            for g, rs in sorted(groups.items()):
                curves = np.array([r[f"{k}_by_age"] for r in rs], float) * 100
                mu = np.nanmean(curves, 0)
                ages = np.arange(len(mu))
                ax.plot(ages, mu, marker="o", label=f"{g[0]} b{g[1]} {g[2]}")
                if len(rs) > 1:
                    sd = np.nanstd(curves, 0, ddof=1)
                    ax.fill_between(ages, mu - sd, mu + sd, alpha=0.15)
            ax.set_xticks(ages)
            ax.set_xlabel("task age (tasks learned since this one)")
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
        axes[0].legend()
        fig.suptitle(f"{args.dataset}: confidence of old-task errors vs. task age (mean ± sd over seeds)")
        fig.tight_layout()
        Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig, dpi=150)
        print(f"figure -> {args.fig}")


if __name__ == "__main__":
    main()
