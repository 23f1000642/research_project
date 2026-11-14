"""RQ1 (docs/paper_plan.md, pre-registered in commit 89d7e9e): do post-hoc temperature calibrators
improve failure detection on old classes, or only calibration? Runs on saved logits, no GPU.

    python analysis/rq1_calibration.py results [--out tables/rq1_calibration.md]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facecl.evaluation import metrics as M  # noqa: E402

from summarize import variant  # noqa: E402

LEGAL = ("ts", "task_ts", "oldnew_ts")
ALL = ("none",) + LEGAL + ("oracle_ts", "oracle_task_ts")
MIN_ITEMS = 20  # fewer calibration items for a group -> fall back to the global temperature


def group_temperatures(logits, y_or_pred_task, groups, global_t, labels):
    """One temperature per group id, fitted on items whose group is that id."""
    temps = {}
    for g in np.unique(groups):
        m = groups == g
        temps[int(g)] = M.fit_temperature(logits[m], labels[m]) if m.sum() >= MIN_ITEMS else global_t
    return temps


def apply_group_t(logits, groups, temps, default):
    t = np.array([temps.get(int(g), default) for g in groups])
    return M.softmax_np(logits / t[:, None]).max(1)


def evaluate_run(run_dir):
    s = json.loads((run_dir / "summary.json").read_text())
    files = sorted((run_dir / "eval").glob("task_*.npz"))
    d = dict(np.load(files[-1]))
    T = s["n_tasks"]
    k = len(s["class_order"]) // T  # classes per task; labels are remapped so task t owns [t*k, (t+1)*k)
    cur = T - 1

    # calibration set: non-probe buffer + current-task validation (what evaluate.py uses)
    keep = ~d["buf_probe"].astype(bool)
    cal_l = np.concatenate([d["val_logits"], d["buf_logits"][keep]]).astype(float)
    cal_y = np.concatenate([d["val_y"], d["buf_y"][keep]]).astype(int)
    cal_task = cal_y // k  # task of the true class (labels are known for calibration data)

    test_l = np.concatenate([d[f"test{j}_logits"] for j in range(T)]).astype(float)
    test_y = np.concatenate([d[f"test{j}_y"] for j in range(T)]).astype(int)
    true_task = test_y // k
    pred_task = test_l.argmax(1) // k
    correct = test_l.argmax(1) == test_y
    old = true_task < cur

    g_t = M.fit_temperature(cal_l, cal_y)
    task_t = group_temperatures(cal_l, None, cal_task, g_t, cal_y)
    on_cal = (cal_task < cur).astype(int)
    on_t = group_temperatures(cal_l, None, on_cal, g_t, cal_y)

    conf = {
        "none": M.softmax_np(test_l).max(1),
        "ts": M.softmax_np(test_l / g_t).max(1),
        "task_ts": apply_group_t(test_l, pred_task, task_t, g_t),
        "oldnew_ts": apply_group_t(test_l, (pred_task < cur).astype(int), on_t, g_t),
    }
    # oracle upper bounds: temperatures fitted on the evaluated old-class test data itself
    o_l, o_y = test_l[old], test_y[old]
    conf["oracle_ts"] = np.zeros(len(test_y))
    conf["oracle_ts"][old] = M.softmax_np(o_l / M.fit_temperature(o_l, o_y)).max(1)
    o_groups = pred_task[old]
    o_t = group_temperatures(o_l, None, o_groups, M.fit_temperature(o_l, o_y), o_y)
    conf["oracle_task_ts"] = np.zeros(len(test_y))
    conf["oracle_task_ts"][old] = apply_group_t(o_l, o_groups, o_t, 1.0)

    out = {}
    for name, c in conf.items():
        res = {}
        for scope, m in (("old", old), ("all", np.ones_like(old))):
            if name.startswith("oracle") and scope == "all":
                continue
            cc, ok = c[m], correct[m]
            res[f"ece_{scope}"] = M.ece(cc, ok) * 100
            res[f"auroc_{scope}"] = M.auroc_error(cc, ok) * 100
            res[f"aurc_{scope}"] = M.aurc(cc, ok) * 100
            res[f"sel90_{scope}"] = M.selective_risk(cc, ok, 0.9) * 100
            res[f"conf_{scope}"] = cc.mean() * 100
        out[name] = res
    out["_acc_old"] = correct[old].mean() * 100
    return s, out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    rows = []
    for rd in sorted(Path(args.results).glob("seq-*/")):
        if not (rd / "summary.json").exists() or not list((rd / "eval").glob("task_*.npz")):
            continue
        s = json.loads((rd / "summary.json").read_text())
        if s.get("status") != "complete" or s["method"] not in ("er", "derpp"):
            continue
        s, res = evaluate_run(rd)
        cfg = yaml.safe_load((rd / "config.yaml").read_text())
        rows.append(((s["dataset"], s["method"], s["buffer_size"], variant(cfg)), s["seed"], res))

    lines = [f"# RQ1: post-hoc calibration vs. failure detection on old classes ({len(rows)} runs)", "",
             "Pooled old-class test samples after the last task; mean over seeds. ECE, AUROC, AURC, "
             "sel. risk @90 % coverage in %.", ""]
    groups = defaultdict(list)
    for key, seed, res in rows:
        groups[key].append(res)
    lines += ["| dataset | method | buffer | variant | n | acc old | calibrator | conf old | ECE old | AUROC old | AURC old | sel90 old | AUROC all |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, rs in sorted(groups.items()):
        for c in ALL:
            mean = lambda m: np.mean([r[c][m] for r in rs]) if m in rs[0][c] else float("nan")
            lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {len(rs)} | "
                         f"{np.mean([r['_acc_old'] for r in rs]):.1f} | {c} | {mean('conf_old'):.1f} | "
                         f"{mean('ece_old'):.2f} | {mean('auroc_old'):.2f} | {mean('aurc_old'):.2f} | "
                         f"{mean('sel90_old'):.2f} | {mean('auroc_all'):.2f} |")

    lines += ["", "## Pre-registered decision (paired over all runs vs. `none`)", ""]
    supported = 0
    for c in LEGAL + ("oracle_ts", "oracle_task_ts"):
        ece0 = np.array([r["none"]["ece_old"] for _, _, r in rows])
        ece1 = np.array([r[c]["ece_old"] for _, _, r in rows])
        dau = np.array([r[c]["auroc_old"] - r["none"]["auroc_old"] for _, _, r in rows])
        rel = np.mean((ece0 - ece1) / ece0) * 100
        p_gt1 = stats.ttest_1samp(dau, 1.0, alternative="greater").pvalue
        calib_ok, detect_flat = rel >= 25, not (p_gt1 < 0.05)
        verdict = "decoupled" if calib_ok and detect_flat else ("detection improves" if p_gt1 < 0.05 else "no calibration gain")
        if c in LEGAL and calib_ok and detect_flat:
            supported += 1
        lines.append(f"- `{c}`: ECE_old relative reduction {rel:+.1f} % (runs improved: "
                     f"{(ece1 < ece0).sum()}/{len(rows)}); AUROC_old change {dau.mean():+.2f} "
                     f"(range {dau.min():+.2f} .. {dau.max():+.2f}; p(change > 1) = {p_gt1:.3f}) -> {verdict}"
                     + ("  [upper bound, not a method]" if c.startswith("oracle") else ""))
    lines.append(f"\nClaim 'calibration and failure detection come apart': "
                 f"{'SUPPORTED' if supported >= 2 else 'NOT SUPPORTED'} ({supported}/3 legal calibrators)")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
