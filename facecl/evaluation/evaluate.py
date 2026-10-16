"""Offline evaluation of a finished run from its saved logits (no GPU, no retraining).

For every time step t (after task t) and seen task j <= t, and every confidence score:
  msp        max softmax probability
  entropy    negative predictive entropy (ranking only)
  maxlogit   largest logit (ranking only)
  energy     logsumexp of logits (ranking only)
  msp_ts     MSP after temperature scaling fitted on buffer replay items + current-task val
  face       MSP * (1 - h(x))           (FACE runs only)
writes results/<run_id>/metrics.json.

Usage:  python -m facecl.evaluation.evaluate results/<run_id> [more run dirs ...]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from ..utils import load_json, save_json
from . import metrics as M

PROB_SCORES = {"msp", "msp_ts", "face"}


def _calibration_set(d):
    """Data a post-hoc method may legitimately use at time t: replay buffer + current-task val."""
    parts_l, parts_y = [d["val_logits"].astype(float)], [d["val_y"].astype(int)]
    if "buf_logits" in d:
        keep = ~d["buf_probe"].astype(bool)
        parts_l.append(d["buf_logits"][keep].astype(float))
        parts_y.append(d["buf_y"][keep].astype(int))
    return np.concatenate(parts_l), np.concatenate(parts_y)


def _scores(logits, extras, temperature):
    probs = M.softmax_np(logits)
    out = {
        "msp": (probs.max(1), probs),
        "entropy": ((probs * np.log(np.clip(probs, 1e-12, 1))).sum(1), None),
        "maxlogit": (logits.max(1), None),
        "energy": (logsumexp(logits, 1), None),
    }
    pt = M.softmax_np(logits / temperature)
    out["msp_ts"] = (pt.max(1), pt)
    if "face_h" in extras:
        out["face"] = (probs.max(1) * (1.0 - extras["face_h"]), None)
    return out


def evaluate_run(run_dir) -> dict:
    run_dir = Path(run_dir)
    summary = load_json(run_dir / "summary.json")
    files = sorted((run_dir / "eval").glob("task_*.npz"))
    per_cell = {}  # score -> metric -> rows[t][j]
    temperatures = []
    for f in files:
        t = int(f.stem.split("_")[1])
        d = dict(np.load(f))
        cal_l, cal_y = _calibration_set(d)
        T_ = M.fit_temperature(cal_l, cal_y)
        temperatures.append(T_)
        n_seen_tasks = sum(1 for k in d if k.startswith("test") and k.endswith("_y"))
        for j in range(n_seen_tasks):
            logits = d[f"test{j}_logits"].astype(float)
            y = d[f"test{j}_y"].astype(int)
            extras = {k[len(f"test{j}_"):]: d[k] for k in d
                      if k.startswith(f"test{j}_") and k not in (f"test{j}_logits", f"test{j}_y")}
            correct = logits.argmax(1) == y
            for name, (conf, probs) in _scores(logits, extras, T_).items():
                res = M.all_quality_metrics(
                    conf, correct, probs=probs, logits=logits if name == "msp" else None, y=y,
                    prob_conf=name in PROB_SCORES)
                for metric, value in res.items():
                    rows = per_cell.setdefault(name, {}).setdefault(metric, [])
                    while len(rows) <= t:
                        rows.append([])
                    rows[t].append(value)

    result = {"run_id": summary["run_id"], "accuracy": M.cl_accuracy_metrics(summary["acc_class_il"]),
              "accuracy_task_il": M.cl_accuracy_metrics(summary["acc_task_il"]),
              "temperatures": temperatures, "per_task": per_cell, "final": {}, "uncertainty_forgetting": {}}
    for name, mets in per_cell.items():
        final = {}
        for metric, rows in mets.items():
            last = np.array(rows[-1], float)
            final[f"{metric}_all"] = float(np.nanmean(last))
            if len(last) > 1:
                final[f"{metric}_old"] = float(np.nanmean(last[:-1]))  # tasks before the last one
        result["final"][name] = final
        if "auroc" in mets:
            result["uncertainty_forgetting"][name] = M.forgetting_of(mets["auroc"])
    save_json(result, run_dir / "metrics.json")
    return result


if __name__ == "__main__":
    for path in sys.argv[1:]:
        r = evaluate_run(path)
        print(path, "final acc", round(r["accuracy"]["final_avg_acc"], 4),
              "| AUROC(old) msp", round(r["final"]["msp"].get("auroc_old", float("nan")), 4),
              *(["| face", round(r["final"]["face"].get("auroc_old", float("nan")), 4)]
                if "face" in r["final"] else []))
