"""Pre-registered confirmation of the post-hoc RQ1 findings (docs/rq1_findings.md), written before
the S3 runs exist. Uses ONLY the runs of experiments/plans/s3_confirm.yaml (new seeds for ER /
DER++, and MIR / iCaRL, never analysed before), selected by run id.

Calibrators fitted on the legal calibration set (non-probe buffer + current-task validation).
All failure-detection metrics are on ALL seen classes after the last task (old-class-only AUROC is
confounded for task-aware calibrators, see docs/rq1_findings.md).

C1 global TS does not change failure detection: mean AUROC_all change within +-1.0 point,
   shown by equivalence testing (two one-sided t-tests, each p < 0.05).
C2 task-wise TS fitted on the buffer worsens calibration: ECE_all above uncalibrated MSP in at
   least 75 % of runs and binomial (sign) test against 50 %, one-sided p < 0.05.
C3 recency bias predicts when task-wise TS helps detection: Spearman correlation between the share
   of old-class errors predicted as a current-task class and the task-wise TS gain in AUROC_all is
   positive with one-sided p < 0.05.
C4 even an oracle global temperature barely helps old-class failure detection: oracle TS (fitted
   on the old-class test data) lowers ECE_old by >= 50 % on average while raising AUROC_old by
   < 3.0 points on average (MSP-type score, old-class AUROC is valid here).
Each claim is reported separately as CONFIRMED / NOT CONFIRMED. No pooling with earlier runs.

    python analysis/confirm_test.py results
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]
import rq1_calibration as R  # noqa: E402
from recency import recency_share  # noqa: E402
from run import expand_plan  # noqa: E402
from facecl.config import load_config  # noqa: E402
from facecl.training.trainer import run_id_for  # noqa: E402

PLAN = ROOT / "experiments" / "plans" / "s3_confirm.yaml"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="results")
    args = ap.parse_args(argv)

    wanted = [run_id_for(load_config(d, m, ov)) for d, m, ov in expand_plan(PLAN)]
    rows, missing = [], []
    for rid in wanted:
        rd = Path(args.results) / rid
        if not (rd / "summary.json").exists() or not list((rd / "eval").glob("task_*.npz")):
            missing.append(rid)
            continue
        share, _ = recency_share(rd)
        rows.append((rid, share, R.evaluate_run(rd)[1]))
    print(f"{len(rows)}/{len(wanted)} planned runs found")
    if missing:
        print("missing:", *missing, sep="\n  ")
        print("DECISION: not possible yet")
        return

    print("\n| run | recency share | AUROC_all none / ts / task_ts | ECE_all none / ts / task_ts |\n|---|---|---|---|")
    for rid, share, r in rows:
        print(f"| {rid} | {share:.1f} | {r['none']['auroc_all']:.1f} / {r['ts']['auroc_all']:.1f} / "
              f"{r['task_ts']['auroc_all']:.1f} | {r['none']['ece_all']:.1f} / {r['ts']['ece_all']:.1f} / "
              f"{r['task_ts']['ece_all']:.1f} |")

    get = lambda c, m: np.array([r[c][m] for _, _, r in rows])
    # C1
    d1 = get("ts", "auroc_all") - get("none", "auroc_all")
    p_lo = stats.ttest_1samp(d1, -1.0, alternative="greater").pvalue
    p_hi = stats.ttest_1samp(d1, 1.0, alternative="less").pvalue
    c1 = max(p_lo, p_hi) < 0.05
    # C2
    worse = int((get("task_ts", "ece_all") > get("none", "ece_all")).sum())
    p2 = stats.binomtest(worse, len(rows), 0.5, alternative="greater").pvalue
    c2 = worse / len(rows) >= 0.75 and p2 < 0.05
    # C3
    share = np.array([s for _, s, _ in rows])
    gain = get("task_ts", "auroc_all") - get("none", "auroc_all")
    rho, p3 = stats.spearmanr(share, gain, alternative="greater")
    c3 = rho > 0 and p3 < 0.05
    # C4
    rel = np.mean((get("none", "ece_old") - get("oracle_ts", "ece_old")) / get("none", "ece_old")) * 100
    d4 = (get("oracle_ts", "auroc_old") - get("none", "auroc_old")).mean()
    c4 = rel >= 50 and d4 < 3.0

    v = lambda ok: "CONFIRMED" if ok else "NOT CONFIRMED"
    print(f"\nC1 global TS, AUROC_all change {d1.mean():+.2f} (TOST p = {max(p_lo, p_hi):.4f}): {v(c1)}")
    print(f"C2 task-wise TS worsens ECE_all in {worse}/{len(rows)} runs (sign test p = {p2:.4f}): {v(c2)}")
    print(f"C3 Spearman(recency share, task-wise TS gain) = {rho:+.2f} (p = {p3:.4f}): {v(c3)}")
    print(f"C4 oracle global TS: ECE_old {rel:+.1f} % lower, AUROC_old {d4:+.2f}: {v(c4)}")


if __name__ == "__main__":
    main()
