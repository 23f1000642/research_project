"""Pre-registered go/no-go test for the FACE pilot (S2a), written before any FACE run on CIFAR-100.

Context (docs/s1_findings.md): H1 as stated was not supported. What did hold is that
misclassification-detection AUROC on old tasks drops sharply once a task is old. FACE is meant to
restore exactly that, so the pilot tests H3 on that quantity only. This re-focusing was decided
after seeing the S1 results and will be reported as such.

Setting: seq-CIFAR-100, buffers 500 and 2000, seeds 0-2, DER++ vs. FACE (= DER++ + probe split +
forgetting head, same alpha/beta/lr, same total buffer). Same seed = same class order, so runs are
paired by seed.

Primary metric: auroc_old = error-detection AUROC on old-task test data after the last task.
Baseline score: for each buffer size, the best of {msp, entropy, maxlogit, energy, msp_ts} on the
DER++ runs, chosen by its mean over seeds (this favours the baseline).

GO (continue with FACE) iff, for BOTH buffer sizes:
  1. FACE score (msp * (1 - h)) on FACE runs beats that baseline by >= 1.0 AUROC point on average,
     and in at least 2 of the 3 seed pairs;
  2. auroc_all (all seen tasks) of the FACE score is not more than 0.5 points below the baseline's;
  3. FACE's final class-IL accuracy is at most 2.0 points below DER++'s on average.
Otherwise NO-GO: stop FACE and switch to backup idea D2. Three seeds give no significance claim;
this is a decision rule only.

    python analysis/h3_pilot_test.py results
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BASELINE_SCORES = ("msp", "entropy", "maxlogit", "energy", "msp_ts")
DATASET = "seq-cifar100"
BUFFERS = (500, 2000)


def load(root):
    runs = defaultdict(dict)  # (method, buffer) -> seed -> metrics
    for d in sorted(Path(root).glob(f"{DATASET}-*/")):
        if not (d / "metrics.json").exists():
            continue
        s = json.loads((d / "summary.json").read_text())
        if s.get("status") == "complete" and s["method"] in ("derpp", "face"):
            runs[(s["method"], s["buffer_size"])][s["seed"]] = json.loads((d / "metrics.json").read_text())
    return runs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="results")
    args = ap.parse_args(argv)
    runs = load(args.results)

    go, complete = True, True
    for buf in BUFFERS:
        der, face = runs.get(("derpp", buf), {}), runs.get(("face", buf), {})
        seeds = sorted(set(der) & set(face))
        print(f"\n=== buffer {buf}: paired seeds {seeds}")
        if len(seeds) < 3:
            print("  fewer than 3 paired seeds: cannot decide")
            complete = False
            continue
        f = lambda runs_, score, key: np.array([runs_[s]["final"][score][key] for s in seeds]) * 100
        means = {sc: f(der, sc, "auroc_old").mean() for sc in BASELINE_SCORES}
        best = max(means, key=means.get)
        base_old, base_all = f(der, best, "auroc_old"), f(der, best, "auroc_all")
        face_old, face_all = f(face, "face", "auroc_old"), f(face, "face", "auroc_all")
        face_msp_old = f(face, "msp", "auroc_old")
        acc_der = np.array([der[s]["accuracy"]["final_avg_acc"] for s in seeds]) * 100
        acc_face = np.array([face[s]["accuracy"]["final_avg_acc"] for s in seeds]) * 100

        print("  DER++ auroc_old by score: " + ", ".join(f"{k} {v:.2f}" for k, v in means.items())
              + f"  -> baseline = {best}")
        diff = face_old - base_old
        print(f"  auroc_old  FACE {face_old.mean():.2f} vs {best} {base_old.mean():.2f}: "
              f"diff {diff.mean():+.2f} per seed {np.round(diff, 2).tolist()}")
        print(f"  (same FACE models scored with plain MSP: {face_msp_old.mean():.2f}, "
              f"i.e. head adds {(face_old - face_msp_old).mean():+.2f})")
        print(f"  auroc_all  FACE {face_all.mean():.2f} vs {base_all.mean():.2f}: diff {(face_all - base_all).mean():+.2f}")
        print(f"  accuracy   FACE {acc_face.mean():.2f} vs DER++ {acc_der.mean():.2f}: diff {(acc_face - acc_der).mean():+.2f}")
        for key in ("aurc_old", "sel_risk_90_old"):  # secondary, lower is better
            print(f"  {key:15s} FACE {f(face, 'face', key).mean():.2f} vs {f(der, best, key).mean():.2f}")

        c1 = diff.mean() >= 1.0 and (diff > 0).sum() >= 2
        c2 = (face_all - base_all).mean() >= -0.5
        c3 = (acc_face - acc_der).mean() >= -2.0
        print(f"  criteria: 1 {'pass' if c1 else 'FAIL'}, 2 {'pass' if c2 else 'FAIL'}, 3 {'pass' if c3 else 'FAIL'}")
        go &= c1 and c2 and c3

    print(f"\nDECISION: {'GO (continue with FACE)' if go else 'NO-GO (switch to D2)'}")


if __name__ == "__main__":
    main()
