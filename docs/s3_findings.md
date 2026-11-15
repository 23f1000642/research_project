# S3 findings: pre-registered confirmation of the post-hoc RQ1 points

Rule: `analysis/confirm_test.py`, committed in `0a2995e` before any S3 run existed. Plan:
`experiments/plans/s3_confirm.yaml` (14 runs, all complete, no failures). Output: `tables/s3_confirm.md`.
Only runs never analysed before: ER / DER++ seeds 3-4 at buffers 500 / 2000; MIR and iCaRL at
buffer 2000, seeds 0-2. No pooling with earlier runs.

Final accuracy (seq-CIFAR-100, buffer 2000): MIR 39.8 +- 1.3, iCaRL 44.7 +- 0.8 (new methods).

## Decision: 1 of 4 claims confirmed

| claim | result | verdict |
|---|---|---|
| C1 global TS leaves AUROC_all within +-1 | mean change +1.02 (TOST p = 0.53) | NOT CONFIRMED |
| C2 task-wise TS worsens ECE_all vs. uncalibrated in >= 75 % of runs | 4 / 14 runs | NOT CONFIRMED |
| C3 recency share predicts task-wise TS gain in AUROC_all | Spearman rho = +0.95, p < 1e-4 | **CONFIRMED** |
| C4 oracle TS: ECE_old -50 % or more while AUROC_old < +3 | ECE_old -83 %, AUROC_old +3.04 | NOT CONFIRMED (by 0.04) |

## Reading (descriptive, not new claims)

* **C3 replicates on new seeds and on two new methods.** Recency share spans 10-13 % (iCaRL,
  nearest-mean classifier) to 71-92 % (ER, MIR); task-wise TS gains +12 to +22 AUROC_all at high share and
  about +1 at iCaRL's low share; DER++ in between. This is the one confirmed finding.
* **C1 failed narrowly and the direction is informative.** Global TS changed AUROC_all by
  0.0 for DER++ and +1.2 to +1.9 for ER / MIR / iCaRL. So the effect is small but not zero, and
  the pre-registered +-1 equivalence margin was too tight for that.
* **C2 rested on a miscount.** `docs/rq1_findings.md` had said task-wise TS raised ECE_all in 22 / 25
  earlier runs vs. uncalibrated. A recount gives 13 / 25 (corrected in that file). In S3, task-wise TS
  lowers ECE for ER / MIR (whose uncalibrated ECE is 44-65) and raises it for DER++. What holds in
  every run, old and new (25 / 25 and 14 / 14), is that task-wise TS calibrates **worse than global TS**.
  That comparison was not pre-registered, so it stays exploratory.
* **C4 missed by 0.04 points.** An oracle global temperature removes most of the old-class
  calibration error (-83 %) but gains only about 3 AUROC points. The qualitative point stands; the
  pre-registered threshold was not met, and we report it as not met.

## Consequence for the paper

Main confirmed result: the benefit of task-aware post-hoc calibration for failure detection in
class-incremental learning is governed by recency bias (confirmed on unseen seeds and methods). The
other points are reported with their pre-registered verdicts, as descriptive context.
