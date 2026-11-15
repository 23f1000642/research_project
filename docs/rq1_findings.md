# RQ1 findings: post-hoc calibration vs. failure detection (saved logits, no GPU)

Pre-registration: `docs/paper_plan.md` (commit `89d7e9e`). Script: `analysis/rq1_calibration.py`.
Full table: `tables/rq1_calibration.md`. 25 runs (ER / DER++; seq-CIFAR-10 b500 in three training
variants, seq-CIFAR-100 b500 / b2000), state after the last task.

Side note: Windows Application Control on the local machine started blocking a scikit-learn DLL.
AUROC is now computed with the Mann-Whitney rank formula; it reproduces the Kaggle (scikit-learn)
values exactly (e.g. 0.526644 / 0.625258 for MSP / energy on one run) and is unit-tested against
the pairwise definition with ties.

## Pre-registered decision: NOT SUPPORTED (0 / 3 legal calibrators)

| calibrator | ECE_old relative change | AUROC_old change (mean, range) | verdict under the rule |
|---|---|---|---|
| global TS | −17 % (better in 20/25 runs) | +0.6 (−0.4 .. +2.8) | ECE gain below the 25 % threshold |
| task-wise TS | +50 % (worse) | +14.9 (−3.8 .. +39.9) | "detection improves" |
| old/new TS | +51 % (worse) | +14.9 (−4.5 .. +40.9) | "detection improves" |
| oracle global TS (bound) | −78 % | +2.0 | |
| oracle task-wise TS (bound) | −79 % | +17.6 | |

## The primary metric was flawed for task-aware calibrators (found after the decision)

AUROC_old pools only test samples whose *true* class is old. For a score that depends on the
*predicted* task, this conditioning leaks the answer: among true-old samples, every prediction in
the current task is an error, so "softer temperature for current-task predictions" becomes a
near-perfect error flag that is not available at deployment. The +15 points above are mostly this
artefact. For MSP and global TS (no dependence on the predicted task) the restriction is harmless.
Consequence for the paper: report failure detection on all seen classes, and treat old-class-only
AUROC as valid only for task-independent scores. This is itself a methodological point.

## All seen classes (valid for every calibrator), mean over seeds

| dataset / method / buffer | AUROC none | global TS | task-wise TS | ECE none | global TS | task-wise TS |
|---|---|---|---|---|---|---|
| C10 DER++ 500 (amp) | 80.2 | 80.2 | 67.1 | 16.9 | 11.0 | 23.6 |
| C10 ER 500 (amp+lr) | 78.2 | 78.8 | 84.4 | 35.1 | 27.2 | 33.7 |
| C100 DER++ 500 | 75.8 | 75.8 | 80.8 | 16.9 | 23.8 | 44.4 |
| C100 DER++ 2000 | 81.0 | 80.9 | 79.6 | 14.2 | 14.8 | 36.2 |
| C100 ER 500 | 67.3 | 69.0 | 88.0 | 62.4 | 44.3 | 50.6 |
| C100 ER 2000 | 74.0 | 75.2 | 86.2 | 45.1 | 32.6 | 42.2 |

What the data say (descriptive; 2–3 seeds per cell):
1. **Global TS: calibration and detection do come apart.** ECE usually improves, failure-detection
   AUROC moves by ≤ 1.7 points. Even an oracle global temperature fitted on the test data lowers
   old-class ECE by 78 % but raises AUROC by only 2 points.
2. **Task-aware TS fitted on the replay buffer calibrates worse than a single global temperature**
   (ECE_all above global TS in 25 of 25 runs; above uncalibrated MSP in only 13 of 25. CORRECTION:
   an earlier version of this note said "ECE up in 22 of 25 runs" vs. uncalibrated; that count was
   wrong, and the pre-registered claim C2 built on it failed in S3, see docs/s3_findings.md). Plausible reason, not tested: buffer items are training
   data the model has memorised, so per-task temperatures fitted on them are too sharp for old tasks.
   This is the problem T-CIL addresses with perturbed exemplars.
3. **Its effect on failure detection depends on the CL method**: large gains for ER (+6 to +21 AUROC),
   losses for DER++ on CIFAR-10 (−13) and at buffer 2000, a gain for DER++ at CIFAR-100 b500.
   Plausible reason, not tested: ER's errors are mostly old-class inputs predicted as current-task
   classes (recency bias), which a softer current-task temperature exposes; DER++ has much less
   recency bias.

## Status of the paper claim

The pre-registered blanket claim ("post-hoc calibrators do not help failure detection") is not
supported. What is supported is narrower and conditional: a single temperature does not help
failure detection (even an oracle one), and task-aware temperatures trade calibration for
detection in a method-dependent way. These three points come from post-hoc inspection and need
confirmation on new runs before they are claims.

## Follow-up: recency bias (exploratory, `analysis/recency.py`, `tables/recency.md`)

Share of old-class errors that predict a current-task class (uniform errors would give
k/(N−1): 22 % on CIFAR-10, 10 % on CIFAR-100): ER 72–88 %, DER++ 35–54 %. Across the 25 runs the
share almost perfectly orders the gain of task-wise TS in failure detection (Spearman ρ = +0.95).
This was found after looking at the data. It and the other post-hoc points are tested on new
runs only, with the rule in `analysis/confirm_test.py` (plan `experiments/plans/s3_confirm.yaml`:
ER / DER++ seeds 3–4 at buffers 500 / 2000, MIR and iCaRL at buffer 2000, seeds 0–2).
