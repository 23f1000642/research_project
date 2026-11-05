# S2a FACE pilot: NO-GO (seq-CIFAR-100, buffers 500 / 2000, seeds 0-2)

Kaggle 2xT4, commit `61d12a3`, 6 FACE runs (~63 min each on a T4 vs. ~50 for DER++). DER++ runs from
S1, paired by seed. Rule: `analysis/h3_pilot_test.py`, committed before these runs.

## Pre-registered test

| buffer | old-task AUROC: FACE score | DER++ best score (MSP) | diff per seed | accuracy FACE vs DER++ |
|---|---|---|---|---|
| 500 | 62.08 | 72.64 | −13.96, −12.25, −5.50 | 35.27 vs 36.91 |
| 2000 | 74.88 | 79.84 | −7.53, −3.99, −3.36 | 51.00 vs 52.06 |

Criteria 1 (≥ +1 AUROC) and 2 (all-task AUROC not worse) fail at both buffer sizes; criterion 3
(accuracy within 2 points) passes. **Decision: NO-GO, switch to backup idea D2.**

The FACE score is worse than plain MSP *on the same FACE models* (−8.4 and −4.6 points), so the
forgetting head itself hurts; it is not only the smaller replay buffer.

## Diagnosis (after the decision; does not change it)

From the saved test-set head outputs h after the last task:
- h is high everywhere, including the current task (mean ≈ 0.5–0.65 on old *and* current tasks), with
  a large spread (sd ≈ 0.3). Multiplying MSP by (1 − h) therefore injects noise into the ranking.
- Buffer 500: the probe has only 41–46 items; h carries no error signal (h on correct vs. wrong items
  0.62–0.65 vs. 0.65–0.71; error-detection AUROC of h alone 0.49–0.58). This is weakness #2 of
  `phase2_method.md` (too few probe items) appearing as predicted.
- Buffer 2000 (181–193 probe items): h does carry signal (AUROC of h alone 0.66–0.71, higher on
  wrong items), but it is weaker than MSP (0.79–0.80) and only moderately correlated with it
  (Spearman −0.43 to −0.50), and the product is worse than MSP alone.

No implementation error was found. A redesigned FACE (larger probe, learned combination instead of
the product) would be a new, post-hoc hypothesis and is not pursued now.

## What remains usable for a paper

S1 + S2a together are a negative result with a clear measurement: old-task error detection drops
sharply at the first task switch, one global temperature cannot repair it, and a learned
sample-level forgetting estimate from a memory-fair probe split is too noisy to help at buffer
sizes 500–2000.
