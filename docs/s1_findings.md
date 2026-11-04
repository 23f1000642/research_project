# S1 H1 pilot: findings (seq-CIFAR-100, 10 × 10 classes, ER / DER++, buffers 500 / 2000, seeds 0-2)

Kaggle 2xT4, commit `612bbad`, amp, constant lr. 12/12 runs complete. Tables: `tables/summary.md`;
figure: `figures/h1_seq-cifar100_s1.png`; test: `analysis/h1_test.py` (rule committed in `0b21620`
before these results were seen).

## Accuracy (class-IL, %, mean ± sd over 3 seeds)

| method | buffer 500 | buffer 2000 | minutes/run (T4, alone) |
|---|---|---|---|
| ER | 21.84 ± 1.98 | 39.11 ± 0.97 | 30 |
| DER++ | 36.91 ± 1.55 | 52.06 ± 1.44 | 50 |

Not yet checked against a published seq-CIFAR-100 table (to do before S2).

## Pre-registered H1 test: NOT SUPPORTED

| group | CER@0.9 slope (points / task of age) | AUROC slope |
|---|---|---|
| DER++ 500 | −1.07 | −3.83 |
| DER++ 2000 | −1.39 | −2.25 |
| ER 500 | +0.46 | −10.69 |
| ER 2000 | +0.62 | −5.94 |
| pooled (12 runs) | −0.34, p = 0.86 | −5.68, p = 0.0001 |

H1 had two parts. The rule required both. The result:
- **Part 1 is false.** Old-task errors do not become *more* confident as the task ages. The share of
  errors made at > 0.9 confidence is flat for ER (≈ 45–55 %) and slowly *falls* for DER++ (≈ 19 → 12 %).
- **Part 2 holds, but as a step, not a trend.** Error-detection AUROC drops sharply the first time a
  task becomes old (age 0 → 1: ER-500 89 → 61, ER-2000 89 → 74, DER++-500 87 → 76, DER++-2000 88 → 81)
  and then stays roughly flat. The negative slopes come almost entirely from that first step.

## What is established by these runs (descriptive, 3 seeds)

After the last task, on old tasks only:

| group | old acc | old mean conf | ECE old | ECE old after TS | AUROC old (MSP) | AUROC old (energy) |
|---|---|---|---|---|---|---|
| ER 500 | 14.5 | 83.0 | 68.5 | 49.0 | 52.7 | 63.3 |
| ER 2000 | 33.8 | 83.1 | 49.3 | 36.0 | 69.8 | 73.8 |
| DER++ 500 | 32.0 | 51.2 | 19.6 | 26.6 | 72.6 | 71.7 |
| DER++ 2000 | 49.1 | 64.7 | 16.0 | 16.7 | 79.8 | 79.0 |

- Old tasks are overconfident for every method (gap 16–69 points), current tasks are not.
- Error detection on old tasks is much worse than on the current task (AUROC 53–80 vs. 85–90).
- Temperature scaling (one T fitted on buffer + current-task validation) helps ER but makes DER++-500
  *worse* on old tasks (ECE 19.6 → 26.6): a single global temperature cannot fix old and new tasks
  at once. This is direct support for the motivation of task-aware calibration (T-CIL) and FACE.

## Consequence

By the decision rule in `phase2_method.md`, H1 as stated is false. Changing H1 after seeing the data
would be a post-hoc hypothesis and must be labelled as such.

**Decision (with the user, after S1):** do not switch to D2 yet. Re-focus on the part that held
(old-task error detection collapses at the first task switch; one global temperature does not fix
it), state openly in the paper that this focus was chosen after S1, and run a small FACE pilot
(S2a, `experiments/plans/s2a_face_pilot.yaml`) with a go/no-go rule fixed in advance in
`analysis/h3_pilot_test.py`. NO-GO there means switching to D2.
