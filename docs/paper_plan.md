# Paper plan after the FACE NO-GO (2026-11-14)

## Why this direction

- FACE (D1) failed its pre-registered pilot (`docs/s2a_findings.md`); backup D2 is already covered by
  arXiv 2610.14258 (`docs/d2_literature.md`).
- Gap found in the literature: recent class-incremental calibration methods (T-CIL, CVPR 2025;
  DATS, arXiv 2509.21161; MT-CC, arXiv 2610.25689, NeurIPS 2026 per its arXiv comment) evaluate
  calibration only (ECE / AECE / NLL). None reports whether they help a model *detect its own
  errors* (misclassification-detection AUROC, AURC, selective risk) on old classes. NPCL (NeurIPS
  2023) and OpenCIL (2024) are closest for "uncertainty forgets" but use domain-IL or OOD detection.
- Working title: **"Calibrated but still blind: post-hoc calibration does not restore failure
  detection on old classes in class-incremental learning."**
- Chosen because it reuses all code and runs, and its main test needs **no GPU**: post-hoc
  calibrators only change confidences, so they can be evaluated on the saved logits.

## Research questions

- **RQ1** Do post-hoc temperature calibrators that are standard or recent in CIL improve error
  detection on old classes, or only calibration?
- **RQ2** How does old-class error detection evolve with task age (S1 showed a step at the first task
  switch for ER/DER++ on CIFAR-100; replicated across more methods and datasets later)?
- **RQ3** How much of the gap could any temperature-based method close at best (oracle temperatures
  fitted on the test data itself)?

## RQ1: pre-registered analysis (written before running it)

Data: every completed run with saved logits (ER, DER++, FACE-backbone excluded; seq-CIFAR-10
buffer 500, seq-CIFAR-100 buffers 500/2000; 3 seeds each; 23 runs at the time of writing; SGD runs
excluded because old-class accuracy is 0). State after the last task.

Calibrators. Each is fitted on data a CIL method may legally keep at the end: the non-probe replay
buffer plus the current task's validation split (same set as `evaluate.py`).
- `none`: MSP.
- `ts`: one global temperature (Guo et al. 2017).
- `task_ts`: one temperature per task, fitted on that task's calibration items; a test input gets the
  temperature of the task its predicted class belongs to (task-aware scaling as in DATS / MT-CC,
  simplified: grouping by predicted task, no learned grouping or distance model).
- `oldnew_ts`: two temperatures, old tasks vs. current task, same assignment rule (MT-CC-style
  simplification with two groups).
- `oracle_ts`, `oracle_task_ts`: the same two scalers fitted on the test data being evaluated
  (not a method; an upper bound for temperature scaling).
Temperatures do not change the argmax, so accuracy is identical for all calibrators.

Primary metrics, computed on the pooled old-class test samples (all tasks before the last):
ECE (15 bins) and error-detection AUROC. Secondary: AURC, selective risk at 90 % coverage; same
metrics on all seen classes.

Decision rule for the claim "calibration and failure detection come apart", for each legal
calibrator c in {ts, task_ts, oldnew_ts}, over runs (paired with `none`):
- calibration effect: mean relative ECE_old reduction >= 25 %, and
- detection effect: mean AUROC_old change < +1.0 point (one-sided paired t-test of change > 1.0 not
  significant at 0.05).
The claim is supported if this holds for at least 2 of the 3 legal calibrators. If a calibrator
improves AUROC_old by >= 1.0 point (significant), that is reported as a positive finding instead.

## Later (GPU, only if RQ1 holds)

Breadth on seq-CIFAR-100 buffer 2000, 3 seeds: add iCaRL, LwF, EWC (online), MIR, and Joint as
reference; re-run with `save_final_model` to evaluate T-CIL itself and MC-dropout. Estimated
~10 GPU-h. TinyImageNet only if budget allows.
