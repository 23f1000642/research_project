# Internal hostile review of manuscript draft v0.1 (2026-11-16)

Reviewer stance: reject unless each point is answered with data or an honest limitation.
Checks run on saved logits (0 GPU). Resulting draft: v0.2.

| # | Attack | Check / evidence | Status |
|---|---|---|---|
| 1 | **Related work misdescribed.** v0.1 said recent CIL calibrators "fit temperatures per task on the replay buffer". | Read MT-CC, DATS, T-CIL (arXiv HTML). T-CIL: one global T on adversarially perturbed exemplars (rejects plain exemplars). DATS: per-class distance-based T, fitted on a reserved slice of each task's validation set. MT-CC: two groups routed by a learned network on old/new class mass, fitted on T-CIL-style validation. None fits per-task T on the buffer; none reports AUROC/AURC. | **Fixed**: intro, related work, limitations rewritten; our calibrator presented as a design-choice probe, not as these methods. |
| 2 | Strawman: buffer-fitted task-TS is a design nobody uses. | Added held-out emulation (`heldout`, Table heldout): per-task T fitted on half of each task's test set. True-task grouping fails (ER AUROC 73.1 -> 54.7); **predicted-task grouping gives ER 73.1 -> 87.7, DER++ 78.4 -> 83.0, ECE close to global TS**. 5-10 items/class suffice (Table size). | **Reframed** as finding (iv), post hoc. Needs confirmation with a proper protocol (see open item A). |
| 3 | Pooled correlation driven only by method identity (Simpson). | Within-method Spearman: ER +0.89, DER++ +0.91; per dataset +0.98 / +0.96. | Answered (Sec. robustness). |
| 4 | Recency share not comparable across datasets (chance 22 % vs 10 %). | share / chance: rho +0.95 (confirmatory). | Answered. |
| 5 | AUROC only; AURC could disagree. | Spearman(share, AURC improvement) +0.91 confirmatory; AUROC/AURC change corr -0.98. | Answered. |
| 6 | Two-group (MT-CC-like) variant might behave differently. | old/new TS gains correlate +1.00 with task-TS; worse ECE than global TS in 39/39. | Answered. |
| 7 | C3 is near-tautological given the mechanism. | Acknowledged in text: C3 follows from the mechanism; the value is the diagnosis plus rule baseline plus remedy. | Addressed in framing. |
| 8 | Novelty of "predicted task as failure signal". | Limited search (arXiv queries) found nothing; stated as limited search. | Partially answered; re-check before submission. |
| 9 | Held-out emulation uses test data and extra memory. | Stated as limitation; fair test must charge memory to the buffer. | Open item A. |
| 10 | C2 rested on a miscount. | Corrected in docs/rq1_findings.md and disclosed in paper. | Disclosed. |
| 11 | Accuracies on seq-CIFAR-100 not compared with a published table. | Only DER++ seq-CIFAR-10 checked (71.2 vs 72.7). | Open item B. |
| 12 | Small n (2-3 seeds per cell), single backbone, CIFAR only. | Limitation. | Open. |
| 13 | Appendix numbers typed by hand. | Size table now generated; accuracy paragraph still copied. | Open (minor). |

## Open items
- **A (GPU, recommended before submission):** pre-registered confirmation of finding (iv) with a fair
  protocol: reserve held-out items per task during training (e.g. 10 per class), charge them to the
  memory budget (buffer reduced by the same number), save logits of held-out items, and compare
  predicted-task-grouped TS vs global TS vs MSP on new seeds.
- **B:** compare seq-CIFAR-100 accuracies with a published Mammoth table.
- **C:** generate the accuracy appendix from `analysis/summarize.py`.
- **D:** compile on Overleaf and fix layout.
