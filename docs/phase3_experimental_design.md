# Phase 3 — Experimental design

Date: 2026-10-07. Status: **plan only. Runtimes below are estimates and will be replaced by measured timings from the Phase 4 pilot.**

Compute: Kaggle free tier, about 30 GPU-hours/week (2×T4 or 1×P100), sessions up to ~12 h (re-check on the account page). With 2×T4 we run **two runs in parallel, one per GPU**.

---

## 1. Staged plan with go/no-go gates

| Stage | Purpose | Gate to continue |
|---|---|---|
| **S0 Sanity** | Split MNIST + seq-CIFAR-10 with ER/DER++; reproduce published DER++ class-IL accuracy within ~2 points | If baselines do not reproduce, fix code before anything else |
| **S1 H1 pilot** | Does "confident forgetting" exist? seq-CIFAR-100, ER & DER++, buffers 500/2000, 3 seeds | H1 supported (old-task CER rises / AUROC drops with task age) → S2. Otherwise write up measurement and pivot to D2 |
| **S2 Main** | Full baseline comparison + FACE on 3 datasets | — |
| **S3 Ablations & sensitivity** | Which parts of FACE matter | — |
| **S4 Robustness** | Domain-IL control, longer sequences, frozen pretrained backbone | — |

## 2. Datasets and task sequences (why each one)

| Dataset | Split | Role | Why |
|---|---|---|---|
| Split MNIST | 5 tasks × 2 classes | **Debugging only** | Too easy to be evidence; fast unit test of the pipeline |
| Seq-CIFAR-10 | 5 × 2 | Main (small) | Standard; direct comparison with DER++ paper |
| Seq-CIFAR-100 | 10 × 10 (also 5 × 20, 20 × 5 in S4) | **Main** | Standard, harder, many tasks → long "task age" for H1 |
| Seq-TinyImageNet | 10 × 20 | Main (scale) | Larger images/classes; reviewers expect it |
| Rotated MNIST (domain-IL) | 20 rotations | **Negative control** | No new classes → no recency leak → FACE should help *less*. Confirms mechanism |
| Frozen ViT features on CIFAR-100 | 10 × 10 | Robustness | Pretrained models forget less; tests whether the effect survives. Caveat: ImageNet pretraining overlaps CIFAR classes |
| SVHN | — | OOD set (optional) | Out-of-distribution detection check |

Permuted MNIST is dropped: it is a domain-IL benchmark already covered by Rotated MNIST, and it is criticised as unrealistic. DomainNet is dropped for compute reasons (noted as a limitation).

**Class order:** each seed defines a different random class order (seed 0..4). Every method sees the same order for the same seed → paired comparisons.

## 3. Train / validation / test protocol (leakage prevention)

- Each task's official training set → **90% train / 10% validation** (stratified, fixed per seed).
- Official test sets are **only** used for final reporting. Never for tuning, early stopping or calibration.
- TinyImageNet: official `val` split is used as the test set (labels for official test are not public).
- Validation of old tasks is **not** available after the task ends (realistic CIL). Post-hoc calibration methods (TS, T-CIL) and FACE get the **same** data: replay buffer + current-task validation.
- No task id at test time (class-IL). Task-IL numbers reported only as a secondary table.
- Hyperparameter tuning uses a separate seed (seed 100) and validation data only.
- **Test logits saved**: after each task, save fp16 logits of the model on every seen task's test set. All uncertainty metrics are computed offline from these files, so adding a new post-hoc baseline never needs a rerun and every number traces to a saved file.

## 4. Model and training

- Backbone: reduced ResNet-18 (standard in CL: DER, Mammoth) for CIFAR/TinyImageNet; small MLP for MNIST.
- Optimiser SGD, constant lr (DER protocol), batch 32 + replay batch 32. Epochs/task: 50 (CIFAR), TinyImageNet epochs to be fixed after timing pilot (same for every method). Mixed precision (AMP) on.
- Buffer: reservoir sampling. Sizes 500 and 2000 (main); 200 and 5000 in sensitivity.

## 5. Methods compared

**Accuracy baselines (retrain):**
- Lower/upper bounds: Fine-tune, Joint (all data at once)
- Regularisation: EWC (online), SI, MAS
- Distillation: LwF
- Replay: ER, DER++, iCaRL, MIR, IDER (if official code runs)
- Ours: FACE (on DER++), FACE-PR, FACE on ER (needs stored logits: +400 B/sample, reported)

**Uncertainty scores (post-hoc, applied to the *same* checkpoint):** MSP, entropy, max-logit, energy, temperature scaling (buffer-validated), T-CIL, MC-dropout (dropout-trained variant), deep ensemble (5 members, CIFAR-100 buffer 2000 only, as a 5×-cost reference), FACE-Snap (upper-bound reference).

**Fairness rule:** FACE's uncertainty is compared with the other scores on both (a) the FACE checkpoint and (b) a plain DER++ checkpoint with the full buffer, so we separate "better score" from "different training".

**Hyperparameters:** baselines use the authors' published best configs for matching settings; FACE inherits DER++'s α, β unchanged and tunes only its own (ρ, head lr, γ) with ≤ 8 configs on seed 100 validation. Same tuning budget cap for any baseline we tune.

## 6. Metrics

Let $a_{t,j}$ = accuracy on task $j$'s test set after training task $t$ (class-IL, over all seen classes).

| Metric | Formula / definition |
|---|---|
| Final average accuracy | $A_T = \frac1T\sum_j a_{T,j}$ |
| Average incremental accuracy | $\bar A = \frac1T \sum_t A_t$ |
| Forgetting | $F = \frac{1}{T-1}\sum_{j<T} \max_{t<T}(a_{t,j} - a_{T,j})$ |
| Backward transfer | $\mathrm{BWT} = \frac{1}{T-1}\sum_{j<T}(a_{T,j} - a_{j,j})$ |
| Forward transfer | Reported for domain-IL only (in class-IL, unseen classes have no output, so FWT is degenerate) |
| ECE (15 bins), adaptive ECE, NLL, Brier | per task $j$ and time $t$ |
| Misclassification-detection AUROC | positives = errors, score = uncertainty |
| AURC / E-AURC | area under risk–coverage curve |
| Selective risk @ 80% / 90% coverage | error rate on the most-confident 80/90% |
| Confident-error rate CER@0.9 | fraction of errors with confidence > 0.9 |
| **Uncertainty forgetting (new)** | $\mathrm{UF} = \frac{1}{T-1}\sum_{j<T}\max_{t<T}(\mathrm{AUROC}_{t,j}-\mathrm{AUROC}_{T,j})$ — the forgetting formula applied to failure detection |
| OOD AUROC (optional) | SVHN vs. seen CIFAR test |
| Efficiency | parameters, buffer bytes, peak GPU memory, train time/task, inference ms/image |

## 7. Ablations (S3)

| ID | Question | Settings |
|---|---|---|
| A1 | Does the probe split matter? | ρ ∈ {0 (head trained on replay items), 0.1, 0.2, 0.3} |
| A2 | Is the product rule needed? | product · logistic calibrator · head score alone |
| A3 | Head input | features · features + logit statistics |
| A4 | Target | soft drop $r$ · binary "forgotten" |
| A5 | Prioritised replay | FACE vs FACE-PR, γ ∈ {0.5, 1, 2} vs MIR |
| A6 | Base learner | FACE on DER++ vs on ER |
| A7 | Cheap vs expensive | FACE vs FACE-Snap |

**Sensitivity:** buffer {200, 500, 2000, 5000}; sequence length (CIFAR-100 in 5/10/20 tasks); head lr and width.

## 8. Seeds, runs and statistics

- Main tables: **5 seeds** (0–4). FACE vs DER++ on seq-CIFAR-100 buffer 2000: **10 seeds** (cheap, most important comparison).
- Ablations, TinyImageNet, robustness: 3 seeds.
- Report mean ± std and 95% t-confidence intervals.
- Tests: **paired t-test** across seeds (same seed = same class order and init). Note: Wilcoxon with n = 5 can never reach p < 0.05 two-sided, which is one reason we use 10 seeds for the key comparison.
- Within a run, bootstrap (1000 resamples of test samples) for AUROC/AURC CIs.
- Multiple comparisons: Holm–Bonferroni across the baselines in each table. Report effect sizes, not only p-values.
- A claim "FACE improves X" is written only if the corrected p < 0.05 **and** the improvement holds on ≥ 2 of 3 datasets.

## 9. Complexity and memory (expected; to be measured)

- FACE extra parameters: MLP head (e.g., 512→128→1) ≈ 66k params ≈ 0.6% of ResNet-18 (~11M).
- Extra memory: none beyond DER++ (probe items are carved from the same buffer).
- Extra compute: one no-grad forward on a probe batch per step + head update; expected ≈ +10–20% train time (to be measured). Inference: one extra MLP call (negligible).
- FACE-Snap: +1 ResNet-18 (~45 MB) per task — reported separately as not memory-fair.

## 10. Compute budget (rough estimate — verify in pilot)

Assumed per-run time on one T4 with AMP: seq-CIFAR-10/100 ≈ 25–40 min; TinyImageNet ≈ 1.5–3 h.

| Block | Runs | Est. GPU-h |
|---|---|---|
| S0 sanity | ~6 | 3 |
| S1 H1 pilot | 2 methods × 2 buffers × 3 seeds = 12 | 6–8 |
| S2 CIFAR-10 + CIFAR-100 | (6 non-buffer × 3 seeds) + (7 buffer methods × 2 buffers × 5 seeds) ≈ 88 per dataset + 10 extra seeds | ≈ 80–120 |
| S2 TinyImageNet | 4 methods × 1 buffer × 3 seeds = 12 | 20–35 |
| S3 ablations | ≈ 50 short runs | 25–35 |
| S4 robustness | ≈ 30 runs (MNIST + ViT features cheap) | 10–15 |
| **Total** | | **≈ 150–220 GPU-h ≈ 3–4 weeks** with two parallel runs on 2×T4 |

If over budget: cut regularisation baselines to 3 seeds on CIFAR-10 only (they are known to be near fine-tune in class-IL), then drop buffer 500 for TinyImageNet.

## 11. Reproducibility

Fixed seeds (python, numpy, torch, cuda), `cudnn.deterministic` for final runs; every run writes its config, git commit hash, library versions, accuracy matrix, saved test logits and wall-clock times to `results/<run_id>/`. An experiment log (`logs/experiment_log.md`) records every run, including failed and discarded ones.

## 12. Implementation choice

Own clean PyTorch codebase (readable, matches the requested structure), following the Mammoth/DER protocol, with baselines validated against published numbers in S0. Mammoth (aimagelab) is used as a reference for checking baseline behaviour.
