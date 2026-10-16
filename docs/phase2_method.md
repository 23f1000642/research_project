# Phase 2 — Research question, hypotheses and method (FACE)

Date: 2026-10-07. Status: **design only — no experiments run yet. Nothing here is a result.**

Labels: **[E]** established in literature · **[H]** our hypothesis (to be tested) · **[S]** speculation.

---

## 1. Research question

> In class-incremental learning, does forgetting damage a model's ability to **recognise its own mistakes on old classes** — beyond what calibration methods repair — and can a **cheap, memory-fair estimate of sample-level forgetting** restore that ability?

## 2. Hypotheses

- **H1 (confident forgetting).** [H] As tasks accumulate, errors on old-task inputs become increasingly *high-confidence*: the fraction of old-task errors with max-softmax > 0.9 grows with task age, and misclassification-detection AUROC on old tasks drops.
- **H2 (calibration ≠ ranking).** [H] Post-hoc temperature methods (TS, T-CIL) reduce ECE but leave misclassification-detection AUROC / AURC almost unchanged, because they rescale confidence rather than reorder it. (Note: a single temperature does *not* provably preserve the MSP ranking across inputs, so this is an empirical claim to test, not a theorem.)
- **H3 (FACE helps).** [H] Discounting confidence by a learned forgetting estimate improves old-task failure detection (AUROC, AURC, selective risk) over MSP, entropy, MC-dropout, TS and T-CIL, at equal memory, with ≤ ~5% extra compute and no loss in accuracy.
- **H4 (optional, prioritised replay).** [H] Replaying samples with high estimated forgetting improves accuracy/forgetting over uniform replay and is competitive with MIR.

**Decision rule.** If H1 is false (no confident-forgetting effect on standard benchmarks), we stop the method work, report the measurement study, and pivot to backup idea D2.

## 3. Expected contributions (conditional on results)

1. **Measurement study:** per-task, over-time uncertainty quality in CIL (ECE, AUROC, AURC, confident-error rate) across EWC/SI/MAS/LwF/ER/DER++/iCaRL/IDER.
2. **FACE** (Forgetting-Aware Confidence Estimation): a plug-in for replay methods that learns to predict sample-level forgetting from data DER++ already stores.
3. **Failure analysis:** when FACE does not help (large buffers, pretrained backbones, task-IL).

---

## 4. Setting and notation

- Tasks $t = 1,\dots,T$; task $t$ has data distribution $\mathcal{D}_t$ over $\mathcal{X}\times C_t$, with disjoint class sets $C_t$. Seen classes $Y_{\le t} = \bigcup_{j\le t} C_j$.
- Class-incremental (CIL): no task id at test time; predict over all of $Y_{\le t}$.
- Model $f_\theta(x) = W\,\phi_\theta(x)$; $p_\theta(y\mid x) = \mathrm{softmax}(f_\theta(x))_y$ over $Y_{\le t}$.
- Replay buffer (DER++ style) $\mathcal{M} = \{(x_i, y_i, z_i, \tau_i)\}$: input, label, **logits at insertion time** $z_i$, and the task index $\tau_i$ when it was inserted.

## 5. The FACE method

### 5.1 Sample-level forgetting target
Each buffer item carries a "photo" of what the past model believed: $z_i$. Define the past model's confidence in the true label, over the classes it knew:
$$p^{\text{snap}}_i = \mathrm{softmax}(z_i)_{y_i} \quad\text{(over } Y_{\le \tau_i}).$$
Sample-level forgetting under the current parameters:
$$r_i(\theta) = \big[\,p^{\text{snap}}_i - p_\theta(y_i \mid x_i)\,\big]_+ \in [0,1].$$
Because $p_\theta$ is normalised over *all* current classes, probability that leaks to newer classes (recency bias) shows up as a drop. $r_i = 0$ means nothing was forgotten.

### 5.2 Probe split (against memorisation bias)
Replayed samples get fitted, so forgetting measured on them **underestimates** forgetting on unseen old-task data. We therefore split the buffer:
$$\mathcal{M} = \mathcal{M}_{\text{rep}} \cup \mathcal{M}_{\text{probe}},\qquad |\mathcal{M}_{\text{probe}}| = \rho\,|\mathcal{M}|.$$
Probe items are **never** used in the classifier loss; they only supervise the forgetting head. The fraction $\rho$ (default 0.1–0.2) is ablated.

### 5.3 Forgetting head
A small MLP $h_\psi : \mathbb{R}^d \to [0,1]$ on detached features:
$$\mathcal{L}_{h}(\psi) = \frac{1}{|B_{\text{probe}}|}\sum_{i\in B_{\text{probe}}} \big(h_\psi(\mathrm{sg}[\phi_\theta(x_i)]) - r_i(\theta)\big)^2 .$$
$\mathrm{sg}$ (stop-gradient) means the head never changes the backbone, so FACE cannot hurt accuracy through this loss.

### 5.4 Forgetting-aware confidence
$$c(x) = \max_y p_\theta(y\mid x)\,\cdot\,\big(1 - h_\psi(\phi_\theta(x))\big), \qquad u(x) = 1 - c(x).$$
Abstain if $c(x) < \kappa$. Heuristic reading [S]: $P(\text{correct}) \approx P(\text{correct if nothing forgotten}) \times P(\text{not forgotten})$. Ablation: replace the product with a 2-feature logistic calibrator $g(p_{\max}, h)$ fitted on the probe set.

### 5.5 Backbone training (unchanged base learner)
DER++ loss on current batch $B$ and replay batch $B_{\text{rep}} \subset \mathcal{M}_{\text{rep}}$:
$$\mathcal{L} = \mathrm{CE}(B) + \alpha\,\| z_i - f_\theta(x_i)\|^2_{B_{\text{rep}}} + \beta\,\mathrm{CE}(B_{\text{rep}}).$$

### 5.6 Optional: forgetting-prioritised replay (FACE-PR)
Sample $i \in \mathcal{M}_{\text{rep}}$ with probability $\propto (\hat r_i + \epsilon)^\gamma$, where $\hat r_i$ is the most recently computed (stale) forgetting of item $i$. Compared against MIR.

### 5.7 Analysis variant: FACE-Snap (not memory-fair)
Store full snapshots $\theta_j$ at task ends; at test time, $\hat r(x) = \max_j g_j(x)\,[\,p_{\theta_j}(\hat y_j\mid x) - p_\theta(\hat y_j \mid x)\,]_+$ with $\hat y_j$ the snapshot's prediction and $g_j$ a Mahalanobis gate ("does $x$ look like task $j$ to snapshot $j$?"). Used only as an upper-bound reference, since snapshot memory (~45 MB each for ResNet-18) dwarfs a replay buffer.

### Algorithm (per training step)
1. Draw current batch $B$ and replay batch $B_{\text{rep}}$; update $\theta$ with $\mathcal{L}$.
2. Draw probe batch $B_{\text{probe}}$; compute $r_i(\theta)$ with no gradient; update $\psi$ with $\mathcal{L}_h$.
3. Reservoir-update the buffer with $(x, y, f_\theta(x), t)$; a new item enters the probe split with probability $\rho$.

Overhead: one extra forward pass on a small probe batch plus a tiny MLP; memory: zero extra images (the probe items are carved out of the same buffer).

---

## 6. Relation to baselines

| Method | Main goal | Uses buffer | Gives a forgetting-aware uncertainty? | How FACE relates |
|---|---|---|---|---|
| EWC / SI / MAS | protect important weights | no | no | orthogonal; FACE-Lite needs a buffer, FACE-Snap works on top of them |
| LwF | distil old outputs on new data | no | no | same as above |
| ER | rehearse old data | yes | no | FACE plugs in |
| DER++ | rehearse + logit distillation | yes (stores logits) | no | FACE reuses its stored logits — no extra memory |
| iCaRL / BiC / WA | fix classifier bias | yes | no (accuracy-oriented) | complementary |
| MIR | replay most-interfered samples | yes | no | compared against FACE-PR |
| TS / T-CIL / DATS | post-hoc calibration | yes (validation) | rescales confidence, not forgetting-specific | main uncertainty competitors |
| IDER | reliable (calibrated) replay | yes | indirectly | main training-time competitor |
| MC-dropout / deep ensemble | generic epistemic UQ | — | not forgetting-specific; ensemble is 5× cost | generic UQ competitors |

## 7. Weaknesses and failure cases (to test, not hide)

1. **H1 may be false.** Strong replay methods (DER++) may keep old-task confidence well-ordered; then FACE has nothing to fix. → Measure first.
2. **Memorisation bias.** Even with a probe split, a small probe (e.g., 20 items at buffer 200) gives noisy targets; the head may overfit. → Buffer sizes 200/500/2000; ρ ablation.
3. **Probe costs replay capacity.** Taking 10–20% of the buffer away from replay may lower accuracy. → Report accuracy honestly; compare to DER++ at the same total buffer.
4. **Stale targets.** $r_i$ is computed with the current $\theta$ but $h$ lags behind; right after a task switch the head is outdated. → Track head error over time.
5. **Feature drift.** The head reads features that keep drifting; old-task features may become unrecognisable. [S]
6. **Product heuristic** in 5.4 is not derived from a probabilistic model. → Logistic-calibrator ablation.
7. **Pretrained backbones** (ViT) forget far less; the effect may vanish. Reviewers will ask. → One frozen-ViT experiment.
8. **Task-IL is not a fit**: with task ids, the recency leak disappears.
9. **Overlap with MIR / forgetting-prediction work** (e.g., forecasting forgotten examples in LMs). Novelty must rest on *uncertainty/failure detection*, not on predicting forgetting per se. → Deeper search before writing.
10. **Snapshot target can itself be wrong** (past model wrong at insertion → $r_i$ small). Acceptable: we measure forgetting, not error.

## 8. Evaluation metrics (detailed in Phase 3)
Accuracy (final, average incremental), forgetting, BWT; per-task ECE; misclassification-detection AUROC; AURC; selective risk at 80/90% coverage; confident-error rate (errors with confidence > 0.9); OOD AUROC (optional); time, memory, parameters.
