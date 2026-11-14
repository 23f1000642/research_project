# D2 novelty check (2026-11-14): the core of D2 is already published as a preprint

D2 idea: treat the per-parameter importance in EWC / SI / MAS as a noisy estimate, measure its
uncertainty (sampling variance), and use that to adjust the regularization.

## Closest prior work

| paper | status | what it does | overlap with D2 |
|---|---|---|---|
| Aryal, "Do Artificial Neural Networks Need Dendritic Nanotubes? Two Reasons for Shrinking Importance Estimates in Continual Learning", arXiv 2610.14258 (Oct 2026) | preprint | per-parameter sampling variance of EWC/MAS importance (v_i = s_i²/n; approximate for SI); fixed and empirical-Bayes shrinkage toward the layer mean; Permuted MNIST, Split CIFAR-100, 5 seeds; EWC gain is a variance effect that vanishes with more samples, MAS gain persists (bias) | **core of D2: done** |
| Rügamer, "On the Epistemic Uncertainty of Overparametrized Neural Networks", arXiv 2605.25234, ICML 2026 | peer-reviewed | removes non-identifiability variance from EWC importance (one synthetic demo) | partial, different variance |
| Liu & Chang, "Elastic Weight Consolidation Done Right for Continual Learning", CVPR 2026 | peer-reviewed | Fisher gradient vanishing; logit reversal | no (point estimate fix) |
| Wiest, "Improved EWC as an Optimization Constraint", arXiv 2611.16039 (Nov 2026) | preprint | removes per-sample loss-gradient weighting in EWC | no |
| Sarkar, EVCLplus, arXiv 2508.16632 (Aug 2025) | preprint | Fisher-weighted penalties on variational posterior mean/variance | no (posterior variance, not estimator noise) |
| UCL (Ahn et al., NeurIPS 2019), UCB (Ebrahimi et al., ICLR 2020), VCL | peer-reviewed | Bayesian CL: posterior uncertainty sets per-weight regularization strength | conceptual overlap: "uncertainty-weighted importance" is an established idea |

## Verdict

D2 as planned is not novel. What remains (log-scale or per-parameter shrinkage, down-weighting
toward zero, an exact SI variance) is an incremental extension of one recent preprint that the
preprint itself names as its next step. Not a strong basis for a journal paper.

Sources read: arXiv abstract/HTML pages of the papers above (via a fetch agent, 2026-11-14).
