# Phase 1 — Literature map (record)

Date: 2026-10-07. Status: abstracts/summaries only unless marked "read". Every entry must be re-verified (full text) before it is cited in the manuscript.

## Closest prior work to our direction (must cite + compare)

| Work | Venue | What it does | Why it matters to us |
|---|---|---|---|
| Li et al., *Calibration of Continual Learning Models* | CVPR-W (CLVision) 2024 | First empirical study of calibration in CL; post-hoc "continual calibration" | Shows CL models are miscalibrated; aggregate ECE only |
| Hwang, Kim, Whang, *T-CIL* | CVPR 2025 | Temperature scaling for class-incremental learning using adversarially perturbed exemplars | Strong post-hoc calibration baseline |
| *DATS: Distance-Aware Temperature Scaling* (arXiv 2509.21161) | preprint 2025 | Distance-aware TS for CIL calibration | Post-hoc calibration baseline (verify) |
| Liu et al., *IDER* | ICLR 2026 | Idempotent experience replay improving calibration + accuracy | Training-time reliability baseline |
| Serra et al., *How to Leverage Predictive Uncertainty Estimates…* | TMLR 2025 | Bias–variance-based memory population | Uncertainty for replay selection is taken |
| Soutif-Cormerais, Carta, van de Weijer, *Temporal Ensembles* | CoLLAs 2023 | EMA weight ensemble for accuracy/stability | Temporal ensembling exists — but not as uncertainty/forgetting signal |
| Aljundi et al., *MIR* (Maximally Interfered Retrieval) | NeurIPS 2019 | Replays samples whose loss would increase most | Closest to forgetting-prioritised replay — must compare |
| Gao et al., *CPCL* | ICML 2025 | Conformal prediction in CL | Coverage-based UQ in CL |
| Shihab et al., *Continual Calibration: Coverage can collapse before accuracy* | arXiv 2604.23987 (2026) | Coverage collapses ~3.4x faster than accuracy (LLM fine-tuning) | Motivating evidence, different domain |
| Ebrahimi et al., UCB | ICLR 2020 | Uncertainty-scaled learning rates in BNNs | Uncertainty-for-stability is taken |
| Ahn et al., UCL | NeurIPS 2019 | Node-wise uncertainty regularisation | same |
| MESU (Bayesian metaplasticity) | Nature Comms 2025 | Boundary-free Bayesian CL | same |
| Tan et al., RCUG (arXiv 2512.09441) | preprint 2025 | VLM-based CIL with uncertainty-guided inference | Uncertainty used for feature selection in PTM-CIL |

## Classic baselines
EWC (Kirkpatrick 2017), SI (Zenke 2017), MAS (Aljundi 2018), LwF (Li & Hoiem 2017), ER, DER/DER++ (Buzzega 2020), iCaRL (Rebuffi 2017), BiC (Wu 2019), WA (Zhao 2020). Unifying view: Benzing 2022 (SI/MAS ≈ sqrt Fisher). Fisher computation matters: van de Ven 2025 (arXiv 2502.11756).

## Gap we target
Calibration work fixes *how much* confidence old-task predictions get (scale). Nobody we found measures/targets whether the model can *rank* its own old-task errors (failure detection) as forgetting accumulates, using a sample-level forgetting signal. To be re-verified in a deeper search before writing.
