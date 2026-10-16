# Forgetting-Aware Confidence in Class-Incremental Learning

Research question: does forgetting make a model *confidently wrong* on old classes, and can a cheap,
memory-fair estimate of sample-level forgetting (FACE) restore its ability to detect its own errors?

Status: **Phase 4 (implementation)**. Pipeline verified on synthetic data and Split MNIST (debug only).
No paper results yet.

## Layout

```
configs/            defaults, per-dataset and per-method YAML (published hyperparameters cited inline)
facecl/             the Python package
  data/             benchmarks, class orders, train/val/test splits, GPU augmentation
  models/           ResNet-18 (DER/Mammoth variant), MLP
  baselines/        SGD, Joint, ER, DER++, MIR, online EWC, SI, MAS, LwF, iCaRL
  methods/          FACE (ours)
  training/         task loop, per-task evaluation dumps, resumable checkpoints
  evaluation/       metrics (accuracy, forgetting, ECE, AUROC, AURC, UF, ...) and offline evaluation
  buffer.py         reservoir buffer with FACE probe split
experiments/        run.py (CLI), plans/ (one YAML per stage), Kaggle launcher + notebook
tests/              unit tests and an end-to-end smoke test of every method
results/ figures/ tables/ logs/ analysis/ manuscript/
docs/               phase 1-3 research documents
```

The code lives in the `facecl` package instead of top-level `datasets/`, `models/` ... folders because a
top-level `datasets` package would shadow the HuggingFace `datasets` library installed on Kaggle.

## Usage

```bash
pip install -r requirements.txt
python -m pytest -q tests
python experiments/run.py --dataset seq-cifar100 --method derpp --set buffer_size=500 seed=0
python experiments/run.py --plan experiments/plans/s1_h1_pilot.yaml
python -m facecl.evaluation.evaluate results/<run_id>
```

Every run writes `results/<run_id>/` (config, environment and git commit, log, accuracy matrices,
timings, `metrics.json`) and appends a row to `logs/experiment_log.md`. Raw per-task logits
(`eval/*.npz`) are too large for git and are archived with the Kaggle outputs.

## Reproducibility

Seeds fix the initialisation (`seed`) and the class order (`order_seed`, defaults to `seed`).
Runs are identified by a hash of their configuration; finished runs are skipped and interrupted runs
resume from the last completed task.

## Documents

- `docs/phase1_literature.md` — literature map and closest prior work
- `docs/phase2_method.md` — research question, hypotheses, FACE method, weaknesses
- `docs/phase3_experimental_design.md` — datasets, protocol, metrics, baselines, statistics, compute budget
