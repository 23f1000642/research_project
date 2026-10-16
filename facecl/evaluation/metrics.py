"""Metric definitions (docs/phase3_experimental_design.md, section 6).

Conventions: `conf` is a confidence (higher = more trusted), `correct` a boolean array.
Uncertainty-ranking metrics (AUROC, AURC) only use the ordering of `conf`.
"""
import warnings

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.metrics import roc_auc_score


# ---- continual-learning accuracy metrics --------------------------------------------
def _square(acc_rows):
    T = max(len(acc_rows), max((len(r) for r in acc_rows), default=0))
    A = np.full((T, T), np.nan)
    for t, row in enumerate(acc_rows):
        A[t, : len(row)] = row
    return A


def cl_accuracy_metrics(acc_rows) -> dict:
    """acc_rows[t][j] = accuracy on task j after training task t (j <= t).

    A single row covering several tasks is the Joint upper bound: only final accuracy applies.
    """
    if len(acc_rows) == 1 and len(acc_rows[0]) > 1:
        return {"final_avg_acc": float(np.mean(acc_rows[0]))}
    A = _square(acc_rows)
    T = A.shape[0]
    out = {
        "final_avg_acc": float(np.nanmean(A[-1])),
        "avg_incremental_acc": float(np.mean([np.nanmean(A[t, : t + 1]) for t in range(T)])),
    }
    if T > 1:
        out["forgetting"] = float(np.mean([np.nanmax(A[:-1, j]) - A[-1, j] for j in range(T - 1)]))
        out["bwt"] = float(np.mean([A[-1, j] - A[j, j] for j in range(T - 1)]))
    return out


def forgetting_of(matrix_rows) -> float:
    """The forgetting formula applied to any per-(t, j) quality score (e.g. AUROC -> UF)."""
    A = _square(matrix_rows)
    T = A.shape[0]
    if T < 2 or np.isnan(A[:-1]).all():
        return float("nan")
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # cells can be NaN (e.g. AUROC with no errors)
        vals = [np.nanmax(A[:-1, j]) - A[-1, j] for j in range(T - 1)]
        return float(np.nanmean(vals))


# ---- calibration ----------------------------------------------------------------------------
def ece(conf, correct, n_bins=15) -> float:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def adaptive_ece(conf, correct, n_bins=15) -> float:
    """Equal-mass bins."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    order = np.argsort(conf)
    total = 0.0
    for chunk in np.array_split(order, n_bins):
        if chunk.size:
            total += chunk.size / conf.size * abs(conf[chunk].mean() - correct[chunk].mean())
    return float(total)


def nll(logits, y) -> float:
    return float(-log_softmax(logits, 1)[np.arange(len(y)), y].mean())


def brier(probs, y) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((probs - onehot) ** 2).sum(1).mean())


# ---- failure detection / selective prediction ----------------------------------------------
def auroc_error(conf, correct) -> float:
    """AUROC for detecting errors with uncertainty = -conf. NaN if only one class present."""
    correct = np.asarray(correct, bool)
    if correct.all() or (~correct).all():
        return float("nan")
    return float(roc_auc_score(~correct, -np.asarray(conf, float)))


def risk_coverage(conf, correct):
    order = np.argsort(-np.asarray(conf, float), kind="stable")
    err = (~np.asarray(correct, bool))[order].astype(float)
    n = len(err)
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(err) / np.arange(1, n + 1)
    return coverage, risk


def aurc(conf, correct) -> float:
    _, risk = risk_coverage(conf, correct)
    return float(risk.mean())


def e_aurc(conf, correct) -> float:
    """AURC minus the AURC of an oracle that ranks every error last."""
    correct = np.asarray(correct, bool)
    oracle = np.where(correct, 1.0, 0.0)
    return aurc(conf, correct) - aurc(oracle, correct)


def selective_risk(conf, correct, coverage=0.9) -> float:
    cov, risk = risk_coverage(conf, correct)
    k = max(int(np.ceil(coverage * len(cov))) - 1, 0)
    return float(risk[k])


def confident_error_rate(conf, correct, threshold=0.9) -> float:
    """Fraction of errors made with confidence above `threshold`."""
    correct = np.asarray(correct, bool)
    if correct.all():
        return float("nan")
    return float((np.asarray(conf)[~correct] > threshold).mean())


# ---- post-hoc temperature scaling ------------------------------------------------------------
def fit_temperature(logits, y) -> float:
    """Single temperature minimising NLL (Guo et al., 2017), searched over log T in [-3, 3]."""
    logits = np.asarray(logits, float)

    def obj(log_t):
        return nll(logits / np.exp(log_t), y)

    res = minimize_scalar(obj, bounds=(-3, 3), method="bounded")
    return float(np.exp(res.x))


def all_quality_metrics(conf, correct, probs=None, logits=None, y=None, prob_conf=True) -> dict:
    """Everything we report for one (score, t, j) cell."""
    out = {
        "auroc": auroc_error(conf, correct),
        "aurc": aurc(conf, correct),
        "e_aurc": e_aurc(conf, correct),
        "sel_risk_80": selective_risk(conf, correct, 0.8),
        "sel_risk_90": selective_risk(conf, correct, 0.9),
    }
    if prob_conf:  # only meaningful when conf is a probability
        out["ece"] = ece(conf, correct)
        out["adaptive_ece"] = adaptive_ece(conf, correct)
        out["cer_0.9"] = confident_error_rate(conf, correct, 0.9)
        out["mean_conf"] = float(np.mean(conf))
    if probs is not None and y is not None:
        out["brier"] = brier(probs, y)
    if logits is not None and y is not None:
        out["nll"] = nll(logits, y)
    return out


def softmax_np(logits):
    return softmax(np.asarray(logits, float), axis=1)
