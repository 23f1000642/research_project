"""Metric sanity checks on hand-computable cases."""
import numpy as np
import pytest

from facecl.evaluation import metrics as M


def test_cl_accuracy_metrics():
    rows = [[0.9], [0.6, 0.8], [0.5, 0.7, 0.85]]
    m = M.cl_accuracy_metrics(rows)
    assert m["final_avg_acc"] == pytest.approx((0.5 + 0.7 + 0.85) / 3)
    assert m["avg_incremental_acc"] == pytest.approx((0.9 + 0.7 + (0.5 + 0.7 + 0.85) / 3) / 3)
    # forgetting: task0 max(0.9, 0.6) - 0.5 = 0.4 ; task1 max(0.8) - 0.7 = 0.1
    assert m["forgetting"] == pytest.approx(0.25)
    assert m["bwt"] == pytest.approx(((0.5 - 0.9) + (0.7 - 0.8)) / 2)


def test_ece_perfect_and_bad():
    conf = np.array([1.0, 1.0, 1.0, 1.0])
    assert M.ece(conf, np.array([1, 1, 1, 1], bool)) == pytest.approx(0.0)
    assert M.ece(conf, np.array([0, 0, 0, 0], bool)) == pytest.approx(1.0)


def test_auroc_and_aurc_perfect_ranking():
    conf = np.array([0.9, 0.8, 0.2, 0.1])
    correct = np.array([1, 1, 0, 0], bool)
    assert M.auroc_error(conf, correct) == pytest.approx(1.0)
    assert M.e_aurc(conf, correct) == pytest.approx(0.0)
    # reversed ranking is the worst case
    assert M.auroc_error(-conf, correct) == pytest.approx(0.0)


def test_selective_risk_and_cer():
    conf = np.array([0.95, 0.9, 0.5, 0.4])
    correct = np.array([0, 1, 1, 0], bool)
    assert M.selective_risk(conf, correct, 0.5) == pytest.approx(0.5)  # top-2: one error
    assert M.confident_error_rate(conf, correct, 0.9) == pytest.approx(0.5)  # errors: 0.95, 0.4


def test_temperature_recovers_scale():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(size=(4000, 5)) * 2
    y = np.array([rng.choice(5, p=p) for p in M.softmax_np(true_logits)])
    t = M.fit_temperature(true_logits * 3.0, y)  # overconfident by 3x
    assert t == pytest.approx(3.0, rel=0.15)


def test_uncertainty_forgetting():
    rows = [[0.9], [0.7, 0.85], [0.6, 0.8, 0.9]]
    assert M.forgetting_of(rows) == pytest.approx(((0.9 - 0.6) + (0.85 - 0.8)) / 2)
