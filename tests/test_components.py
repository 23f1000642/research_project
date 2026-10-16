"""Unit tests for data splits, buffer, FACE target and an end-to-end smoke run of every method."""
import numpy as np
import pytest
import torch

from facecl.buffer import ReservoirBuffer
from facecl.config import load_config
from facecl.data import Augmenter, build_benchmark
from facecl.evaluation import evaluate_run
from facecl.training import run

METHODS = ["sgd", "joint", "er", "derpp", "mir", "ewc_on", "si", "mas", "lwf", "icarl", "face", "face_pr"]


def _cfg(method, tmp_path, **ov):
    base = {"seed": 0, "buffer_size": 40, "results_dir": str(tmp_path), "device": "cpu",
            "data_root": str(tmp_path / "data"), "n_epochs": 2, "head_refit_steps": 5,
            "fisher_samples": 30, "importance_samples": 30,
            "experiment_log": str(tmp_path / "experiment_log.md")}
    # strengths for methods that (deliberately) have no default on the synthetic dataset
    base.update({"ewc_on": {"e_lambda": 1.0}, "si": {"c": 0.5}, "lwf": {"alpha": 1.0}}.get(method, {}))
    base.update(ov)
    return load_config("synthetic", method, base)


def test_split_has_no_leakage(tmp_path):
    bench = build_benchmark(_cfg("sgd", tmp_path))
    for t in range(bench.n_tasks):
        xtr, ytr = bench.task_data(t, "train")
        xva, yva = bench.task_data(t, "val")
        assert set(ytr.tolist()) == set(bench.task_classes(t)) == set(yva.tolist())
        # no identical image appears in both train and val
        tr = {bytes(a.numpy().tobytes()) for a in xtr}
        assert not any(bytes(a.numpy().tobytes()) in tr for a in xva)
    assert abs(len(bench._val[1]) / (len(bench._val[1]) + len(bench._train[1])) - 0.1) < 0.02


def test_class_order_depends_on_order_seed(tmp_path):
    a = build_benchmark(_cfg("sgd", tmp_path, order_seed=1)).class_order
    b = build_benchmark(_cfg("sgd", tmp_path, order_seed=2)).class_order
    c = build_benchmark(_cfg("sgd", tmp_path, order_seed=1)).class_order
    assert not np.array_equal(a, b) and np.array_equal(a, c)


def test_reservoir_buffer_is_uniform_and_probe_fraction():
    torch.manual_seed(0)
    counts = np.zeros(1000)
    for _ in range(200):
        buf = ReservoirBuffer(100, (1, 2, 2), 3, "cpu", probe_fraction=0.2)
        buf.add(torch.zeros(1000, 1, 2, 2, dtype=torch.uint8), torch.arange(1000), task=0)
        counts[buf.y.numpy()] += 1
        assert len(buf) == 100
    # every stream item should be kept with probability 100/1000
    assert abs(counts.mean() / 200 - 0.1) < 0.005
    assert counts[:500].sum() == pytest.approx(counts[500:].sum(), rel=0.1)
    assert 0.1 < buf.probe.float().mean() < 0.3


def test_augmenter_shapes_and_eval_identity():
    aug = Augmenter(crop_padding=4, hflip=True)
    x = torch.randint(0, 255, (5, 3, 32, 32), dtype=torch.uint8)
    out = aug(x, train=True)
    assert out.shape == (5, 3, 32, 32) and out.dtype == torch.float32
    assert torch.equal(aug(x, train=False), x.float() / 255)


def test_face_target_values(tmp_path):
    from facecl.baselines import build_method
    from facecl.models import build_model

    cfg = _cfg("face", tmp_path)
    bench = build_benchmark(cfg)
    m = build_method("face", build_model(cfg, bench), bench, cfg, "cpu")
    # one item stored at task 0 (2 classes seen) with confident logits for its label 1
    m.buffer.add(torch.zeros(1, *bench.in_shape, dtype=torch.uint8), torch.tensor([1]), task=0,
                 logits=torch.tensor([[0.0, 4.0, 9.0, 9.0, 9.0, 9.0]]))
    idx = torch.tensor([0])
    p_snap = torch.softmax(torch.tensor([0.0, 4.0]), 0)[1]
    # now (task 1, 4 classes seen) the model puts most mass on a new class
    now = torch.tensor([[0.0, 1.0, 5.0, 0.0, 99.0, 99.0]])
    p_now = torch.softmax(now[0, :4], 0)[1]
    r = m.forgetting_target(idx, now, t=1)
    assert r.item() == pytest.approx((p_snap - p_now).item(), abs=1e-6)
    # no forgetting when the current model is at least as confident
    assert m.forgetting_target(idx, torch.tensor([[0.0, 9.0, 0, 0, 0, 0]]), t=1).item() == 0.0


@pytest.mark.parametrize("method", METHODS)
def test_smoke_every_method(method, tmp_path):
    run_dir = run(_cfg(method, tmp_path))
    res = evaluate_run(run_dir)
    acc = res["accuracy"]["final_avg_acc"]
    assert 0.0 <= acc <= 1.0
    if method == "face":
        assert "face" in res["final"]
    # re-running a finished configuration is a no-op
    assert run(_cfg(method, tmp_path)) == run_dir


def test_per_sample_grads_match_loop():
    """The vmapped per-sample squared gradients used by EWC/MAS equal an explicit loop."""
    import torch.nn.functional as F

    from facecl.baselines.regularization import per_sample_sq_grads
    from facecl.models.networks import MLP

    torch.manual_seed(0)
    net = MLP(4, in_dim=6, hidden=5)
    x = torch.rand(7, 1, 2, 3)
    y = torch.randint(0, 4, (7,))
    aux = torch.stack([y.float(), torch.rand(7)], 1)

    def loss(logits, a):
        return a[1] * F.cross_entropy(logits, a[0].long().unsqueeze(0))

    sq, ab = per_sample_sq_grads(net, x, loss, aux, chunk=3)
    net.eval()
    ref_sq = {k: torch.zeros_like(v) for k, v in net.named_parameters()}
    ref_ab = {k: torch.zeros_like(v) for k, v in net.named_parameters()}
    for i in range(7):
        net.zero_grad()
        loss(net(x[i:i + 1]), aux[i]).backward()
        for k, p in net.named_parameters():
            ref_sq[k] += p.grad ** 2
            ref_ab[k] += p.grad.abs()
    for k in ref_sq:
        assert torch.allclose(sq[k], ref_sq[k], atol=1e-6)
        assert torch.allclose(ab[k], ref_ab[k], atol=1e-6)
