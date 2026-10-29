"""Regularisation and distillation baselines: online EWC, SI, MAS, LwF."""
import copy

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from .base import ContinualMethod


def _params(net):
    return {k: v for k, v in net.named_parameters() if v.requires_grad}


def per_sample_sq_grads(net, x, per_sample_loss, aux=None, chunk=32):
    """Sum over samples of squared (or absolute) per-sample gradients.

    `per_sample_loss(logits_1xC, aux_i)` returns a scalar loss for one sample; this is vmapped,
    `aux` is an optional per-sample tensor [N, ...] (e.g. target and weight) batched with x.
    Model is put in eval mode so BatchNorm uses running statistics (needed for vmap).
    Returns {param_name: tensor} of summed squared gradients and summed absolute gradients.
    """
    was_training = net.training
    net.eval()
    params = {k: v.detach() for k, v in _params(net).items()}
    buffers = {k: v.detach() for k, v in net.named_buffers()}

    def loss_fn(p, xi, ai):
        logits = functional_call(net, (p, buffers), (xi.unsqueeze(0),))
        return per_sample_loss(logits, ai)

    g_fn = vmap(grad(loss_fn), in_dims=(None, 0, 0))
    sq = {k: torch.zeros_like(v) for k, v in params.items()}
    ab = {k: torch.zeros_like(v) for k, v in params.items()}
    if aux is None:
        aux = torch.zeros(x.shape[0], device=x.device)
    for s in range(0, x.shape[0], chunk):
        g = g_fn(params, x[s:s + chunk], aux[s:s + chunk])
        for k in sq:
            sq[k] += (g[k] ** 2).sum(0)
            ab[k] += g[k].abs().sum(0)
    net.train(was_training)
    return sq, ab


class _QuadraticPenalty(ContinualMethod):
    """Shared machinery: penalty = sum_k importance_k * (theta_k - anchor_k)^2."""

    strength_key = "e_lambda"

    def __init__(self, model, bench, cfg, device):
        super().__init__(model, bench, cfg, device)
        self.importance = None
        self.anchor = None

    def penalty(self):
        if self.importance is None:
            return torch.zeros((), device=self.device)
        total = 0.0
        for k, p in _params(self.net).items():
            total = total + (self.importance[k] * (p - self.anchor[k]) ** 2).sum()
        return total

    def observe(self, x, y, t):
        logits = self.net(self.aug(x))
        loss = self.ce(logits, y, t) + float(self.cfg[self.strength_key]) * self.penalty()
        self.step(loss)
        return loss.item()

    def _set_anchor(self):
        self.anchor = {k: v.detach().clone() for k, v in _params(self.net).items()}

    def state_dict(self):
        return {"net": self.net.state_dict(), "importance": self.importance, "anchor": self.anchor}

    def load_state_dict(self, state):
        self.net.load_state_dict(state["net"])
        self.importance, self.anchor = state["importance"], state["anchor"]


class EWCOnline(_QuadraticPenalty):
    """Online EWC (Schwarz et al., 2018): F <- gamma * F + F_task; loss += lambda * sum F (theta - theta*)^2.

    The penalty scale (no 1/2) matches Mammoth so its published lambda values apply.
    Fisher type (cfg.fisher):
      "mammoth"   true-label squared gradients weighted by p(y_true|x) (Mammoth's estimator, default);
      "sampled"   y ~ p(y|x), an unbiased estimate of the true Fisher;
      "empirical" true-label squared gradients.
    See van de Ven (2025) on why this choice matters.
    """

    def end_task(self, t, x, y):
        n = min(int(self.cfg.get("fisher_samples", 2000)), x.shape[0])
        sel = torch.randperm(x.shape[0], device=x.device)[:n]
        xs, ys = self.aug(x[sel], train=False), y[sel]
        n_seen = self.n_seen(t)
        kind = self.cfg.get("fisher", "mammoth")
        with torch.no_grad():
            was = self.net.training
            self.net.eval()
            probs = F.softmax(self.net(xs)[:, :n_seen], 1)
            self.net.train(was)
        weight = torch.ones(n, device=xs.device)
        if kind == "sampled":
            targets = torch.multinomial(probs, 1).squeeze(1)
        elif kind in ("empirical", "mammoth"):
            targets = ys
            if kind == "mammoth":
                weight = probs.gather(1, ys[:, None]).squeeze(1)
        else:
            raise ValueError(f"unknown fisher type {kind}")
        aux = torch.stack([targets.float(), weight.sqrt()], 1)

        def nll(logits, a):
            # scaling the loss by sqrt(w) scales the squared gradient by w
            return a[1] * F.cross_entropy(logits[:, :n_seen], a[0].long().unsqueeze(0))

        sq, _ = per_sample_sq_grads(self.net, xs, nll, aux, int(self.cfg.get("grad_chunk", 32)))
        fisher = {k: v / n for k, v in sq.items()}
        gamma = float(self.cfg.get("gamma", 1.0))
        if self.importance is None:
            self.importance = fisher
        else:
            self.importance = {k: gamma * self.importance[k] + fisher[k] for k in fisher}
        self._set_anchor()


class MAS(_QuadraticPenalty):
    """Memory Aware Synapses (Aljundi et al., 2018): importance = mean |d ||f(x)||^2 / d theta|."""

    def __init__(self, model, bench, cfg, device):
        super().__init__(model, bench, cfg, device)
        self.n_tasks_seen = 0

    def end_task(self, t, x, y):
        n = min(int(self.cfg.get("importance_samples", 2000)), x.shape[0])
        sel = torch.randperm(x.shape[0], device=x.device)[:n]
        xs = self.aug(x[sel], train=False)
        n_seen = self.n_seen(t)

        def out_norm(logits, _):
            return (logits[:, :n_seen] ** 2).sum()

        _, ab = per_sample_sq_grads(self.net, xs, out_norm, chunk=int(self.cfg.get("grad_chunk", 32)))
        new = {k: v / n for k, v in ab.items()}
        if self.importance is None:
            self.importance = new
        else:  # running average over tasks, as in the original MAS
            m = self.n_tasks_seen
            self.importance = {k: (m * self.importance[k] + new[k]) / (m + 1) for k in new}
        self.n_tasks_seen += 1
        self._set_anchor()

    def state_dict(self):
        s = super().state_dict()
        s["n_tasks_seen"] = self.n_tasks_seen
        return s

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.n_tasks_seen = state["n_tasks_seen"]


class SI(_QuadraticPenalty):
    """Synaptic Intelligence (Zenke et al., 2017), following the Mammoth implementation:
    omega += lr * g^2 per step (path integral for plain SGD), and at task end
    Omega += omega / ((theta - theta_start)^2 + xi).
    """

    strength_key = "c"

    def begin_task(self, t):
        super().begin_task(t)
        self.start = {k: v.detach().clone() for k, v in _params(self.net).items()}
        self.small_omega = {k: torch.zeros_like(v) for k, v in self.start.items()}

    def observe(self, x, y, t):
        logits = self.net(self.aug(x))
        loss = self.ce(logits, y, t) + float(self.cfg["c"]) * self.penalty()
        lr = self.lr()

        def accumulate():
            for k, p in _params(self.net).items():
                if p.grad is not None:
                    self.small_omega[k] += lr * p.grad.detach().float() ** 2

        self.step(loss, before_step=accumulate)
        return loss.item()

    def end_task(self, t, x, y):
        xi = float(self.cfg.get("xi", 1.0))
        new = {k: self.small_omega[k] / ((p.detach() - self.start[k]) ** 2 + xi)
               for k, p in _params(self.net).items()}
        if self.importance is None:
            self.importance = new
        else:
            self.importance = {k: self.importance[k] + new[k] for k in new}
        self._set_anchor()


class LwF(ContinualMethod):
    """Learning without Forgetting (Li & Hoiem, 2017), class-incremental form:
    CE over seen classes + alpha * T^2 * KL(old || new) on the old classes' logits.
    """

    def __init__(self, model, bench, cfg, device):
        super().__init__(model, bench, cfg, device)
        self.old = None

    def observe(self, x, y, t):
        xa = self.aug(x)
        logits = self.net(xa)
        loss = F.cross_entropy(logits[:, : self.n_seen(t)], y)
        if self.old is not None and t > 0:
            n_old = self.n_seen(t - 1)
            T = float(self.cfg.get("softmax_temp", 2.0))
            with torch.no_grad():
                old_logits = self.old(xa)[:, :n_old]
            kd = F.kl_div(F.log_softmax(logits[:, :n_old] / T, 1), F.softmax(old_logits / T, 1),
                          reduction="batchmean") * T * T
            loss = loss + float(self.cfg["alpha"]) * kd
        self.step(loss)
        return loss.item()

    def end_task(self, t, x, y):
        self.old = copy.deepcopy(self.net).eval()
        for p in self.old.parameters():
            p.requires_grad_(False)

    def state_dict(self):
        return {"net": self.net.state_dict(), "old": None if self.old is None else self.old.state_dict()}

    def load_state_dict(self, state):
        self.net.load_state_dict(state["net"])
        if state["old"] is not None:
            self.old = copy.deepcopy(self.net).eval()
            self.old.load_state_dict(state["old"])
