"""Replay baselines: ER, DER++ and MIR (with the reservoir buffer)."""
import torch
import torch.nn.functional as F
from torch.func import functional_call

from ..buffer import ReservoirBuffer
from .base import ContinualMethod


class SGD(ContinualMethod):
    """Fine-tuning lower bound (also used by the Joint upper bound)."""

    def observe(self, x, y, t):
        logits = self.net(self.aug(x))
        loss = self.ce(logits, y, t)
        self.step(loss)
        return loss.item()


class ER(ContinualMethod):
    """Experience Replay: current batch + a buffer batch in one forward pass."""

    uses_buffer = True

    def __init__(self, model, bench, cfg, device, store_logits=False, probe_fraction=0.0):
        super().__init__(model, bench, cfg, device)
        self.buffer = ReservoirBuffer(cfg["buffer_size"], bench.in_shape, bench.n_classes, device,
                                      store_logits=store_logits, probe_fraction=probe_fraction)
        self.mb = int(cfg.get("minibatch_size", cfg["batch_size"]))

    def replay_indices(self, n):
        return self.buffer.sample(n, probe=False)

    def observe(self, x, y, t):
        inputs, labels = x, y
        if not self.buffer.is_empty():
            idx = self.replay_indices(self.mb)
            if idx.numel() > 0:
                inputs = torch.cat([x, self.buffer.x[idx]])
                labels = torch.cat([y, self.buffer.y[idx]])
        loss = self.ce(self.net(self.aug(inputs)), labels, t)
        self.step(loss)
        self.buffer.add(x, y, t)
        return loss.item()

    def buffer_snapshot(self):
        n = len(self.buffer)
        if n == 0:
            return None
        return {"x": self.buffer.x[:n], "y": self.buffer.y[:n], "task": self.buffer.task[:n],
                "probe": self.buffer.probe[:n]}

    def state_dict(self):
        return {"net": self.net.state_dict(), "buffer": self.buffer.state_dict()}

    def load_state_dict(self, state):
        self.net.load_state_dict(state["net"])
        self.buffer.load_state_dict(state["buffer"])


class DERpp(ER):
    """Dark Experience Replay++ (Buzzega et al., NeurIPS 2020).

    loss = CE(current) + alpha * MSE(f(x'), z') + beta * CE(f(x''), y'')
    with (x', z') and (x'', y'') two independent buffer samples, z' the logits stored at insertion.
    """

    def __init__(self, model, bench, cfg, device, probe_fraction=0.0):
        super().__init__(model, bench, cfg, device, store_logits=True, probe_fraction=probe_fraction)
        self.alpha = float(cfg["alpha"])
        self.beta = float(cfg["beta"])

    def replay_losses(self, t):
        """alpha/beta terms; returns (loss, idx2, logits2) so subclasses can reuse the beta forward."""
        loss = torch.zeros((), device=self.device)
        idx2, out2 = None, None
        if self.buffer.is_empty():
            return loss, idx2, out2
        idx1 = self.replay_indices(self.mb)
        if idx1.numel() > 0:
            out1 = self.net(self.aug(self.buffer.x[idx1]))
            loss = loss + self.alpha * F.mse_loss(out1, self.buffer.logits[idx1])
        idx2 = self.replay_indices(self.mb)
        if idx2.numel() > 0:
            out2 = self.net(self.aug(self.buffer.x[idx2]))
            loss = loss + self.beta * self.ce(out2, self.buffer.y[idx2], t)
        return loss, idx2, out2

    def observe(self, x, y, t):
        logits = self.net(self.aug(x))
        loss = self.ce(logits, y, t)
        replay_loss, _, _ = self.replay_losses(t)
        loss = loss + replay_loss
        self.step(loss)
        self.buffer.add(x, y, t, logits=logits.detach())
        return loss.item()


class MIR(ER):
    """Maximally Interfered Retrieval (Aljundi et al., NeurIPS 2019) on top of ER.

    From a random candidate subset of the buffer, replay the items whose loss would increase
    most after a virtual SGD step on the current batch.
    """

    def __init__(self, model, bench, cfg, device):
        super().__init__(model, bench, cfg, device)
        self.n_candidates = int(cfg.get("mir_candidates", 50))

    def replay_indices(self, n):
        cand = self.buffer.sample(self.n_candidates, probe=False)
        if cand.numel() <= n:
            return cand
        return self._most_interfered(cand, n)

    def _most_interfered(self, cand, n):
        x_cur, y_cur = self._current
        params = {k: v for k, v in self.net.named_parameters()}
        buffers = {k: v for k, v in self.net.named_buffers()}
        was_training = self.net.training
        self.net.eval()  # no BN statistic updates during the virtual step
        loss = self.ce(functional_call(self.net, (params, buffers), (self.aug(x_cur),)), y_cur, self.task)
        grads = torch.autograd.grad(loss, list(params.values()))
        lr = self.lr()
        virtual = {k: (v - lr * g).detach() for (k, v), g in zip(params.items(), grads)}
        with torch.no_grad():
            xc = self.aug(self.buffer.x[cand], train=False)
            yc = self.buffer.y[cand]
            pre = F.cross_entropy(functional_call(self.net, (params, buffers), (xc,)), yc, reduction="none")
            post = F.cross_entropy(functional_call(self.net, (virtual, buffers), (xc,)), yc, reduction="none")
        self.net.train(was_training)
        return cand[torch.topk(post - pre, n).indices]

    def observe(self, x, y, t):
        self._current = (x, y)
        return super().observe(x, y, t)
