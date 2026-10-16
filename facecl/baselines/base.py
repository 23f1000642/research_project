"""Common interface for every continual-learning method (baselines and FACE).

The trainer calls, for each task t:
    x, y = method.task_train_data(t, x, y)
    method.begin_task(t)
    for each mini-batch: method.observe(x_batch_uint8, y_batch, t)
    method.end_task(t, x, y)
and for evaluation:
    method.predict(x_float, t) -> logits over all outputs (trainer masks unseen classes)
    method.extra_scores(x_float, t) -> {name: tensor[N]} (e.g. FACE forgetting estimate)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.augment import Augmenter


class ContinualMethod:
    uses_buffer = False

    def __init__(self, model: nn.Module, bench, cfg: dict, device):
        self.net = model.to(device)
        self.bench = bench
        self.cfg = cfg
        self.device = device
        self.aug = Augmenter(bench.crop_padding, bench.hflip)
        self.opt = None
        self.task = 0

    # ---- hooks -------------------------------------------------------------------
    def task_train_data(self, t, x, y):
        return x, y

    def begin_task(self, t: int):
        self.task = t
        self.opt = self.make_optimizer()

    def end_task(self, t: int, x, y):
        pass

    def observe(self, x, y, t: int) -> float:
        raise NotImplementedError

    # ---- evaluation ----------------------------------------------------------------
    @torch.no_grad()
    def predict(self, x, t: int):
        return self.net(x)

    @torch.no_grad()
    def extra_scores(self, x, t: int) -> dict:
        return {}

    def buffer_snapshot(self):
        """Buffer contents to save for offline post-hoc calibration (None if no buffer)."""
        return None

    # ---- helpers ---------------------------------------------------------------------
    def make_optimizer(self):
        return torch.optim.SGD(self.net.parameters(), lr=float(self.cfg["lr"]),
                               momentum=float(self.cfg.get("momentum", 0.0)),
                               weight_decay=float(self.cfg.get("weight_decay", 0.0)))

    def n_seen(self, t: int) -> int:
        return self.bench.n_seen_classes(t)

    def ce(self, logits, y, t: int):
        """Cross-entropy; optionally restricted to classes seen so far (cfg.mask_future)."""
        if self.cfg.get("mask_future", False):
            logits = logits[:, : self.n_seen(t)]
        return F.cross_entropy(logits, y)

    def step(self, loss):
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

    def state_dict(self) -> dict:
        return {"net": self.net.state_dict()}

    def load_state_dict(self, state: dict):
        self.net.load_state_dict(state["net"])
