"""iCaRL (Rebuffi et al., CVPR 2017).

* class-balanced exemplar memory chosen by herding on L2-normalised features;
* training on current data + exemplars with binary cross-entropy, where old-class targets
  are the sigmoid outputs of the previous model (distillation);
* nearest-mean-of-exemplars classification.

For uncertainty evaluation we expose NMC "logits" = -||f(x) - mu_c||^2 / temperature on
normalised features (cfg.nmc_temp); the argmax equals the NMC prediction.
"""
import copy

import torch
import torch.nn.functional as F

from .base import ContinualMethod


class ICaRL(ContinualMethod):
    uses_buffer = True

    def __init__(self, model, bench, cfg, device):
        super().__init__(model, bench, cfg, device)
        self.mem_size = int(cfg["buffer_size"])
        self.mem_x = torch.zeros((0, *bench.in_shape), dtype=torch.uint8, device=device)
        self.mem_y = torch.zeros(0, dtype=torch.long, device=device)
        self.old = None
        self.class_means = None

    def task_train_data(self, t, x, y):
        if self.mem_x.shape[0] == 0:
            return x, y
        return torch.cat([x, self.mem_x]), torch.cat([y, self.mem_y])

    def begin_task(self, t):
        super().begin_task(t)
        self.class_means = None

    def observe(self, x, y, t):
        xa = self.aug(x)
        n_seen = self.n_seen(t)
        logits = self.net(xa)[:, :n_seen]
        targets = F.one_hot(y, n_seen).float()
        if self.old is not None and t > 0:
            n_old = self.n_seen(t - 1)
            with torch.no_grad():
                targets[:, :n_old] = torch.sigmoid(self.old(xa)[:, :n_old])
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        self.step(loss)
        return loss.item()

    @torch.no_grad()
    def _features(self, x_uint8, bs=256):
        was = self.net.training
        self.net.eval()
        feats = []
        for s in range(0, x_uint8.shape[0], bs):
            xb = self.aug(x_uint8[s:s + bs], train=False)
            feats.append(F.normalize(self.net.features(xb), dim=1))
        self.net.train(was)
        return torch.cat(feats)

    @torch.no_grad()
    def end_task(self, t, x, y):
        n_seen = self.n_seen(t)
        per_class = self.mem_size // n_seen
        new_x, new_y = [], []
        # shrink old exemplar sets (herding order is preserved, keep the first m)
        for c in range(self.n_seen(t - 1) if t > 0 else 0):
            idx = (self.mem_y == c).nonzero().squeeze(1)[:per_class]
            new_x.append(self.mem_x[idx])
            new_y.append(self.mem_y[idx])
        # herding for the new classes (x also holds old exemplars, but those have old labels)
        for c in self.bench.task_classes(t):
            xc = x[y == c]
            feats = self._features(xc)
            mu = feats.mean(0)
            chosen, running = [], torch.zeros_like(mu)
            available = torch.ones(feats.shape[0], dtype=torch.bool, device=feats.device)
            for k in range(1, min(per_class, feats.shape[0]) + 1):
                dist = ((mu[None] - (running[None] + feats) / k) ** 2).sum(1)
                dist[~available] = float("inf")
                i = int(dist.argmin())
                chosen.append(i)
                available[i] = False
                running += feats[i]
            new_x.append(xc[chosen])
            new_y.append(torch.full((len(chosen),), c, dtype=torch.long, device=x.device))
        self.mem_x, self.mem_y = torch.cat(new_x), torch.cat(new_y)
        self.old = copy.deepcopy(self.net).eval()
        for p in self.old.parameters():
            p.requires_grad_(False)
        self._compute_means(n_seen)

    @torch.no_grad()
    def _compute_means(self, n_seen):
        feats = self._features(self.mem_x)
        means = torch.zeros((n_seen, feats.shape[1]), device=feats.device)
        for c in range(n_seen):
            means[c] = F.normalize(feats[self.mem_y == c].mean(0), dim=0)
        self.class_means = means

    @torch.no_grad()
    def predict(self, x, t):
        if self.class_means is None:
            return self.net(x)
        feats = F.normalize(self.net.features(x), dim=1)
        d2 = torch.cdist(feats, self.class_means) ** 2
        logits = torch.full((x.shape[0], self.bench.n_classes), -1e4, device=x.device)
        logits[:, : self.class_means.shape[0]] = -d2 / float(self.cfg.get("nmc_temp", 0.1))
        return logits

    def buffer_snapshot(self):
        if self.mem_x.shape[0] == 0:
            return None
        return {"x": self.mem_x, "y": self.mem_y,
                "task": self.mem_y // self.bench.classes_per_task,
                "probe": torch.zeros_like(self.mem_y, dtype=torch.bool)}

    def state_dict(self):
        return {"net": self.net.state_dict(), "mem_x": self.mem_x, "mem_y": self.mem_y,
                "old": None if self.old is None else self.old.state_dict(),
                "class_means": self.class_means}

    def load_state_dict(self, state):
        self.net.load_state_dict(state["net"])
        self.mem_x, self.mem_y = state["mem_x"], state["mem_y"]
        self.class_means = state["class_means"]
        if state["old"] is not None:
            self.old = copy.deepcopy(self.net).eval()
            self.old.load_state_dict(state["old"])
