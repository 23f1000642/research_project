"""FACE: Forgetting-Aware Confidence Estimation (see docs/phase2_method.md).

Built on DER++. Additions:
  * probe split: a fraction rho of buffer slots is never replayed; it only supervises the head;
  * sample-level forgetting target  r_i = [p_snap(y_i) - p_now(y_i)]_+ , where p_snap is the
    softmax of the logits stored at insertion over the classes seen then, and p_now the current
    softmax over the classes seen now;
  * forgetting head h(stopgrad(features)) in [0,1], trained with MSE on probe items;
  * confidence c(x) = max_y p(y|x) * (1 - h(x))  (computed offline from saved logits + h);
  * optional forgetting-prioritised replay (FACE-PR): replay sampling prob ~ (r_hat + eps)^gamma.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..baselines.replay import DERpp


class ForgettingHead(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, feats):
        return torch.sigmoid(self.net(feats)).squeeze(1)


class FACE(DERpp):
    def __init__(self, model, bench, cfg, device):
        super().__init__(model, bench, cfg, device, probe_fraction=float(cfg.get("probe_fraction", 0.1)))
        self.head = ForgettingHead(model.feat_dim, int(cfg.get("head_hidden", 128))).to(device)
        self.head_opt = torch.optim.Adam(self.head.parameters(), lr=float(cfg.get("head_lr", 1e-3)))
        self.probe_bs = int(cfg.get("probe_batch_size", 32))
        self.prioritized = bool(cfg.get("prioritized_replay", False))
        self.pr_gamma = float(cfg.get("pr_gamma", 1.0))
        self.pr_eps = float(cfg.get("pr_eps", 0.01))

    # ---- forgetting target -----------------------------------------------------------
    def forgetting_target(self, idx, logits_now, t):
        """r_i for buffer items `idx` given current logits (any dimension >= n_seen(t))."""
        y = self.buffer.y[idx]
        p_now = F.softmax(logits_now[:, : self.n_seen(t)].float(), 1).gather(1, y[:, None]).squeeze(1)
        stored = self.buffer.logits[idx]
        tasks = self.buffer.task[idx]
        p_snap = torch.empty_like(p_now)
        for tau in tasks.unique().tolist():  # classes seen at insertion time differ per task
            m = tasks == tau
            n_tau = self.n_seen(int(tau))
            p_snap[m] = F.softmax(stored[m, :n_tau], 1).gather(1, y[m, None]).squeeze(1)
        return (p_snap - p_now).clamp_min(0.0)

    # ---- replay sampling -----------------------------------------------------------------
    def replay_indices(self, n):
        if not self.prioritized:
            return self.buffer.sample(n, probe=False)
        w = (self.buffer.score + self.pr_eps) ** self.pr_gamma
        return self.buffer.sample(n, probe=False, weights=w)

    # ---- training ----------------------------------------------------------------------------
    def observe(self, x, y, t):
        logits = self.net(self.aug(x))
        loss = self.ce(logits, y, t)
        replay_loss, idx2, out2 = self.replay_losses(t)
        loss = loss + replay_loss
        self.step(loss)
        if self.prioritized and idx2 is not None and idx2.numel() > 0:
            with torch.no_grad():  # stale forgetting scores for prioritised replay
                self.buffer.score[idx2] = self.forgetting_target(idx2, out2.detach(), t)
        self.buffer.add(x, y, t, logits=logits.detach())
        self.train_head(t, steps=1)
        return loss.item()

    def train_head(self, t, steps=1):
        for _ in range(steps):
            idx = self.buffer.sample(self.probe_bs, probe=True)
            if idx.numel() == 0:
                return
            with torch.no_grad():
                was = self.net.training
                self.net.eval()  # eval mode: probe items must not touch BatchNorm statistics
                logits, feats = self.net(self.aug(self.buffer.x[idx], train=False), return_features=True)
                self.net.train(was)
                target = self.forgetting_target(idx, logits, t)
            with torch.autocast(device_type=self.device.type, enabled=False):  # small head: fp32
                pred = self.head(feats.float())
                head_loss = F.mse_loss(pred, target.float())
            self.head_opt.zero_grad(set_to_none=True)
            head_loss.backward()
            self.head_opt.step()

    def end_task(self, t, x, y):
        # refit the head on the final parameters of this task (they no longer change before testing)
        self.train_head(t, steps=int(self.cfg.get("head_refit_steps", 200)))

    # ---- evaluation ------------------------------------------------------------------------------
    @torch.no_grad()
    def extra_scores(self, x, t):
        feats = self.net.features(x)
        return {"face_h": self.head(feats)}

    def state_dict(self):
        s = super().state_dict()
        s["head"] = self.head.state_dict()
        s["head_opt"] = self.head_opt.state_dict()
        return s

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.head.load_state_dict(state["head"])
        self.head_opt.load_state_dict(state["head_opt"])
