"""Reservoir replay buffer.

Each slot stores: the un-augmented input (uint8), label, logits at insertion time (DER/FACE),
the task index at insertion, a `probe` flag (FACE: never replayed, only supervises the
forgetting head) and a per-item score used by prioritised replay.
"""
import torch


class ReservoirBuffer:
    def __init__(self, size: int, in_shape, n_classes: int, device, store_logits: bool = False,
                 probe_fraction: float = 0.0):
        self.size = int(size)
        self.device = device
        self.probe_fraction = float(probe_fraction)
        self.x = torch.zeros((self.size, *in_shape), dtype=torch.uint8, device=device)
        self.y = torch.zeros(self.size, dtype=torch.long, device=device)
        self.task = torch.zeros(self.size, dtype=torch.long, device=device)
        self.probe = torch.zeros(self.size, dtype=torch.bool, device=device)
        self.score = torch.zeros(self.size, device=device)
        self.logits = (torch.zeros((self.size, n_classes), device=device) if store_logits else None)
        self.seen = 0  # number of stream examples offered to the buffer so far

    def __len__(self):
        return min(self.seen, self.size)

    def is_empty(self):
        return len(self) == 0

    @torch.no_grad()
    def add(self, x, y, task: int, logits=None):
        """Reservoir sampling (Vitter's algorithm R), one example at a time."""
        for i in range(x.shape[0]):
            if self.seen < self.size:
                slot = self.seen
            else:
                slot = int(torch.randint(0, self.seen + 1, (1,)).item())
            self.seen += 1
            if slot >= self.size:
                continue
            self.x[slot] = x[i].to(self.device)
            self.y[slot] = y[i]
            self.task[slot] = task
            self.score[slot] = 0.0
            self.probe[slot] = bool(torch.rand(1).item() < self.probe_fraction)
            if self.logits is not None and logits is not None:
                self.logits[slot] = logits[i].detach().float()

    def indices(self, probe=None):
        n = len(self)
        idx = torch.arange(n, device=self.device)
        if probe is None:
            return idx
        return idx[self.probe[:n] == probe]

    def sample(self, n: int, probe=False, weights=None):
        """Sample up to n indices without replacement from the replay (probe=False) or probe part."""
        pool = self.indices(probe)
        if pool.numel() == 0:
            return pool
        n = min(n, pool.numel())
        if weights is None:
            choice = torch.randperm(pool.numel(), device=self.device)[:n]
        else:
            w = weights[pool].clamp_min(1e-12)
            choice = torch.multinomial(w, n, replacement=False)
        return pool[choice]

    def state_dict(self):
        state = {k: getattr(self, k) for k in ("x", "y", "task", "probe", "score")}
        state["logits"] = self.logits
        state["seen"] = self.seen
        return state

    def load_state_dict(self, state):
        for k in ("x", "y", "task", "probe", "score"):
            getattr(self, k).copy_(state[k])
        if self.logits is not None and state.get("logits") is not None:
            self.logits.copy_(state["logits"])
        self.seen = int(state["seen"])
