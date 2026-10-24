"""Continual-learning benchmarks built from standard image datasets.

Leakage rules enforced here:
  * each task's official training set is split 90/10 into train/val (stratified, fixed per order_seed);
  * official test sets are only exposed through `task_data(t, "test")`;
  * class order is a permutation drawn from `order_seed`; labels are remapped so that
    task t owns the contiguous label range [t*k, (t+1)*k).
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

SUPPORTED = ("seq-mnist", "seq-cifar10", "seq-cifar100", "seq-tinyimg", "rot-mnist", "synthetic")


@dataclass
class ContinualBenchmark:
    name: str
    setting: str  # "class-il" or "domain-il"
    n_tasks: int
    classes_per_task: int
    n_classes: int  # size of the output layer
    in_shape: tuple
    mean: tuple
    std: tuple
    crop_padding: int
    hflip: bool
    class_order: np.ndarray
    _train: tuple = field(repr=False, default=None)
    _val: tuple = field(repr=False, default=None)
    _test: tuple = field(repr=False, default=None)
    _rotations: np.ndarray = field(repr=False, default=None)

    # ---- label bookkeeping -------------------------------------------------------
    def task_classes(self, t: int) -> list:
        if self.setting == "domain-il":
            return list(range(self.n_classes))
        k = self.classes_per_task
        return list(range(t * k, (t + 1) * k))

    def n_seen_classes(self, t: int) -> int:
        """Number of output units active after training task t (0-indexed)."""
        if self.setting == "domain-il":
            return self.n_classes
        return (t + 1) * self.classes_per_task

    # ---- data access -------------------------------------------------------------
    def task_data(self, t: int, split: str):
        """Return (x uint8 [N,C,H,W], y long [N]) for task t and split in {train,val,test}."""
        x, y = {"train": self._train, "val": self._val, "test": self._test}[split]
        if self.setting == "domain-il":
            return self._rotate(x, float(self._rotations[t])), y.clone()
        mask = (y >= t * self.classes_per_task) & (y < (t + 1) * self.classes_per_task)
        return x[mask], y[mask]

    def joint_data(self, split: str, upto: int = None):
        """All data of tasks 0..upto (inclusive) concatenated — used by the Joint upper bound."""
        upto = self.n_tasks - 1 if upto is None else upto
        parts = [self.task_data(t, split) for t in range(upto + 1)]
        return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])

    @staticmethod
    def _rotate(x: torch.Tensor, angle: float) -> torch.Tensor:
        import torchvision.transforms.functional as TF

        return TF.rotate(x, angle)


# ---------------------------------------------------------------------------------
def _stratified_split(y: np.ndarray, val_frac: float, rng: np.random.Generator):
    train_idx, val_idx = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_frac))
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    return np.sort(np.concatenate(train_idx)), np.sort(np.concatenate(val_idx))


def _to_uint8_chw(arr: np.ndarray) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(arr))
    if t.ndim == 3:  # [N,H,W] grayscale
        t = t[:, None]
    elif t.shape[-1] in (1, 3) and t.shape[1] not in (1, 3):  # NHWC -> NCHW
        t = t.permute(0, 3, 1, 2)
    return t.contiguous().to(torch.uint8)


def _load_torchvision(name: str, root: str):
    import torchvision.datasets as D

    cls = {"mnist": D.MNIST, "cifar10": D.CIFAR10, "cifar100": D.CIFAR100}[name]
    tr = cls(root, train=True, download=True)
    te = cls(root, train=False, download=True)
    xtr = tr.data.numpy() if torch.is_tensor(tr.data) else tr.data
    xte = te.data.numpy() if torch.is_tensor(te.data) else te.data
    return xtr, np.asarray(tr.targets), xte, np.asarray(te.targets)


def _load_tinyimagenet(root: str, cache_dir: str):
    """Standard tiny-imagenet-200 layout. The official `val` split is our test set."""
    cache = Path(cache_dir) / "tinyimagenet_uint8.npz"
    if cache.exists():
        d = np.load(cache)
        return d["xtr"], d["ytr"], d["xte"], d["yte"]
    from PIL import Image

    root = Path(root)
    wnids = sorted((root / "wnids.txt").read_text().split())
    index = {w: i for i, w in enumerate(wnids)}
    xtr, ytr = [], []
    for w in wnids:
        for p in sorted((root / "train" / w / "images").glob("*.JPEG")):
            xtr.append(np.asarray(Image.open(p).convert("RGB")))
            ytr.append(index[w])
    xte, yte = [], []
    for line in (root / "val" / "val_annotations.txt").read_text().splitlines():
        fname, w = line.split("\t")[:2]
        xte.append(np.asarray(Image.open(root / "val" / "images" / fname).convert("RGB")))
        yte.append(index[w])
    out = (np.stack(xtr), np.array(ytr), np.stack(xte), np.array(yte))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, xtr=out[0], ytr=out[1], xte=out[2], yte=out[3])
    return out


def _synthetic(n_classes: int, rng: np.random.Generator):
    """Tiny learnable dataset for unit tests: class-dependent mean images + noise."""
    protos = rng.integers(0, 256, size=(n_classes, 8, 8)).astype(np.float32)

    def make(n_per_class):
        xs, ys = [], []
        for c in range(n_classes):
            noise = rng.normal(0, 25, size=(n_per_class, 8, 8))
            xs.append(np.clip(protos[c] + noise, 0, 255))
            ys.append(np.full(n_per_class, c))
        return np.concatenate(xs).astype(np.uint8), np.concatenate(ys)

    xtr, ytr = make(60)
    xte, yte = make(20)
    return xtr, ytr, xte, yte


SPECS = {
    # name: (source, setting, n_tasks, classes_per_task, n_classes, mean, std, crop_pad, hflip)
    "seq-mnist": ("mnist", "class-il", 5, 2, 10, (0.0,), (1.0,), 0, False),
    "rot-mnist": ("mnist", "domain-il", 20, 10, 10, (0.0,), (1.0,), 0, False),
    "seq-cifar10": ("cifar10", "class-il", 5, 2, 10,
                    (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615), 4, True),
    "seq-cifar100": ("cifar100", "class-il", 10, 10, 100,
                     (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762), 4, True),
    "seq-tinyimg": ("tinyimagenet", "class-il", 10, 20, 200,
                    (0.4802, 0.4480, 0.3975), (0.2770, 0.2691, 0.2821), 4, True),
    "synthetic": ("synthetic", "class-il", 3, 2, 6, (0.0,), (1.0,), 0, False),
}


def build_benchmark(cfg: dict) -> ContinualBenchmark:
    name = cfg["dataset"]
    if name not in SPECS:
        raise ValueError(f"unknown dataset {name}; supported: {SUPPORTED}")
    src, setting, n_tasks, k, n_classes, mean, std, pad, flip = SPECS[name]
    n_tasks = int(cfg.get("n_tasks") or n_tasks)
    if setting == "class-il" and cfg.get("n_tasks"):
        k = n_classes // n_tasks  # e.g. CIFAR-100 as 5x20 or 20x5
    data_root = os.path.expanduser(cfg.get("data_root", "./data"))
    rng = np.random.default_rng(int(cfg["order_seed"]))

    if src == "synthetic":
        xtr, ytr, xte, yte = _synthetic(n_classes, np.random.default_rng(12345))
    elif src == "tinyimagenet":
        xtr, ytr, xte, yte = _load_tinyimagenet(cfg["tinyimagenet_root"], data_root)
    else:
        xtr, ytr, xte, yte = _load_torchvision(src, data_root)

    # class order: a permutation per order_seed (identity for domain-IL)
    if setting == "class-il" and cfg.get("shuffle_classes", True):
        order = rng.permutation(n_classes)
    else:
        order = np.arange(n_classes)
    remap = np.empty(n_classes, dtype=np.int64)
    remap[order] = np.arange(n_classes)
    ytr, yte = remap[ytr], remap[yte]

    tr_idx, va_idx = _stratified_split(ytr, float(cfg.get("val_fraction", 0.1)), rng)
    xtr_t, xte_t = _to_uint8_chw(xtr), _to_uint8_chw(xte)
    ytr_t, yte_t = torch.as_tensor(ytr, dtype=torch.long), torch.as_tensor(yte, dtype=torch.long)

    rotations = None
    if setting == "domain-il":
        rotations = rng.uniform(0, 180, size=n_tasks)

    return ContinualBenchmark(
        name=name, setting=setting, n_tasks=n_tasks, classes_per_task=k, n_classes=n_classes,
        in_shape=tuple(xtr_t.shape[1:]), mean=mean, std=std, crop_padding=pad, hflip=flip,
        class_order=order,
        _train=(xtr_t[tr_idx], ytr_t[tr_idx]), _val=(xtr_t[va_idx], ytr_t[va_idx]),
        _test=(xte_t, yte_t), _rotations=rotations,
    )
