"""Layered YAML configuration.

Precedence (later wins):
  configs/defaults.yaml
  configs/datasets/<dataset>.yaml
  configs/methods/<method>.yaml            (top-level keys)
  configs/methods/<method>.yaml -> per_dataset[<dataset>][<buffer_size>|"any"]
  command-line overrides  key=value
"""
from pathlib import Path

import yaml

from .utils import PROJECT_ROOT

CONFIG_DIR = PROJECT_ROOT / "configs"


def _read(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def parse_overrides(items) -> dict:
    """Turn ["lr=0.1", "seed=3"] into {"lr": 0.1, "seed": 3} using YAML typing."""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(value)
    return out


def _method_config(method: str) -> dict:
    """Method YAML, resolving `inherit: <other method>` (e.g. FACE inherits DER++'s settings)."""
    own = _read(CONFIG_DIR / "methods" / f"{method}.yaml")
    parent = own.pop("inherit", None)
    if parent is None:
        return own
    merged = _method_config(parent)
    per = merged.pop("per_dataset", {}) or {}
    for ds, blocks in (own.pop("per_dataset", {}) or {}).items():
        for key, values in (blocks or {}).items():
            per.setdefault(ds, {}).setdefault(key, {}).update(values or {})
    merged.update(own)
    merged["per_dataset"] = per
    return merged


def load_config(dataset: str, method: str, overrides=None) -> dict:
    overrides = dict(overrides or {})
    cfg = _read(CONFIG_DIR / "defaults.yaml")
    cfg.update(_read(CONFIG_DIR / "datasets" / f"{dataset}.yaml"))
    method_cfg = _method_config(method)
    per_dataset = method_cfg.pop("per_dataset", {}) or {}
    cfg.update(method_cfg)
    # buffer size may come from the command line, and selects the per-dataset block
    buffer_size = overrides.get("buffer_size", cfg.get("buffer_size", 0))
    block = per_dataset.get(dataset, {}) or {}
    cfg.update(block.get("any", {}) or {})
    cfg.update(block.get(buffer_size, {}) or {})
    cfg.update(overrides)
    cfg["dataset"] = dataset
    cfg["method"] = method
    if cfg.get("order_seed") is None:
        cfg["order_seed"] = cfg["seed"]
    return cfg
