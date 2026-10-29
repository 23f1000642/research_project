"""Task-sequential training loop with per-task evaluation, logging and resumable checkpoints.

Outputs in results/<run_id>/:
  config.yaml        full resolved configuration
  env.json           library versions, GPU, git commit
  train.log          human-readable log
  eval/task_XX.npz   after task XX: masked test logits of every seen task (fp16), labels,
                     extra scores (e.g. FACE head), current-task validation logits and the
                     buffer's logits/labels/task/probe flags (for offline post-hoc calibration)
  last.pt            resumable checkpoint (overwritten after every task)
  summary.json       accuracy matrices, timings, memory, parameter counts; status=complete
"""
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from ..baselines import build_method
from ..data import build_benchmark
from ..models import build_model
from ..utils import (PROJECT_ROOT, config_hash, environment_info, get_device, load_json,
                     rng_state, save_json, set_rng_state, set_seed, setup_logger)

# keys that do not change the experiment itself (excluded from the run id hash)
_NON_EXPERIMENT_KEYS = {"results_dir", "data_root", "tinyimagenet_root", "device", "resume",
                        "data_on_device", "eval_batch_size", "save_checkpoint", "log_every",
                        "experiment_log", "keep_checkpoint", "save_final_model"}


def run_id_for(cfg: dict) -> str:
    exp = {k: v for k, v in cfg.items() if k not in _NON_EXPERIMENT_KEYS}
    return f"{cfg['dataset']}-{cfg['method']}-b{cfg.get('buffer_size', 0)}-s{cfg['seed']}-{config_hash(exp)}"


@torch.no_grad()
def _forward_all(method, x_uint8, t, bs, device):
    """Masked logits + extra scores for a uint8 tensor, in eval mode."""
    net = method.net
    was = net.training
    net.eval()
    n_seen = method.n_seen(t)
    logits, extras = [], {}
    for s in range(0, x_uint8.shape[0], bs):
        xb = method.aug(x_uint8[s:s + bs].to(device, non_blocking=True), train=False)
        logits.append(method.predict(xb, t)[:, :n_seen].float().cpu())
        for k, v in method.extra_scores(xb, t).items():
            extras.setdefault(k, []).append(v.float().cpu())
    net.train(was)
    out = torch.cat(logits) if logits else torch.zeros((0, n_seen))
    return out, {k: torch.cat(v) for k, v in extras.items()}


def evaluate_after_task(method, bench, t, cfg, device, eval_dir: Path):
    bs = int(cfg.get("eval_batch_size", 512))
    acc_cil, acc_til, arrays = [], [], {}
    t0 = time.perf_counter()
    n_test = 0
    for j in range(t + 1):
        x, y = bench.task_data(j, "test")
        logits, extras = _forward_all(method, x, t, bs, device)
        y = y.cpu()
        acc_cil.append(float((logits.argmax(1) == y).float().mean()))
        cls = bench.task_classes(j)
        til_pred = logits[:, cls[0]:cls[-1] + 1].argmax(1) + cls[0]
        acc_til.append(float((til_pred == y).float().mean()))
        arrays[f"test{j}_logits"] = logits.half().numpy()
        arrays[f"test{j}_y"] = y.numpy().astype(np.int16)
        for k, v in extras.items():
            arrays[f"test{j}_{k}"] = v.numpy().astype(np.float32)
        n_test += y.shape[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    infer_ms = 1000.0 * (time.perf_counter() - t0) / max(n_test, 1)

    xv, yv = bench.task_data(t, "val")
    vlog, vext = _forward_all(method, xv, t, bs, device)
    arrays["val_logits"] = vlog.half().numpy()
    arrays["val_y"] = yv.numpy().astype(np.int16)
    for k, v in vext.items():
        arrays[f"val_{k}"] = v.numpy().astype(np.float32)

    snap = method.buffer_snapshot()
    if snap is not None:
        blog, bext = _forward_all(method, snap["x"], t, bs, device)
        arrays["buf_logits"] = blog.half().numpy()
        arrays["buf_y"] = snap["y"].cpu().numpy().astype(np.int16)
        arrays["buf_task"] = snap["task"].cpu().numpy().astype(np.int16)
        arrays["buf_probe"] = snap["probe"].cpu().numpy()
        for k, v in bext.items():
            arrays[f"buf_{k}"] = v.numpy().astype(np.float32)

    eval_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(eval_dir / f"task_{t:02d}.npz", **arrays)
    return acc_cil, acc_til, infer_ms


def _count_params(method):
    n = sum(p.numel() for p in method.net.parameters())
    head = getattr(method, "head", None)
    n_head = sum(p.numel() for p in head.parameters()) if head is not None else 0
    return n, n_head


def _buffer_bytes(method):
    buf = getattr(method, "buffer", None)
    if buf is not None:
        n = len(buf)
        per = buf.x[0].numel() + (buf.logits.shape[1] * 4 if buf.logits is not None else 0) + 8 + 8
        return int(n * per)
    if hasattr(method, "mem_x"):
        return int(method.mem_x.numel() + method.mem_y.numel() * 8)
    return 0


def _append_experiment_log(row: dict, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("| date | run_id | dataset | method | buffer | seed | status | final acc (CIL) "
                        "| minutes | commit |\n|---|---|---|---|---|---|---|---|---|---|\n")
    with open(path, "a") as f:
        f.write("| {date} | {run_id} | {dataset} | {method} | {buffer} | {seed} | {status} | {acc} "
                "| {minutes} | {commit} |\n".format(**row))


def run(cfg: dict) -> Path:
    device = get_device(cfg.get("device", "auto"))
    rid = run_id_for(cfg)
    run_dir = Path(cfg.get("results_dir", PROJECT_ROOT / "results")) / rid
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and load_json(summary_path).get("status") == "complete":
        print(f"[skip] {rid} already complete")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logger(run_dir / "train.log")
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)
    env = environment_info()
    save_json(env, run_dir / "env.json")

    set_seed(int(cfg["seed"]), bool(cfg.get("deterministic", False)))
    bench = build_benchmark(cfg)
    model = build_model(cfg, bench)
    method = build_method(cfg["method"], model, bench, cfg, device)
    log.info(f"run {rid} on {device} | {bench.name} {bench.setting} T={bench.n_tasks} "
             f"k={bench.classes_per_task} | class order {bench.class_order.tolist()}")

    acc_cil, acc_til, timings = [], [], []
    start_task = 0
    ckpt_path = run_dir / "last.pt"
    if ckpt_path.exists() and cfg.get("resume", True):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        method.load_state_dict(ckpt["method"])
        set_rng_state(ckpt["rng"])
        acc_cil, acc_til, timings = ckpt["acc_cil"], ckpt["acc_til"], ckpt["timings"]
        start_task = ckpt["task"] + 1
        log.info(f"resumed after task {ckpt['task']}")

    joint = cfg["method"] == "joint"
    tasks = [bench.n_tasks - 1] if joint else range(start_task, bench.n_tasks)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    on_device = bool(cfg.get("data_on_device", True))
    bs, epochs = int(cfg["batch_size"]), int(cfg["n_epochs"])
    t_start = time.perf_counter()

    for t in tasks:
        x, y = bench.joint_data("train") if joint else bench.task_data(t, "train")
        x, y = (x.to(device), y.to(device)) if on_device else (x, y)
        x, y = method.task_train_data(t, x, y)
        method.begin_task(t)
        method.net.train()
        t0 = time.perf_counter()
        for ep in range(epochs):
            perm = torch.randperm(x.shape[0], device=x.device)
            losses = []
            for s in range(0, x.shape[0], bs):
                idx = perm[s:s + bs]
                xb, yb = x[idx].to(device, non_blocking=True), y[idx].to(device, non_blocking=True)
                with method.autocast():
                    losses.append(method.observe(xb, yb, t))
            method.end_epoch(t, ep)
            if (ep + 1) % int(cfg.get("log_every", 10)) == 0 or ep == epochs - 1:
                log.info(f"task {t} epoch {ep + 1}/{epochs} loss {np.mean(losses):.4f}")
        method.end_task(t, x, y)
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_s = time.perf_counter() - t0

        cil, til, infer_ms = evaluate_after_task(method, bench, t, cfg, device, run_dir / "eval")
        acc_cil.append(cil)
        acc_til.append(til)
        timings.append({"task": t, "train_seconds": train_s, "infer_ms_per_image": infer_ms})
        log.info(f"after task {t}: class-IL acc per task {np.round(cil, 4).tolist()} "
                 f"(mean {np.mean(cil):.4f}) | task-IL mean {np.mean(til):.4f} | train {train_s / 60:.1f} min")
        if cfg.get("save_checkpoint", True) and not joint:
            torch.save({"task": t, "method": method.state_dict(), "rng": rng_state(),
                        "acc_cil": acc_cil, "acc_til": acc_til, "timings": timings}, ckpt_path)

    n_params, n_head = _count_params(method)
    summary = {
        "run_id": rid, "status": "complete", "dataset": cfg["dataset"], "method": cfg["method"],
        "seed": cfg["seed"], "order_seed": cfg["order_seed"], "buffer_size": cfg.get("buffer_size", 0),
        "setting": bench.setting, "n_tasks": bench.n_tasks, "joint": joint,
        "acc_class_il": acc_cil, "acc_task_il": acc_til, "timings": timings,
        "wall_minutes_this_session": (time.perf_counter() - t_start) / 60,
        "params_backbone": n_params, "params_extra": n_head, "buffer_bytes": _buffer_bytes(method),
        "peak_gpu_mem_mb": (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None,
        "env": env, "class_order": bench.class_order.tolist(),
    }
    save_json(summary, summary_path)
    if cfg.get("save_final_model", False):
        torch.save(method.net.state_dict(), run_dir / "model_final.pt")
    if ckpt_path.exists() and not cfg.get("keep_checkpoint", False):
        ckpt_path.unlink()
    _append_experiment_log({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "run_id": rid, "dataset": cfg["dataset"],
        "method": cfg["method"], "buffer": cfg.get("buffer_size", 0), "seed": cfg["seed"],
        "status": "complete", "acc": f"{np.mean(acc_cil[-1]):.4f}",
        "minutes": f"{sum(x['train_seconds'] for x in timings) / 60:.1f}", "commit": env["git_commit"],
    }, cfg.get("experiment_log", PROJECT_ROOT / "logs" / "experiment_log.md"))
    log.info(f"done: final class-IL accuracy {np.mean(acc_cil[-1]):.4f} -> {summary_path}")
    return run_dir
