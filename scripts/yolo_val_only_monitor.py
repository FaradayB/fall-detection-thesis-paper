#!/usr/bin/env python3
"""
Validation-Only Resource Monitor for Ultralytics YOLO
----------------------------------------------------
- NO TRAINING. This script only runs validation on already-trained weights.
- Based on your prior run layout:
    runs_2/train/YOLO_noD_Reversed_lr.<lr>/weights/{best.pt|last.pt}
- For each learning rate, loads the weights and runs `model.val()` with callbacks that capture:
  * per-batch inference time (ms)
  * total validation wall time (s)
  * process CPU% / RAM (MB), system CPU% / RAM%
  * GPU util% and VRAM used (MB) via NVML (if available)
  * torch CUDA peak mem (allocated/reserved)

Outputs (per LR run):
- <project>/<name>/val_monitor/resource_summary.json
- <project>/<name>/val_monitor/metrics.csv

Aggregate:
- <project>/val_monitor_summary.csv (one row per LR)

Prereqs:
  pip install ultralytics psutil pynvml pandas
  # + matching torch/torchvision for your CUDA
"""

import os
import json
import time
import statistics as stats
from datetime import datetime
from pathlib import Path

import psutil
import torch
import pandas as pd

# Optional GPU telemetry via NVML
try:
    import pynvml
    _NVML_READY = True
    pynvml.nvmlInit()
except Exception:
    _NVML_READY = False

# ------------------ Callback for validation monitoring -----------------------

_PROC = psutil.Process(os.getpid())

def _prime_cpu_percent_samplers():
    try:
        _PROC.cpu_percent(None)
    except Exception:
        pass
    try:
        psutil.cpu_percent(None)
    except Exception:
        pass

def get_cpu_ram_snapshot():
    """Return (proc_cpu%, proc_rss_MB, sys_cpu%, sys_mem%)."""
    try:
        p_cpu = _PROC.cpu_percent(None)
        rss_mb = _PROC.memory_info().rss / (1024**2)
    except Exception:
        p_cpu, rss_mb = None, None
    try:
        sys_cpu = psutil.cpu_percent(None)
        sys_mem = psutil.virtual_memory().percent
    except Exception:
        sys_cpu, sys_mem = None, None
    return p_cpu, rss_mb, sys_cpu, sys_mem

def update_gpu_maxima(max_gpu_util, max_gpu_mem_used):
    """Update dicts with max GPU util% and VRAM used (MB) per device index."""
    if not torch.cuda.is_available():
        return
    try:
        n = torch.cuda.device_count()
    except Exception:
        n = 0
    if n == 0:
        return

    if _NVML_READY:
        for i in range(n):
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                max_gpu_util[i] = max(max_gpu_util.get(i, 0), int(util.gpu))
                used_mb = mem.used / (1024**2)
                max_gpu_mem_used[i] = max(max_gpu_mem_used.get(i, 0.0), used_mb)
            except Exception:
                pass
    else:
        for i in range(n):
            try:
                used_mb = torch.cuda.memory_allocated(i) / (1024**2)
                max_gpu_mem_used[i] = max(max_gpu_mem_used.get(i, 0.0), used_mb)
            except Exception:
                pass

class ValResourceCallback:
    def __init__(self):
        self.batch_times = []
        self.max_proc_ram_mb = 0.0
        self.max_proc_cpu = 0.0
        self.max_sys_cpu = 0.0
        self.max_sys_mem = 0.0
        self.max_gpu_util = {}      # {gpu_index: percent}
        self.max_gpu_mem_used = {}  # {gpu_index: MB}
        self._t0 = None
        self._val_start = None
        self.summary = {}

    def on_val_start(self, trainer):
        _prime_cpu_percent_samplers()
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        self._val_start = time.perf_counter()
        update_gpu_maxima(self.max_gpu_util, self.max_gpu_mem_used)

    def on_val_batch_start(self, trainer):
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        self._t0 = time.perf_counter()

    def on_val_batch_end(self, trainer):
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        if self._t0 is not None:
            dt = time.perf_counter() - self._t0
            self.batch_times.append(dt)

        p_cpu, rss_mb, sys_cpu, sys_mem = get_cpu_ram_snapshot()
        if rss_mb is not None:
            self.max_proc_ram_mb = max(self.max_proc_ram_mb, rss_mb)
        if p_cpu is not None:
            self.max_proc_cpu = max(self.max_proc_cpu, p_cpu)
        if sys_cpu is not None:
            self.max_sys_cpu = max(self.max_sys_cpu, sys_cpu)
        if sys_mem is not None:
            self.max_sys_mem = max(self.max_sys_mem, sys_mem)

        update_gpu_maxima(self.max_gpu_util, self.max_gpu_mem_used)

    def on_val_end(self, trainer):
        total_val_time = time.perf_counter() - (self._val_start or time.perf_counter())
        if torch.cuda.is_available():
            try:
                peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024**2)
                peak_res_mb = torch.cuda.max_memory_reserved() / (1024**2)
            except Exception:
                peak_alloc_mb, peak_res_mb = None, None
        else:
            peak_alloc_mb = peak_res_mb = None

        if self.batch_times:
            bt_ms = [t * 1000.0 for t in self.batch_times]
            try:
                p50 = stats.median(bt_ms)
            except Exception:
                p50 = None
            try:
                p95 = stats.quantiles(bt_ms, n=20)[-1] if len(bt_ms) >= 2 else None
            except Exception:
                p95 = None
            avg = sum(bt_ms) / len(bt_ms)
        else:
            avg = p50 = p95 = None

        self.summary = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "num_batches": len(self.batch_times),
            "batch_time_ms_avg": avg,
            "batch_time_ms_p50": p50,
            "batch_time_ms_p95": p95,
            "total_val_time_s": total_val_time,
            "max_process_cpu_percent": self.max_proc_cpu,
            "max_process_ram_mb": self.max_proc_ram_mb,
            "max_system_cpu_percent": self.max_sys_cpu,
            "max_system_mem_percent": self.max_sys_mem,
            "max_gpu_util_percent": self.max_gpu_util,
            "max_gpu_mem_used_mb": self.max_gpu_mem_used,
            "torch_peak_cuda_alloc_mb": peak_alloc_mb,
            "torch_peak_cuda_reserved_mb": peak_res_mb,
            "nvml_available": _NVML_READY,
        }

def run_validation_with_metrics(weights_path, data_yaml, imgsz=640, batch=16, device="cuda", half=True, workers=4, out_dir="val_monitor"):
    """Run Ultralytics model.val() with resource/timing callbacks, save to out_dir."""
    from ultralytics import YOLO

    cb = ValResourceCallback()

    model = YOLO(weights_path)

    # Register callbacks via the public API to support Ultralytics versions that
    # do not accept 'callbacks=' in model.val/train.
    for event, fn in [
        ("on_val_start", cb.on_val_start),
        ("on_val_batch_start", cb.on_val_batch_start),
        ("on_val_batch_end", cb.on_val_batch_end),
        ("on_val_end", cb.on_val_end),
    ]:
        try:
            model.add_callback(event, fn)
        except Exception:
            pass

    results = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device,
        half=half,
        workers=workers,
        verbose=True,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resource summary
    with open(out_dir / "resource_summary.json", "w", encoding="utf-8") as f:
        json.dump(cb.summary, f, indent=2)

    # Ultralytics metrics
    try:
        pd.DataFrame([results.results_dict]).to_csv(out_dir / "metrics.csv", index=False)
    except Exception as e:
        with open(out_dir / "metrics.txt", "w", encoding="utf-8") as f:
            f.write(str(results))

    return cb.summary, out_dir

# --------------------------- Main (validation only) --------------------------

def main():
    # ====== USER CONFIG (based on your prior setup) ======
    DATA_YAML = "dataset_paper_new/data_nano_rev.yaml"
    VAL_BATCH = 16
    IMGSZ = 640
    DEVICE = "cuda"      # or 0 / "cpu"
    PROJECT = "runs_2/train"
    NAME_PREFIX = "YOLO_noD_Reversed_lr"
    USE_HALF_FOR_VAL = True
    # Pick the LRs for which you already have trained runs:
    LEARNING_RATES = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    # Which weight file to prefer
    PREFER_WEIGHTS = "best.pt"     # fallback to last.pt if best doesn't exist
    # =====================================================

    project_dir = Path(PROJECT)
    project_dir.mkdir(parents=True, exist_ok=True)

    aggregate_rows = []

    for lr in LEARNING_RATES:
        run_name = f"{NAME_PREFIX}.{lr}"
        run_dir = project_dir / run_name
        weights_dir = run_dir / "weights"

        preferred = weights_dir / PREFER_WEIGHTS
        fallback  = weights_dir / "last.pt"
        if preferred.exists():
            weights_path = preferred
        elif fallback.exists():
            weights_path = fallback
        else:
            print(f"[SKIP] No weights found for lr={lr} at {weights_dir}")
            continue

        print(f"\n=== Validating {weights_path} (lr={lr}) ===\n")
        val_out_dir = run_dir / "val_monitor"

        summary, out_dir = run_validation_with_metrics(
            weights_path=str(weights_path),
            data_yaml=DATA_YAML,
            imgsz=IMGSZ,
            batch=VAL_BATCH,
            device=DEVICE,
            half=USE_HALF_FOR_VAL,
            workers=4,
            out_dir=str(val_out_dir),
        )

        # Load Ultralytics metrics (if available) to aggregate
        metrics_csv = out_dir / "metrics.csv"
        metrics = {}
        if metrics_csv.exists():
            try:
                df = pd.read_csv(metrics_csv)
                metrics = df.iloc[0].to_dict()
            except Exception:
                pass

        # Flatten GPU dicts for convenience (max across devices)
        def _max_or_none(d):
            try:
                return max(d.values()) if isinstance(d, dict) and len(d) > 0 else None
            except Exception:
                return None

        row = {
            "lr0": lr,
            "weights": str(weights_path),
            "run_dir": str(run_dir),
            "total_val_time_s": summary.get("total_val_time_s"),
            "batch_time_ms_avg": summary.get("batch_time_ms_avg"),
            "batch_time_ms_p50": summary.get("batch_time_ms_p50"),
            "batch_time_ms_p95": summary.get("batch_time_ms_p95"),
            "max_process_cpu_percent": summary.get("max_process_cpu_percent"),
            "max_process_ram_mb": summary.get("max_process_ram_mb"),
            "max_system_cpu_percent": summary.get("max_system_cpu_percent"),
            "max_system_mem_percent": summary.get("max_system_mem_percent"),
            "max_gpu_util_percent": _max_or_none(summary.get("max_gpu_util_percent")),
            "max_gpu_mem_used_mb": _max_or_none(summary.get("max_gpu_mem_used_mb")),
            "torch_peak_cuda_alloc_mb": summary.get("torch_peak_cuda_alloc_mb"),
            "torch_peak_cuda_reserved_mb": summary.get("torch_peak_cuda_reserved_mb"),
        }

        # Merge Ultralytics val metrics if present (e.g., precision/recall/mAP50/mAP50-95)
        row.update(metrics)
        aggregate_rows.append(row)

    # Save aggregate CSV at project root for convenience
    agg_path = project_dir / "val_monitor_summary.csv"
    if aggregate_rows:
        pd.DataFrame(aggregate_rows).to_csv(agg_path, index=False)
        print(f"\nSaved aggregate summary: {agg_path.resolve()}")
    else:
        print("\nNo validations run (no weights found). Check your run directories and learning-rate list.")

if __name__ == "__main__":
    main()
