#!/usr/bin/env python3
"""
YOLO Validation Resource Monitor (Ultralytics YOLOv8/YOLO11)
-----------------------------------------------------------
What it does:
- Hooks into Ultralytics validation via callbacks.
- Measures per-batch inference time (ms), total validation wall time (s).
- Tracks CPU% (process & system), RAM (process RSS), system RAM%.
- Tracks GPU utilization% and VRAM usage via NVIDIA NVML (if available).
- Also records PyTorch peak CUDA memory (allocated & reserved).

Outputs:
- yolo_val_metrics/resource_summary.json  -> resource & timing summary
- yolo_val_metrics/metrics.csv            -> Ultralytics validation metrics (mAP, precision, recall, etc.)

Usage:
1) Install deps (example):
   pip install ultralytics psutil pynvml pandas
   # and the correct torch/torchvision for your CUDA
2) Edit the USER CONFIG at the bottom, then run:
   python yolo_val_resource_monitor.py

Notes:
- GPU utilization requires NVIDIA + NVML (pynvml). If unavailable, the script still runs and
  will fall back to torch.cuda memory stats where possible.
- "Inference time per batch" reflects the time between on_val_batch_start and on_val_batch_end
  in Ultralytics' validation loop (includes model forward & NMS for that batch).
"""

import os
import json
import time
import statistics as stats
from datetime import datetime
from pathlib import Path

import psutil
import torch

# NVML (GPU stats)
try:
    import pynvml
    _NVML_READY = True
    pynvml.nvmlInit()
except Exception:
    _NVML_READY = False

# --- Helpers -----------------------------------------------------------------

_PROC = psutil.Process(os.getpid())

def _prime_cpu_percent_samplers():
    # Prime samplers so first reading isn't 0.0
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
                util = pynvml.nvmlDeviceGetUtilizationRates(h)  # .gpu, .memory (percents)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)         # .used, .total (bytes)
                max_gpu_util[i] = max(max_gpu_util.get(i, 0), int(util.gpu))
                used_mb = mem.used / (1024**2)
                max_gpu_mem_used[i] = max(max_gpu_mem_used.get(i, 0.0), used_mb)
            except Exception:
                # If a particular call fails, skip
                pass
    else:
        # Fallback: track per-device torch allocated bytes (not total board usage)
        for i in range(n):
            try:
                used_mb = torch.cuda.memory_allocated(i) / (1024**2)
                max_gpu_mem_used[i] = max(max_gpu_mem_used.get(i, 0.0), used_mb)
            except Exception:
                pass

# --- Callback collector ------------------------------------------------------

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

    # Ultralytics callback signatures pass a "trainer" argument
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
        # Synchronize to make timing accurate
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

        # Summarize batch timing
        if self.batch_times:
            bt_ms = [t * 1000.0 for t in self.batch_times]
            try:
                p50 = stats.median(bt_ms)
            except Exception:
                p50 = None
            try:
                # p95 via quantiles if enough samples
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
            "max_gpu_util_percent": self.max_gpu_util,        # dict per GPU index
            "max_gpu_mem_used_mb": self.max_gpu_mem_used,     # dict per GPU index
            "torch_peak_cuda_alloc_mb": peak_alloc_mb,
            "torch_peak_cuda_reserved_mb": peak_res_mb,
            "nvml_available": _NVML_READY,
        }
        print(json.dumps(self.summary, indent=2))

# --- Runner ------------------------------------------------------------------

def run_validation_with_metrics(model_path, data_yaml, imgsz=640, batch=16, device=0, half=True, workers=4, project_dir="yolo_val_metrics"):
    from ultralytics import YOLO
    import pandas as pd

    cb = ValResourceCallback()
    callbacks = {
        "on_val_start": cb.on_val_start,
        "on_val_batch_start": cb.on_val_batch_start,
        "on_val_batch_end": cb.on_val_batch_end,
        "on_val_end": cb.on_val_end,
    }

    model = YOLO(model_path)

    # Run validation with our callbacks
    results = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device,
        half=half,
        workers=workers,
        callbacks=callbacks,
        verbose=True,
    )

    # Save outputs
    out_dir = Path(project_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resource summary (JSON)
    with open(out_dir / "resource_summary.json", "w", encoding="utf-8") as f:
        json.dump(cb.summary, f, indent=2)

    # Ultralytics metrics (CSV)
    try:
        pd.DataFrame([results.results_dict]).to_csv(out_dir / "metrics.csv", index=False)
    except Exception as e:
        # Fallback: write str(results)
        with open(out_dir / "metrics.txt", "w", encoding="utf-8") as f:
            f.write(str(results))

    print(f"\nSaved results to: {out_dir.resolve()}")
    return cb.summary

# --- Entry point -------------------------------------------------------------

if __name__ == "__main__":
    # ====== USER CONFIG (edit these) ======
    MODEL_PATH = "yolov8n.pt"      # path to your trained weights (.pt)
    DATA_YAML  = "caucafall.yaml"  # path to your dataset YAML
    IMGSZ      = 640
    BATCH      = 16
    DEVICE     = 0                 # e.g., 0 for first GPU, or "cpu"
    HALF       = True              # use half precision on supported GPUs
    WORKERS    = 4
    # =====================================

    # Run validation with resource monitoring
    run_validation_with_metrics(
        model_path=MODEL_PATH,
        data_yaml=DATA_YAML,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        half=HALF,
        workers=WORKERS,
    )
