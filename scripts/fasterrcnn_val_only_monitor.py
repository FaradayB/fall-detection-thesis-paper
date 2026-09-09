#!/usr/bin/env python3
"""
Faster R-CNN Validation-Only Resource Monitor
---------------------------------------------
- NO TRAINING. This script runs validation on a torchvision Faster R-CNN model using your
  VOCLikeDataset setup, and captures:
    * per-batch inference time (ms)
    * total validation wall time (s)
    * process CPU% / RAM (MB), system CPU% / RAM%
    * GPU util% and VRAM used (MB) via NVML (if available)
    * torch CUDA peak memory (allocated/reserved)

- Saves:
  runs/train/Faster_R-CNN_Optimized/val_monitor/resource_summary.json
  runs/train/Faster_R-CNN_Optimized/val_monitor/metrics.csv

Prereqs:
  pip install psutil pynvml pandas torchmetrics tqdm
"""

import os
import json
import time
import csv
from pathlib import Path

import psutil
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, f1_score, confusion_matrix
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pandas as pd

# Optional GPU telemetry via NVML
try:
    import pynvml
    _NVML_READY = True
    pynvml.nvmlInit()
except Exception:
    _NVML_READY = False

# =========================
# CONFIG (edit as needed)
# =========================

CLASS_NAMES = ["fall", "no_fall"]
CLASS_MAP = {name.lower().replace(" ", "_"): idx + 1 for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES) + 1  # +1 background

# Dataset
DATASET_BASE = 'new_dataset'  # expects images/{train,val} and labels_voc/{train,val}
INPUT_SIZE = 640
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# Validation loader
VAL_BATCH_SIZE = 1
NUM_WORKERS = 2

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_HALF = False  # torchvision Faster R-CNN usually runs FP32

# Weights (optional): set to a .pt path of your trained model (state_dict)
# Example default from your training script:
WEIGHTS_PATH = 'runs/train/Faster_R-CNN_Optimized/model_final.pt'  # set to None if not available

# Output dirs
RUN_DIR = Path('runs/train/Faster_R-CNN_Optimized')
VAL_OUT_DIR = RUN_DIR / 'val_monitor'

# =========================
# Dataset & transforms
# =========================

def get_transform():
    return T.Compose([T.Resize((INPUT_SIZE, INPUT_SIZE)), T.ToTensor()])

class VOCLikeDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir, transforms=None, class_map=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms
        self.class_map = class_map or CLASS_MAP
        self.files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".jpg",".png",".jpeg"))])
        
    def __len__(self): 
        return len(self.files)
        
    def __getitem__(self, idx):
        img_file = self.files[idx]
        img = Image.open(os.path.join(self.images_dir, img_file)).convert("RGB")
        xml_path = os.path.join(self.labels_dir, img_file.rsplit('.',1)[0] + '.xml')
        
        boxes, labels = [], []
        try:
            tree = ET.parse(xml_path)
            for obj in tree.getroot().findall('object'):
                cls = obj.find('name').text.lower().replace(" ", "_")
                if cls not in self.class_map:
                    # unknown class: skip
                    continue
                labels.append(self.class_map[cls])
                b = obj.find('bndbox')
                boxes.append([
                    float(b.find('xmin').text), 
                    float(b.find('ymin').text),
                    float(b.find('xmax').text), 
                    float(b.find('ymax').text)
                ])
        except Exception as e:
            # missing/invalid xml -> empty targets
            pass
        
        if self.transforms:
            img = self.transforms(img)
            
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }
        return img, target

# =========================
# Resource monitoring utils
# =========================

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
        # Fallback: track per-device torch allocated bytes (not total board usage)
        for i in range(n):
            try:
                used_mb = torch.cuda.memory_allocated(i) / (1024**2)
                max_gpu_mem_used[i] = max(max_gpu_mem_used.get(i, 0.0), used_mb)
            except Exception:
                pass

# =========================
# Validation with monitor
# =========================

def validate_with_monitor(model, loader, device):
    """
    Validates the model and records resource/timing stats.
    Returns resource_summary (dict) and metrics dict (precision, recall, mAP50, mAP50-95).
    """
    model.eval()
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    mp50 = MeanAveragePrecision(iou_thresholds=[0.5]).to(device)
    mp_all = MeanAveragePrecision().to(device)

    sum_box = sum_cls = sum_d = 0.0
    count = 0

    all_t, all_p, all_s = [], [], []  # for PR/F1/precision/recall

    # Resource tracking
    batch_times = []
    max_proc_ram_mb = 0.0
    max_proc_cpu = 0.0
    max_sys_cpu = 0.0
    max_sys_mem = 0.0
    max_gpu_util = {}
    max_gpu_mem_used = {}

    _prime_cpu_percent_samplers()
    val_t0 = time.perf_counter()
    update_gpu_maxima(max_gpu_util, max_gpu_mem_used)

    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc="Validating"):
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # ----------------- Timed inference -----------------
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = model(imgs)  # eval forward
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            batch_times.append(dt)

            # ----------------- Optional: compute val "loss" (not timed) -----------------
            # Switch briefly to train mode to get losses, then back to eval
            model.train()
            loss_dict = model(imgs, targets)
            model.eval()

            sum_box += float(loss_dict.get('loss_box_reg', 0.0))
            sum_cls += float(loss_dict.get('loss_classifier', 0.0))
            sum_d   += float(loss_dict.get('loss_objectness', 0.0)) + float(loss_dict.get('loss_rpn_box_reg', 0.0))
            count += 1

            # --------------- Metrics update --------------------
            for out, tgt in zip(outputs, targets):
                keep = out['scores'] > SCORE_THRESHOLD
                pred = {
                    'boxes': out['boxes'][keep], 
                    'scores': out['scores'][keep], 
                    'labels': out['labels'][keep]
                }
                gt = {'boxes': tgt['boxes'], 'labels': tgt['labels']}
                mp50.update([pred], [gt])
                mp_all.update([pred], [gt])

                pb, pl, ps = pred['boxes'].cpu(), pred['labels'].cpu(), pred['scores'].cpu()
                gb, gl = gt['boxes'].cpu(), gt['labels'].cpu()

                # simple matching for per-sample PR bookkeeping
                ious = box_iou(pb, gb) if (len(pb) and len(gb)) else None
                gt_used = set()
                for i in range(len(pb)):
                    score = float(ps[i].item())
                    label = int(pl[i].item())
                    if ious is not None and ious.numel() > 0:
                        max_iou, gt_idx = ious[i].max(0)
                        if float(max_iou.item()) >= IOU_THRESHOLD and int(gt_idx.item()) not in gt_used:
                            all_p.append(label); all_t.append(int(gl[gt_idx].item())); all_s.append(score)
                            gt_used.add(int(gt_idx.item()))
                            continue
                    # false positive
                    all_p.append(label); all_t.append(0); all_s.append(score)
                # remaining false negatives
                for j in range(len(gb)):
                    if j not in gt_used:
                        all_p.append(0); all_t.append(int(gl[j].item())); all_s.append(0.0)

            # --------------- Resource snapshots per batch ------
            p_cpu, rss_mb, sys_cpu, sys_mem = get_cpu_ram_snapshot()
            if rss_mb is not None:
                max_proc_ram_mb = max(max_proc_ram_mb, rss_mb)
            if p_cpu is not None:
                max_proc_cpu = max(max_proc_cpu, p_cpu)
            if sys_cpu is not None:
                max_sys_cpu = max(max_sys_cpu, sys_cpu)
            if sys_mem is not None:
                max_sys_mem = max(max_sys_mem, sys_mem)

            update_gpu_maxima(max_gpu_util, max_gpu_mem_used)

    total_val_time = time.perf_counter() - val_t0

    # Torch CUDA peak mem
    if torch.cuda.is_available():
        try:
            peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024**2)
            peak_res_mb = torch.cuda.max_memory_reserved() / (1024**2)
        except Exception:
            peak_alloc_mb = peak_res_mb = None
    else:
        peak_alloc_mb = peak_res_mb = None

    # Metrics
    res50 = float(mp50.compute()['map'].item())
    res_all = float(mp_all.compute()['map'].item())

    labels = list(range(1, NUM_CLASSES))
    precision = precision_score(all_t, all_p, labels=labels, average='weighted', zero_division=0)
    recall = recall_score(all_t, all_p, labels=labels, average='weighted', zero_division=0)

    losses = {
        'box': (sum_box / count) if count else None,
        'cls': (sum_cls / count) if count else None,
        'dfl': (sum_d  / count) if count else None,  # dfl = objectness + rpn_box_reg (naming for compatibility with your tables)
    }
    metrics = {'precision': precision, 'recall': recall, 'mAP50': res50, 'mAP50-95': res_all}

    # Summarize batch timings
    if len(batch_times) > 0:
        bt_ms = [t * 1000.0 for t in batch_times]
        bt_avg = float(sum(bt_ms) / len(bt_ms))
        bt_p50 = float(np.median(bt_ms))
        bt_p95 = float(np.percentile(bt_ms, 95))
    else:
        bt_avg = bt_p50 = bt_p95 = None

    summary = {
        "num_batches": len(batch_times),
        "batch_time_ms_avg": bt_avg,
        "batch_time_ms_p50": bt_p50,
        "batch_time_ms_p95": bt_p95,
        "total_val_time_s": float(total_val_time),
        "max_process_cpu_percent": float(max_proc_cpu),
        "max_process_ram_mb": float(max_proc_ram_mb),
        "max_system_cpu_percent": float(max_sys_cpu),
        "max_system_mem_percent": float(max_sys_mem),
        "max_gpu_util_percent": {k: float(v) for k, v in max_gpu_util.items()},
        "max_gpu_mem_used_mb": {k: float(v) for k, v in max_gpu_mem_used.items()},
        "torch_peak_cuda_alloc_mb": float(peak_alloc_mb) if peak_alloc_mb is not None else None,
        "torch_peak_cuda_reserved_mb": float(peak_res_mb) if peak_res_mb is not None else None,
        "nvml_available": _NVML_READY,
        "losses": losses,
        "metrics": metrics,
    }
    return summary, metrics

# =========================
# Main
# =========================

def build_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def main():
    print(f"Using device: {DEVICE}")
    VAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Dataset / loader
    val_ds = VOCLikeDataset(f"{DATASET_BASE}/images/val", f"{DATASET_BASE}/labels_voc/val", get_transform())
    val_loader = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False,
                            collate_fn=lambda x: tuple(zip(*x)), num_workers=NUM_WORKERS)

    print(f"Validation samples: {len(val_ds)}")

    # Model
    model = build_model(NUM_CLASSES)
    model.to(DEVICE)

    # Load weights if available
    if WEIGHTS_PATH and Path(WEIGHTS_PATH).exists():
        print(f"Loading weights: {WEIGHTS_PATH}")
        sd = torch.load(WEIGHTS_PATH, map_location=DEVICE)
        # Handle both state_dict-only and checkpoint dicts
        if isinstance(sd, dict) and all(k.startswith('backbone.') or 'roi_heads' in k or 'rpn' in k for k in sd.keys()):
            model.load_state_dict(sd, strict=False)
        elif isinstance(sd, dict) and 'model' in sd:
            model.load_state_dict(sd['model'], strict=False)
        else:
            # try direct
            try:
                model.load_state_dict(sd, strict=False)
            except Exception as e:
                print(f"Warning: couldn't load state_dict strictly: {e}")

    # Validate with monitoring
    summary, metrics = validate_with_monitor(model, val_loader, DEVICE)

    # Save outputs
    with open(VAL_OUT_DIR / "resource_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # metrics.csv (flat)
    flat = {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "mAP50": metrics["mAP50"],
        "mAP50-95": metrics["mAP50-95"],
        "batch_time_ms_avg": summary["batch_time_ms_avg"],
        "batch_time_ms_p50": summary["batch_time_ms_p50"],
        "batch_time_ms_p95": summary["batch_time_ms_p95"],
        "total_val_time_s": summary["total_val_time_s"],
        "max_process_cpu_percent": summary["max_process_cpu_percent"],
        "max_process_ram_mb": summary["max_process_ram_mb"],
        "max_system_cpu_percent": summary["max_system_cpu_percent"],
        "max_system_mem_percent": summary["max_system_mem_percent"],
        "torch_peak_cuda_alloc_mb": summary["torch_peak_cuda_alloc_mb"],
        "torch_peak_cuda_reserved_mb": summary["torch_peak_cuda_reserved_mb"],
    }
    pd.DataFrame([flat]).to_csv(VAL_OUT_DIR / "metrics.csv", index=False)

    print(f"\nSaved validation monitor outputs to: {VAL_OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
