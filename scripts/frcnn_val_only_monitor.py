#!/usr/bin/env python3
"""
This is the Faster R-CNN version of the "validate only, sweep across learning
rates" script. For every LR in LEARNING_RATES it looks for a checkpoint under
runs_2/train/FRCNN_noD_Normal_lr.<lr>/ (trying model_final.pt, then best.pt,
then last.pt), loads it, and validates it - no training happens here at all.

While validating, it tracks the same things as the other monitor scripts:
per-batch inference time, total wall time, CPU/RAM, GPU usage through NVML,
PyTorch's peak CUDA memory, plus the actual detection metrics (mAP50,
mAP50-95, precision, recall) and the averaged validation losses.

Each LR gets its own resource_summary.json and metrics.csv inside its
val_monitor folder, and everything is also collected into one CSV:
runs_2/train/val_monitor_summary_FRCNN_valonly.csv.

Needs: torch, torchvision, psutil, pynvml, torchmetrics, pandas, tqdm,
pillow, matplotlib, scipy.
"""

import os
import json
import time
import statistics as stats
from datetime import datetime
from pathlib import Path

import psutil
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix, precision_recall_curve, f1_score
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pandas as pd

# Optional NVML for GPU telemetry
try:
    import pynvml
    _NVML_READY = True
    pynvml.nvmlInit()
except Exception:
    _NVML_READY = False

# Config block - the values I actually change between runs.
CLASS_NAMES = ["fall", "no_fall"]
CLASS_MAP = {name.lower().replace(" ", "_"): idx + 1 for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES) + 1  # + background
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
INPUT_SIZE = 640

VAL_BATCH_SIZE = 1
DATASET_BASE = 'dataset_paper_new'

PROJECT_DIR = Path('runs_2/train')
NAME_PREFIX = 'FRCNN_noD_Normal_lr'
LEARNING_RATES = [0.1, 0.01, 0.001]
WEIGHTS_PREFERENCE = ["model_final.pt", "best.pt", "last.pt"]  # tried in this order

# Same VOC-style dataset loader used in the other Faster R-CNN script.
def get_transform():
    return T.Compose([T.Resize((INPUT_SIZE, INPUT_SIZE)), T.ToTensor()])

class VOCLikeDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir, transforms=None, class_map=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms
        self.class_map = class_map or CLASS_MAP
        self.files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])
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
                    continue
                labels.append(self.class_map[cls])
                b = obj.find('bndbox')
                boxes.append([
                    float(b.find('xmin').text), 
                    float(b.find('ymin').text),
                    float(b.find('xmax').text), 
                    float(b.find('ymax').text)
                ])
        except Exception:
            pass
        if self.transforms:
            img = self.transforms(img)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }
        return img, target

# CPU/RAM/GPU monitoring helpers, written more compactly here than in the
# other scripts but doing the exact same job.
_PROC = psutil.Process(os.getpid())

def _prime_cpu_percent_samplers():
    try: _PROC.cpu_percent(None)
    except Exception: pass
    try: psutil.cpu_percent(None)
    except Exception: pass

def get_cpu_ram_snapshot():
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

# A small class version of the same monitor, just with start/on_batch_start/
# on_batch_end/finish methods instead of Ultralytics-style callbacks, since
# torchvision's Faster R-CNN doesn't have a built-in callback system.
class ValResourceMonitor:
    def __init__(self):
        self.batch_times = []
        self.max_proc_ram_mb = 0.0
        self.max_proc_cpu = 0.0
        self.max_sys_cpu = 0.0
        self.max_sys_mem = 0.0
        self.max_gpu_util = {}
        self.max_gpu_mem_used = {}
        self._t0 = None
        self._val_start = None
        self.summary = {}
    def start(self):
        _prime_cpu_percent_samplers()
        if torch.cuda.is_available():
            try: torch.cuda.reset_peak_memory_stats()
            except Exception: pass
        self._val_start = time.perf_counter()
        update_gpu_maxima(self.max_gpu_util, self.max_gpu_mem_used)
    def on_batch_start(self):
        if torch.cuda.is_available():
            try: torch.cuda.synchronize()
            except Exception: pass
        self._t0 = time.perf_counter()
    def on_batch_end(self):
        if torch.cuda.is_available():
            try: torch.cuda.synchronize()
            except Exception: pass
        if self._t0 is not None:
            self.batch_times.append(time.perf_counter() - self._t0)
        p_cpu, rss_mb, sys_cpu, sys_mem = get_cpu_ram_snapshot()
        if rss_mb is not None: self.max_proc_ram_mb = max(self.max_proc_ram_mb, rss_mb)
        if p_cpu is not None: self.max_proc_cpu = max(self.max_proc_cpu, p_cpu)
        if sys_cpu is not None: self.max_sys_cpu = max(self.max_sys_cpu, sys_cpu)
        if sys_mem is not None: self.max_sys_mem = max(self.max_sys_mem, sys_mem)
        update_gpu_maxima(self.max_gpu_util, self.max_gpu_mem_used)
    def finish(self):
        total_val_time = time.perf_counter() - (self._val_start or time.perf_counter())
        if torch.cuda.is_available():
            try:
                peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024**2)
                peak_res_mb = torch.cuda.max_memory_reserved() / (1024**2)
            except Exception:
                peak_alloc_mb = peak_res_mb = None
        else:
            peak_alloc_mb = peak_res_mb = None
        if self.batch_times:
            bt_ms = [t * 1000.0 for t in self.batch_times]
            try: p50 = stats.median(bt_ms)
            except Exception: p50 = None
            try: p95 = stats.quantiles(bt_ms, n=20)[-1] if len(bt_ms) >= 2 else None
            except Exception: p95 = None
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
        return self.summary

# The actual validation loop - runs the model, scores it, and times it.
def validate_with_monitor(model, loader, device, save_dir, score_thresh=SCORE_THRESHOLD, iou_thresh=IOU_THRESHOLD):
    """Validate one model and write its resource_summary.json and metrics.csv
    into save_dir/val_monitor/.
    """
    save_dir = Path(save_dir)
    vm_dir = save_dir / 'val_monitor'
    vm_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    mp50 = MeanAveragePrecision(iou_thresholds=[0.5]).to(device)
    mp_all = MeanAveragePrecision().to(device)
    sum_b = sum_c = sum_d = 0.0
    count = 0
    all_t, all_p, all_s = [], [], []

    def match_predictions(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_thresh=0.5):
        # Pairs up predictions with ground truth boxes by IoU overlap, so we
        # can tell which predictions were correct, which were false alarms,
        # and which ground-truth boxes got missed entirely.
        matches = []
        if len(pred_boxes) == 0:
            for j in range(len(gt_boxes)):
                matches.append((0, gt_labels[j].item(), 0.0))
            return matches
        if len(gt_boxes) == 0:
            for i in range(len(pred_boxes)):
                matches.append((pred_labels[i].item(), 0, pred_scores[i].item()))
            return matches
        ious = box_iou(pred_boxes, gt_boxes)
        gt_used = set()
        for i in range(len(pred_boxes)):
            score = pred_scores[i].item()
            label = pred_labels[i].item()
            if ious.numel() > 0:
                max_iou, gt_idx = ious[i].max(0)
                if max_iou >= iou_thresh and gt_idx.item() not in gt_used:
                    matches.append((label, gt_labels[gt_idx].item(), score))
                    gt_used.add(gt_idx.item())
                    continue
            matches.append((label, 0, score))
        for j in range(len(gt_boxes)):
            if j not in gt_used:
                matches.append((0, gt_labels[j].item(), 0.0))
        return matches

    monitor = ValResourceMonitor()
    monitor.start()

    with torch.no_grad():
        for imgs, targets in loader:
            monitor.on_batch_start()

            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Get the actual predictions first.
            outputs = model(imgs)

            # torchvision only gives you loss values in train mode, so switch
            # over briefly to compute a validation loss on the same batch.
            model.train()
            loss_dict = model(imgs, targets)
            model.eval()
            sum_b += float(loss_dict.get('loss_box_reg', 0.0))
            sum_c += float(loss_dict.get('loss_classifier', 0.0))
            sum_d += float(loss_dict.get('loss_objectness', 0.0)) + float(loss_dict.get('loss_rpn_box_reg', 0.0))
            count += 1

            # Feed this batch's predictions into the running metrics.
            for out, tgt in zip(outputs, targets):
                keep = out['scores'] > score_thresh
                pred = {'boxes': out['boxes'][keep], 'scores': out['scores'][keep], 'labels': out['labels'][keep]}
                gt = {'boxes': tgt['boxes'], 'labels': tgt['labels']}
                mp50.update([pred], [gt])
                mp_all.update([pred], [gt])

                pb, pl, ps = pred['boxes'].detach().cpu(), pred['labels'].detach().cpu(), pred['scores'].detach().cpu()
                gb, gl = gt['boxes'].detach().cpu(), gt['labels'].detach().cpu()
                for p_label, g_label, p_score in match_predictions(pb, pl, ps, gb, gl, iou_thresh=iou_thresh):
                    all_p.append(p_label)
                    all_t.append(g_label)
                    all_s.append(p_score)

            monitor.on_batch_end()

    summary = monitor.finish()

    # Work out the final numbers now that all batches are done.
    res50 = float(mp50.compute()['map'].item())
    res_all = float(mp_all.compute()['map'].item())
    valid_labels = list(range(1, NUM_CLASSES))
    precision = precision_score(all_t, all_p, labels=valid_labels, average='weighted', zero_division=0)
    recall = recall_score(all_t, all_p, labels=valid_labels, average='weighted', zero_division=0)
    losses = {'box': (sum_b/count) if count else 0.0, 'cls': (sum_c/count) if count else 0.0, 'dfl': (sum_d/count) if count else 0.0}
    metrics = {'precision': precision, 'recall': recall, 'mAP50': res50, 'mAP50-95': res_all}

    # Save outputs
    with open(vm_dir / 'resource_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame([{**losses, **metrics}]).to_csv(vm_dir / 'metrics.csv', index=False)

    return losses, metrics, summary

# Builds a fresh model and loads whichever checkpoint format it turns out to be.
def build_model_and_load(weights_path, device):
    """Build the Faster R-CNN model and load weights from weights_path,
    handling a couple of different checkpoint formats I've saved things in.
    """
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    model.to(device)

    state = torch.load(weights_path, map_location=device)
    loaded = False

    # First just try treating it as a plain state_dict.
    if isinstance(state, dict):
        try:
            model.load_state_dict(state)
            loaded = True
        except Exception:
            pass
        # Didn't work, so maybe it's a checkpoint dict with the state_dict
        # nested under one of these keys instead.
        if not loaded:
            for key in ('state_dict', 'model_state', 'model'):
                if key in state and isinstance(state[key], dict):
                    try:
                        model.load_state_dict(state[key])
                        loaded = True
                        break
                    except Exception:
                        pass

    if not loaded:
        raise RuntimeError(f"Could not load weights from {weights_path}. "
                           "Expected a state_dict or a checkpoint with 'state_dict'/'model_state' keys.")
    model.eval()
    return model

# Runs validation for every learning rate we have a checkpoint for.
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    val_ds = VOCLikeDataset(f"{DATASET_BASE}/images/val", f"{DATASET_BASE}/labels_voc/val", get_transform())
    val_loader = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    print(f"Validation samples: {len(val_ds)}")

    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    agg_rows = []

    for lr in LEARNING_RATES:
        run_dir = PROJECT_DIR / f"{NAME_PREFIX}.{lr}"
        if not run_dir.exists():
            print(f"[SKIP] Run dir not found for lr={lr}: {run_dir}")
            continue

        # Look for a usable weights file, in order of preference.
        weights_path = None
        for fname in WEIGHTS_PREFERENCE:
            candidate = run_dir / fname
            if candidate.exists():
                weights_path = candidate
                break
        if weights_path is None:
            # Some of the older runs saved weights under a 'weights' subfolder
            # instead of directly in the run directory, so check there too.
            for fname in WEIGHTS_PREFERENCE:
                candidate = run_dir / "weights" / fname
                if candidate.exists():
                    weights_path = candidate
                    break
        if weights_path is None:
            print(f"[SKIP] No weights found for lr={lr} in {run_dir} or weights/")
            continue

        print(f"\n=== Validating {weights_path} (lr={lr}) ===")
        model = build_model_and_load(weights_path, device)

        # Validate with monitoring
        _, metrics, summary = validate_with_monitor(model, val_loader, device, run_dir)

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
            "max_gpu_util_percent": max(summary.get("max_gpu_util_percent", {}).values(), default=None) if isinstance(summary.get("max_gpu_util_percent"), dict) else None,
            "max_gpu_mem_used_mb": max(summary.get("max_gpu_mem_used_mb", {}).values(), default=None) if isinstance(summary.get("max_gpu_mem_used_mb"), dict) else None,
            "torch_peak_cuda_alloc_mb": summary.get("torch_peak_cuda_alloc_mb"),
            "torch_peak_cuda_reserved_mb": summary.get("torch_peak_cuda_reserved_mb"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "mAP50": metrics.get("mAP50"),
            "mAP50-95": metrics.get("mAP50-95"),
        }
        agg_rows.append(row)

    # Save aggregate CSV
    agg_path = PROJECT_DIR / 'val_monitor_summary_FRCNN_valonly.csv'
    if agg_rows:
        pd.DataFrame(agg_rows).to_csv(agg_path, index=False)
        print(f"\nSaved aggregate summary: {agg_path.resolve()}")
    else:
        print("\nNo validations run. Check your run directories and weights.")

if __name__ == '__main__':
    main()
