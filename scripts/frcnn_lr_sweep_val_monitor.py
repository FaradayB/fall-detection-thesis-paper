#!/usr/bin/env python3
"""
This is the full Faster R-CNN pipeline: for each learning rate in
LEARNING_RATES it trains the model from scratch, validates it every epoch
(recording resource usage the whole time), applies early stopping on mAP50,
and once training is done it generates the usual set of plots (confusion
matrix, F1/precision/recall vs. confidence, PR curve, label distribution,
and a results panel across epochs).

The resource monitoring records the same stuff as the other scripts here:
per-batch inference time, total validation time, CPU/RAM, GPU utilization
and VRAM through NVML, and PyTorch's peak CUDA memory.

Each run gets resource_summary.json and metrics.csv under
runs_2/train/FRCNN_noD_Normal_lr.<lr>/val_monitor/, and once every LR is
done, everything gets combined into runs_2/train/val_monitor_summary_FRCNN.csv.

Needs: torch, torchvision, psutil, pynvml, torchmetrics, pandas, tqdm,
pillow, matplotlib, scipy.
"""

import os
import time
import csv
import json
import statistics as stats
from datetime import datetime
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
from sklearn.metrics import precision_score, recall_score, confusion_matrix, precision_recall_curve, f1_score
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
import xml.etree.ElementTree as ET
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d as uf
import pickle
import pandas as pd

# Optional NVML for GPU telemetry
try:
    import pynvml
    _NVML_READY = True
    pynvml.nvmlInit()
except Exception:
    _NVML_READY = False

# Config - the knobs I actually turn between experiments.

CLASS_NAMES = ["fall", "no_fall"]
CLASS_MAP = {name.lower().replace(" ", "_"): idx + 1 for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES) + 1  # + background

VISUAL_DEBUG = False
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
INPUT_SIZE = 640

NUM_EPOCHS = 100
LEARNING_RATES = [0.1, 0.01, 0.001]  # sweep
BATCH_SIZE = 16
VAL_BATCH_SIZE = 1
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005
PATIENCE = 5
WARMUP_EPOCH = 3

DATASET_BASE = 'dataset_paper_new'

PROJECT_DIR = Path('runs_2/train')
NAME_PREFIX = 'FRCNN_noD_Normal_lr'

# Dataset loader, reading Pascal VOC-style XML annotations.

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
            # if the xml is missing or malformed, just treat this image as
            # having no annotated objects instead of crashing the whole run
            boxes, labels = [], []

        if self.transforms:
            img = self.transforms(img)
            
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }
        return img, target

# Resource monitoring - tracks CPU, RAM, and GPU usage during validation.
# Same pattern as the other scripts, just packaged as a class here since
# torchvision doesn't have a callback system like Ultralytics does.

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
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        self._val_start = time.perf_counter()
        update_gpu_maxima(self.max_gpu_util, self.max_gpu_mem_used)

    def on_batch_start(self):
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        self._t0 = time.perf_counter()

    def on_batch_end(self):
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

    def finish(self):
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
        return self.summary

# Runs one full validation pass, computing both accuracy metrics and
# resource usage at the same time, since we're already looping over batches.

def validate_with_monitor(model, loader, device, save_dir, score_thresh=SCORE_THRESHOLD, iou_thresh=IOU_THRESHOLD):
    """Validate the model: precision, recall, mAP50, mAP50-95, and losses,
    plus the resource/timing numbers. Writes resource_summary.json and
    metrics.csv into save_dir/val_monitor/.
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
            
            # Get the predictions first.
            outputs = model(imgs)

            # torchvision only returns losses in train mode, so switch over
            # briefly, run the same batch again just to get the loss values,
            # then switch back to eval.
            model.train()
            loss_dict = model(imgs, targets)
            model.eval()
            
            sum_b += float(loss_dict.get('loss_box_reg', 0.0))
            sum_c += float(loss_dict.get('loss_classifier', 0.0))
            sum_d += float(loss_dict.get('loss_objectness', 0.0)) + float(loss_dict.get('loss_rpn_box_reg', 0.0))
            count += 1

            # Update the running accuracy metrics with this batch's predictions.
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

    res50 = float(mp50.compute()['map'].item())
    res_all = float(mp_all.compute()['map'].item())
    valid_labels = list(range(1, NUM_CLASSES))
    precision = precision_score(all_t, all_p, labels=valid_labels, average='weighted', zero_division=0)
    recall = recall_score(all_t, all_p, labels=valid_labels, average='weighted', zero_division=0)
    losses = {'box': (sum_b/count) if count else 0.0, 'cls': (sum_c/count) if count else 0.0, 'dfl': (sum_d/count) if count else 0.0}
    metrics = {'precision': precision, 'recall': recall, 'mAP50': res50, 'mAP50-95': res_all}

    # Save resource summary
    with open(vm_dir / 'resource_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # Save metrics CSV (single row)
    df = pd.DataFrame([{**losses, **metrics}])
    df.to_csv(vm_dir / 'metrics.csv', index=False)

    return losses, metrics, summary, (all_t, all_p, all_s)

# Plotting helpers below - these just turn the numbers above into the charts
# I actually put in the thesis. All optional, training still works without
# calling any of them.

def create_confusion_matrix(y_true, y_pred, save_path, normalize=False):
    labels = list(range(1, NUM_CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
        title = 'Confusion Matrix (Normalized)'
        fmt = '.2f'
        vmax = 1.0
    else:
        title = 'Confusion Matrix'
        fmt = 'd'
        vmax = None
    fig_size = max(8, len(CLASS_NAMES) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm, cmap='Blues', vmax=vmax)
    for (i, j), v in np.ndenumerate(cm):
        color = 'white' if (normalize and v > 0.5) or (not normalize and v > cm.max()/2) else 'black'
        if normalize:
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', color=color, fontsize=8)
        else:
            ax.text(j, i, f'{v}', ha='center', va='center', color=color, fontsize=8)
    ax.set_xticks(range(len(CLASS_NAMES)), labels=CLASS_NAMES, rotation=45, ha='right')
    ax.set_yticks(range(len(CLASS_NAMES)), labels=CLASS_NAMES)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.colorbar(im)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_confidence_curves(all_t, all_p, all_s, plot_dir):
    labels = list(range(1, NUM_CLASSES))
    thr = np.linspace(0, 1, 200)
    f1_curves = {}
    f1_all = []
    yta, ypa, ysa = np.array(all_t), np.array(all_p), np.array(all_s)
    for t in thr:
        yp = np.where(ysa >= t, ypa, 0)
        f1_all.append(f1_score(yta, yp, labels=labels, average='weighted', zero_division=0))
        for label in labels:
            f1_curves.setdefault(label, [])
            f1_curves[label].append(f1_score((yta == label).astype(int), (yp == label).astype(int), zero_division=0))
    fig, ax = plt.subplots(figsize=(12, 8))
    top_classes = min(5, len(CLASS_NAMES))
    for i, class_name in enumerate(CLASS_NAMES[:top_classes], start=1):
        ax.plot(thr, f1_curves[i], label=class_name, alpha=0.7)
    best = int(np.nanargmax(f1_all))
    ax.plot(thr, f1_all, linewidth=3, label=f'All Classes {f1_all[best]:.2f}@{thr[best]:.2f}')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('F1')
    ax.set_title(f'F1-Confidence Curve (Top {top_classes} Classes)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(plot_dir/'F1_curve.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    for metric_name, metric_func in [('Precision', precision_score), ('Recall', recall_score)]:
        fig, ax = plt.subplots(figsize=(12, 8))
        for cls, class_name in zip(labels, CLASS_NAMES):
            mask = (yta == cls).astype(int)
            scores = np.where(ypa == cls, ysa, 0.0)
            if len(np.unique(mask)) > 1:
                prec_ci, rec_ci, thr_ci = precision_recall_curve(mask, scores)
                y_vals = prec_ci[:-1] if metric_name == 'Precision' else rec_ci[:-1]
                x_vals = thr_ci
                area = np.trapz(y_vals, x_vals) if len(x_vals) > 1 else 0
                ax.plot(x_vals, y_vals, label=f"{class_name} {area:.3f}", alpha=0.7)
        metric_all = []
        for t in thr:
            yp_thresh = np.where(ysa >= t, ypa, 0)
            metric_all.append(metric_func(yta, yp_thresh, labels=labels, average='weighted', zero_division=0))
        best_idx = int(np.nanargmax(metric_all))
        area_all = np.trapz(metric_all, thr)
        ax.plot(thr, metric_all, linewidth=3, label=f"All Classes {area_all:.3f} at {thr[best_idx]:.3f}")
        ax.set_xlabel('Confidence')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name}–Confidence Curve')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        fig.savefig(plot_dir/f'{metric_name[0]}_curve.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

def create_pr_curve(all_t, all_p, all_s, save_path):
    labels = list(range(1, NUM_CLASSES))
    yta, ypa, ysa = np.array(all_t), np.array(all_p), np.array(all_s)
    fig, ax = plt.subplots(figsize=(12, 8))
    ap_scores = []
    for cls, class_name in zip(labels, CLASS_NAMES):
        mask = (yta == cls).astype(int)
        scores = np.where(ypa == cls, ysa, 0)
        if len(np.unique(mask)) > 1:
            prec, rec, _ = precision_recall_curve(mask, scores)
            ap = np.trapz(prec, rec) if len(rec) > 1 else 0
            ap_scores.append(ap)
            ax.plot(rec, prec, label=f"{class_name} AP={ap:.3f}", alpha=0.7)
        else:
            ap_scores.append(0)
    overall_map = np.mean(ap_scores) if ap_scores else 0.0
    ax.plot([], [], linewidth=3, label=f'mAP = {overall_map:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_label_distribution(dataset_base, save_path):
    ann = []
    labels_path = Path(f"{dataset_base}/labels_voc/train")
    if not labels_path.exists():
        return
    for xml_file in labels_path.glob('*.xml'):
        try:
            tree = ET.parse(xml_file)
            for obj in tree.getroot().findall('object'):
                class_name = obj.find('name').text
                ann.append(class_name)
        except Exception:
            pass
    if not ann:
        return
    counts = pd.Series(ann).value_counts()
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = counts.plot.bar(ax=ax, edgecolor='black')
    ax.set_ylabel('Number of Instances')
    ax.set_xlabel('Class')
    ax.set_title('Label Distribution in Training Set')
    ax.tick_params(axis='x', rotation=45)
    for bar in bars.patches:
        height = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2., height+0.1, f'{int(height)}', ha='center', va='bottom')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_results_panel(csv_file, save_path):
    if not os.path.exists(csv_file):
        return
    df = pd.read_csv(csv_file)
    smooth = lambda x: uf(x.astype(float), size=5, mode='nearest')
    cols = [
        ('train/box_loss', 'Box Loss'),
        ('train/cls_loss', 'Cls Loss'),
        ('train/dfl_loss', 'DFL Loss'),
        ('metrics/precision(B)', 'Precision'),
        ('metrics/recall(B)', 'Recall'),
        ('metrics/mAP50(B)', 'mAP50'),
        ('metrics/mAP50-95(B)', 'mAP50-95'),
        ('val/box_loss', 'Val Box'),
        ('val/cls_loss', 'Val Cls'),
        ('val/dfl_loss', 'Val DFL')
    ]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for ax, (col, name) in zip(axes, cols):
        if col in df.columns:
            raw = df[col].astype(float)
            ax.plot(df['epoch'], raw, marker='o', label='raw', alpha=0.7, markersize=3)
            ax.plot(df['epoch'], smooth(raw), linestyle='--', label='smooth', linewidth=2)
            ax.set_title(name)
            ax.set_xlabel('Epoch')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No data for\n{name}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(name)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# This is the actual training loop for one learning rate: trains epoch by
# epoch, validates after each one, and stops early if mAP50 stalls.

def train_and_validate_for_lr(lr0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = VOCLikeDataset(f"{DATASET_BASE}/images/train", f"{DATASET_BASE}/labels_voc/train", get_transform())
    val_ds = VOCLikeDataset(f"{DATASET_BASE}/images/val", f"{DATASET_BASE}/labels_voc/val", get_transform())
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Same setup as build_model() in the other scripts - pretrained backbone,
    # swap the head to predict our own classes.
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr0, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    run_dir = PROJECT_DIR / f'{NAME_PREFIX}.{lr0}'
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_file = run_dir / 'results.csv'

    headers = [
        'epoch', 'time', 'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
        'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
        'val/box_loss', 'val/cls_loss', 'val/dfl_loss', 'lr/pg0', 'lr/pg1', 'lr/pg2'
    ]

    best_map = 0.0
    patience_counter = 0

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            t0 = time.time()

            # Ramp the learning rate up gradually for the first few epochs
            # instead of hitting it at full strength immediately - helps
            # avoid the loss spiking right at the start of training.
            if epoch <= WARMUP_EPOCH:
                warmup_lr = lr0 * (epoch / WARMUP_EPOCH)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                    param_group['momentum'] = 0.8

            sum_box = sum_cls = sum_dfl = 0.0
            train_pbar = tqdm(train_loader, desc=f'LR {lr0} | Epoch {epoch}/{NUM_EPOCHS}')
            for imgs, targets in train_pbar:
                imgs = [img.to(device) for img in imgs]
                targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
                loss_dict = model(imgs, targets)
                losses = sum(loss_dict.values())
                sum_box += float(loss_dict.get('loss_box_reg', 0.0))
                sum_cls += float(loss_dict.get('loss_classifier', 0.0))
                sum_dfl += float(loss_dict.get('loss_objectness', 0.0)) + float(loss_dict.get('loss_rpn_box_reg', 0.0))
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                train_pbar.set_postfix({
                    'box': f'{sum_box/(train_pbar.n+1):.3f}',
                    'cls': f'{sum_cls/(train_pbar.n+1):.3f}'
                })

            train_box = sum_box / max(1, len(train_loader))
            train_cls = sum_cls / max(1, len(train_loader))
            train_dfl = sum_dfl / max(1, len(train_loader))
            epoch_time = time.time() - t0

            val_losses, metrics, summary, (all_t, all_p, all_s) = validate_with_monitor(model, val_loader, device, run_dir)

            # Ultralytics logs three param-group LRs (pg0/pg1/pg2), so I'm
            # matching that column layout here even though this optimizer
            # really only has one LR - just repeat it if there aren't three.
            lrs = [group['lr'] for group in optimizer.param_groups]
            pg0, pg1, pg2 = (lrs + [lrs[-1]] * 3)[:3]


            row = [
                epoch, epoch_time,
                train_box, train_cls, train_dfl,
                metrics['precision'], metrics['recall'], metrics['mAP50'], metrics['mAP50-95'],
                val_losses['box'], val_losses['cls'], val_losses['dfl'],
                pg0, pg1, pg2
            ]
            writer.writerow(row)

            # Console summary
            print(f"Epoch {epoch}/{NUM_EPOCHS} - {NAME_PREFIX}.{lr0}")
            print(f"  Train - Box: {train_box:.4f}, Cls: {train_cls:.4f}, DFL: {train_dfl:.4f}")
            print(f"  Val   - Box: {val_losses['box']:.4f}, Cls: {val_losses['cls']:.4f}, DFL: {val_losses['dfl']:.4f}")
            print(f"  Metrics - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
            print(f"  mAP50: {metrics['mAP50']:.4f}, mAP50-95: {metrics['mAP50-95']:.4f}")
            print(f"  Val total time: {summary['total_val_time_s']:.2f}s, avg batch {summary['batch_time_ms_avg']:.1f} ms")

            scheduler.step()

            # Stop early if mAP50 hasn't improved in PATIENCE epochs, no point
            # burning more GPU time once it's plateaued.
            if metrics['mAP50'] > best_map:
                best_map = metrics['mAP50']
                patience_counter = 0
                torch.save(model.state_dict(), run_dir / 'best.pt')
            else:
                patience_counter += 1
                if patience_counter > PATIENCE:
                    print(f"Early stopping at epoch {epoch} for lr={lr0}")
                    break

        # Always keep the last epoch's weights too, even if it wasn't the best.
        torch.save(model.state_dict(), run_dir / 'last.pt')

        # Generate the plots using predictions from the last validation pass.
        plot_dir = run_dir / 'plots'
        plot_dir.mkdir(exist_ok=True)
        create_confusion_matrix(all_t, all_p, plot_dir / 'confusion_matrix.png', normalize=False)
        create_confusion_matrix(all_t, all_p, plot_dir / 'confusion_matrix_normalized.png', normalize=True)
        create_confidence_curves(all_t, all_p, all_s, plot_dir)
        create_pr_curve(all_t, all_p, all_s, plot_dir / 'PR_curve.png')
        create_label_distribution(DATASET_BASE, plot_dir / 'labels.png')
        create_results_panel(csv_file, plot_dir / 'results.png')

    return run_dir

# Trains and validates every learning rate in the sweep, then builds one
# combined CSV so I can compare them side by side.
def main():
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    agg_rows = []

    for lr in LEARNING_RATES:
        run_dir = train_and_validate_for_lr(lr)

        vm_dir = Path(run_dir) / 'val_monitor'
        metrics_csv = vm_dir / 'metrics.csv'
        summary_json = vm_dir / 'resource_summary.json'
        row = {"lr0": lr, "run_dir": str(run_dir)}
        if metrics_csv.exists():
            try:
                df = pd.read_csv(metrics_csv)
                row.update(df.iloc[0].to_dict())
            except Exception:
                pass
        if summary_json.exists():
            try:
                with open(summary_json, 'r', encoding='utf-8') as f:
                    summ = json.load(f)
                # flatten GPU maxima (max across devices)
                def _max_or_none(d):
                    try:
                        return max(d.values()) if isinstance(d, dict) and len(d) > 0 else None
                    except Exception:
                        return None
                row.update({
                    "total_val_time_s": summ.get("total_val_time_s"),
                    "batch_time_ms_avg": summ.get("batch_time_ms_avg"),
                    "batch_time_ms_p50": summ.get("batch_time_ms_p50"),
                    "batch_time_ms_p95": summ.get("batch_time_ms_p95"),
                    "max_process_cpu_percent": summ.get("max_process_cpu_percent"),
                    "max_process_ram_mb": summ.get("max_process_ram_mb"),
                    "max_system_cpu_percent": summ.get("max_system_cpu_percent"),
                    "max_system_mem_percent": summ.get("max_system_mem_percent"),
                    "max_gpu_util_percent": _max_or_none(summ.get("max_gpu_util_percent")),
                    "max_gpu_mem_used_mb": _max_or_none(summ.get("max_gpu_mem_used_mb")),
                    "torch_peak_cuda_alloc_mb": summ.get("torch_peak_cuda_alloc_mb"),
                    "torch_peak_cuda_reserved_mb": summ.get("torch_peak_cuda_reserved_mb"),
                })
            except Exception:
                pass
        agg_rows.append(row)

    # Save aggregate
    agg_path = PROJECT_DIR / 'val_monitor_summary_FRCNN.csv'
    pd.DataFrame(agg_rows).to_csv(agg_path, index=False)
    print(f"\nSaved aggregate summary: {agg_path.resolve()}")

if __name__ == '__main__':
    main()
