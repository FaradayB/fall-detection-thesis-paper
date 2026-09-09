# Fall Detection: YOLO vs Faster R-CNN

Experiment code for comparing YOLO (Ultralytics) and Faster R-CNN object detectors on a fall-detection dataset, including learning-rate sweeps and resource-usage monitoring during validation.

## Structure

```
notebooks/   Data prep, training, and inference notebooks
scripts/     Standalone validation/monitoring scripts (CLI, no notebook needed)
results/     Summary metrics (xlsx) from validation monitoring runs
```

### Notebooks
- `datasetsplit.ipynb` - splits the raw dataset into train/val sets
- `convert_yolo_to_voc.ipynb` - converts YOLO-format labels to Pascal VOC format (for Faster R-CNN)
- `trainmodel.ipynb` - main training notebook
- `modelfall.ipynb` / `modelfallnRCNN.ipynb` - model definitions/experiments for YOLO and Faster R-CNN respectively
- `old_modelfallRCNN.ipynb` - earlier/legacy Faster R-CNN model notebook, kept for reference
- `inference.ipynb` - runs inference with trained weights

### Scripts
Each script under `scripts/` runs validation with resource monitoring (CPU/RAM/GPU utilization, VRAM, inference timing) and writes `resource_summary.json` + `metrics.csv` per run:

- `yolo_val_resource_monitor.py` - single YOLO model validation with resource monitoring
- `yolo_val_only_monitor.py` - validation-only pass over existing YOLO checkpoints across learning rates
- `yolo_lr_sweep_val_monitor.py` - trains YOLO across a learning-rate sweep, then validates each
- `fasterrcnn_val_only_monitor.py` - validation-only pass for a single Faster R-CNN checkpoint
- `frcnn_val_only_monitor.py` - validation-only pass over existing Faster R-CNN checkpoints across learning rates
- `frcnn_lr_sweep_val_monitor.py` - trains Faster R-CNN across a learning-rate sweep, then validates each

## Setup

Pretrained YOLO weights (`yolo11n.pt`, `yolov8n.pt`) aren't tracked in this repo — the `ultralytics` package downloads them automatically on first use (e.g. `YOLO("yolo11n.pt")`).

Training run outputs (`runs/`, `runs_2/`, `val_logs/` - checkpoints, plots, logs) are also untracked; re-run the notebooks/scripts above to regenerate them.
