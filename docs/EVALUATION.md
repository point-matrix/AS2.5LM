# Evaluation metrics

Produced by `scripts/benchmark.py` (speed, memory, power) and `scripts/evaluate.py` (accuracy),
combined by `scripts/report.py` into `summary.json`, `frame_metrics.csv` and charts.
Full-run results: [results/](../results/README.md).

## Speed

| Metric | Definition |
|---|---|
| Stage latency | Per frame, GPU synchronised at every boundary: upload, projection, SalsaNext model, KNN + labels, Grid Engine on GPU, copy to CPU. Scans are read before the timer starts; warm-up frames excluded |
| Median / p95 / p99 / max | Over all timed frames |
| Real-time | Share of frames with end-to-end < 100 ms (one scan of a 10 Hz LiDAR) and Grid Engine < 30 ms (target) |
| Jitter | Standard deviation of latency |
| Throughput | Total LiDAR points ÷ total processing time (upload + SalsaNext + Grid Engine) |

## Size, weight and power (SWaP)

| Metric | Definition |
|---|---|
| Peak GPU memory | PyTorch peak allocated / reserved during the timed run, plus the CUDA context measured separately |
| GPU power | Board power from NVML every 50 ms; only readings inside frame-processing intervals count. Idle measured for 2 s first |
| Energy per frame | Mean active power × mean end-to-end latency |

## Segmentation

Ground truth: SemanticKITTI `.label` files mapped to SalsaNext ids with `learning_map`; points labelled
unlabeled/outlier are ignored.

| Metric | Definition |
|---|---|
| IoU (per class) | TP / (TP + FP + FN), accumulated over all frames |
| mIoU | Mean IoU over the classes present in the ground truth of the evaluated frames |
| Overall accuracy | Correct points / labelled points |
| Variants | Pixel lookup (FP32), KNN (FP32), KNN (FP16): the effect of KNN and of FP16 |
| By distance | Accuracy and mIoU for points 0–10, 10–30, 30–60, 60+ m from the sensor |

## The grid

| Metric | Definition |
|---|---|
| Grid-level class accuracy | Share of cells whose class equals the majority ground-truth class of their points; also by area and by cell size |
| Missed obstacles | Ground-truth obstacle points (priority ≥ 4) whose cell has traversability ≥ 0.5, or no cell. Reported for vehicles + people (classes 1–8) and for all obstacles |
| False alarms | Road / parking points whose cell has traversability < 0.5 |

## Record frames

| Record | Rule |
|---|---|
| Highest overall accuracy | Per-frame accuracy |
| Highest car / person IoU | Per-frame IoU, only frames with ≥ 50 car / ≥ 20 person ground-truth points |
| Fastest | Lowest end-to-end latency |
| Stress test | Most car + pedestrian points (ground truth when every frame has labels, otherwise predicted) |
