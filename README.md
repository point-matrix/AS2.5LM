# AS2.5LM

**Real-time semantic 2.5D mapping from LiDAR.** A raw LiDAR scan is labelled point by point
with SalsaNext, then turned into an adaptive 2.5D grid (5 cm cells near the vehicle, up to 40 cm
far away) carrying elevation, roughness, semantic class and traversability for every cell.
The grid engine is written in C++/CUDA and runs in **4.65 ms per frame**; the whole pipeline,
from raw scan to grid, runs in **31.1 ms** on a Tesla T4.

Built for Smart India Hackathon (SIH).

## Results

Full SemanticKITTI sequence 08 (4,071 frames, the model's validation split), Tesla T4.
Details, charts and per-frame data: [`results/`](results/README.md).

| | |
|---|---|
| Grid Engine latency | **4.65 ms** median, 6.66 ms p99 (target < 30 ms) |
| End-to-end latency | **31.1 ms** median, 33.4 ms p99, **32 frames/s** |
| Real-time margin | 100% of frames under 100 ms (one scan of a 10 Hz LiDAR) |
| Segmentation | **mIoU 59.0%**, 89.3% of points correct (FP16 = FP32) |
| Obstacle safety | 0.47% of vehicle / pedestrian points shown as passable |
| Memory and power | ~583 MB GPU memory, 1.8 J per frame |
| CUDA vs reference | Bit-identical grids (checked on every run) |

![Latency per frame across the sequence](results/figures/latency_over_sequence.png)

## Pipeline

```mermaid
flowchart LR
    classDef default fill:#111111,stroke:#FF6600,stroke-width:2px,color:#FFFFFF;
    style SEG fill:#000000,stroke:#FF6600,stroke-width:2px,color:#FF6600,stroke-dasharray: 5 5;
    style GRID fill:#000000,stroke:#FF6600,stroke-width:2px,color:#FF6600,stroke-dasharray: 5 5;

    A["Raw LiDAR scan<br/>KITTI .bin, ~123k points"] --> SEG

    subgraph SEG["Semantic segmentation (PyTorch)"]
        direction TB
        B["Range-image projection<br/>64 x 2048, nearest point per pixel"] --> C["SalsaNext, FP16<br/>20-class probabilities"]
        C --> D["KNN post-processing<br/>class + confidence per point"]
    end

    SEG --> GRID

    subgraph GRID["Grid Engine (C++/CUDA)"]
        direction TB
        E["40 cm tiles, cell size by distance<br/>5 / 10 / 20 / 40 cm"] --> F["Per-cell statistics<br/>heights, variance, class vote"]
        F --> G["Split obstacle cells<br/>merge flat ground beyond 10 m"]
        G --> H["Elevation, roughness,<br/>traversability per cell"]
    end

    GRID --> I["LGF1 frames<br/>points + ~52k cells"]
    GRID --> K["Metrics<br/>summary.json, charts"]
    I --> J["Dashboard (React)<br/>2D / 3D views, layers, drivable area"]
```

How the frames reach the dashboard:

```mermaid
flowchart LR
    classDef default fill:#111111,stroke:#FF6600,stroke-width:2px,color:#FFFFFF;
    style OFF fill:#000000,stroke:#FF6600,stroke-width:2px,color:#FF6600,stroke-dasharray: 5 5;

    subgraph OFF["Offline (Kaggle, Tesla T4)"]
        S1["Sequence 08 scans"] --> P1["export_clip.py<br/>frames 001500-002499"]
    end
    P1 --> DS[("Hugging Face dataset<br/>manifest.json + frames")]
    U["User uploads a .bin"] --> SP["Hugging Face Space (ZeroGPU)<br/>same pipeline, one frame"]
    DS --> W["Dashboard on Vercel<br/>decodeFrame.ts"]
    SP --> W
```

More: [architecture](docs/ARCHITECTURE.md) · [grid engine algorithm](docs/GRID_ENGINE.md) ·
[evaluation metrics](docs/EVALUATION.md) · [frame format](docs/FORMAT.md) · [deployment](docs/DEPLOY.md)

## What each cell contains

| Field | Meaning |
|---|---|
| position, size | footprint of the cell; 5 cm within 10 m, 10 / 20 / 40 cm further out |
| semantic class, confidence | confidence-weighted vote of the points in the cell (19 SemanticKITTI classes) |
| min / max / mean height, variance | elevation statistics of the cell's points |
| terrain complexity | roughness 0-1 from height range and spread |
| traversability | 0 (blocked) to 1 (clear): class weight x (1 - 0.5 x roughness) |
| point count | LiDAR returns in the cell |

## Repository layout

```
src/as25lm/          Python package
  config.py          classes, cell sizes, thresholds (single source of truth)
  segmentation/      SalsaNext wrapper: projection, model, KNN (vendored model code)
  grid/              Grid Engine: NumPy reference + C++/CUDA extension (csrc/)
  pipeline.py        scan -> labels -> grid, with stage timings
  metrics.py         IoU, distance bands, grid accuracy, obstacle safety, GPU power
  frame_format.py    LGF1 frames for the dashboard
  report.py, export.py
scripts/             benchmark, evaluate, report, check_parity, export_clip,
                     convert_weights, build_space, render_figures (slide images)
notebooks/           kaggle_run.ipynb: runs the scripts on a Kaggle T4
space/               Hugging Face Space (live single-frame API)
dashboard/           decodeFrame.ts, drivability.ts (for the React app)
tests/               pytest suite on synthetic scenes (no data needed)
results/             full-run results: summary.json, frame_metrics.csv, figures/
docs/                architecture, algorithm, metrics, format, deployment
```

## Quick start

Requirements: Python 3.10+, PyTorch 2.1+. The CUDA Grid Engine needs an NVIDIA GPU and the
CUDA toolkit (`nvcc`); without them use `--engine numpy` (same output, slower).

```bash
pip install -e ".[eval,export,dev]"
pytest                                   # CUDA parity test is skipped without nvcc
```

Data and weights are not in the repository:

- KITTI odometry velodyne scans and SemanticKITTI labels (sequence 08 for evaluation)
- SalsaNext pretrained model (`pretrained/SalsaNext`, `arch_cfg.yaml`), converted once:

```bash
python scripts/convert_weights.py --checkpoint pretrained/SalsaNext --out weights/salsanext_weights.pth
```

Run the evaluation (Tesla T4 on Kaggle: see [`notebooks/kaggle_run.ipynb`](notebooks/kaggle_run.ipynb)):

```bash
DATA=/path/to/dataset/sequences/08
W=weights/salsanext_weights.pth
python scripts/check_parity.py --scans $DATA/velodyne --weights $W
python scripts/benchmark.py    --scans $DATA/velodyne --labels $DATA/labels --weights $W --out runs/seq08
python scripts/evaluate.py     --scans $DATA/velodyne --labels $DATA/labels --weights $W --out runs/seq08
python scripts/report.py       --results runs/seq08
```

Export the dashboard clip and build the live Space: [docs/DEPLOY.md](docs/DEPLOY.md).

## Demo
- **Frontend Repository:** [point-matrix/connect](https://github.com/point-matrix/connect)
- Dashboard: given in ppt(not shown here for safety purposes)
- Clip data: https://huggingface.co/datasets/ranbyDipz/sih-lidar-clip

## Acknowledgements

- [SalsaNext](https://github.com/tiagoCortinhal/SalsaNext) (Cortinhal et al.): segmentation model,
  KNN post-processing and pretrained weights (MIT; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md))
- [SemanticKITTI](http://www.semantic-kitti.org/) and [KITTI](https://www.cvlibs.net/datasets/kitti/): data

## License

Not licensed for reuse at this time. Third-party code keeps its own license (see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).
