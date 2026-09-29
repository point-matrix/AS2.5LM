# Architecture

```mermaid
flowchart TB
    scan["KITTI .bin scan (x, y, z, remission)"] --> proj
    subgraph seg["as25lm.segmentation"]
        proj["project(): 64 x 2048 range image<br/>nearest point per pixel, empty = 0"] --> model["SalsaNext (FP16 autocast)<br/>softmax already inside forward()"]
        model --> knn["KNN post-processing<br/>5 neighbours, 5 x 5 window, 1 m cutoff"]
    end
    knn -->|"labels + confidence (GPU tensors)"| grid
    subgraph g["as25lm.grid"]
        grid["cuda_engine.GridEngineCUDA<br/>(or reference.run_reference on CPU)"]
    end
    grid --> cells["FINAL_DTYPE cells (numpy)"]
    cells --> fmt["frame_format: LGF1 frames"]
    cells --> met["metrics / report"]
```

## Components

| Module | Role |
|---|---|
| `config.py` | Every constant: classes, colours, priorities, traversability weights, cell sizes, distance bands, thresholds, sensor and KNN settings, SemanticKITTI label map |
| `segmentation/segmenter.py` | `Segmenter`: projection, SalsaNext, KNN; safe weight loading; optional stage timing |
| `segmentation/salsanext.py`, `knn.py` | Model and KNN from the SalsaNext repository (MIT) |
| `grid/reference.py` | NumPy Grid Engine: the specification; also `cell_index_of_points` |
| `grid/cuda_engine.py` + `grid/csrc/` | C++/CUDA Grid Engine, compiled on first use with `torch.utils.cpp_extension` |
| `pipeline.py` | `Pipeline.process(raw)`: scan → labels → cells, with stage timings |
| `metrics.py` | IoU / mIoU, distance bands, grid-level accuracy, obstacle safety, `PowerSampler` |
| `report.py` | Latency summaries, record frames, charts |
| `frame_format.py`, `export.py` | LGF1 frames and the dashboard clip |
| `data.py` | Reading scans and labels |

## Segmentation fixes

The original inference notebook had several problems, fixed here and verified on real data:

| Problem | Fix |
|---|---|
| Softmax applied twice (SalsaNext's `forward()` already ends with softmax), squashing every confidence to ≤ 0.125 | Model output used directly; confidence now averages 0.93 |
| Empty range-image pixels fed as −1; SalsaNext was trained with 0 | Empty pixels are 0 after normalisation |
| Several points on one pixel: GPU index assignment kept an undefined one | Nearest point kept, ties by lowest index (matches SalsaNext's CPU loader) |
| No KNN post-processing | KNN from the SalsaNext repo: +3.2 mIoU |
| `strict=False` checkpoint loading could hide a wrong file | Strict loading; original checkpoint converted to a weights-only file that loads safely |
| Latency timers included disk reads and zip compression | I/O outside timers, GPU synchronised, warm-up excluded |

## Grid Engine versions

The first Grid Engine was pure Python / NumPy (~212 ms per frame on CPU). The current engine:

- runs in C++/CUDA (4.65 ms per frame on a T4) and stays bit-identical to its NumPy reference;
- uses SalsaNext's class ids (the first version was off by one class);
- assigns cell size per 40 cm tile instead of per point, so cells never overlap and no points are lost;
- uses 5 / 10 / 20 / 40 cm cells, so every level nests exactly;
- computes exact statistics for split and merged cells, and fixed physical roughness limits;
- keeps 5 cm cells within 10 m (no merging there).

Algorithm: [GRID_ENGINE.md](GRID_ENGINE.md).

## Deployment

```mermaid
flowchart LR
    K["Kaggle T4<br/>scripts/export_clip.py"] --> D[("HF dataset<br/>manifest.json + frames")]
    U["Uploaded .bin"] --> S["HF ZeroGPU Space<br/>space/app.py"]
    D --> V["React dashboard<br/>Vercel"]
    S --> V
```

The Space runs SalsaNext on the GPU and the NumPy Grid Engine on the CPU (ZeroGPU cannot compile
CUDA extensions); both produce the same cells. Steps: [DEPLOY.md](DEPLOY.md).
