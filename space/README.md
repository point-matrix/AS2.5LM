---
title: AS2.5LM live frame
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# AS2.5LM: live single frame

Upload one LiDAR scan. The Space runs SalsaNext semantic segmentation on the GPU (ZeroGPU,
FP16, KNN post-processing) and the Grid Engine on the CPU (NumPy reference, bit-identical to the
CUDA engine), and returns one **LGF1 frame** (`.lgf.gz`) plus a summary.

| `layout` | File | Columns | Labels |
|---|---|---|---|
| `raw` (default) | `.bin` (KITTI float32) or `.npy` | x, y, z, intensity | predicted by SalsaNext |
| `labeled` | `.npy` | x, y, z, intensity, class | provided (0 = unlabeled, 1–19); SalsaNext is skipped |

If ZeroGPU refuses a GPU (the caller's daily quota is used up, or no GPU is free), SalsaNext runs
on the Space's CPU instead: slower, same output format. The summary then shows `device: "CPU"` and
`gpu_fallback` with the reason. GPU time requested per call: `GPU_SECONDS` (default 10).

Limits: 500,000 points, finite coordinates within 10 km; pickled `.npy` files are rejected.

| Endpoint | Input | Output |
|---|---|---|
| `/infer` | `scan_file` (`.bin` / `.npy`), optional `layout` (`raw` / `labeled`) | `[frame file (.lgf.gz), summary JSON]` |
| `/ping` | nothing | `"ok"` (call on page load to wake the Space) |

Format and client example: `docs/FORMAT.md` in the project repository.
Third-party code: see `THIRD_PARTY_NOTICES.md`.
