---
title: AS2.5LM live frame
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: REPLACE_WITH_VERSION_FROM_YOUR_SPACE
python_version: "3.12"
app_file: app.py
pinned: false
---

# AS2.5LM: live single frame

Upload a raw KITTI velodyne `.bin` scan. The Space runs SalsaNext semantic segmentation on
the GPU (ZeroGPU, FP16, KNN post-processing) and the Grid Engine on the CPU (NumPy reference,
bit-identical to the CUDA engine), and returns one **LGF1 frame** (`.lgf.gz`) plus a summary.

| Endpoint | Input | Output |
|---|---|---|
| `/infer` | `.bin` file (`scan_file`) | `[frame file (.lgf.gz), summary JSON]` |
| `/ping` | nothing | `"ok"` (call on page load to wake the Space) |

Format and client example: `docs/FORMAT.md` in the project repository.
Third-party code: see `THIRD_PARTY_NOTICES.md`.
