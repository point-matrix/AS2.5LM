# LGF1 frame format

One file = one LiDAR frame: the segmented point cloud and the 2.5D adaptive grid.
The same format comes from both sources the dashboard uses:

| Source | Where | What |
|---|---|---|
| Precomputed clip | Hugging Face dataset repo, written by `scripts/export_clip.py` | `manifest.json` + `lite/<frame>.lgf.gz` + `full/<frame>.lgf.gz` |
| Live upload | Hugging Face ZeroGPU Space, `/infer` endpoint | one `<frame>_*.lgf.gz` per uploaded `.bin` |

Read it with `dashboard/decodeFrame.ts` (no dependencies) or `as25lm.frame_format.decode_frame`
in Python. Files are gzip-compressed;
`loadFrame(url)` fetches, decompresses (browser `DecompressionStream`) and decodes.

## Byte layout

All numbers little-endian.

| Bytes | Content |
|---|---|
| 0–3 | ASCII `LGF1` |
| 4–7 | `uint32` header length H |
| 8 … 8+H | UTF-8 JSON header, padded with spaces so that 8 + H is a multiple of 8 |
| … | arrays ("sections"), each starting at a multiple of 8; positions listed in `header.sections` |

`header.sections[i] = { name, dtype, shape, offset, bytes }`: `offset` is from the start of the
uncompressed file, so each section can be viewed directly as a typed array without copying.

## Header

| Field | Meaning |
|---|---|
| `frame_id` | e.g. `"001533"` (live: uploaded file name) |
| `source` | `"precomputed"` or `"live"` |
| `sequence` | KITTI sequence, e.g. `"08"` (null for live) |
| `n_points_total` | points in the original scan |
| `point_stride` | 1 = all points stored; 4 = every 4th point (lite playback files) |
| `n_cells` | grid cells |
| `level_sizes_m` | cell size by level: `[0.4, 0.2, 0.1, 0.05]` (level 3 = 5 cm) |
| `sensor_height_m` | 1.73. KITTI heights are relative to the LiDAR; add this to z to put the road near 0 m |
| `stats` | engine counts: `n_points_in`, `n_points_used`, `n_stage1_cells`, `n_splits`, `n_merges`, `n_final_cells`, `n_unlabeled_dropped` |
| `timing_ms` | clip: `upload, salsa_project, salsa_model, salsa_post, grid_gpu, grid_to_cpu, salsanext_total, grid_engine_total, end_to_end` (Tesla T4). Live: `segmentation_ms, gpu_wait_ms, grid_engine_cpu_ms` |
| `device` | GPU name, or the live Space's device description |
| `accuracy` | clip only: `{accuracy, car_iou, person_iou, gt_car_points, gt_person_points}` against ground truth; IoU is null when the class isn't in the frame |
| `clip_index` | clip only: position in the manifest |

## Sections

**Points** (N = stored points; sensor frame; x forward, y left, z up)

| Section | Type | Shape | Units |
|---|---|---|---|
| `points.xyz_cm` | int16 | N×3 | centimetres |
| `points.intensity` | uint8 | N | 0–255 = remission 0–1 |
| `points.class` | uint8 | N | predicted class id |
| `points.confidence` | uint8 | N | 0–255 = 0–1, probability of the predicted class |
| `points.gt_class` | uint8 | N | ground-truth class id; **clip only** |

**Cells** (M cells; they never overlap)

| Section | Type | Shape | Units |
|---|---|---|---|
| `cells.ixy` | int16 | M×2 | cell index at its own level |
| `cells.level` | uint8 | M | 3 = 5 cm, 2 = 10 cm, 1 = 20 cm, 0 = 40 cm |
| `cells.class` | uint8 | M | semantic class id (confidence-weighted vote of its points) |
| `cells.z_cm` | int16 | M×3 | min, max, mean height in centimetres |
| `cells.z_std_mm` | uint16 | M | height standard deviation, millimetres |
| `cells.traversability` | uint8 | M | 0–255 = 0–1 (0 blocked, 1 clear) |
| `cells.complexity` | uint8 | M | terrain complexity, 0–255 = 0–1 |
| `cells.confidence` | uint8 | M | semantic confidence, 0–255 = 0–1 |
| `cells.point_count` | uint16 | M | LiDAR points in the cell |

Cell footprint: `size = level_sizes_m[level]`, `x_min = ix × size`, `x_max = (ix + 1) × size`, same for y.
`cellInfo(frame, i)` returns all of this in metres and 0–1 units.

Precision: positions and heights ±0.5 cm; fractions ±0.002. Enough for display and hover values.

## Class ids (SalsaNext / SemanticKITTI)

0 unlabeled, 1 car, 2 bicycle, 3 motorcycle, 4 truck, 5 other-vehicle, 6 person, 7 bicyclist,
8 motorcyclist, 9 road, 10 parking, 11 sidewalk, 12 other-ground, 13 building, 14 fence,
15 vegetation, 16 trunk, 17 terrain, 18 pole, 19 traffic-sign.
Names, colours, priority and traversability weight: `CLASSES` in `decodeFrame.ts` and `classes` in the manifest.

## Precomputed clip

```
manifest.json
lite/001500.lgf.gz  …  lite/002499.lgf.gz   grid + every 4th point   (~475 KB each)
full/001500.lgf.gz  …  full/002499.lgf.gz   grid + all points        (~880 KB each)
```

`manifest.json`: `fps` (10), `frame_count`, `first_frame`, `last_frame`, `lite_point_stride`,
`sensor_height_m`, `level_sizes_m`, `classes`, `bookmarks` (record frames: label, frame, index, value),
`summary` (device, FP16, KNN, median latencies, total sizes, mean accuracy) and `frames[]`
(paths relative to the manifest, sizes, per-frame `timing_ms`, `stats`, `accuracy`).

```ts
const m = await loadManifest('https://huggingface.co/datasets/USER/REPO/resolve/main/manifest.json');
const frame = await loadFrame(frameUrl(m, i, playing ? 'lite' : 'full'));
```

**Bandwidth.** A lite frame is ~475 KB, so 10 fps needs ~4.7 MB/s. Prefetch a few seconds ahead,
or play at 5 fps on slow networks. Load the `full` file only when paused.

## Live upload (Space)

```ts
import { Client, handle_file } from '@gradio/client';
const client = await Client.connect('USER/SPACE');     // on page load: client.predict('/ping') wakes it
const r = await client.predict('/infer', { scan_file: handle_file(file) });
const [frameFile, summary] = r.data as [{ url: string }, any];
const frame = await loadFrame(frameFile.url);           // same decoder as the clip
```

Live frames store all points (`point_stride` 1) and have no ground truth or accuracy.

## Rendering notes

- Heights: add `sensor_height_m` to every z so the road sits near 0 m.
- 3D blocks: box from `z_min` to `z_max` over the cell footprint. Give flat cells a minimum
  thickness (2–5 cm) so ground stays visible.
- ~52k cells per frame: use instanced meshes (three.js `InstancedMesh`), not one mesh per cell.
- Empty space (no cell) means no LiDAR returns, not free space. Show it as "unknown".

## Drivable / non-drivable layer

`dashboard/drivability.ts` classifies each cell as drivable / caution / blocked from its class, terrain
complexity and confidence, with an **on-road** (default) and an **off-road** vehicle profile.
It runs in the browser, so profiles and thresholds can change without re-running the pipeline.
`frameDrivability(frame)` returns one code per cell; `drivableArea(frame, codes)` gives m² per state.
