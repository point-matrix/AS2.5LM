# Grid Engine

Turns labelled points into an adaptive 2.5D grid. Implemented twice with identical results:
`grid/reference.py` (NumPy, the specification) and `grid/csrc/grid_engine.cu` (CUDA).

## Cell sizes

| Level | Cell size | Used for (tile-centre distance) |
|---|---|---|
| 3 | 5 cm | 0–10 m |
| 2 | 10 cm | 10–30 m |
| 1 | 20 cm | 30–60 m |
| 0 | 40 cm | 60 m and beyond |

The ground plane is divided into 40 cm tiles. Each tile gets one cell size from the distance of its
centre to the sensor, so cells never overlap and every point lands in exactly one cell. Each level is
exactly half the previous one, so cells nest as a true quadtree.

## Steps

1. **Drop unlabeled points** (class 0).
2. **Key every point.** 5 cm index `i = floor(x / 0.05)` (float64); tile `i >> 3`; tile-centre
   distance → level; cell key `level << 44 | (i >> (3 − level)) + OFF << 22 | …`.
3. **Build cells.** Stable sort by key; per cell: count, Σz, Σz², min z, max z, and the
   confidence-weighted class vote (winning class, its score, total score).
4. **Split.** Cells with priority ≥ 4 (vehicles, people, structures), level < 3 and ≥ 2 points are
   replaced by their non-empty quarter cells, recomputed from their own points.
5. **Merge (one pass).** A parent whose children are all unsplit cells of one class with priority ≤ 2
   becomes a single cell with exact combined statistics. Parents with tile centre within 10 m never
   merge, so the near field keeps its 5 cm cells.
6. **Final fields**, one row per cell:

```
elevation           = Σz / n
elevation_variance  = max(Σz² / n − elevation², 0)
height_range        = max z − min z
terrain_complexity  = 0.6 · min(height_range / 0.30 m, 1) + 0.4 · min(std / 0.10 m, 1)
traversability      = clip(class_weight · (1 − 0.5 · terrain_complexity), 0, 1)
semantic_confidence = winning class score / total score
```

Priorities and traversability weights per class: `config.py`.

## CUDA implementation

| Stage | Kernel(s) | Median on T4 |
|---|---|---|
| Filter + point keys | `k_point_keys` | 0.34 ms |
| Cells | stable sort (ATen/CUB) + `k_reduce_cells` | 0.52 ms |
| Split | `k_split_flags`, `k_rekey`, second sort + reduce | 0.61 ms |
| Merge | `k_parent_keys`, sort, `k_merge_groups` | 1.12 ms |
| Final fields + dashboard layout | `k_finalize`, `k_pack` | 0.47 ms |

(Stage medians from a 50-frame profile; `scripts/benchmark.py` reports them under `extras`.)

- **Bit-identical to NumPy.** Both use stable sorts and sum each cell's points in the same order,
  and the extension is compiled with `--fmad=false` so no multiply-add is fused. `scripts/check_parity.py`
  and `tests/test_cuda_parity.py` compare every field.
- **Dashboard layout on the GPU.** `k_pack` writes each row in numpy's `FINAL_DTYPE` byte layout, so the
  copy to the CPU is a single memcpy (1.5 ms) instead of a per-column conversion (~13 ms).
- **Input straight from SalsaNext.** Labels and confidences are GPU tensors; nothing goes through the CPU.
