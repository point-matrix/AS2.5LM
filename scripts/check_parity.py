#!/usr/bin/env python
"""
Check that the CUDA Grid Engine is bit-identical to the NumPy reference on real frames.

Integer fields must match exactly, float fields within 1e-9 (expected: identical), and the
FINAL_DTYPE array written on the GPU must equal the reference field by field.
Exits with status 1 on any mismatch.

Example:
  python scripts/check_parity.py --scans .../08/velodyne --weights weights/salsanext_weights.pth --frames 5
"""

import sys

import numpy as np
import torch

from as25lm import cli, config as C
from as25lm.data import frame_id, load_scan
from as25lm.grid.reference import run_reference, to_structured

INT_FIELDS = {"level", "semantic_class", "semantic_priority", "point_count"}


def main():
    p = cli.parser(__doc__)
    cli.add_data_args(p)
    cli.add_model_args(p)
    args = p.parse_args()
    if args.frames is None:
        args.frames = 5
    args.engine = "cuda"
    pipe = cli.make_pipeline(args)
    names = [n for n, _ in C.FINAL_FIELDS]

    ok_all = True
    for path in cli.scan_list(args):
        raw = load_scan(path)
        pts = torch.from_numpy(raw).to(pipe.device)
        labels, conf, _ = pipe.seg.segment(pts, fp16=pipe.fp16)
        res = pipe.engine.run(pts[:, :3], labels, conf)
        m_gpu = res.matrix_numpy()
        m_ref, st_ref, _ = run_reference(raw[:, :3], labels.cpu().numpy(), conf.float().cpu().numpy())
        if m_gpu.shape != m_ref.shape or res.stats != st_ref:
            ok_all = False
            print(f"{frame_id(path)}: MISMATCH shape {m_gpu.shape} vs {m_ref.shape}\n  gpu {res.stats}\n  ref {st_ref}")
            continue
        diff = np.abs(m_gpu - m_ref).max(axis=0) if len(m_ref) else np.zeros(len(names))
        bad = [n for i, n in enumerate(names) if diff[i] > (0.0 if n in INT_FIELDS else 1e-9)]
        cells_gpu, cells_ref = res.to_numpy(), to_structured(m_ref)
        packed_bad = [n for n in names if not np.array_equal(cells_gpu[n], cells_ref[n])]
        ok_all &= not bad and not packed_bad
        status = "bit-identical" if diff.max() == 0 else ("OK" if not bad else f"MISMATCH in {bad}")
        packed = "dashboard array identical" if not packed_bad else f"dashboard array MISMATCH in {packed_bad}"
        print(f"{frame_id(path)}: {len(m_gpu):6d} cells, splits {st_ref['n_splits']:5d}, "
              f"merges {st_ref['n_merges']:6d}, max diff {diff.max():.2e} -> {status}; {packed}")
    print("\nCUDA engine matches the reference" if ok_all else "\nMISMATCH")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
