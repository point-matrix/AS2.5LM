# -*- coding: utf-8 -*-
"""
Grid Engine: CUDA wrapper
=========================
Builds the C++/CUDA extension once (JIT, cached in build_dir) and runs it on
GPU tensors, so SalsaNext output goes straight in without a CPU round trip.
Output is bit-identical to the NumPy reference (grid/reference.py).

    engine = GridEngineCUDA()                      # compiles on first use
    res = engine.run(xyz, labels, conf)            # all CUDA tensors
    cells = res.to_numpy()                         # FINAL_DTYPE array
"""

import os

import numpy as np
import torch

from .. import config as C
from .reference import grid_config_from_bounds, to_structured

_EXT = None
_PINNED = None
STAGES = ("filter+keys", "cells", "split", "merge", "finalize")
STAT_NAMES = ("n_points_in", "n_points_used", "n_stage1_cells", "n_splits",
              "n_merges", "n_final_cells", "n_unlabeled_dropped")


def build_extension(build_dir=None, verbose=False):
    """Compile (or load the cached) extension. Takes ~1-2 min the first time."""
    global _EXT
    if _EXT is not None:
        return _EXT
    from torch.utils.cpp_extension import load

    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
    if build_dir:
        os.makedirs(build_dir, exist_ok=True)
    _EXT = load(
        name="as25lm_grid_cuda",
        sources=[os.path.join(src, "bindings.cpp"), os.path.join(src, "grid_engine.cu")],
        extra_cflags=["-O3"],
        # --fmad=false keeps float64 results bit-identical to the NumPy reference
        extra_cuda_cflags=["-O3", "--fmad=false"],
        build_directory=build_dir,
        verbose=verbose,
    )
    return _EXT


def _pinned_buffer(numel):
    """Reusable page-locked float64 host buffer (allocating one is slow)."""
    global _PINNED
    if _PINNED is None or _PINNED.numel() < numel:
        _PINNED = torch.empty(int(numel * 1.5) + 1024, dtype=torch.float64, pin_memory=True)
    return _PINNED[:numel]


class GridResult:
    """Output of one frame. `matrix` stays on the GPU until to_numpy()."""

    def __init__(self, matrix, bounds, stats, stage_ms, packed=None):
        self.matrix = matrix                 # (n_cells, 20) float64, CUDA
        self.bounds = bounds                 # (4,) xmin, ymin, xmax, ymax, CUDA
        self.stats = dict(zip(STAT_NAMES, stats.tolist()))
        self.stage_ms = dict(zip(STAGES, stage_ms.tolist())) if stage_ms.numel() else {}
        self.packed = packed if packed is not None and packed.numel() else None  # FINAL_DTYPE bytes, CUDA

    def _host_view(self):
        """Copy the cell matrix into a reused pinned buffer (overwritten next frame)."""
        m = self.matrix
        if m.numel() == 0:
            return np.zeros((0, C.N_FINAL_FIELDS))
        host = _pinned_buffer(m.numel()).view(m.shape)
        host.copy_(m, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return host.numpy()

    def matrix_numpy(self):
        """(n_cells, 20) float64 host copy, columns in FINAL_DTYPE order."""
        return self._host_view().copy()

    def to_numpy(self):
        """FINAL_DTYPE structured array (same schema as the v1 engine).

        The GPU has already written the rows in FINAL_DTYPE's byte layout, so
        this is a single copy into a fresh array with no per-column work.
        """
        if self.packed is None:
            return self.to_numpy_columns()
        n = self.matrix.shape[0]
        out = np.empty(n, dtype=C.FINAL_DTYPE)
        torch.from_numpy(out.view(np.uint8).reshape(-1)).copy_(self.packed)
        return out

    def to_numpy_columns(self):
        """Older path: copy the float64 matrix, then convert column by column on the CPU."""
        return to_structured(self._host_view())

    def grid_config(self):
        if self.matrix.numel() == 0:
            return None
        return grid_config_from_bounds(self.bounds.double().cpu().numpy())


class GridEngineCUDA:
    def __init__(self, device="cuda", build_dir=None, verbose=False):
        self.ext = build_extension(build_dir=build_dir, verbose=verbose)
        self.device = torch.device(device)
        self.pri_lut = torch.as_tensor(C.SEMANTIC_PRIORITY_LUT, dtype=torch.int32, device=self.device)
        self.trav_lut = torch.as_tensor(C.TRAVERSABILITY_LUT, dtype=torch.float64, device=self.device)
        self.params = C.cuda_params()
        self.pack_layout, self.pack_itemsize = _pack_layout(self.device)
        self._no_pack = torch.empty(0, dtype=torch.int32, device=self.device)

    def run(self, xyz, labels, conf, profile=False, pack=True):
        """xyz (N,>=3), labels (N,) in 0..19, conf (N,) -- CUDA tensors preferred.

        pack=True also writes the rows in FINAL_DTYPE byte layout on the GPU,
        which makes to_numpy() a single copy.
        """
        xyz = torch.as_tensor(xyz, device=self.device)
        labels = torch.as_tensor(labels, device=self.device)
        conf = torch.as_tensor(conf, device=self.device)
        layout = self.pack_layout if pack else self._no_pack
        out = self.ext.run(xyz, labels, conf, self.pri_lut, self.trav_lut, self.params, profile,
                           layout, self.pack_itemsize if pack else 0)
        return GridResult(*out)


def _pack_layout(device):
    """(2, 20) int32 tensor: byte offset of each FINAL_DTYPE field, and 1 for int32 fields."""
    dt = C.FINAL_DTYPE
    offsets, is_int = [], []
    for name, ftype in C.FINAL_FIELDS:
        ftype = np.dtype(ftype)
        if ftype not in (np.dtype(np.float64), np.dtype(np.int32)):
            raise TypeError(f"field {name}: only float64 and int32 can be packed on the GPU")
        offsets.append(dt.fields[name][1])
        is_int.append(1 if ftype == np.dtype(np.int32) else 0)
    if any(o % 4 for o in offsets) or dt.itemsize % 4:
        raise ValueError("FINAL_DTYPE field offsets must be multiples of 4 bytes")
    return torch.tensor([offsets, is_int], dtype=torch.int32, device=device).contiguous(), dt.itemsize
