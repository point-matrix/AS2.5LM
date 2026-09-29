# -*- coding: utf-8 -*-
"""
End-to-end pipeline: raw scan -> SalsaNext labels -> adaptive 2.5D grid.

    seg = Segmenter("salsanext_weights.pth", device="cuda")
    pipe = Pipeline(seg, engine="cuda")        # or engine="numpy" (CPU, bit-identical cells)
    result = pipe.process(load_scan(path), timed=True)
    result.cells, result.timing_ms, result.stats
"""

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .grid.reference import run_reference, to_structured


@dataclass
class FrameResult:
    labels: torch.Tensor          # (N,) class ids 0..19
    conf: torch.Tensor            # (N,) probability of the label
    pix_labels: torch.Tensor      # (N,) plain range-image lookup (before KNN)
    cells: np.ndarray             # FINAL_DTYPE structured array
    stats: dict                   # grid engine counts
    timing_ms: dict = field(default_factory=dict)
    grid_result: object = None    # CUDA GridResult (stage timings, GPU matrix), if engine="cuda"


def add_totals(t):
    """Add salsanext_total, grid_engine_total and end_to_end to a stage-timing dict (ms)."""
    t["salsanext_total"] = t.get("salsa_project", 0) + t.get("salsa_model", 0) + t.get("salsa_post", 0)
    t["grid_engine_total"] = t.get("grid_gpu", 0) + t.get("grid_cpu", 0) + t.get("grid_to_cpu", 0)
    t["end_to_end"] = t.get("upload", 0) + t["salsanext_total"] + t["grid_engine_total"]
    return t


class Pipeline:
    def __init__(self, segmenter, engine="cuda", fp16=True, build_dir=None, verbose_build=False):
        self.seg = segmenter
        self.fp16 = fp16
        self.engine_kind = engine
        self.device = segmenter.device
        if engine == "cuda":
            from .grid.cuda_engine import GridEngineCUDA
            self.engine = GridEngineCUDA(device=self.device, build_dir=build_dir, verbose=verbose_build)
        elif engine == "numpy":
            self.engine = None
        else:
            raise ValueError("engine must be 'cuda' or 'numpy'")

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    def grid_cells(self, raw, labels, conf):
        """Grid for given per-point labels/confidences (untimed) -> (cells, stats)."""
        if self.engine is not None:
            pts = torch.from_numpy(np.ascontiguousarray(np.asarray(raw)[:, :3])).to(self.device)
            res = self.engine.run(pts, torch.as_tensor(labels, device=self.device),
                                  torch.as_tensor(conf, device=self.device))
            return res.to_numpy(), res.stats
        lab = labels.cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
        cf = conf.float().cpu().numpy() if torch.is_tensor(conf) else np.asarray(conf)
        matrix, stats, _ = run_reference(np.asarray(raw)[:, :3], lab, cf)
        return to_structured(matrix), stats

    def process(self, raw, timed=False, fp16=None):
        """raw (N, 4) float32 numpy -> FrameResult. timed=True records stage times (ms)."""
        fp16 = self.fp16 if fp16 is None else fp16
        T = {} if timed else None
        t = self._sync() if timed else 0.0
        pts = torch.from_numpy(np.ascontiguousarray(raw)).to(self.device)
        if timed:
            t1 = self._sync(); T["upload"] = (t1 - t) * 1000
        labels, conf, pix = self.seg.segment(pts, fp16=fp16, timing=T)

        if self.engine is not None:
            t = self._sync() if timed else 0.0
            res = self.engine.run(pts[:, :3], labels, conf)
            if timed:
                t1 = self._sync(); T["grid_gpu"] = (t1 - t) * 1000; t = t1
            cells = res.to_numpy()
            if timed:
                t1 = self._sync(); T["grid_to_cpu"] = (t1 - t) * 1000
            stats = res.stats
        else:
            res = None
            t = self._sync() if timed else 0.0
            lab_np, conf_np = labels.cpu().numpy(), conf.float().cpu().numpy()
            if timed:
                t1 = time.perf_counter(); T["grid_to_cpu"] = (t1 - t) * 1000; t = t1
            matrix, stats, _ = run_reference(np.asarray(raw)[:, :3], lab_np, conf_np)
            cells = to_structured(matrix)
            if timed:
                T["grid_cpu"] = (time.perf_counter() - t) * 1000
        return FrameResult(labels, conf, pix, cells, stats, add_totals(T) if timed else {}, res)
