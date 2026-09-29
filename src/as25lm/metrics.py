# -*- coding: utf-8 -*-
"""
Evaluation metrics (definitions: docs/EVALUATION.md).

  SegmentationEval  per-point confusion matrices: IoU, mIoU over classes present, accuracy,
                    accuracy by distance band, per-frame accuracy / car IoU / person IoU
  GridEval          what the grid shows: grid-level class accuracy and obstacle safety
  PowerSampler      GPU board power via NVML in a background thread
"""

import threading
import time

import numpy as np

from . import config as C
from .grid.reference import cell_index_of_points

NC = C.N_CLASSES


def confusion(gt, pred, n=NC):
    """(n, n) counts, rows = ground truth, columns = prediction."""
    gt = np.asarray(gt, np.int64)
    pred = np.asarray(pred, np.int64)
    return np.bincount(gt * n + pred, minlength=n * n).reshape(n, n)


def iou_per_class(cm):
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    return tp / np.maximum(cm.sum(0) + cm.sum(1) - tp, 1)


def miou_present(cm):
    """Mean IoU over classes 1..19 that have ground-truth points; (miou, classes_present)."""
    iou, present = iou_per_class(cm)[1:], cm.sum(1)[1:] > 0
    return (float(iou[present].mean()) if present.any() else float("nan")), int(present.sum())


def accuracy(cm):
    total = cm.sum()
    return float(np.trace(cm) / total) if total else float("nan")


def _pct(a, b):
    return a / b if b else float("nan")


class SegmentationEval:
    """Accumulates confusion matrices for one or more label variants.

    Only points with a ground-truth class (> 0) count.
    """

    def __init__(self, variants, active):
        self.variants = list(variants)
        self.active = active
        self.cms = {v: np.zeros((NC, NC), np.int64) for v in self.variants}
        self.bands = [np.zeros((NC, NC), np.int64) for _ in C.DISTANCE_BAND_NAMES]
        self.frames = []

    def add(self, frame_id, gt, preds, xyz):
        """gt (N,), preds {variant: (N,)}, xyz (N, >=2) sensor-frame points."""
        gt = np.asarray(gt, np.int64)
        m = gt > 0
        for v in self.variants:
            self.cms[v] += confusion(gt[m], np.asarray(preds[v])[m])
        a = np.asarray(preds[self.active], np.int64)
        r = np.hypot(np.asarray(xyz)[:, 0], np.asarray(xyz)[:, 1])
        band = np.digitize(r, C.DISTANCE_BAND_EDGES_M)
        for b in range(len(self.bands)):
            mb = m & (band == b)
            self.bands[b] += confusion(gt[mb], a[mb])
        cm_f = confusion(gt[m], a[m])
        iou_f = iou_per_class(cm_f)
        self.frames.append(dict(frame=frame_id, accuracy=accuracy(cm_f),
                                car_iou=float(iou_f[C.CAR]), eval_car_points=int(cm_f[C.CAR].sum()),
                                person_iou=float(iou_f[C.PERSON]), eval_person_points=int(cm_f[C.PERSON].sum())))

    def summary(self):
        names = [C.LABEL_NAMES[c] for c in range(1, NC)]
        present = self.cms[self.active].sum(1)[1:] > 0
        out = dict(frames_evaluated=len(self.frames), active=self.active, variants={})
        for v in self.variants:
            miou, k = miou_present(self.cms[v])
            out["variants"][v] = dict(miou=miou, overall_accuracy=accuracy(self.cms[v]), classes_present=k)
        iou = iou_per_class(self.cms[self.active])[1:]
        out["per_class_iou"] = {n: (float(x) if p else None) for n, x, p in zip(names, iou, present)}
        out["per_class_gt_points"] = {n: int(x) for n, x in zip(names, self.cms[self.active].sum(1)[1:])}
        fp = self.cms[self.active].sum(0)[1:]
        out["absent_classes_predicted_points"] = {n: int(f) for n, f, p in zip(names, fp, present) if not p and f}
        out["by_distance"] = {}
        for name, cm in zip(C.DISTANCE_BAND_NAMES, self.bands):
            if cm.sum():
                miou, k = miou_present(cm)
                out["by_distance"][name] = dict(accuracy=accuracy(cm), miou=miou, classes_present=k,
                                                points=int(cm.sum()))
        return out


class GridEval:
    """Grid-level class accuracy and obstacle safety, from the cells the dashboard shows."""

    def __init__(self, blocked_below=C.TRAV_BLOCKED):
        self.blocked_below = blocked_below
        self.is_obstacle = np.array([C.SEMANTIC_PRIORITY[c] >= 4 and c != 0 for c in range(NC)])
        self.is_dynamic = np.isin(np.arange(NC), range(1, 9))          # vehicles and people
        self.is_drivable = np.isin(np.arange(NC), C.DRIVABLE_CLASSES)   # road, parking
        self.s = dict(obs=0, obs_missed=0, dyn=0, dyn_missed=0, drv=0, drv_blocked=0)
        self.g = dict(cells=0, correct=0, area=0.0, area_correct=0.0)
        self.by_size = {s: [0, 0] for s in sorted(C.LEVEL_SIZE_M.values())}
        self.frames = []

    def add(self, frame_id, gt, xyz, cells):
        gt = np.asarray(gt, np.int64)
        idx = cell_index_of_points(np.asarray(xyz)[:, :3], cells)
        on = idx >= 0
        trav = np.where(on, cells["traversability"][np.clip(idx, 0, None)], np.nan)
        blocked = trav < self.blocked_below                    # False where there is no cell
        obs, dyn, drv = self.is_obstacle[gt], self.is_dynamic[gt], self.is_drivable[gt]
        f = dict(obs=int(obs.sum()), obs_missed=int((obs & ~blocked).sum()),
                 dyn=int(dyn.sum()), dyn_missed=int((dyn & ~blocked).sum()),
                 drv=int(drv.sum()), drv_blocked=int((drv & blocked).sum()))
        for k, v in f.items():
            self.s[k] += v

        sel = on & (gt > 0)
        votes = np.bincount(idx[sel] * NC + gt[sel], minlength=len(cells) * NC).reshape(len(cells), NC)
        has = votes.sum(1) > 0
        ok = has & (votes.argmax(1) == cells["semantic_class"])
        area = cells["resolution"] ** 2
        self.g["cells"] += int(has.sum()); self.g["correct"] += int(ok.sum())
        self.g["area"] += float(area[has].sum()); self.g["area_correct"] += float(area[ok].sum())
        for s in self.by_size:
            size_sel = np.isclose(cells["resolution"], s)
            self.by_size[s][0] += int((has & size_sel).sum())
            self.by_size[s][1] += int((ok & size_sel).sum())
        self.frames.append(dict(frame=frame_id, obstacle_points=f["obs"], obstacle_missed=f["obs_missed"],
                                dynamic_points=f["dyn"], dynamic_missed=f["dyn_missed"],
                                grid_cells_evaluated=int(has.sum()),
                                grid_cell_accuracy=_pct(int(ok.sum()), int(has.sum()))))

    def summary(self):
        s, g = self.s, self.g
        return dict(
            grid_accuracy=dict(by_cell=_pct(g["correct"], g["cells"]), by_area=_pct(g["area_correct"], g["area"]),
                               cells=g["cells"],
                               by_size={f"{k * 100:g} cm": _pct(c, n) for k, (n, c) in self.by_size.items()}),
            safety=dict(threshold=self.blocked_below,
                        missed_vehicles_people=_pct(s["dyn_missed"], s["dyn"]),
                        missed_all_obstacles=_pct(s["obs_missed"], s["obs"]),
                        false_alarm_road_parking=_pct(s["drv_blocked"], s["drv"]), **s))


def frame_accuracy(gt, pred):
    """Accuracy, car IoU and person IoU for one frame (None where the class is absent)."""
    gt = np.asarray(gt, np.int64)
    pred = np.asarray(pred, np.int64)
    m = gt > 0
    if not m.any():
        return None

    def iou(c):
        if not (gt[m] == c).any():
            return None
        inter = int(((gt == c) & (pred == c) & m).sum())
        union = int((((gt == c) | (pred == c)) & m).sum())
        return inter / union

    return dict(accuracy=float((gt[m] == pred[m]).mean()), car_iou=iou(C.CAR), person_iou=iou(C.PERSON),
                gt_car_points=int((gt == C.CAR).sum()), gt_person_points=int((gt == C.PERSON).sum()))


class PowerSampler:
    """GPU board power (W) every `interval` s in a background thread (NVML; no-op if unavailable).

        with PowerSampler() as p:
            ...                      # record (start, end) processing intervals
        watts = p.watts_during(intervals)
    """

    def __init__(self, interval=0.05, gpu_index=0):
        self.interval, self.t, self.w = interval, [], []
        self._stop, self._th = threading.Event(), None
        self.limit_w = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self.limit_w = pynvml.nvmlDeviceGetEnforcedPowerLimit(self._h) / 1000
            pynvml.nvmlDeviceGetPowerUsage(self._h)
            self.available = True
        except Exception:
            self.available = False

    def _run(self):
        while not self._stop.is_set():
            try:
                w = self._nvml.nvmlDeviceGetPowerUsage(self._h) / 1000.0
                self.t.append(time.perf_counter())
                self.w.append(w)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        if self.available:
            self._th = threading.Thread(target=self._run, daemon=True)
            self._th.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._th is not None:
            self._th.join()

    def watts_during(self, intervals):
        """Readings taken inside the given (start, end) perf_counter intervals."""
        if not self.w or not intervals:
            return np.array([])
        t, w = np.array(self.t), np.array(self.w)
        starts, ends = np.array(intervals).T
        i = np.searchsorted(starts, t, side="right") - 1
        inside = (i >= 0) & (t <= ends[np.clip(i, 0, None)])
        return w[inside]
