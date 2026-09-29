# -*- coding: utf-8 -*-
"""Reading KITTI velodyne scans and SemanticKITTI labels."""

import glob
import os

import numpy as np

from . import config as C

MAX_POINTS = 200_000          # a KITTI HDL-64 scan has ~120k


def frame_id(path):
    """'.../velodyne/001533.bin' -> '001533'."""
    return os.path.basename(path).split(".")[0]


def load_scan(path, validate=False):
    """KITTI velodyne .bin -> (N, 4) float32 [x, y, z, remission].

    validate=True rejects malformed files and drops non-finite rows (use for uploads).
    """
    data = np.fromfile(path, dtype=np.float32)
    if not validate:
        return data.reshape(-1, 4)
    if data.size == 0 or data.size % 4:
        raise ValueError("Not a KITTI .bin scan: the file must hold float32 x, y, z, remission rows.")
    raw = data.reshape(-1, 4)
    raw = raw[np.isfinite(raw).all(axis=1)]
    if len(raw) == 0:
        raise ValueError("The scan has no valid points.")
    if len(raw) > MAX_POINTS:
        raise ValueError(f"The scan has {len(raw):,} points; the limit is {MAX_POINTS:,}.")
    return np.ascontiguousarray(raw)


def load_labels(path):
    """SemanticKITTI .label -> (N,) SalsaNext class ids (learning_map applied, 0 = unlabeled)."""
    raw = np.fromfile(path, dtype=np.uint32) & 0xFFFF
    return C.LEARNING_MAP_LUT[raw]


def list_scans(scan_dir, start=None, count=None):
    """Sorted .bin paths; optionally frames start .. start + count - 1 by frame number."""
    files = sorted(glob.glob(os.path.join(scan_dir, "*.bin")))
    if start is not None:
        files = [f for f in files if int(frame_id(f)) >= start]
    if count is not None:
        files = files[:count]
    return files


def label_path(label_dir, scan_path):
    """Matching .label path, or None if there is no label folder / file."""
    if not label_dir:
        return None
    p = os.path.join(label_dir, frame_id(scan_path) + ".label")
    return p if os.path.exists(p) else None
