# -*- coding: utf-8 -*-
"""
LGF1 frame format -- one LiDAR frame (points + grid cells) for the dashboard
===========================================================================
Written by the clip export (scripts/export_clip.py) and the live Space (single
upload), read by dashboard/decodeFrame.ts in the browser. Spec: docs/FORMAT.md.

Layout (little-endian):
    bytes 0-3   "LGF1"
    bytes 4-7   uint32 header length H
    bytes 8..   UTF-8 JSON header, padded with spaces so 8 + H is a multiple of 8
    then each array ("section") starts at a multiple of 8; the header's
    "sections" list gives name, dtype, shape, byte offset and byte length.
Files may be gzip-compressed as a whole (.lgf.gz).
"""

import gzip
import json
import struct

import numpy as np

from . import config as C

MAGIC = b"LGF1"
VERSION = 1
ALIGN = 8
# Display hint: KITTI heights are relative to the LiDAR, mounted ~1.73 m above the road.
SENSOR_HEIGHT_M = C.SENSOR_HEIGHT_M

_DTYPES = {"int16": np.int16, "uint8": np.uint8, "uint16": np.uint16, "float32": np.float32}


def _q(x, scale, dtype):
    info = np.iinfo(dtype)
    return np.clip(np.rint(np.asarray(x, np.float64) * scale), info.min, info.max).astype(dtype)


def _cell_indices(cells):
    res = cells["resolution"]
    ix = np.floor(cells["x_min"] / res + 0.5).astype(np.int64)
    iy = np.floor(cells["y_min"] / res + 0.5).astype(np.int64)
    return np.stack([ix, iy], axis=1)


def build_sections(points_xyz, points_intensity, points_class, points_conf, cells,
                   points_gt=None, point_stride=1):
    """Quantise points and cells into the arrays stored in a frame."""
    sl = slice(None, None, max(int(point_stride), 1))
    xyz = np.asarray(points_xyz)[sl, :3]
    sec = {
        "points.xyz_cm": _q(xyz, 100.0, np.int16),
        "points.intensity": _q(np.clip(np.asarray(points_intensity)[sl], 0, 1), 255.0, np.uint8),
        "points.class": np.asarray(points_class)[sl].astype(np.uint8),
        "points.confidence": _q(np.clip(np.asarray(points_conf)[sl], 0, 1), 255.0, np.uint8),
    }
    if points_gt is not None:
        sec["points.gt_class"] = np.asarray(points_gt)[sl].astype(np.uint8)
    ixy = _cell_indices(cells)
    sec.update({
        "cells.ixy": np.clip(ixy, -32768, 32767).astype(np.int16),
        "cells.level": cells["level"].astype(np.uint8),
        "cells.class": cells["semantic_class"].astype(np.uint8),
        "cells.z_cm": _q(np.stack([cells["min_height"], cells["max_height"], cells["elevation"]], axis=1),
                         100.0, np.int16),
        "cells.z_std_mm": _q(np.sqrt(np.maximum(cells["elevation_variance"], 0)), 1000.0, np.uint16),
        "cells.traversability": _q(cells["traversability"], 255.0, np.uint8),
        "cells.complexity": _q(cells["terrain_complexity"], 255.0, np.uint8),
        "cells.confidence": _q(cells["semantic_confidence"], 255.0, np.uint8),
        "cells.point_count": np.clip(cells["point_count"], 0, 65535).astype(np.uint16),
    })
    return sec


def encode_frame(header, sections):
    """Serialise a header dict and {name: ndarray} sections to LGF1 bytes."""
    names = list(sections)
    arrays = [np.ascontiguousarray(sections[n]) for n in names]
    for n, a in zip(names, arrays):
        if a.dtype.name not in _DTYPES:
            raise TypeError(f"section {n}: unsupported dtype {a.dtype}")

    def header_bytes(offsets):
        h = dict(header)
        h["format"], h["version"] = "LGF1", VERSION
        h["sections"] = [dict(name=n, dtype=a.dtype.name, shape=list(a.shape), offset=o, bytes=a.nbytes)
                         for n, a, o in zip(names, arrays, offsets)]
        raw = json.dumps(h, separators=(",", ":")).encode("utf-8")
        pad = (-(8 + len(raw))) % ALIGN
        return raw + b" " * pad

    # Offsets depend on header length and vice versa: iterate until stable.
    offsets = [0] * len(arrays)
    for _ in range(10):
        hb = header_bytes(offsets)
        pos, new = 8 + len(hb), []
        for a in arrays:
            pos += (-pos) % ALIGN
            new.append(pos)
            pos += a.nbytes
        if new == offsets:
            break
        offsets = new
    out = bytearray(MAGIC + struct.pack("<I", len(hb)) + hb)
    for a, o in zip(arrays, offsets):
        out += b"\0" * (o - len(out))
        out += a.tobytes()
    return bytes(out)


def decode_frame(buf):
    """LGF1 bytes (optionally gzip) -> (header dict, {name: ndarray})."""
    if buf[:2] == b"\x1f\x8b":
        buf = gzip.decompress(buf)
    if buf[:4] != MAGIC:
        raise ValueError("not an LGF1 frame")
    (hlen,) = struct.unpack_from("<I", buf, 4)
    header = json.loads(buf[8:8 + hlen].decode("utf-8"))
    arrays = {}
    for s in header["sections"]:
        a = np.frombuffer(buf, dtype=_DTYPES[s["dtype"]], count=int(np.prod(s["shape"])), offset=s["offset"])
        arrays[s["name"]] = a.reshape(s["shape"])
    return header, arrays


def frame_header(frame_id, source, n_points_total, point_stride, cells, stats=None,
                 timing_ms=None, device=None, accuracy=None, sequence=None, extra=None):
    """Header fields shared by exported and live frames."""
    h = dict(
        frame_id=str(frame_id), source=source, sequence=sequence,
        n_points_total=int(n_points_total), point_stride=int(max(point_stride, 1)),
        n_cells=int(len(cells)),
        level_sizes_m=[C.LEVEL_SIZE_M[L] for L in range(C.MAX_LEVEL + 1)],
        sensor_height_m=SENSOR_HEIGHT_M,
        units=dict(xyz="cm", z="cm", z_std="mm", fractions="0-255 = 0-1"),
        stats=stats or {}, timing_ms=timing_ms or {}, device=device, accuracy=accuracy,
    )
    if extra:
        h.update(extra)
    return h


def write_frame(path, header, sections, compress=True):
    data = encode_frame(header, sections)
    if compress:
        data = gzip.compress(data, compresslevel=6)
    with open(path, "wb") as f:
        f.write(data)
    return len(data)


def class_table():
    """Class list for the manifest / dashboard legend."""
    return [dict(id=i, name=C.LABEL_NAMES[i], color=C.CLASS_COLORS[i],
                 priority=int(C.SEMANTIC_PRIORITY[i]), traversability=float(C.TRAVERSABILITY_WEIGHT[i]))
            for i in range(C.N_CLASSES)]
