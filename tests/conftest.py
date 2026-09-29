"""Synthetic LiDAR scenes for tests (no KITTI data needed)."""

import numpy as np
import pytest


def synthetic_scan(seed=0):
    """Ground rings like an HDL-64 plus a few objects; returns (raw (N,4) float32, labels, conf)."""
    rng = np.random.default_rng(seed)
    pts, lab = [], []
    # ground: 40 rings from 3 m to 80 m, road inside |y| < 6 m, sidewalk / terrain outside
    for r in np.geomspace(3, 80, 40):
        n = int(400 + 30 * r)
        a = rng.uniform(-np.pi, np.pi, n)
        x, y = r * np.cos(a), r * np.sin(a)
        z = -1.73 + rng.normal(0, 0.01, n)
        c = np.where(np.abs(y) < 6, 9, np.where(np.abs(y) < 8, 11, 17))
        pts.append(np.stack([x, y, z], 1)); lab.append(c)
    # objects: two cars, a building wall, a person, a pole, a bush
    def box(cx, cy, sx, sy, z0, z1, n, cls):
        p = np.stack([rng.uniform(cx - sx, cx + sx, n), rng.uniform(cy - sy, cy + sy, n),
                      rng.uniform(z0, z1, n)], 1)
        pts.append(p); lab.append(np.full(n, cls))
    box(8, 2, 2.0, 0.9, -1.7, -0.2, 3000, 1)
    box(25, -3, 2.0, 0.9, -1.7, -0.2, 1500, 1)
    box(15, 12, 10, 0.3, -1.7, 4.0, 4000, 13)
    box(6, -7, 0.3, 0.3, -1.7, 0.1, 500, 6)
    box(12, 7, 0.1, 0.1, -1.7, 3.0, 300, 18)
    box(40, 20, 1.5, 1.5, -1.7, 0.5, 800, 15)
    xyz = np.concatenate(pts).astype(np.float32)
    labels = np.concatenate(lab).astype(np.int64)
    rem = rng.uniform(0, 1, len(xyz)).astype(np.float32)
    conf = rng.uniform(0.5, 1.0, len(xyz)).astype(np.float32)
    return np.concatenate([xyz, rem[:, None]], 1), labels, conf


@pytest.fixture(scope="session")
def scene():
    return synthetic_scan(0)
