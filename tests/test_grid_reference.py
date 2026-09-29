"""Grid Engine invariants on a synthetic scene (NumPy reference)."""

import numpy as np

from as25lm import config as C
from as25lm.grid.reference import cell_index_of_points, pack_key, run_reference, to_structured


def _grid(scene):
    raw, labels, conf = scene
    matrix, stats, _ = run_reference(raw[:, :3], labels, conf)
    return raw, labels, conf, to_structured(matrix), stats


def test_every_point_kept(scene):
    raw, labels, _, cells, stats = _grid(scene)
    assert cells["point_count"].sum() == (labels != 0).sum() == stats["n_points_used"]


def test_cells_never_overlap(scene):
    _, _, _, cells, _ = _grid(scene)
    lvl = cells["level"].astype(np.int64)
    ix = np.floor(cells["x_min"] / cells["resolution"] + 0.5).astype(np.int64)
    iy = np.floor(cells["y_min"] / cells["resolution"] + 0.5).astype(np.int64)
    k = 1 << (C.MAX_LEVEL - lvl)
    rep = k * k
    off = np.concatenate([np.arange(r) for r in rep])
    kk = np.repeat(k, rep)
    fine = pack_key(3, np.repeat(ix * k, rep) + off % kk, np.repeat(iy * k, rep) + off // kk)
    assert np.unique(fine).size == fine.size


def test_stats_match_brute_force(scene):
    raw, labels, conf, cells, _ = _grid(scene)
    keep = labels != 0
    idx = cell_index_of_points(raw[keep, :3], cells)
    assert (idx >= 0).all()
    z = raw[keep, 2].astype(np.float64)
    cnt = np.bincount(idx, minlength=len(cells))
    assert np.array_equal(cnt, cells["point_count"].astype(np.int64))
    assert np.allclose(np.bincount(idx, weights=z, minlength=len(cells)) / cnt, cells["elevation"], atol=1e-9)
    zmin = np.full(len(cells), np.inf); np.minimum.at(zmin, idx, z)
    assert np.array_equal(zmin, cells["min_height"])
    sc = np.zeros((len(cells), C.N_CLASSES)); np.add.at(sc, (idx, labels[keep]), conf[keep])
    assert np.array_equal(sc.argmax(1), cells["semantic_class"])


def test_points_inside_their_cells(scene):
    raw, labels, _, cells, _ = _grid(scene)
    keep = labels != 0
    c = cells[cell_index_of_points(raw[keep, :3], cells)]
    x, y = raw[keep, 0].astype(np.float64), raw[keep, 1].astype(np.float64)
    assert np.all((x >= c["x_min"]) & (x < c["x_max"]) & (y >= c["y_min"]) & (y < c["y_max"]))


def test_near_field_stays_5cm(scene):
    _, _, _, cells, _ = _grid(scene)
    L = cells["level"].astype(np.int64)
    ix = np.floor(cells["x_min"] / cells["resolution"] + 0.5).astype(np.int64)
    iy = np.floor(cells["y_min"] / cells["resolution"] + 0.5).astype(np.int64)
    cx, cy = ((ix >> L) + 0.5) * C.TILE_M, ((iy >> L) + 0.5) * C.TILE_M
    inner = cx * cx + cy * cy < C.NO_MERGE_RADIUS_M ** 2
    assert inner.any() and np.all(L[inner] == 3)


def test_unlabeled_points_dropped(scene):
    raw, labels, conf = scene
    lab = labels.copy()
    lab[::50] = 0
    _, stats, _ = run_reference(raw[:, :3], lab, conf)
    assert stats["n_unlabeled_dropped"] == (lab == 0).sum()
    assert stats["n_points_used"] == (lab != 0).sum()
