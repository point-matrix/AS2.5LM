# -*- coding: utf-8 -*-
"""
Grid Engine: NumPy reference implementation
===========================================
This is the specification the CUDA engine must reproduce. Every step mirrors
a CUDA kernel, including summation order, so results match bit for bit:
sums use np.bincount (sequential, in sorted point order), sorts are stable,
and all geometry is float64.

Pipeline
  1. Drop unlabeled points (class 0).
  2. Per point: finest-cell index (5 cm), 40 cm tile, tile-centre distance ->
     base level. Every point in a tile uses the same cell size.
  3. Group points by base-level cell -> stage-1 leaf cells.
  4. Split: leaves with priority >= 4, level < 3 and >= 2 points are replaced
     by their non-empty quadrant children (one level finer), recomputed from
     the points themselves.
  5. Merge (single pass): a parent whose children are all leaves, all the
     same class and all priority <= 2 becomes one leaf with exact combined
     statistics. Parents whose tile centre is within NO_MERGE_RADIUS_M never
     merge, so the near field keeps its 5 cm cells.
  6. Final per-cell fields (FINAL_DTYPE order).
"""

import numpy as np
from .. import config as C

_NC = C.N_CLASSES


# ── key helpers ────────────────────────────────────────────────
def pack_key(level, ix, iy):
    return ((np.asarray(level, np.int64) << C.LEVEL_SHIFT)
            | ((np.asarray(ix, np.int64) + C.KEY_OFF) << C.KEY_BITS)
            | (np.asarray(iy, np.int64) + C.KEY_OFF))


def unpack_key(key):
    key = np.asarray(key, np.int64)
    level = key >> C.LEVEL_SHIFT
    ix = ((key >> C.KEY_BITS) & C.KEY_MASK) - C.KEY_OFF
    iy = (key & C.KEY_MASK) - C.KEY_OFF
    return level, ix, iy


def parent_key(key):
    """Parent cell key, or -1 for level-0 cells."""
    level, ix, iy = unpack_key(key)
    out = pack_key(np.maximum(level - 1, 0), ix >> 1, iy >> 1)
    return np.where(level > 0, out, -1)


# ── stage helpers ──────────────────────────────────────────────
def point_cells(xyz):
    """Finest-cell indices and base-level cell key for every point."""
    x = xyz[:, 0].astype(np.float64)
    y = xyz[:, 1].astype(np.float64)
    lim = C.KEY_OFF - 1
    i3x = np.clip(np.floor(x / C.FINEST_CELL_M).astype(np.int64), -lim, lim)
    i3y = np.clip(np.floor(y / C.FINEST_CELL_M).astype(np.int64), -lim, lim)

    tx = i3x >> C.MAX_LEVEL
    ty = i3y >> C.MAX_LEVEL
    cx = (tx.astype(np.float64) + 0.5) * C.TILE_M
    cy = (ty.astype(np.float64) + 0.5) * C.TILE_M
    r2 = cx * cx + cy * cy

    level = np.zeros(len(x), dtype=np.int64)
    level[r2 < C.BAND_EDGE_L1_M * C.BAND_EDGE_L1_M] = 1
    level[r2 < C.BAND_EDGE_L2_M * C.BAND_EDGE_L2_M] = 2
    level[r2 < C.BAND_EDGE_L3_M * C.BAND_EDGE_L3_M] = 3
    sh = C.MAX_LEVEL - level
    key = pack_key(level, i3x >> sh, i3y >> sh)
    return i3x, i3y, key


def near_field(key):
    """True for cells whose 40 cm tile centre is within NO_MERGE_RADIUS_M."""
    level, ix, iy = unpack_key(key)
    cx = ((ix >> level).astype(np.float64) + 0.5) * C.TILE_M
    cy = ((iy >> level).astype(np.float64) + 0.5) * C.TILE_M
    return cx * cx + cy * cy < C.NO_MERGE_RADIUS_M * C.NO_MERGE_RADIUS_M


def reduce_sorted(sorted_keys, perm, z, labels, conf):
    """Per-cell statistics for points already sorted by cell key.

    Returns (cells, seg) where seg[j] is the cell of sorted point j.
    """
    n = len(sorted_keys)
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    counts = np.diff(np.r_[starts, n])
    ns = len(starts)
    seg = np.repeat(np.arange(ns, dtype=np.int64), counts)

    zz = z[perm].astype(np.float64)
    s1 = np.bincount(seg, weights=zz, minlength=ns)
    s2 = np.bincount(seg, weights=zz * zz, minlength=ns)
    zmin = np.minimum.reduceat(zz, starts)
    zmax = np.maximum.reduceat(zz, starts)

    sc = np.bincount(seg * _NC + labels[perm].astype(np.int64),
                     weights=conf[perm].astype(np.float64),
                     minlength=ns * _NC).reshape(ns, _NC)
    cls = np.argmax(sc, axis=1).astype(np.int64)
    total = np.zeros(ns)
    for c in range(_NC):                  # sequential, like the CUDA loop
        total = total + sc[:, c]
    wscore = sc[np.arange(ns), cls]

    cells = dict(key=sorted_keys[starts].copy(), cnt=counts.astype(np.int64),
                 sum=s1, sumsq=s2, zmin=zmin, zmax=zmax,
                 cls=cls, wscore=wscore, total=total)
    return cells, seg


def _select(cells, idx):
    return {k: v[idx] for k, v in cells.items()}


def _concat(a, b):
    return {k: np.concatenate([a[k], b[k]]) for k in a}


def finalize(cells):
    """Per-cell output matrix, columns in FINAL_DTYPE order."""
    level, ix, iy = unpack_key(cells['key'])
    size = C.FINEST_CELL_M * np.exp2((C.MAX_LEVEL - level).astype(np.float64))
    x_min = ix.astype(np.float64) * size
    x_max = (ix + 1).astype(np.float64) * size
    y_min = iy.astype(np.float64) * size
    y_max = (iy + 1).astype(np.float64) * size

    cnt = cells['cnt'].astype(np.float64)
    mean = cells['sum'] / cnt
    var = np.maximum(cells['sumsq'] / cnt - mean * mean, 0.0)
    hr = cells['zmax'] - cells['zmin']
    std = np.sqrt(var)
    complexity = (C.TERRAIN_W_RANGE * np.minimum(hr / C.TERRAIN_RANGE_LIMIT_M, 1.0)
                  + C.TERRAIN_W_STD * np.minimum(std / C.TERRAIN_STD_LIMIT_M, 1.0))
    cls = cells['cls']
    trav = C.TRAVERSABILITY_LUT[cls] * (1.0 - C.TRAV_TERRAIN_PENALTY * complexity)
    trav = np.minimum(np.maximum(trav, 0.0), 1.0)
    total = cells['total']
    conf = np.where(total > 0, cells['wscore'] / np.where(total > 0, total, 1.0), 0.0)

    cols = [x_min, x_max, y_min, y_max,
            (x_min + x_max) * 0.5, (y_min + y_max) * 0.5,
            size, level.astype(np.float64),
            mean, cells['zmin'], cells['zmax'], var,
            cls.astype(np.float64), conf, np.ones_like(mean),
            complexity, trav, cnt,
            C.SEMANTIC_PRIORITY_LUT[cls].astype(np.float64), hr]
    return np.stack(cols, axis=1)


# ── public API ─────────────────────────────────────────────────
def run_reference(xyz, labels, conf):
    """Run the full v2 Grid Engine on CPU.

    xyz: (N, 3) float, labels: (N,) int in 0..19, conf: (N,) float.
    Returns (matrix, stats, bounds): matrix is (n_cells, 20) float64 in
    FINAL_DTYPE column order, sorted by cell key.
    """
    xyz = np.asarray(xyz)[:, :3]
    labels = np.asarray(labels).astype(np.int64)
    conf = np.asarray(conf)
    n_in = len(labels)
    if labels.size and (labels.min() < 0 or labels.max() >= _NC):
        raise ValueError(f"labels must be in 0..{_NC - 1}")

    keep = labels != C.UNLABELED
    xyz, labels, conf = xyz[keep], labels[keep], conf[keep]
    n = len(labels)
    stats = dict(n_points_in=n_in, n_points_used=n, n_unlabeled_dropped=n_in - n,
                 n_stage1_cells=0, n_splits=0, n_merges=0, n_final_cells=0)
    if n == 0:
        return np.zeros((0, C.N_FINAL_FIELDS)), stats, None

    bounds = np.array([xyz[:, 0].min(), xyz[:, 1].min(),
                       xyz[:, 0].max(), xyz[:, 1].max()], dtype=np.float64)
    z = xyz[:, 2]

    # Stage 1: base-level cells
    i3x, i3y, key0 = point_cells(xyz)
    order = np.argsort(key0, kind='stable')
    ks = key0[order]
    cells1, seg1 = reduce_sorted(ks, order, z, labels, conf)
    stats['n_stage1_cells'] = len(cells1['key'])

    # Stage 2: split high-priority cells into quadrant children
    level1 = cells1['key'] >> C.LEVEL_SHIFT
    split = ((C.SEMANTIC_PRIORITY_LUT[cells1['cls']] >= C.SPLIT_PRIORITY_THRESHOLD)
             & (level1 < C.MAX_LEVEL)
             & (cells1['cnt'] >= C.SPLIT_MIN_POINTS))
    n_splits = int(split.sum())
    stats['n_splits'] = n_splits
    if n_splits:
        pt_split = split[seg1]
        child_level = (ks >> C.LEVEL_SHIFT) + 1
        sh = np.maximum(C.MAX_LEVEL - child_level, 0)
        p = order
        child_key = pack_key(child_level, i3x[p] >> sh, i3y[p] >> sh)
        key1 = np.where(pt_split, child_key, ks)
        o2 = np.argsort(key1, kind='stable')
        cells2, _ = reduce_sorted(key1[o2], order[o2], z, labels, conf)
    else:
        cells2 = cells1

    # Stage 3: merge (single pass)
    n2 = len(cells2['key'])
    par2 = parent_key(cells2['key'])
    leaf_m = np.flatnonzero(par2 >= 0)
    mkey = par2[leaf_m]
    msrc = leaf_m.astype(np.int64)
    if n_splits:
        par_s = parent_key(cells1['key'][split])
        par_s = par_s[par_s >= 0]
        mkey = np.concatenate([mkey, par_s])
        msrc = np.concatenate([msrc, np.full(len(par_s), -1, np.int64)])

    removed = np.zeros(n2, dtype=bool)
    merged = None
    if len(mkey):
        mo = np.argsort(mkey, kind='stable')
        mks, msrc_s = mkey[mo], msrc[mo]
        gstart = np.r_[0, np.flatnonzero(mks[1:] != mks[:-1]) + 1]
        gcnt = np.diff(np.r_[gstart, len(mks)])
        ng = len(gstart)
        gid = np.repeat(np.arange(ng), gcnt)

        is_int = msrc_s < 0
        src = np.where(is_int, 0, msrc_s)
        m_cls = cells2['cls'][src]
        m_pri = C.SEMANTIC_PRIORITY_LUT[m_cls]
        has_int = np.maximum.reduceat(is_int.astype(np.int8), gstart) > 0
        pri_ok = np.maximum.reduceat(np.where(is_int, 0, m_pri), gstart) <= C.MERGE_PRIORITY_THRESHOLD
        cls_ok = (np.maximum.reduceat(m_cls, gstart) == np.minimum.reduceat(m_cls, gstart))
        elig = ~has_int & pri_ok & cls_ok & ~near_field(mks[gstart])

        n_merges = int(elig.sum())
        stats['n_merges'] = n_merges
        if n_merges:
            mem_elig = elig[gid]
            removed[msrc_s[mem_elig]] = True
            g = gid[mem_elig]
            s = msrc_s[mem_elig]
            eg = np.flatnonzero(elig)
            remap = np.full(ng, -1, np.int64)
            remap[eg] = np.arange(n_merges)
            g = remap[g]

            def gsum(v):
                return np.bincount(g, weights=v, minlength=n_merges)

            zmin = np.full(n_merges, np.inf)
            zmax = np.full(n_merges, -np.inf)
            np.minimum.at(zmin, g, cells2['zmin'][s])
            np.maximum.at(zmax, g, cells2['zmax'][s])
            merged = dict(
                key=mks[gstart[eg]],
                cnt=np.bincount(g, weights=cells2['cnt'][s].astype(np.float64),
                                minlength=n_merges).astype(np.int64),
                sum=gsum(cells2['sum'][s]), sumsq=gsum(cells2['sumsq'][s]),
                zmin=zmin, zmax=zmax,
                cls=cells2['cls'][s][np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1]],
                wscore=gsum(cells2['wscore'][s]), total=gsum(cells2['total'][s]))

    final = _select(cells2, np.flatnonzero(~removed))
    if merged is not None:
        final = _concat(final, merged)
    fo = np.argsort(final['key'], kind='stable')
    final = _select(final, fo)
    stats['n_final_cells'] = len(final['key'])
    return finalize(final), stats, bounds


def grid_config_from_bounds(bounds):
    """Padded grid extent snapped to the 40 cm tile, for the dashboard."""
    if bounds is None:
        return None
    pad, t = C.BOUNDARY_PADDING_M, C.TILE_M
    return {
        'grid_x_min': float(np.floor((bounds[0] - pad) / t) * t),
        'grid_y_min': float(np.floor((bounds[1] - pad) / t) * t),
        'grid_x_max': float(np.ceil((bounds[2] + pad) / t) * t),
        'grid_y_max': float(np.ceil((bounds[3] + pad) / t) * t),
        'tile_size': t,
        'level_sizes': dict(C.LEVEL_SIZE_M),
    }


def cell_index_of_points(xyz, cells):
    """Row in `cells` (FINAL_DTYPE) of the grid cell covering each point's location, -1 if none.

    Cells never overlap, so each point matches at most one level. Uses the same
    float64 cell indexing as the engine. Any point is looked up by position, so a
    point dropped as unlabeled still maps to a cell if other points created one there.
    """
    n = len(xyz)
    out = np.full(n, -1, dtype=np.int64)
    if len(cells) == 0 or n == 0:
        return out
    level = cells['level'].astype(np.int64)
    cix = np.floor(cells['x_min'] / cells['resolution'] + 0.5).astype(np.int64)
    ciy = np.floor(cells['y_min'] / cells['resolution'] + 0.5).astype(np.int64)
    keys = pack_key(level, cix, ciy)
    order = np.argsort(keys)
    sk = keys[order]
    lim = C.KEY_OFF - 1
    i3x = np.clip(np.floor(np.asarray(xyz)[:, 0].astype(np.float64) / C.FINEST_CELL_M).astype(np.int64), -lim, lim)
    i3y = np.clip(np.floor(np.asarray(xyz)[:, 1].astype(np.float64) / C.FINEST_CELL_M).astype(np.int64), -lim, lim)
    for lvl in range(C.MAX_LEVEL + 1):
        sh = C.MAX_LEVEL - lvl
        k = pack_key(lvl, i3x >> sh, i3y >> sh)
        pos = np.minimum(np.searchsorted(sk, k), len(sk) - 1)
        hit = (sk[pos] == k) & (out < 0)
        out[hit] = order[pos[hit]]
    return out


def to_structured(matrix):
    """(n, 20) matrix -> FINAL_DTYPE structured array."""
    out = np.empty(len(matrix), dtype=C.FINAL_DTYPE)
    for i, (name, dt) in enumerate(C.FINAL_FIELDS):
        out[name] = matrix[:, i].astype(dt) if dt != np.float64 else matrix[:, i]
    return out
