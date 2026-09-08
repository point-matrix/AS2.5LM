# -*- coding: utf-8 -*-
"""
Grid Engine -- Geometry & Coordinate Processing
=================================================
Input loading, validation, coordinate computation, and radial grid
assignment (Phases 1-3).
"""

import numpy as np
from .config import (
    DEFAULT_DISTANCE_BANDS, COARSEST_RES, BOUNDARY_PADDING_M,
    LABEL_NAMES, N_CLASSES,
)


def load_and_validate(input_path):
    """
    Load a point cloud file and validate its structure.

    Expected columns: [X, Y, Z, Label, Confidence, Intensity]

    Returns
    -------
    points_xyz : (N, 3) float64
    labels : (N,) int32
    confidence : (N,) float32
    intensity : (N,) float32
    metadata : dict
    """
    raw = np.load(input_path).astype(np.float64)
    assert raw.ndim == 2 and raw.shape[1] == 6, (
        f"Expected (N, 6), got {raw.shape}")

    points_xyz  = raw[:, :3].astype(np.float64)
    labels      = raw[:, 3].astype(np.int32)
    confidence  = raw[:, 4].astype(np.float32)
    intensity   = raw[:, 5].astype(np.float32)

    n_points = len(points_xyz)

    # Validate label range
    valid_labels = sorted(LABEL_NAMES.keys())
    assert np.all((labels >= 0) & (labels < N_CLASSES)), "Invalid labels"

    # Validate confidence range
    assert np.all((confidence >= 0) & (confidence <= 1.0 + 1e-6)), \
        "Confidence out of [0,1]"

    metadata = {
        "n_points": n_points,
        "sensor_origin": [0.0, 0.0, 0.0],
        "label_names": LABEL_NAMES,
        "valid_labels": valid_labels,
        "map_bounds": {
            "x_min": float(points_xyz[:, 0].min()),
            "x_max": float(points_xyz[:, 0].max()),
            "y_min": float(points_xyz[:, 1].min()),
            "y_max": float(points_xyz[:, 1].max()),
            "z_min": float(points_xyz[:, 2].min()),
            "z_max": float(points_xyz[:, 2].max()),
        },
        "source_file": input_path,
    }
    return points_xyz, labels, confidence, intensity, metadata


def compute_radial_distance(points_xyz, origin=(0.0, 0.0)):
    """Compute 2D radial distance from sensor origin."""
    dx = points_xyz[:, 0] - origin[0]
    dy = points_xyz[:, 1] - origin[1]
    return np.sqrt(dx**2 + dy**2)


def assign_resolution(radial_distance, distance_bands=None):
    """
    Assign each point a resolution based on its radial distance band.

    Returns
    -------
    point_resolution : (N,) float64
    """
    if distance_bands is None:
        distance_bands = DEFAULT_DISTANCE_BANDS

    n = len(radial_distance)
    point_resolution = np.full(n, COARSEST_RES, dtype=np.float64)

    for d_min, d_max, res, _level in distance_bands:
        mask = (radial_distance >= d_min) & (radial_distance < d_max)
        point_resolution[mask] = res

    return point_resolution


def compute_grid_bounds(points_xyz, padding=BOUNDARY_PADDING_M):
    """Compute aligned grid boundaries with padding."""
    x_min = np.floor((points_xyz[:, 0].min() - padding) / COARSEST_RES) * COARSEST_RES
    y_min = np.floor((points_xyz[:, 1].min() - padding) / COARSEST_RES) * COARSEST_RES
    x_max = np.ceil((points_xyz[:, 0].max() + padding) / COARSEST_RES) * COARSEST_RES
    y_max = np.ceil((points_xyz[:, 1].max() + padding) / COARSEST_RES) * COARSEST_RES
    return float(x_min), float(y_min), float(x_max), float(y_max)


def project_to_cells(points_xyz, point_resolution, grid_bounds):
    """
    Project 3D points into 2.5D grid cells and compute per-cell elevation
    statistics.

    This implements the 3D -> 2.5D projection (Phase 3).

    Returns
    -------
    cell_data : (n_cells, 12) float64 array
    cell_columns : list of str
    point_to_cell : (N,) int64  -- maps each point to its cell index
    sorted_order, group_starts, group_ends : sorting arrays
    """
    x_min_g, y_min_g, x_max_g, y_max_g = grid_bounds
    N = len(points_xyz)

    # Compute cell column and row for each point
    px, py, pz = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    cell_col = np.floor((px - x_min_g) / point_resolution).astype(np.int64)
    cell_row = np.floor((py - y_min_g) / point_resolution).astype(np.int64)

    # Quantise resolution for grouping (avoid float key issues)
    res_quant = np.round(point_resolution * 10000).astype(np.int64)

    # Unique cell key: encode (res_quant, col, row) into one int64
    max_col = int(np.ceil((x_max_g - x_min_g) / 0.0625)) + 1
    max_row = int(np.ceil((y_max_g - y_min_g) / 0.0625)) + 1
    cell_key = res_quant * (max_col * max_row) + cell_col * max_row + cell_row

    # Sort by cell key for grouped aggregation
    sorted_order = np.argsort(cell_key, kind='mergesort')
    sorted_keys = cell_key[sorted_order]

    # Find group boundaries
    diff = np.diff(sorted_keys)
    breaks = np.where(diff != 0)[0] + 1
    group_starts = np.concatenate([[0], breaks])
    group_ends = np.concatenate([breaks, [N]])
    n_cells = len(group_starts)

    # Map points to cell index
    point_to_cell = np.empty(N, dtype=np.int64)
    for gi in range(n_cells):
        s, e = group_starts[gi], group_ends[gi]
        point_to_cell[sorted_order[s:e]] = gi

    # Compute per-cell statistics
    cell_columns = [
        'x_min', 'y_min', 'x_max', 'y_max', 'resolution',
        'point_count', 'elevation', 'min_height', 'max_height',
        'height_variance', 'height_range', 'occupancy',
    ]
    cell_data = np.zeros((n_cells, len(cell_columns)), dtype=np.float64)

    for gi in range(n_cells):
        s, e = group_starts[gi], group_ends[gi]
        pts = sorted_order[s:e]
        rep = pts[0]  # representative point
        res = point_resolution[rep]

        cx = int(np.floor((px[rep] - x_min_g) / res))
        cy = int(np.floor((py[rep] - y_min_g) / res))

        cell_data[gi, 0] = x_min_g + cx * res         # x_min
        cell_data[gi, 1] = y_min_g + cy * res         # y_min
        cell_data[gi, 2] = cell_data[gi, 0] + res     # x_max
        cell_data[gi, 3] = cell_data[gi, 1] + res     # y_max
        cell_data[gi, 4] = res                         # resolution
        cell_data[gi, 5] = e - s                       # point_count

        z_vals = pz[pts]
        cell_data[gi, 6]  = z_vals.mean()              # elevation
        cell_data[gi, 7]  = z_vals.min()                # min_height
        cell_data[gi, 8]  = z_vals.max()                # max_height
        cell_data[gi, 9]  = z_vals.var()                # height_variance
        cell_data[gi, 10] = z_vals.max() - z_vals.min() # height_range
        cell_data[gi, 11] = 1.0                         # occupancy

    return cell_data, cell_columns, point_to_cell, sorted_order, group_starts, group_ends
