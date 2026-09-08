# -*- coding: utf-8 -*-
"""
Grid Engine -- Pipeline Orchestrator
======================================
Single high-level function that runs the complete grid engine
pipeline from raw input to final variable-resolution 2.5D grid.
"""

import numpy as np
import time
from .config import (
    LEVEL_RESOLUTION, N_CLASSES, SEMANTIC_PRIORITY,
    TRAVERSABILITY_WEIGHT, FINAL_DTYPE, COARSEST_RES,
    DEFAULT_DISTANCE_BANDS,
)
from .geometry import (
    load_and_validate, compute_radial_distance, assign_resolution,
    compute_grid_bounds, project_to_cells,
)
from .fusion import fuse_semantics
from .quadtree import build_quadtree, refine_quadtree


def build_grid_engine(input_path, config=None, verbose=True):
    """
    Build the complete variable-resolution 2.5D semantic elevation
    grid from a point cloud file.

    Parameters
    ----------
    input_path : str
        Path to .npy file with columns [X, Y, Z, Label, Confidence, Intensity].
    config : dict or None
        Optional overrides (distance_bands, etc.).
    verbose : bool
        Print progress messages.

    Returns
    -------
    result : dict with keys:
        'final_cells'    - structured array of final leaf cells
        'grid_config'    - grid configuration dict
        'metadata'       - input metadata
        'timings'        - per-stage timing dict
        'stats'          - summary statistics dict
    """
    config = config or {}
    timings = {}
    t_total = time.time()

    def log(msg):
        if verbose:
            print(msg)

    # ================================================================
    # STAGE 1: Input Loading & Validation
    # ================================================================
    t = time.time()
    log("Stage 1: Loading input...")
    points_xyz, labels, confidence, intensity, metadata = load_and_validate(input_path)
    N_POINTS = len(points_xyz)
    timings['input'] = time.time() - t
    log(f"  {N_POINTS:,} points loaded ({timings['input']*1000:.1f} ms)")

    # ================================================================
    # STAGE 2: Coordinate Processing & Radial Resolution
    # ================================================================
    t = time.time()
    log("Stage 2: Coordinate processing & radial resolution...")
    radial_distance = compute_radial_distance(points_xyz)

    distance_bands = config.get('distance_bands', DEFAULT_DISTANCE_BANDS)
    point_resolution = assign_resolution(radial_distance, distance_bands)

    grid_bounds = compute_grid_bounds(points_xyz)
    x_min_g, y_min_g, x_max_g, y_max_g = grid_bounds

    grid_config = {
        'grid_x_min': x_min_g,
        'grid_y_min': y_min_g,
        'grid_x_max': x_max_g,
        'grid_y_max': y_max_g,
        'coarsest_resolution': COARSEST_RES,
        'distance_bands': distance_bands,
    }
    timings['coordinates'] = time.time() - t
    log(f"  Grid: [{x_min_g},{x_max_g}] x [{y_min_g},{y_max_g}] "
        f"({timings['coordinates']*1000:.1f} ms)")

    # ================================================================
    # STAGE 3: 3D -> 2.5D Projection
    # ================================================================
    t = time.time()
    log("Stage 3: 3D -> 2.5D projection...")
    cell_data, cell_columns, point_to_cell, sorted_order, \
        group_starts, group_ends = project_to_cells(
            points_xyz, point_resolution, grid_bounds)
    n_cells = cell_data.shape[0]
    timings['projection'] = time.time() - t
    log(f"  {n_cells:,} cells created ({timings['projection']*1000:.1f} ms)")

    # ================================================================
    # STAGE 4: Semantic Fusion
    # ================================================================
    t = time.time()
    log("Stage 4: Semantic fusion...")
    cell_data, cell_columns, cell_scores = fuse_semantics(
        cell_data, cell_columns, point_to_cell, labels, confidence)
    timings['fusion'] = time.time() - t
    log(f"  Semantics fused ({timings['fusion']*1000:.1f} ms)")

    # ================================================================
    # STAGE 5: Quadtree Construction
    # ================================================================
    t = time.time()
    log("Stage 5: Building quadtree...")
    qt_nodes, qt_children, qt_count = build_quadtree(
        cell_data, cell_columns, grid_bounds)
    n_leaf_pre = (qt_nodes['is_leaf'] == 1).sum()
    n_internal_pre = (qt_nodes['is_leaf'] == 0).sum()
    timings['quadtree_build'] = time.time() - t
    log(f"  {qt_count:,} nodes ({n_leaf_pre:,} leaves, {n_internal_pre:,} internal) "
        f"({timings['quadtree_build']*1000:.1f} ms)")

    # ================================================================
    # STAGE 6: Semantic Refinement (Split / Merge)
    # ================================================================
    t = time.time()
    log("Stage 6: Semantic refinement...")
    qt_nodes, qt_children, qt_count, n_splits, n_merges = refine_quadtree(
        qt_nodes, qt_count, cell_data, cell_columns,
        points_xyz, labels, confidence)
    n_leaf_post = (qt_nodes['is_leaf'] == 1).sum()
    timings['refinement'] = time.time() - t
    log(f"  {n_splits:,} splits, {n_merges:,} merges -> {n_leaf_post:,} leaves "
        f"({timings['refinement']*1000:.1f} ms)")

    # ================================================================
    # STAGE 7: Final Grid Assembly
    # ================================================================
    t = time.time()
    log("Stage 7: Assembling final grid...")
    final_cells = _assemble_final_cells(
        qt_nodes, qt_count, cell_data, cell_columns)
    n_final = len(final_cells)
    timings['finalization'] = time.time() - t
    log(f"  {n_final:,} final leaf cells ({timings['finalization']*1000:.1f} ms)")

    timings['total'] = time.time() - t_total

    # Summary statistics
    stats = {
        'n_input_points': N_POINTS,
        'n_final_cells': n_final,
        'n_splits': n_splits,
        'n_merges': n_merges,
        'qt_nodes_total': qt_count,
        'qt_leaves': n_leaf_post,
        'point_retention_pct': 100.0 * final_cells['point_count'].sum() / N_POINTS,
        'min_resolution': float(final_cells['resolution'].min()),
        'max_resolution': float(final_cells['resolution'].max()),
        'mean_resolution': float(final_cells['resolution'].mean()),
    }

    log(f"\nGrid Engine complete in {timings['total']*1000:.0f} ms")
    log(f"  {stats['n_final_cells']:,} cells, "
        f"{stats['point_retention_pct']:.2f}% retention")

    return {
        'final_cells': final_cells,
        'grid_config': grid_config,
        'metadata': metadata,
        'timings': timings,
        'stats': stats,
        # Intermediate data for comparison / visualization
        '_qt_nodes': qt_nodes,
        '_qt_children': qt_children,
        '_qt_count': qt_count,
        '_cell_data_phase4': cell_data,
        '_cell_columns': cell_columns,
        '_points_xyz': points_xyz,
        '_labels': labels,
        '_confidence': confidence,
        '_intensity': intensity,
        '_radial_distance': radial_distance,
        '_point_resolution': point_resolution,
    }


def build_radial_only_grid(input_path, config=None):
    """
    Build a radial-only variable-resolution grid (no quadtree refinement).
    Used for comparison purposes.

    Returns
    -------
    cell_data : (n_cells, 15) array
    cell_columns : list
    grid_config : dict
    """
    config = config or {}
    points_xyz, labels, confidence, intensity, metadata = load_and_validate(input_path)
    radial_distance = compute_radial_distance(points_xyz)
    distance_bands = config.get('distance_bands', DEFAULT_DISTANCE_BANDS)
    point_resolution = assign_resolution(radial_distance, distance_bands)
    grid_bounds = compute_grid_bounds(points_xyz)

    cell_data, cell_columns, point_to_cell, _, _, _ = project_to_cells(
        points_xyz, point_resolution, grid_bounds)

    cell_data, cell_columns, _ = fuse_semantics(
        cell_data, cell_columns, point_to_cell, labels, confidence)

    grid_config = {
        'grid_x_min': grid_bounds[0],
        'grid_y_min': grid_bounds[1],
        'grid_x_max': grid_bounds[2],
        'grid_y_max': grid_bounds[3],
    }
    return cell_data, cell_columns, grid_config


def _assemble_final_cells(qt_nodes, qt_count, cell_data, cell_columns):
    """
    Extract leaf cells from refined quadtree and produce final
    structured array with all required fields.
    """
    nodes = qt_nodes[:qt_count]
    leaf_mask = (nodes['is_leaf'] == 1)
    leaf_nodes = nodes[leaf_mask]
    n_leaves = len(leaf_nodes)

    COL = {str(name): idx for idx, name in enumerate(cell_columns)}

    final = np.zeros(n_leaves, dtype=FINAL_DTYPE)

    # Spatial
    final['x_min']    = leaf_nodes['x_min']
    final['x_max']    = leaf_nodes['x_max']
    final['y_min']    = leaf_nodes['y_min']
    final['y_max']    = leaf_nodes['y_max']
    final['x_center'] = (leaf_nodes['x_min'] + leaf_nodes['x_max']) / 2.0
    final['y_center'] = (leaf_nodes['y_min'] + leaf_nodes['y_max']) / 2.0
    final['resolution']  = leaf_nodes['resolution']
    final['level']       = leaf_nodes['level']
    final['point_count'] = leaf_nodes['point_count']

    # From quadtree
    final['elevation']            = leaf_nodes['elevation']
    final['height_range']         = leaf_nodes['height_range']
    final['semantic_class']       = leaf_nodes['semantic_class']
    final['semantic_confidence']  = leaf_nodes['semantic_confidence']
    final['semantic_priority']    = leaf_nodes['semantic_priority']

    # Enrich from cell_data where available
    leaf_idx = leaf_nodes['leaf_idx']
    has_orig = (leaf_idx >= 0)
    no_orig  = (leaf_idx < 0)

    if has_orig.sum() > 0:
        oi = leaf_idx[has_orig].astype(np.int64)
        final['min_height'][has_orig]        = cell_data[oi, COL['min_height']]
        final['max_height'][has_orig]        = cell_data[oi, COL['max_height']]
        final['elevation_variance'][has_orig] = cell_data[oi, COL['height_variance']]
        final['occupancy'][has_orig]          = cell_data[oi, COL['occupancy']]

    if no_orig.sum() > 0:
        elev = final['elevation'][no_orig]
        hr   = final['height_range'][no_orig]
        pc   = final['point_count'][no_orig]
        final['min_height'][no_orig]        = elev - hr / 2.0
        final['max_height'][no_orig]        = elev + hr / 2.0
        final['elevation_variance'][no_orig] = (hr ** 2) / 12.0
        final['occupancy'][no_orig]          = np.where(pc > 0, 1.0, 0.0)

    # Terrain complexity
    hr = final['height_range']
    ev = final['elevation_variance']
    hr_max = hr.max() if hr.max() > 0 else 1.0
    ev_max = ev.max() if ev.max() > 0 else 1.0
    final['terrain_complexity'] = 0.6 * (hr / hr_max) + 0.4 * (ev / ev_max)

    # Traversability
    trav_lut = np.array([TRAVERSABILITY_WEIGHT[i] for i in range(N_CLASSES)])
    class_trav = trav_lut[final['semantic_class']]
    terrain_pen = 1.0 - 0.5 * final['terrain_complexity']
    final['traversability'] = np.clip(class_trav * terrain_pen, 0.0, 1.0)

    return final
