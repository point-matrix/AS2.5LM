# -*- coding: utf-8 -*-
"""
Grid Engine -- Quadtree Construction & Refinement
====================================================
Bottom-up quadtree build (Phase 5) and semantic adaptive
refinement with split/merge (Phase 6).
"""

import numpy as np
from collections import defaultdict
from .config import (
    LEVEL_RESOLUTION, RES_TO_LEVEL, MAX_DEPTH, N_CLASSES,
    SEMANTIC_PRIORITY, NODE_DTYPE,
    SPLIT_PRIORITY_THRESHOLD, MERGE_PRIORITY_THRESHOLD, SPLIT_MIN_POINTS,
)


def build_quadtree(cell_data, cell_columns, grid_bounds):
    """
    Build a quadtree from Phase-4 cell data (bottom-up construction).

    Returns
    -------
    nodes : structured array with NODE_DTYPE
    children_array : int64 array of child indices
    node_count : int
    """
    x_min_g, y_min_g, x_max_g, y_max_g = grid_bounds
    COL = {name: idx for idx, name in enumerate(cell_columns)}
    n_cells = cell_data.shape[0]

    # Assign quadtree level from resolution
    leaf_resolution = cell_data[:, COL['resolution']]
    leaf_level = np.zeros(n_cells, dtype=np.int32)
    for res_val, lvl in RES_TO_LEVEL.items():
        mask = np.isclose(leaf_resolution, res_val)
        leaf_level[mask] = lvl

    # Pre-allocate
    max_nodes = n_cells * 2
    nodes = np.zeros(max_nodes, dtype=NODE_DTYPE)

    # Bulk-insert leaf cells
    nodes['x_min'][:n_cells]      = cell_data[:, COL['x_min']]
    nodes['y_min'][:n_cells]      = cell_data[:, COL['y_min']]
    nodes['x_max'][:n_cells]      = cell_data[:, COL['x_max']]
    nodes['y_max'][:n_cells]      = cell_data[:, COL['y_max']]
    nodes['level'][:n_cells]      = leaf_level
    nodes['resolution'][:n_cells] = leaf_resolution
    nodes['is_leaf'][:n_cells]    = 1
    nodes['point_count'][:n_cells]         = cell_data[:, COL['point_count']]
    nodes['elevation'][:n_cells]           = cell_data[:, COL['elevation']]
    nodes['height_range'][:n_cells]        = cell_data[:, COL['height_range']]
    nodes['semantic_class'][:n_cells]      = cell_data[:, COL['semantic_class']].astype(np.int32)
    nodes['semantic_confidence'][:n_cells] = cell_data[:, COL['semantic_confidence']]
    nodes['semantic_priority'][:n_cells]   = cell_data[:, COL['semantic_priority']].astype(np.int32)
    nodes['parent_idx'][:n_cells]  = -1
    nodes['child_start'][:n_cells] = -1
    nodes['child_count'][:n_cells] = 0
    nodes['leaf_idx'][:n_cells]    = np.arange(n_cells)
    node_count = n_cells

    # Hash index
    def grid_key(level, xm, ym):
        res = LEVEL_RESOLUTION[level]
        col = int(round((xm - x_min_g) / res))
        row = int(round((ym - y_min_g) / res))
        return (level, col, row)

    node_hash = {}
    for ni in range(node_count):
        lvl = int(nodes[ni]['level'])
        key = grid_key(lvl, nodes[ni]['x_min'], nodes[ni]['y_min'])
        node_hash[key] = ni

    # Bottom-up: build internal nodes
    for parent_level in range(MAX_DEPTH - 1, -1, -1):
        child_level = parent_level + 1
        parent_res = LEVEL_RESOLUTION[parent_level]

        child_indices = [ni for ni in range(node_count)
                         if nodes[ni]['level'] == child_level]
        if not child_indices:
            continue

        parent_groups = defaultdict(list)
        for ni in child_indices:
            pcol = int(np.floor((nodes[ni]['x_min'] - x_min_g) / parent_res))
            prow = int(np.floor((nodes[ni]['y_min'] - y_min_g) / parent_res))
            parent_groups[(pcol, prow)].append(ni)

        for (pcol, prow), children in parent_groups.items():
            px_min = x_min_g + pcol * parent_res
            py_min = y_min_g + prow * parent_res

            parent_key = grid_key(parent_level, px_min, py_min)
            existing = node_hash.get(parent_key, None)

            # Aggregate children
            all_points = 0.0
            weighted_elev = 0.0
            max_hr = 0.0
            max_priority = 0
            class_scores = np.zeros(N_CLASSES, dtype=np.float64)

            for ci in children:
                cp = nodes[ci]['point_count']
                all_points += cp
                weighted_elev += nodes[ci]['elevation'] * cp
                max_hr = max(max_hr, float(nodes[ci]['height_range']))
                max_priority = max(max_priority, int(nodes[ci]['semantic_priority']))
                sc = int(nodes[ci]['semantic_class'])
                class_scores[sc] += nodes[ci]['semantic_confidence'] * cp

            if existing is not None and nodes[existing]['is_leaf'] == 1:
                ni = existing
                own_pts = nodes[ni]['point_count']
                all_points += own_pts
                weighted_elev += nodes[ni]['elevation'] * own_pts
                max_hr = max(max_hr, float(nodes[ni]['height_range']))
                max_priority = max(max_priority, int(nodes[ni]['semantic_priority']))
                sc = int(nodes[ni]['semantic_class'])
                class_scores[sc] += nodes[ni]['semantic_confidence'] * own_pts
                nodes[ni]['is_leaf'] = 0
                for ci in children:
                    nodes[ci]['parent_idx'] = ni
            else:
                ni = node_count
                nodes[ni]['x_min'] = px_min
                nodes[ni]['y_min'] = py_min
                nodes[ni]['x_max'] = px_min + parent_res
                nodes[ni]['y_max'] = py_min + parent_res
                nodes[ni]['level'] = parent_level
                nodes[ni]['resolution'] = parent_res
                nodes[ni]['is_leaf'] = 0
                nodes[ni]['parent_idx'] = -1
                nodes[ni]['leaf_idx'] = -1
                for ci in children:
                    nodes[ci]['parent_idx'] = ni
                node_hash[parent_key] = ni
                node_count += 1

            nodes[ni]['point_count'] = all_points
            nodes[ni]['elevation'] = weighted_elev / all_points if all_points > 0 else 0
            nodes[ni]['height_range'] = max_hr
            nodes[ni]['semantic_priority'] = max_priority
            winner = int(np.argmax(class_scores))
            nodes[ni]['semantic_class'] = winner
            total_score = class_scores.sum()
            nodes[ni]['semantic_confidence'] = (
                class_scores[winner] / total_score if total_score > 0 else 0
            )
            nodes[ni]['child_count'] = len(children)

    nodes = nodes[:node_count].copy()

    # Build children index array
    children_lists = defaultdict(list)
    for ni in range(node_count):
        pi = nodes[ni]['parent_idx']
        if pi >= 0:
            children_lists[int(pi)].append(ni)

    total_ch = sum(len(v) for v in children_lists.values())
    children_array = np.full(total_ch, -1, dtype=np.int64)
    offset = 0
    for ni in range(node_count):
        if ni in children_lists:
            ch = sorted(children_lists[ni])
            children_array[offset:offset + len(ch)] = ch
            nodes[ni]['child_start'] = offset
            nodes[ni]['child_count'] = len(ch)
            offset += len(ch)
        elif nodes[ni]['is_leaf'] == 1:
            nodes[ni]['child_start'] = -1
            nodes[ni]['child_count'] = 0
    children_array = children_array[:offset]

    return nodes, children_array, node_count


def refine_quadtree(nodes, node_count, cell_data, cell_columns,
                    points_xyz, labels, confidence):
    """
    Semantic adaptive refinement: split high-priority cells, merge
    uniform low-priority cells (Phase 6).

    Returns
    -------
    nodes : refined node array
    children_array : updated children index
    node_count : int
    n_splits : int
    n_merges : int
    """
    COL = {str(name): idx for idx, name in enumerate(cell_columns)}
    N_PTS = points_xyz.shape[0]

    # ── Build point -> cell mapping ────────────────────────────
    # (reuse point_to_cell from caller if possible; here we rebuild
    #  from the original cell leaf_idx)
    cell_to_points = defaultdict(list)
    # We need the original point_to_cell, but that info is in the leaf_idx.
    # We'll rebuild from the original cell_data: leaf cells with leaf_idx >= 0
    # correspond to original Phase-4 cells.

    # For the split step, we need to know which points are in each cell.
    # Build from cell_data boundaries for efficiency.
    px = points_xyz[:, 0]
    py = points_xyz[:, 1]

    # ── Identify split candidates ──────────────────────────────
    leaf_mask = (nodes[:node_count]['is_leaf'] == 1)
    split_mask = (
        leaf_mask &
        (nodes[:node_count]['semantic_priority'] >= SPLIT_PRIORITY_THRESHOLD) &
        (nodes[:node_count]['level'] < MAX_DEPTH) &
        (nodes[:node_count]['point_count'] >= SPLIT_MIN_POINTS)
    )
    split_indices = np.where(split_mask)[0]

    # Pre-allocate for new nodes
    max_new = len(split_indices) * 4
    new_nodes = np.zeros(max_new, dtype=nodes.dtype)
    new_count = 0
    n_splits = 0

    # We need to find points for each split candidate.
    # Build a spatial index of points for fast lookup.
    # Pre-compute point-cell assignments for split candidates only.

    for si in split_indices:
        node = nodes[si]
        parent_level = int(node['level'])
        child_level = parent_level + 1
        child_res = LEVEL_RESOLUTION[child_level]

        # Find points in this cell using vectorized spatial query
        in_cell = (
            (px >= node['x_min']) & (px < node['x_max']) &
            (py >= node['y_min']) & (py < node['y_max'])
        )
        in_cell |= (
            (px == node['x_max']) &
            (py >= node['y_min']) & (py <= node['y_max'])
        )
        in_cell |= (
            (py == node['y_max']) &
            (px >= node['x_min']) & (px <= node['x_max'])
        )
        pt_indices = np.where(in_cell)[0]

        if len(pt_indices) < SPLIT_MIN_POINTS:
            continue

        x_mid = (node['x_min'] + node['x_max']) / 2
        y_mid = (node['y_min'] + node['y_max']) / 2

        quadrants = [
            (node['x_min'], node['y_min'], x_mid, y_mid),
            (x_mid, node['y_min'], node['x_max'], y_mid),
            (node['x_min'], y_mid, x_mid, node['y_max']),
            (x_mid, y_mid, node['x_max'], node['y_max']),
        ]

        pt_pos = points_xyz[pt_indices]
        pt_lab = labels[pt_indices]
        pt_conf = confidence[pt_indices]

        children_created = 0
        for qx0, qy0, qx1, qy1 in quadrants:
            in_q = (
                (pt_pos[:, 0] >= qx0) & (pt_pos[:, 0] < qx1) &
                (pt_pos[:, 1] >= qy0) & (pt_pos[:, 1] < qy1)
            )
            if qx1 == node['x_max']:
                in_q |= (
                    (pt_pos[:, 0] == qx1) &
                    (pt_pos[:, 1] >= qy0) & (pt_pos[:, 1] <= qy1)
                )
            if qy1 == node['y_max']:
                in_q |= (
                    (pt_pos[:, 1] == qy1) &
                    (pt_pos[:, 0] >= qx0) & (pt_pos[:, 0] <= qx1)
                )

            q_idx = np.where(in_q)[0]
            if len(q_idx) == 0:
                continue

            q_xyz = pt_pos[q_idx]
            q_lab = pt_lab[q_idx]
            q_conf = pt_conf[q_idx]
            z_vals = q_xyz[:, 2]

            cs = np.zeros(N_CLASSES, dtype=np.float64)
            np.add.at(cs, q_lab, q_conf.astype(np.float64))
            w = int(np.argmax(cs))
            ts = cs.sum()

            ci = node_count + new_count
            new_nodes[new_count]['x_min'] = qx0
            new_nodes[new_count]['y_min'] = qy0
            new_nodes[new_count]['x_max'] = qx1
            new_nodes[new_count]['y_max'] = qy1
            new_nodes[new_count]['level'] = child_level
            new_nodes[new_count]['resolution'] = child_res
            new_nodes[new_count]['is_leaf'] = 1
            new_nodes[new_count]['point_count'] = len(q_idx)
            new_nodes[new_count]['elevation'] = z_vals.mean()
            new_nodes[new_count]['height_range'] = z_vals.max() - z_vals.min()
            new_nodes[new_count]['semantic_class'] = w
            new_nodes[new_count]['semantic_confidence'] = cs[w] / ts if ts > 0 else 0
            new_nodes[new_count]['semantic_priority'] = int(SEMANTIC_PRIORITY[w])
            new_nodes[new_count]['parent_idx'] = si
            new_nodes[new_count]['child_start'] = -1
            new_nodes[new_count]['child_count'] = 0
            new_nodes[new_count]['leaf_idx'] = -1
            new_count += 1
            children_created += 1

        if children_created > 0:
            nodes[si]['is_leaf'] = 0
            nodes[si]['child_count'] = children_created
            n_splits += 1

    # Append new nodes
    if new_count > 0:
        extended = np.zeros(node_count + new_count, dtype=nodes.dtype)
        extended[:node_count] = nodes[:node_count]
        extended[node_count:node_count + new_count] = new_nodes[:new_count]
        nodes = extended
        node_count += new_count

    # ── Merge candidates ───────────────────────────────────────
    ch_lists = defaultdict(list)
    for ni in range(node_count):
        pi = int(nodes[ni]['parent_idx'])
        if pi >= 0:
            ch_lists[pi].append(ni)

    n_merges = 0
    n_ch_removed = 0
    for ni in range(node_count):
        if nodes[ni]['is_leaf'] == 1 or nodes[ni]['child_count'] == 0:
            continue
        ch = ch_lists.get(ni, [])
        if not ch:
            continue
        if not all(nodes[ci]['is_leaf'] == 1 for ci in ch):
            continue
        if not all(nodes[ci]['semantic_priority'] <= MERGE_PRIORITY_THRESHOLD for ci in ch):
            continue
        classes = set(int(nodes[ci]['semantic_class']) for ci in ch)
        if len(classes) > 1:
            continue

        total_pts = sum(float(nodes[ci]['point_count']) for ci in ch)
        w_elev = sum(float(nodes[ci]['elevation']) * float(nodes[ci]['point_count']) for ci in ch)

        nodes[ni]['is_leaf'] = 1
        nodes[ni]['point_count'] = total_pts
        if total_pts > 0:
            nodes[ni]['elevation'] = w_elev / total_pts
        nodes[ni]['height_range'] = max(float(nodes[ci]['height_range']) for ci in ch)
        nodes[ni]['semantic_confidence'] = max(float(nodes[ci]['semantic_confidence']) for ci in ch)
        nodes[ni]['semantic_priority'] = max(int(nodes[ci]['semantic_priority']) for ci in ch)

        for ci in ch:
            nodes[ci]['is_leaf'] = -1
            nodes[ci]['parent_idx'] = -1
            n_ch_removed += 1
        nodes[ni]['child_count'] = 0
        nodes[ni]['child_start'] = -1
        n_merges += 1

    # ── Compact ────────────────────────────────────────────────
    active_mask = (nodes[:node_count]['is_leaf'] != -1)
    active_idx = np.where(active_mask)[0]
    n_active = len(active_idx)

    old_to_new = np.full(node_count, -1, dtype=np.int64)
    old_to_new[active_idx] = np.arange(n_active)

    compact = nodes[active_idx].copy()
    for i in range(n_active):
        pi = int(compact[i]['parent_idx'])
        if pi >= 0:
            compact[i]['parent_idx'] = old_to_new[pi]

    nodes = compact
    node_count = n_active

    # Rebuild children array
    ch_lists_new = defaultdict(list)
    for ni in range(node_count):
        pi = int(nodes[ni]['parent_idx'])
        if pi >= 0:
            ch_lists_new[pi].append(ni)

    total_ch = sum(len(v) for v in ch_lists_new.values())
    children_array = np.full(total_ch, -1, dtype=np.int64)
    offset = 0
    for ni in range(node_count):
        if ni in ch_lists_new:
            ch = sorted(ch_lists_new[ni])
            children_array[offset:offset + len(ch)] = ch
            nodes[ni]['child_start'] = offset
            nodes[ni]['child_count'] = len(ch)
            offset += len(ch)
        elif nodes[ni]['is_leaf'] == 1:
            nodes[ni]['child_start'] = -1
            nodes[ni]['child_count'] = 0
    children_array = children_array[:offset]

    return nodes, children_array, node_count, n_splits, n_merges
