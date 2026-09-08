# -*- coding: utf-8 -*-
"""
Grid Engine -- Semantic Fusion
================================
Confidence-weighted semantic voting per cell (Phase 4).
"""

import numpy as np
from .config import N_CLASSES, SEMANTIC_PRIORITY


def fuse_semantics(cell_data, cell_columns, point_to_cell, labels, confidence):
    """
    Fuse per-point semantic predictions into per-cell labels using
    confidence-weighted voting.

    Extends cell_data with 3 new columns:
      semantic_class, semantic_confidence, semantic_priority

    Returns
    -------
    cell_data_ext : (n_cells, 15) float64
    cell_columns_ext : list of str
    cell_scores : (n_cells, N_CLASSES) float64
    """
    n_cells = cell_data.shape[0]

    # Accumulate confidence-weighted scores per cell per class
    cell_scores = np.zeros((n_cells, N_CLASSES), dtype=np.float64)
    np.add.at(cell_scores, (point_to_cell, labels), confidence.astype(np.float64))

    # Winning class
    cell_class = np.argmax(cell_scores, axis=1).astype(np.int32)

    # Confidence = winner / total
    total = cell_scores.sum(axis=1)
    winner = np.max(cell_scores, axis=1)
    cell_conf = np.where(total > 0, winner / total, 0.0)

    # Priority lookup
    priority_arr = np.array([SEMANTIC_PRIORITY[i] for i in range(N_CLASSES)])
    cell_priority = priority_arr[cell_class]

    # Extend cell_data
    cell_data_ext = np.column_stack([
        cell_data,
        cell_class.astype(np.float64),
        cell_conf,
        cell_priority.astype(np.float64),
    ])
    cell_columns_ext = list(cell_columns) + [
        'semantic_class', 'semantic_confidence', 'semantic_priority'
    ]

    return cell_data_ext, cell_columns_ext, cell_scores
