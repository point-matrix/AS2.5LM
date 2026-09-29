# -*- coding: utf-8 -*-
"""
Configuration: the single source of truth for every stage.

Classes follow SalsaNext / SemanticKITTI (0 = unlabeled, 1 = car ... 19 = traffic-sign).
Grid cells are 5 / 10 / 20 / 40 cm by distance; each level nests exactly in the next.
"""

import numpy as np

# ── Sensor and segmentation (SalsaNext pretrained arch_cfg.yaml) ─────────────
SENSOR = dict(height=64, width=2048, fov_up=3.0, fov_down=-25.0,
              img_means=[12.12, 10.88, 0.23, -1.04, 0.21],   # range, x, y, z, remission
              img_stds=[12.32, 11.47, 6.91, 0.86, 0.16])
KNN_PARAMS = dict(knn=5, search=5, sigma=1.0, cutoff=1.0)
SENSOR_HEIGHT_M = 1.73          # KITTI LiDAR height above the road (display offset)

# SemanticKITTI raw label id -> SalsaNext class id (data_cfg.yaml "learning_map")
LEARNING_MAP = {0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5, 30: 6, 31: 7,
                32: 8, 40: 9, 44: 10, 48: 11, 49: 12, 50: 13, 51: 14, 52: 0, 60: 9, 70: 15,
                71: 16, 72: 17, 80: 18, 81: 19, 99: 0, 252: 1, 253: 7, 254: 6, 255: 8,
                256: 5, 257: 5, 258: 4, 259: 5}
LEARNING_MAP_LUT = np.zeros(max(LEARNING_MAP) + 1, dtype=np.int64)
for _raw, _cls in LEARNING_MAP.items():
    LEARNING_MAP_LUT[_raw] = _cls

# ── Grid cell sizes / quadtree levels ────────────────────────────────────────
# Level 3 is the finest. size(L) = FINEST_CELL_M * 2 ** (MAX_LEVEL - L)
MAX_LEVEL = 3
FINEST_CELL_M = 0.05
TILE_M = FINEST_CELL_M * 2 ** MAX_LEVEL          # 0.40 m, the level-0 cell
LEVEL_SIZE_M = {L: FINEST_CELL_M * 2 ** (MAX_LEVEL - L) for L in range(MAX_LEVEL + 1)}

# ── Distance bands (tile-centre distance in the XY plane) ────────────────────
# [0, 10) m -> 5 cm, [10, 30) -> 10 cm, [30, 60) -> 20 cm, >= 60 m -> 40 cm
BAND_EDGE_L3_M = 10.0
BAND_EDGE_L2_M = 30.0
BAND_EDGE_L1_M = 60.0

# ── Semantic classes ─────────────────────────────────────────────────────────
LABEL_NAMES = {
    0: "unlabeled", 1: "car", 2: "bicycle", 3: "motorcycle", 4: "truck", 5: "other-vehicle",
    6: "person", 7: "bicyclist", 8: "motorcyclist", 9: "road", 10: "parking", 11: "sidewalk",
    12: "other-ground", 13: "building", 14: "fence", 15: "vegetation", 16: "trunk",
    17: "terrain", 18: "pole", 19: "traffic-sign",
}
N_CLASSES = len(LABEL_NAMES)        # 20
UNLABELED = 0
CAR, PERSON = 1, 6

CLASS_COLORS = {
    0: "#000000",
    1: "#ff0000", 2: "#ff8000", 3: "#ff8000", 4: "#800000", 5: "#800000",
    6: "#0000ff", 7: "#0000ff", 8: "#0000ff",
    9: "#808080", 10: "#606060", 11: "#a0a0a0", 12: "#c0c0c0",
    13: "#ffd700", 14: "#8b4513", 15: "#008000", 16: "#8b4513",
    17: "#90ee90", 18: "#d3d3d3", 19: "#ff00ff",
}

SEMANTIC_PRIORITY = {
    0: 5,
    1: 5, 2: 5, 3: 5, 4: 5, 5: 5,      # vehicles
    6: 5, 7: 5, 8: 5,                  # people
    9: 1, 10: 1, 11: 1, 12: 1,         # ground
    13: 4, 14: 4,                      # building, fence
    15: 2,                             # vegetation
    16: 4,                             # trunk
    17: 1,                             # terrain
    18: 4, 19: 4,                      # pole, traffic-sign
}

TRAVERSABILITY_WEIGHT = {
    0: 0.00,
    1: 0.10, 2: 0.10, 3: 0.10, 4: 0.10, 5: 0.10,
    6: 0.05, 7: 0.05, 8: 0.05,
    9: 0.95, 10: 0.90, 11: 0.85, 12: 0.70,
    13: 0.05, 14: 0.05,
    15: 0.40,
    16: 0.05,
    17: 0.50,
    18: 0.05, 19: 0.05,
}

# ── Refinement ───────────────────────────────────────────────────────────────
SPLIT_PRIORITY_THRESHOLD = 4
MERGE_PRIORITY_THRESHOLD = 2
SPLIT_MIN_POINTS = 2
# No merging for parent cells whose 40 cm tile centre is closer than this, so
# the near field keeps its 5 cm cells. 0 disables the rule.
NO_MERGE_RADIUS_M = BAND_EDGE_L3_M

# ── Terrain complexity (fixed physical limits) ───────────────────────────────
# complexity = W_RANGE * min(height_range / RANGE_LIMIT, 1) + W_STD * min(height_std / STD_LIMIT, 1)
TERRAIN_RANGE_LIMIT_M = 0.30
TERRAIN_STD_LIMIT_M = 0.10
TERRAIN_W_RANGE = 0.6
TERRAIN_W_STD = 0.4
TRAV_TERRAIN_PENALTY = 0.5          # traversability *= 1 - 0.5 * complexity

# ── Grid bounds (informational, for the dashboard) ──────────────────────────
BOUNDARY_PADDING_M = 1.0

# ── Evaluation ───────────────────────────────────────────────────────────────
TRAV_BLOCKED = 0.5                  # cells below this traversability count as blocked
DISTANCE_BAND_EDGES_M = (10.0, 30.0, 60.0)
DISTANCE_BAND_NAMES = ("0-10 m", "10-30 m", "30-60 m", "60+ m")
DRIVABLE_CLASSES = (9, 10)          # road, parking (false-alarm metric)
MIN_CAR_POINTS, MIN_PERSON_POINTS = 50, 20

# ── Final cell dtype ─────────────────────────────────────────────────────────
FINAL_FIELDS = [
    ('x_min',                np.float64),
    ('x_max',                np.float64),
    ('y_min',                np.float64),
    ('y_max',                np.float64),
    ('x_center',             np.float64),
    ('y_center',             np.float64),
    ('resolution',           np.float64),
    ('level',                np.int32),
    ('elevation',            np.float64),
    ('min_height',           np.float64),
    ('max_height',           np.float64),
    ('elevation_variance',   np.float64),
    ('semantic_class',       np.int32),
    ('semantic_confidence',  np.float64),
    ('occupancy',            np.float64),
    ('terrain_complexity',   np.float64),
    ('traversability',       np.float64),
    ('point_count',          np.float64),
    ('semantic_priority',    np.int32),
    ('height_range',         np.float64),
]
FINAL_DTYPE = np.dtype(FINAL_FIELDS)
N_FINAL_FIELDS = len(FINAL_FIELDS)   # 20; column order of the engine's output matrix

# ── Lookup tables ────────────────────────────────────────────────────────────
SEMANTIC_PRIORITY_LUT = np.array([SEMANTIC_PRIORITY[i] for i in range(N_CLASSES)], dtype=np.int32)
TRAVERSABILITY_LUT = np.array([TRAVERSABILITY_WEIGHT[i] for i in range(N_CLASSES)], dtype=np.float64)

# ── Cell key packing (shared by the NumPy reference and CUDA) ───────────────
# key = level << 44 | (ix + OFF) << 22 | (iy + OFF), ix/iy in level-L cell units
KEY_BITS = 22
KEY_OFF = 1 << (KEY_BITS - 1)           # +-2,097,151 cells (+-104 km at 5 cm)
KEY_MASK = (1 << KEY_BITS) - 1
LEVEL_SHIFT = 2 * KEY_BITS


def cuda_params():
    """Scalar parameters passed to the CUDA engine, in the order it expects."""
    return [
        float(FINEST_CELL_M), float(TILE_M),
        float(BAND_EDGE_L3_M), float(BAND_EDGE_L2_M), float(BAND_EDGE_L1_M),
        float(SPLIT_PRIORITY_THRESHOLD), float(SPLIT_MIN_POINTS),
        float(MERGE_PRIORITY_THRESHOLD),
        float(TERRAIN_RANGE_LIMIT_M), float(TERRAIN_STD_LIMIT_M),
        float(TERRAIN_W_RANGE), float(TERRAIN_W_STD), float(TRAV_TERRAIN_PENALTY),
        float(NO_MERGE_RADIUS_M),
    ]
