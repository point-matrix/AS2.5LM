# -*- coding: utf-8 -*-
"""
Grid Engine — Configuration
============================
Default parameters for the variable-resolution 2.5D grid engine.
"""

import numpy as np

# ── Distance-based resolution bands ────────────────────────────
# (min_dist, max_dist, resolution_m, quadtree_level)
DEFAULT_DISTANCE_BANDS = [
    (0,   10,  0.05, 3),   # 0–10 m  → 5 cm
    (10,  30,  0.10, 2),   # 10–30 m → 10 cm
    (30,  60,  0.25, 1),   # 30–60 m → 25 cm
    (60, 100,  0.50, 0),   # 60–100 m → 50 cm
]

# ── Quadtree levels ────────────────────────────────────────────
LEVEL_RESOLUTION = {
    3: 0.05,    # 5.0 cm
    2: 0.10,    # 10.0 cm
    1: 0.25,    # 25.0 cm
    0: 0.50,    # 50.0 cm
}
RES_TO_LEVEL = {v: k for k, v in LEVEL_RESOLUTION.items()}
MAX_DEPTH = max(LEVEL_RESOLUTION.keys())  # 3
COARSEST_RES = 0.50

# ── Semantic labels ────────────────────────────────────────────
# This is the ONLY semantic interface.
LABEL_NAMES = {
    0: "Car",
    1: "Bicycle",
    2: "Motorcycle",
    3: "Truck",
    4: "Other-vehicle",
    5: "Person",
    6: "Bicyclist",
    7: "Motorcyclist",
    8: "Road",
    9: "Parking",
    10: "Sidewalk",
    11: "Other-ground",
    12: "Building",
    13: "Fence",
    14: "Vegetation",
    15: "Trunk",
    16: "Terrain",
    17: "Pole",
    18: "Traffic-sign",
}
N_CLASSES = len(LABEL_NAMES)

CLASS_COLORS = {
    0: "#ff0000",   # Car - red
    1: "#ff8000",   # Bicycle - orange
    2: "#ff8000",   # Motorcycle - orange
    3: "#800000",   # Truck - dark red
    4: "#800000",   # Other-vehicle - dark red
    5: "#0000ff",   # Person - blue
    6: "#0000ff",   # Bicyclist - blue
    7: "#0000ff",   # Motorcyclist - blue
    8: "#808080",   # Road - gray
    9: "#606060",   # Parking - dark gray
    10: "#a0a0a0",  # Sidewalk - light gray
    11: "#c0c0c0",  # Other-ground - lighter gray
    12: "#ffd700",  # Building - yellow
    13: "#8b4513",  # Fence - brown
    14: "#008000",  # Vegetation - green
    15: "#8b4513",  # Trunk - brown
    16: "#90ee90",  # Terrain - light green
    17: "#d3d3d3",  # Pole - very light gray
    18: "#ff00ff",  # Traffic-sign - magenta
}

SEMANTIC_PRIORITY = {
    0: 5,   # Car
    1: 5,   # Bicycle
    2: 5,   # Motorcycle
    3: 5,   # Truck
    4: 5,   # Other-vehicle
    5: 5,   # Person
    6: 5,   # Bicyclist
    7: 5,   # Motorcyclist
    8: 1,   # Road
    9: 1,   # Parking
    10: 1,  # Sidewalk
    11: 1,  # Other-ground
    12: 4,  # Building
    13: 4,  # Fence
    14: 2,  # Vegetation
    15: 4,  # Trunk
    16: 1,  # Terrain
    17: 4,  # Pole
    18: 4,  # Traffic-sign
}

TRAVERSABILITY_WEIGHT = {
    0: 0.10,   # Car
    1: 0.10,   # Bicycle
    2: 0.10,   # Motorcycle
    3: 0.10,   # Truck
    4: 0.10,   # Other-vehicle
    5: 0.05,   # Person
    6: 0.05,   # Bicyclist
    7: 0.05,   # Motorcyclist
    8: 0.95,   # Road
    9: 0.90,   # Parking
    10: 0.85,  # Sidewalk
    11: 0.70,  # Other-ground
    12: 0.05,  # Building
    13: 0.05,  # Fence
    14: 0.40,  # Vegetation
    15: 0.05,  # Trunk
    16: 0.50,  # Terrain
    17: 0.05,  # Pole
    18: 0.05,  # Traffic-sign
}

# ── Refinement parameters ──────────────────────────────────────
SPLIT_PRIORITY_THRESHOLD = 4
MERGE_PRIORITY_THRESHOLD = 2
SPLIT_MIN_POINTS = 2

# ── Grid padding ───────────────────────────────────────────────
BOUNDARY_PADDING_M = 1.0

# ── Quadtree node dtype ────────────────────────────────────────
NODE_FIELDS = [
    ('x_min', np.float64),
    ('y_min', np.float64),
    ('x_max', np.float64),
    ('y_max', np.float64),
    ('level', np.int32),
    ('resolution', np.float64),
    ('is_leaf', np.int32),
    ('point_count', np.float64),
    ('elevation', np.float64),
    ('height_range', np.float64),
    ('semantic_class', np.int32),
    ('semantic_confidence', np.float64),
    ('semantic_priority', np.int32),
    ('parent_idx', np.int64),
    ('child_start', np.int64),
    ('child_count', np.int32),
    ('leaf_idx', np.int64),
]
NODE_DTYPE = np.dtype(NODE_FIELDS)

# ── Final cell dtype ───────────────────────────────────────────
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
