"""Metric definitions on small hand-made inputs."""

import numpy as np

from as25lm import config as C
from as25lm.grid.reference import run_reference, to_structured
from as25lm.metrics import GridEval, SegmentationEval, confusion, frame_accuracy, iou_per_class, miou_present


def test_iou_and_miou_over_present_classes():
    gt = np.array([1, 1, 1, 9, 9, 9, 9, 0])
    pred = np.array([1, 1, 9, 9, 9, 9, 1, 5])
    cm = confusion(gt[gt > 0], pred[gt > 0])
    iou = iou_per_class(cm)
    assert np.isclose(iou[1], 2 / 4) and np.isclose(iou[9], 3 / 5)
    miou, k = miou_present(cm)
    assert k == 2 and np.isclose(miou, (2 / 4 + 3 / 5) / 2)       # absent classes are left out


def test_frame_accuracy_absent_class_is_none():
    r = frame_accuracy(np.array([9, 9, 11]), np.array([9, 11, 11]))
    assert np.isclose(r["accuracy"], 2 / 3) and r["car_iou"] is None and r["person_iou"] is None


def test_segmentation_eval_bands(scene):
    raw, labels, _ = scene
    ev = SegmentationEval(["a"], "a")
    ev.add("0", labels, {"a": labels}, raw)
    s = ev.summary()
    assert s["variants"]["a"]["overall_accuracy"] == 1.0
    assert sum(b["points"] for b in s["by_distance"].values()) == (labels > 0).sum()


def test_grid_eval_perfect_labels(scene):
    raw, labels, conf = scene
    cells = to_structured(run_reference(raw[:, :3], labels, conf)[0])
    ge = GridEval()
    ge.add("0", labels, raw, cells)
    s = ge.summary()
    assert s["safety"]["obs"] == np.isin(labels, [1, 6, 13, 18]).sum()
    assert s["safety"]["missed_vehicles_people"] < 0.05
    assert 0.9 < s["grid_accuracy"]["by_cell"] <= 1.0
    assert C.TRAV_BLOCKED == s["safety"]["threshold"]
