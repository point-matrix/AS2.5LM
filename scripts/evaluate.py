#!/usr/bin/env python
"""
Accuracy against SemanticKITTI ground truth.

Per-point: IoU per class, mIoU over classes present, overall accuracy, by distance band,
for pixel lookup (FP32), KNN (FP32) and KNN (FP16). Grid: cell class vs majority ground
truth, and obstacle safety (missed obstacles, false alarms) at traversability < 0.5.
Writes <out>/accuracy_frames.csv, <out>/accuracy.json and <out>/confusion.npy.

Example:
  python scripts/evaluate.py --scans .../08/velodyne --labels .../08/labels \
      --weights weights/salsanext_weights.pth --out results/run
"""

import os

import numpy as np
import pandas as pd
import torch

from as25lm import cli
from as25lm.data import frame_id, label_path, load_labels, load_scan
from as25lm.metrics import GridEval, SegmentationEval
from as25lm.report import write_json


def main():
    p = cli.parser(__doc__)
    cli.add_data_args(p, labels_required=True)
    cli.add_model_args(p)
    p.add_argument("--out", required=True, help="output folder")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    scans = [s for s in cli.scan_list(args) if label_path(args.labels, s)]
    if not scans:
        raise SystemExit("No scans with matching .label files.")
    pipe = cli.make_pipeline(args)
    seg = pipe.seg
    knn = seg.knn is not None
    variants = ["pixel lookup, FP32", "KNN, FP32", "KNN, FP16"] if knn else ["pixel lookup, FP32", "pixel lookup, FP16"]
    active = ("KNN" if knn else "pixel lookup") + (", FP32" if args.fp32 else ", FP16")
    seg_eval, grid_eval = SegmentationEval(variants, active), GridEval()

    from tqdm import tqdm
    for path in tqdm(scans, desc="Evaluate"):
        raw = load_scan(path)
        gt = load_labels(label_path(args.labels, path))
        pts = torch.from_numpy(raw).to(pipe.device)
        lab32, conf32, pix32 = seg.segment(pts, fp16=False)
        lab16, conf16, pix16 = seg.segment(pts, fp16=True)
        preds = ({"pixel lookup, FP32": pix32, "KNN, FP32": lab32, "KNN, FP16": lab16} if knn
                 else {"pixel lookup, FP32": pix32, "pixel lookup, FP16": pix16})
        preds = {k: v.cpu().numpy() for k, v in preds.items()}
        seg_eval.add(frame_id(path), gt, preds, raw)
        a_lab, a_conf = (lab32, conf32) if args.fp32 else (lab16, conf16)
        cells, _ = pipe.grid_cells(raw, a_lab, a_conf)
        grid_eval.add(frame_id(path), gt, raw, cells)

    frames = pd.DataFrame(seg_eval.frames).set_index("frame").join(pd.DataFrame(grid_eval.frames).set_index("frame"))
    frames.to_csv(os.path.join(args.out, "accuracy_frames.csv"))
    out = dict(accuracy=seg_eval.summary(), **grid_eval.summary())
    write_json(os.path.join(args.out, "accuracy.json"), out)
    np.save(os.path.join(args.out, "confusion.npy"), seg_eval.cms[active])

    a = out["accuracy"]
    print(f"{a['frames_evaluated']} frames evaluated")
    for v, r in a["variants"].items():
        tag = "   <- used by the pipeline" if v == active else ""
        print(f"  {v:20s}: mIoU {r['miou'] * 100:.1f}% over {r['classes_present']} classes, "
              f"accuracy {r['overall_accuracy'] * 100:.1f}%{tag}")
    for band, r in a["by_distance"].items():
        print(f"  {band:8s}: accuracy {r['accuracy'] * 100:5.1f}%, mIoU {r['miou'] * 100:5.1f}%")
    s, g = out["safety"], out["grid_accuracy"]
    print(f"  grid cells correct: {g['by_cell'] * 100:.1f}% (by area {g['by_area'] * 100:.1f}%)")
    print(f"  missed obstacles: vehicles + people {s['missed_vehicles_people'] * 100:.2f}%, "
          f"all {s['missed_all_obstacles'] * 100:.2f}%; false alarms road/parking "
          f"{s['false_alarm_road_parking'] * 100:.2f}%")
    print(f"Wrote {args.out}/accuracy_frames.csv, accuracy.json, confusion.npy")


if __name__ == "__main__":
    main()
