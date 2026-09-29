# -*- coding: utf-8 -*-
"""
Dashboard clip export: LGF1 frames + manifest.json (format: docs/FORMAT.md).

For each frame: lite/<frame>.lgf.gz (grid + every Nth point, for playback) and
full/<frame>.lgf.gz (grid + all points, loaded when paused).
"""

import os

import numpy as np

from . import config as C
from . import frame_format as FF
from .data import frame_id, label_path, load_labels, load_scan
from .metrics import frame_accuracy
from .report import write_json


def _bookmarks(frames_meta):
    import pandas as pd
    fm = pd.DataFrame([dict(index=m["index"], e2e=m["timing_ms"].get("end_to_end", np.nan), **(m["accuracy"] or {}))
                       for m in frames_meta]).set_index("index")
    out = []

    def add(label, idx, value):
        out.append(dict(label=label, frame=frames_meta[int(idx)]["id"], index=int(idx),
                        value=None if value is None or (isinstance(value, float) and np.isnan(value)) else float(value)))

    if "accuracy" in fm:
        add("Highest overall accuracy", fm["accuracy"].idxmax(), fm["accuracy"].max())
        cars = fm[fm["gt_car_points"] >= C.MIN_CAR_POINTS].dropna(subset=["car_iou"])
        if len(cars):
            add("Highest car IoU", cars["car_iou"].idxmax(), cars["car_iou"].max())
        peds = fm[fm["gt_person_points"] >= C.MIN_PERSON_POINTS].dropna(subset=["person_iou"])
        if len(peds):
            add("Highest person IoU", peds["person_iou"].idxmax(), peds["person_iou"].max())
        stress = (fm["gt_car_points"] + fm["gt_person_points"]).idxmax()
        add("Stress test (most car + pedestrian points)", stress,
            fm.loc[stress, "gt_car_points"] + fm.loc[stress, "gt_person_points"])
    if fm["e2e"].notna().any():
        add("Fastest end to end (ms)", fm["e2e"].idxmin(), fm["e2e"].min())
        add("Slowest end to end (ms)", fm["e2e"].idxmax(), fm["e2e"].max())
    return out


def export_clip(pipeline, scans, out_dir, label_dir=None, lite_stride=4, sequence=None,
                preview_mp4=False, progress=True, warmup=5):
    """Run the pipeline on `scans` and write the clip; returns the manifest dict."""
    for sub in ("lite", "full"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    device = pipeline.seg.device
    device_name = __import__("torch").cuda.get_device_name(0) if device.type == "cuda" else "CPU"

    video = None
    if preview_mp4:
        video = _PreviewVideo(os.path.join(out_dir, "preview_traversability.mp4"))
        if not video.ok:
            print("ffmpeg not found: skipping the preview video")
            video = None

    for path in scans[:warmup]:                     # keep the cold start out of the per-frame timings
        pipeline.process(load_scan(path))

    it = scans
    if progress:
        from tqdm import tqdm
        it = tqdm(scans, desc="Clip export")
    frames_meta = []
    for i, path in enumerate(it):
        fid = frame_id(path)
        raw = load_scan(path)
        res = pipeline.process(raw, timed=True)
        timing = {k: round(v, 3) for k, v in res.timing_ms.items()}
        lab = res.labels.cpu().numpy()
        conf = res.conf.float().cpu().numpy()
        lp = label_path(label_dir, path)
        gt = load_labels(lp) if lp else None
        acc = frame_accuracy(gt, lab) if gt is not None else None
        sizes = {}
        for kind, stride in (("lite", lite_stride), ("full", 1)):
            header = FF.frame_header(fid, "precomputed", len(raw), stride, res.cells, stats=res.stats,
                                     timing_ms=timing, device=device_name, accuracy=acc, sequence=sequence,
                                     extra=dict(clip_index=i))
            sections = FF.build_sections(raw[:, :3], raw[:, 3], lab, conf, res.cells, points_gt=gt,
                                         point_stride=stride)
            sizes[kind] = FF.write_frame(os.path.join(out_dir, kind, fid + ".lgf.gz"), header, sections)
        frames_meta.append(dict(index=i, id=fid, lite=f"lite/{fid}.lgf.gz", full=f"full/{fid}.lgf.gz",
                                lite_bytes=sizes["lite"], full_bytes=sizes["full"], n_points=len(raw),
                                n_cells=len(res.cells), timing_ms=timing, stats=res.stats, accuracy=acc))
        if video:
            video.add(res.cells, fid, timing)
    if video:
        video.close()

    accs = [m["accuracy"]["accuracy"] for m in frames_meta if m["accuracy"]]
    manifest = dict(
        format="LGF1-clip", version=1, sequence=sequence, fps=10, frame_count=len(frames_meta),
        first_frame=frames_meta[0]["id"], last_frame=frames_meta[-1]["id"], lite_point_stride=lite_stride,
        sensor_height_m=FF.SENSOR_HEIGHT_M, level_sizes_m=[C.LEVEL_SIZE_M[L] for L in range(C.MAX_LEVEL + 1)],
        classes=FF.class_table(), bookmarks=_bookmarks(frames_meta),
        summary=dict(device=device_name, fp16=pipeline.fp16, knn=pipeline.seg.knn is not None,
                     engine=pipeline.engine_kind,
                     median_end_to_end_ms=float(np.median([m["timing_ms"]["end_to_end"] for m in frames_meta])),
                     median_grid_engine_ms=float(np.median([m["timing_ms"]["grid_engine_total"] for m in frames_meta])),
                     lite_total_mb=sum(m["lite_bytes"] for m in frames_meta) / 2**20,
                     full_total_mb=sum(m["full_bytes"] for m in frames_meta) / 2**20,
                     mean_accuracy=float(np.mean(accs)) if accs else None),
        frames=frames_meta)
    write_json(os.path.join(out_dir, "manifest.json"), manifest)
    return manifest


def upload_to_hf(out_dir, repo_id, token):
    """Upload the clip folder to a public Hugging Face dataset repo; returns the manifest URL."""
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.upload_large_folder(repo_id=repo_id, repo_type="dataset", folder_path=out_dir)
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/manifest.json"


class _PreviewVideo:
    """Top-down traversability video of the clip (needs ffmpeg)."""

    def __init__(self, path, view_m=40.0):
        import shutil
        self.ok = shutil.which("ffmpeg") is not None
        if not self.ok:
            return
        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter
        self.plt, self.view = plt, view_m
        self.fig, self.ax = plt.subplots(figsize=(8, 8), dpi=100)
        self.writer = FFMpegWriter(fps=10, bitrate=6000)
        self.writer.setup(self.fig, path, dpi=100)

    def add(self, cells, fid, timing):
        from matplotlib.collections import PolyCollection
        v = self.view
        c = cells[(np.abs(cells["x_center"]) < v) & (np.abs(cells["y_center"]) < v)]
        verts = np.stack([np.stack([c["x_min"], c["y_min"]], 1), np.stack([c["x_max"], c["y_min"]], 1),
                          np.stack([c["x_max"], c["y_max"]], 1), np.stack([c["x_min"], c["y_max"]], 1)], 1)
        self.ax.clear()
        pc = PolyCollection(verts, array=c["traversability"], cmap="RdYlGn", edgecolors="none")
        pc.set_clim(0, 1)
        self.ax.add_collection(pc)
        self.ax.set_xlim(-v, v); self.ax.set_ylim(-v, v); self.ax.set_aspect("equal")
        self.ax.set_title(f"Frame {fid}  |  Grid Engine {timing.get('grid_engine_total', 0):.1f} ms  |  "
                          f"end to end {timing.get('end_to_end', 0):.1f} ms", loc="left", fontsize=10)
        self.writer.grab_frame()

    def close(self):
        self.writer.finish()
        self.plt.close(self.fig)
