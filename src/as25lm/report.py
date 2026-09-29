# -*- coding: utf-8 -*-
"""Results: latency summary, record frames, summary.json / frame_metrics.csv, and charts."""

import json
import os

import numpy as np

from . import config as C

LATENCY_COLUMNS = ["upload", "salsa_project", "salsa_model", "salsa_post", "salsanext_total",
                   "grid_gpu", "grid_cpu", "grid_to_cpu", "grid_engine_total", "end_to_end"]
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#6b7075"


def to_jsonable(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, float) and np.isnan(o):
        return None
    return str(o)


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=to_jsonable)


def latency_summary(df):
    cols = [c for c in LATENCY_COLUMNS if c in df]
    d = df[cols]
    return {c: dict(median_ms=round(float(d[c].median()), 2), p95_ms=round(float(d[c].quantile(0.95)), 2),
                    p99_ms=round(float(d[c].quantile(0.99)), 2), max_ms=round(float(d[c].max()), 2),
                    mean_ms=round(float(d[c].mean()), 2)) for c in cols}


def real_time_summary(df):
    e2e, grid = df["end_to_end"], df["grid_engine_total"]
    return dict(pct_within_100ms=float((e2e < 100).mean()), pct_grid_within_30ms=float((grid < 30).mean()),
                e2e_p99_ms=float(e2e.quantile(0.99)), e2e_max_ms=float(e2e.max()),
                grid_p99_ms=float(grid.quantile(0.99)), grid_max_ms=float(grid.max()),
                slowest_frame=str(e2e.idxmax()), e2e_std_ms=float(e2e.std()), grid_std_ms=float(grid.std()),
                vram_first_mb=float(df["vram_mb"].iloc[0]) if "vram_mb" in df else None,
                vram_last_mb=float(df["vram_mb"].iloc[-1]) if "vram_mb" in df else None,
                vram_max_mb=float(df["vram_mb"].max()) if "vram_mb" in df else None)


def records(bench_df, eval_df=None):
    """Record frames: accuracy, car IoU, person IoU, fastest, stress test."""
    out = {}
    if eval_df is not None and len(eval_df):
        fa = eval_df["accuracy"].idxmax()
        out["highest_accuracy"] = dict(frame=str(fa), accuracy=float(eval_df.loc[fa, "accuracy"]))
        cars = eval_df[eval_df["eval_car_points"] >= C.MIN_CAR_POINTS]
        if len(cars):
            fc = cars["car_iou"].idxmax()
            out["highest_car_iou"] = dict(frame=str(fc), iou=float(cars.loc[fc, "car_iou"]),
                                          car_points=int(cars.loc[fc, "eval_car_points"]), frames_with_cars=len(cars))
        peds = eval_df[eval_df["eval_person_points"] >= C.MIN_PERSON_POINTS]
        if len(peds):
            fp = peds["person_iou"].idxmax()
            out["highest_person_iou"] = dict(frame=str(fp), iou=float(peds.loc[fp, "person_iou"]),
                                             person_points=int(peds.loc[fp, "eval_person_points"]),
                                             frames_with_people=len(peds))
    e2e = bench_df["end_to_end"]
    ff = e2e.idxmin()
    out["fastest"] = dict(frame=str(ff), end_to_end_ms=float(e2e[ff]),
                          salsanext_ms=float(bench_df.loc[ff, "salsanext_total"]),
                          grid_engine_ms=float(bench_df.loc[ff, "grid_engine_total"]))
    use_gt = "gt_car_points" in bench_df and bench_df["gt_car_points"].notna().all()
    pre = "gt_" if use_gt else "pred_"
    if pre + "car_points" in bench_df:
        cars_p = bench_df[pre + "car_points"].astype(int)
        peds_p = bench_df[pre + "person_points"].astype(int)
        fs = (cars_p + peds_p).idxmax()
        out["stress_test"] = dict(frame=str(fs), source="ground truth" if use_gt else "predicted",
                                  car_points=int(cars_p[fs]), pedestrian_points=int(peds_p[fs]),
                                  total_points=int(bench_df.loc[fs, "n_points"]), end_to_end_ms=float(e2e[fs]),
                                  salsanext_ms=float(bench_df.loc[fs, "salsanext_total"]),
                                  grid_engine_ms=float(bench_df.loc[fs, "grid_engine_total"]))
    return out


# ── Charts ───────────────────────────────────────────────────────────────────

def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def plot_latency_over_sequence(df, path, gpu="GPU", sequence="08"):
    import matplotlib.pyplot as plt
    e2e, grid = df["end_to_end"].values, df["grid_engine_total"].values
    fig, ax = plt.subplots(figsize=(12, 4.2))
    x = np.arange(len(df))
    ax.plot(x, e2e, color=BLUE, lw=1.2, label="End to end (SalsaNext + Grid Engine)")
    ax.plot(x, grid, color=ORANGE, lw=1.2, label="Grid Engine")
    ax.axhline(100, color=GREY, ls="--", lw=1)
    ax.axhline(30, color=GREY, ls=":", lw=1)
    ax.text(len(df) - 1, 101, "100 ms: one LiDAR scan at 10 Hz", ha="right", va="bottom", color=GREY, fontsize=9)
    ax.text(len(df) - 1, 28.5, "30 ms: Grid Engine target", ha="right", va="top", color=GREY, fontsize=9)
    ax.set_ylim(0, max(110, float(e2e.max()) * 1.1))
    ax.set_xlim(0, max(len(df) - 1, 1))
    ax.set_xlabel(f"Frame (sequence {sequence}, {df.index[0]}-{df.index[-1]})")
    ax.set_ylabel("Latency per frame (ms)")
    ax.set_title(f"Latency per frame on {gpu}", loc="left")
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0, 0.93))
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_latency_histogram(df, path):
    import matplotlib.pyplot as plt
    e2e = df["end_to_end"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(e2e.values, bins=50, color=BLUE, edgecolor="white", linewidth=0.5)
    top = ax.get_ylim()[1]
    for q, label, ls in [(e2e.median(), "median", "-"), (e2e.quantile(0.99), "p99", "--")]:
        ax.axvline(q, color=GREY, ls=ls, lw=1)
        ax.text(q, top * 0.97, f" {label} {q:.1f} ms", color=GREY, fontsize=9, va="top")
    ax.set_xlabel("End-to-end latency (ms)")
    ax.set_ylabel("Frames")
    ax.set_title(f"End-to-end latency, {len(df):,} frames", loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, path, title):
    import matplotlib.pyplot as plt
    cm = np.asarray(cm, np.float64)[1:, 1:]
    rows = cm.sum(1) > 0
    cm_n = cm[rows] / cm[rows].sum(1, keepdims=True)
    names = [C.LABEL_NAMES[c] for c in range(1, C.N_CLASSES)]
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=9)
    ax.set_yticks(range(int(rows.sum())), [n for n, p in zip(names, rows) if p], fontsize=9)
    for i in range(cm_n.shape[0]):
        for j in range(cm_n.shape[1]):
            if cm_n[i, j] >= 0.01:
                ax.text(j, i, f"{cm_n[i, j] * 100:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if cm_n[i, j] > 0.5 else "#1b1f23")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    ax.set_title(title, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Share of ground-truth points")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_class_iou(summary, path):
    """Horizontal bars, sorted, one colour (a single measure)."""
    import matplotlib.pyplot as plt
    iou = {k: v for k, v in summary["accuracy"]["per_class_iou"].items() if v is not None}
    items = sorted(iou.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(8, 6.2))
    y = np.arange(len(items))
    ax.barh(y, [v * 100 for _, v in items], color=BLUE, height=0.7)
    ax.set_yticks(y, [k for k, _ in items], fontsize=9)
    for i, (_, v) in enumerate(items):
        ax.text(v * 100 + 1, i, f"{v * 100:.1f}", va="center", fontsize=8, color="#1b1f23")
    active = summary["accuracy"]["active"]
    miou = summary["accuracy"]["variants"][active]["miou"]
    ax.axvline(miou * 100, color=GREY, ls="--", lw=1)
    ax.text(miou * 100 + 1, len(items) - 0.4, f"mIoU {miou * 100:.1f}", color=GREY, fontsize=9, va="center")
    ax.set_xlim(0, 105)
    ax.set_xlabel("IoU (%)")
    frames = summary["accuracy"]["frames_evaluated"]
    ax.set_title(f"Per-class IoU ({active}, {frames:,} frames)", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_accuracy_by_distance(summary, path):
    """Accuracy and mIoU per distance band, grouped bars."""
    import matplotlib.pyplot as plt
    bands = summary["accuracy"]["by_distance"]
    names = list(bands)
    acc = [bands[n]["accuracy"] * 100 for n in names]
    miou = [bands[n]["miou"] * 100 for n in names]
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    b1 = ax.bar(x - w / 2 - 0.01, acc, w, color=BLUE, label="Overall accuracy")
    b2 = ax.bar(x + w / 2 + 0.01, miou, w, color=ORANGE, label="mIoU")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1, f"{r.get_height():.0f}",
                    ha="center", va="bottom", fontsize=8, color="#1b1f23")
    ax.set_xticks(x, [f"{n}\n{bands[n]['points'] / 1e6:,.1f} M pts" for n in names], fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("%")
    ax.set_title("Segmentation quality by distance from the sensor", loc="left")
    ax.legend(frameon=False, loc="upper right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figures(results_dir, summary=None, frame_df=None, confusion_cm=None):
    """Write every chart the available data allows into results_dir/figures/."""
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    made = []
    run = (summary or {}).get("run", {})
    if frame_df is not None and "end_to_end" in frame_df:
        p = os.path.join(fig_dir, "latency_over_sequence.png")
        plot_latency_over_sequence(frame_df, p, run.get("gpu", "GPU"), run.get("sequence", "08")); made.append(p)
        p = os.path.join(fig_dir, "latency_histogram.png")
        plot_latency_histogram(frame_df, p); made.append(p)
    if summary and "accuracy" in summary:
        p = os.path.join(fig_dir, "per_class_iou.png")
        plot_per_class_iou(summary, p); made.append(p)
        if summary["accuracy"].get("by_distance"):
            p = os.path.join(fig_dir, "accuracy_by_distance.png")
            plot_accuracy_by_distance(summary, p); made.append(p)
    if confusion_cm is not None and summary and "accuracy" in summary:
        a = summary["accuracy"]
        p = os.path.join(fig_dir, "confusion_matrix.png")
        plot_confusion(confusion_cm, p, f"Confusion matrix, % of each ground-truth class "
                                        f"({a['active']}, {a['frames_evaluated']:,} frames)"); made.append(p)
    return made
