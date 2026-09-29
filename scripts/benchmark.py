#!/usr/bin/env python
"""
Latency, GPU memory, power and throughput of the full pipeline.

Scans are read before each timer starts; every timestamp waits for the GPU; warm-up
frames are excluded. Writes <out>/benchmark_frames.csv and <out>/benchmark.json.

Example:
  python scripts/benchmark.py --scans .../08/velodyne --labels .../08/labels \
      --weights weights/salsanext_weights.pth --out results/run
"""

import os
import time

import numpy as np
import pandas as pd
import torch

from as25lm import cli, config as C
from as25lm.data import frame_id, label_path, load_labels, load_scan
from as25lm.grid.reference import run_reference
from as25lm.metrics import PowerSampler
from as25lm.report import latency_summary, real_time_summary, write_json


def main():
    p = cli.parser(__doc__)
    cli.add_data_args(p)
    cli.add_model_args(p)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--no-extras", action="store_true",
                   help="skip the extra comparisons (GPU stage profile, FP32 vs FP16, CPU reference)")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    scans = cli.scan_list(args)
    pipe = cli.make_pipeline(args)
    cuda = pipe.device.type == "cuda"
    gpu = torch.cuda.get_device_name(0) if cuda else "CPU"
    print(f"{len(scans)} frames | {gpu} | engine {args.engine} | {'FP32' if args.fp32 else 'FP16'} | "
          f"KNN {'off' if args.no_knn else 'on'}")

    with PowerSampler() as idle:
        time.sleep(2.0)
    idle_w = float(np.mean(idle.w)) if idle.w else None

    for path in scans[:args.warmup]:
        pipe.process(load_scan(path))
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    rows, intervals = [], []
    with PowerSampler() as power:
        for path in scans:
            raw = load_scan(path)                                   # disk read before the timer starts
            lp = label_path(args.labels, path)
            gt = load_labels(lp) if lp else None
            t0 = time.perf_counter()
            res = pipe.process(raw, timed=True)
            intervals.append((t0, time.perf_counter()))
            row = dict(res.timing_ms, frame=frame_id(path), n_points=len(raw), n_cells=len(res.cells))
            pc, sc = res.cells["point_count"], res.cells["semantic_class"]
            row["pred_car_points"] = int(pc[sc == C.CAR].sum())
            row["pred_person_points"] = int(pc[sc == C.PERSON].sum())
            row["gt_car_points"] = int((gt == C.CAR).sum()) if gt is not None else np.nan
            row["gt_person_points"] = int((gt == C.PERSON).sum()) if gt is not None else np.nan
            row["vram_mb"] = torch.cuda.memory_allocated() / 2**20 if cuda else np.nan
            rows.append(row)
    df = pd.DataFrame(rows).set_index("frame")
    df.to_csv(os.path.join(args.out, "benchmark_frames.csv"))

    lat = latency_summary(df)
    e2e, grid = df["end_to_end"], df["grid_engine_total"]
    out = dict(run=dict(gpu=gpu, frames=len(df), first_frame=df.index[0], last_frame=df.index[-1],
                        sequence=args.sequence, fp16=not args.fp32, knn=not args.no_knn, engine=args.engine),
               latency_ms=lat, real_time=real_time_summary(df))

    swap = {}
    if cuda:
        free_b, total_b = torch.cuda.mem_get_info()
        context_mb = max((total_b - free_b) / 2**20 - torch.cuda.memory_reserved() / 2**20, 0)
        peak_alloc = torch.cuda.max_memory_allocated() / 2**20
        peak_res = torch.cuda.max_memory_reserved() / 2**20
        swap.update(peak_vram_allocated_mb=peak_alloc, peak_vram_reserved_mb=peak_res, cuda_context_mb=context_mb,
                    device_peak_mb=peak_res + context_mb, gpu_memory_total_mb=total_b / 2**20)
    active_w = power.watts_during(intervals)
    if len(active_w):
        mean_w = float(active_w.mean())
        e_frame = mean_w * e2e.mean() / 1000
        swap.update(idle_w=idle_w, mean_active_w=mean_w, peak_w=float(active_w.max()), power_limit_w=power.limit_w,
                    energy_per_frame_j=e_frame, frames_per_joule=1 / e_frame, power_readings=len(active_w))
    out["swap"] = swap
    total_points, total_s = int(df["n_points"].sum()), e2e.sum() / 1000
    out["throughput"] = dict(frames=len(df), points=total_points, seconds=total_s,
                             points_per_s=total_points / total_s, frames_per_s=len(df) / total_s,
                             grid_engine_points_per_s=total_points / (grid.sum() / 1000))

    if not args.no_extras:
        out["extras"] = extras(pipe, scans[:50], cuda)

    write_json(os.path.join(args.out, "benchmark.json"), out)
    print(pd.DataFrame(lat).T.to_string())
    verdict = "PASS" if lat["grid_engine_total"]["p95_ms"] < 30 else "FAIL"
    print(f"\nGrid Engine: median {lat['grid_engine_total']['median_ms']} ms, "
          f"p95 {lat['grid_engine_total']['p95_ms']} ms -> {verdict} (target < 30 ms)")
    print(f"End to end: median {lat['end_to_end']['median_ms']} ms "
          f"({1000 / lat['end_to_end']['median_ms']:.1f} FPS)")
    if swap:
        print("SWaP:", {k: round(v, 2) if isinstance(v, float) else v for k, v in swap.items()})
    print(f"Wrote {args.out}/benchmark_frames.csv and benchmark.json")


def extras(pipe, scans, cuda):
    """GPU stage profile, FP32 vs FP16 model time, and the NumPy reference on CPU."""
    out = {}
    raws = [load_scan(s) for s in scans]
    if pipe.engine is not None:
        stage_rows = []
        for raw in raws:
            pts = torch.from_numpy(raw).to(pipe.device)
            labels, conf, _ = pipe.seg.segment(pts, fp16=pipe.fp16)
            stage_rows.append(pipe.engine.run(pts[:, :3], labels, conf, profile=True).stage_ms)
        out["grid_gpu_stages_median_ms"] = pd.DataFrame(stage_rows).median().round(3).to_dict()
    if cuda:
        projs = [pipe.seg.project(torch.from_numpy(raw).to(pipe.device))[0] for raw in raws]
        model_ms = {}
        for fp16 in (False, True):
            for pr in projs[:5]:
                pipe.seg.probabilities(pr, fp16)
            ts = []
            for pr in projs:
                torch.cuda.synchronize(); t = time.perf_counter()
                pipe.seg.probabilities(pr, fp16)
                torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1000)
            model_ms["fp16" if fp16 else "fp32"] = float(np.median(ts))
        out["salsanext_model_median_ms"] = model_ms
    cpu_ms = []
    for raw in raws[:10]:
        labels, conf, _ = pipe.seg.segment(torch.from_numpy(raw).to(pipe.device), fp16=pipe.fp16)
        lab, cf = labels.cpu().numpy(), conf.float().cpu().numpy()
        t = time.perf_counter()
        run_reference(raw[:, :3], lab, cf)
        cpu_ms.append((time.perf_counter() - t) * 1000)
    out["grid_numpy_cpu_median_ms"] = float(np.median(cpu_ms))
    print("Extras:", out)
    return out


if __name__ == "__main__":
    main()
