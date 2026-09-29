#!/usr/bin/env python
"""
Combine benchmark + evaluation outputs into summary.json and frame_metrics.csv, and draw the charts.

Reads from --results: benchmark.json, benchmark_frames.csv, accuracy.json, accuracy_frames.csv,
confusion.npy (whatever exists). If only summary.json and frame_metrics.csv are present
(e.g. results/ in this repo), it just redraws the charts.

Example:
  python scripts/report.py --results results/run
"""

import json
import os

import numpy as np
import pandas as pd

from as25lm import cli
from as25lm.report import make_figures, records, write_json


def read_frames(path):
    return pd.read_csv(path, dtype={"frame": str}).set_index("frame")


def main():
    p = cli.parser(__doc__)
    p.add_argument("--results", required=True, help="folder with benchmark / evaluation outputs")
    args = p.parse_args()
    d = args.results
    f = lambda name: os.path.join(d, name)

    if os.path.exists(f("benchmark.json")):
        bench = json.load(open(f("benchmark.json")))
        bdf = read_frames(f("benchmark_frames.csv"))
        acc = json.load(open(f("accuracy.json"))) if os.path.exists(f("accuracy.json")) else {}
        adf = read_frames(f("accuracy_frames.csv")) if os.path.exists(f("accuracy_frames.csv")) else None
        summary = dict(run=bench["run"], latency_ms=bench["latency_ms"], **acc,
                       swap=bench.get("swap", {}), throughput=bench["throughput"],
                       real_time=bench["real_time"], records=records(bdf, adf))
        if "extras" in bench:
            summary["extras"] = bench["extras"]
        frame_df = bdf.join(adf, how="left") if adf is not None else bdf
        write_json(f("summary.json"), summary)
        frame_df.to_csv(f("frame_metrics.csv"))
        print(f"Wrote {f('summary.json')} and {f('frame_metrics.csv')} ({len(frame_df):,} frames)")
    else:
        summary = json.load(open(f("summary.json")))
        frame_df = read_frames(f("frame_metrics.csv"))

    cm = np.load(f("confusion.npy")) if os.path.exists(f("confusion.npy")) else None
    for path in make_figures(d, summary, frame_df, cm):
        print("chart:", path)


if __name__ == "__main__":
    main()
