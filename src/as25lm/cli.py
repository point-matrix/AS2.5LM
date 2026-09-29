# -*- coding: utf-8 -*-
"""Shared command-line options for the scripts in scripts/."""

import argparse

from . import config as C
from .data import list_scans


def parser(description):
    return argparse.ArgumentParser(description=description,
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def add_data_args(p, labels_required=False):
    g = p.add_argument_group("data")
    g.add_argument("--scans", required=True, help="folder of KITTI velodyne .bin scans (one sequence)")
    g.add_argument("--labels", required=labels_required, default=None,
                   help="folder of SemanticKITTI .label files for the same sequence")
    g.add_argument("--start", type=int, default=None, help="first frame number")
    g.add_argument("--frames", type=int, default=None, help="number of frames (default: all)")
    g.add_argument("--sequence", default="08", help="sequence id, for reports")


def add_model_args(p, engine_default="cuda"):
    g = p.add_argument_group("model")
    g.add_argument("--weights", required=True, help="SalsaNext weights (see scripts/convert_weights.py)")
    g.add_argument("--trusted-checkpoint", action="store_true",
                   help="allow loading the original training checkpoint (full unpickling)")
    g.add_argument("--arch-cfg", default=None, help="SalsaNext arch_cfg.yaml (sensor and KNN settings)")
    g.add_argument("--no-knn", action="store_true", help="plain range-image lookup instead of KNN")
    g.add_argument("--fp32", action="store_true", help="run SalsaNext in FP32 (default: FP16 on CUDA)")
    g.add_argument("--device", default="cuda", help="cuda or cpu")
    g.add_argument("--engine", default=engine_default, choices=["cuda", "numpy"],
                   help="grid engine: C++/CUDA extension or NumPy reference (bit-identical)")
    g.add_argument("--build-dir", default=None, help="where the CUDA extension is compiled and cached")


def make_segmenter(args):
    from .segmentation import Segmenter, read_arch_cfg
    sensor, knn = (read_arch_cfg(args.arch_cfg) if args.arch_cfg else (C.SENSOR, C.KNN_PARAMS))
    return Segmenter(args.weights, device=args.device, sensor=sensor, knn_params=knn,
                     use_knn=not args.no_knn, trusted_checkpoint=args.trusted_checkpoint)


def make_pipeline(args, segmenter=None):
    from .pipeline import Pipeline
    seg = segmenter or make_segmenter(args)
    return Pipeline(seg, engine=args.engine, fp16=not args.fp32, build_dir=args.build_dir)


def scan_list(args):
    scans = list_scans(args.scans, args.start, args.frames)
    if not scans:
        raise SystemExit(f"No .bin scans found in {args.scans}")
    return scans
