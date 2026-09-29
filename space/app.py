# -*- coding: utf-8 -*-
"""
AS2.5LM live single-frame API (Hugging Face ZeroGPU Space).

Upload a raw KITTI velodyne .bin -> SalsaNext (GPU, FP16) + KNN -> Grid Engine
(NumPy, CPU; bit-identical to the CUDA engine) -> one LGF1 frame (.lgf.gz), the same
format as the precomputed clip. The dashboard calls "/infer" with @gradio/client and
decodes the result with decodeFrame.ts. Build the deployable folder with
scripts/build_space.py.
"""

import gzip
import os
import tempfile
import time
from pathlib import Path

import gradio as gr
import torch

from as25lm import frame_format as FF
from as25lm.data import load_scan
from as25lm.grid.reference import run_reference, to_structured
from as25lm.segmentation import Segmenter

try:
    import spaces                     # present on Hugging Face Spaces
    gpu = spaces.GPU
except ImportError:                   # local runs: no-op decorator
    def gpu(fn=None, **_):
        return fn if fn is not None else (lambda f: f)

HERE = Path(__file__).parent
FORCE_CPU = os.environ.get("FORCE_CPU") == "1"
DEVICE = "cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu"
USE_FP16 = DEVICE == "cuda"
BENCHMARK_NOTE = os.environ.get(
    "BENCHMARK_NOTE",
    "Live demo: the Grid Engine runs on CPU here. Real-time benchmark (Tesla T4, CUDA, 4,071 frames): "
    "Grid Engine 4.65 ms, end to end 31.1 ms per frame.")

# ZeroGPU wants models placed on CUDA at import time (outside the @gpu function).
SEG = Segmenter(HERE / "salsanext_weights.pth", DEVICE)


@gpu(duration=30)
def segment_scan(raw):
    """SalsaNext + KNN for one scan. On ZeroGPU a GPU is attached only while this runs."""
    t0 = time.perf_counter()
    pts = torch.from_numpy(raw).to(DEVICE)
    labels, conf, _ = SEG.segment(pts, fp16=USE_FP16)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    device = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
    ms = (time.perf_counter() - t0) * 1000
    return labels.cpu().numpy(), conf.float().cpu().numpy(), ms, device


def infer(scan_file):
    """Run the full pipeline on an uploaded .bin; return the .lgf.gz frame and a summary."""
    if scan_file is None:
        raise gr.Error("Upload a KITTI velodyne .bin file.")
    path = scan_file if isinstance(scan_file, str) else scan_file.name
    t_start = time.perf_counter()
    try:
        raw = load_scan(path, validate=True)
    except ValueError as e:
        raise gr.Error(str(e))
    t_seg = time.perf_counter()
    labels, conf, seg_ms, device = segment_scan(raw)
    wait_ms = (time.perf_counter() - t_seg) * 1000 - seg_ms     # GPU hand-over and queueing
    t_grid = time.perf_counter()
    matrix, stats, _ = run_reference(raw[:, :3], labels, conf)
    cells = to_structured(matrix)
    grid_ms = (time.perf_counter() - t_grid) * 1000
    t_enc = time.perf_counter()
    timing = dict(segmentation_ms=round(seg_ms, 2), gpu_wait_ms=round(max(wait_ms, 0), 2),
                  grid_engine_cpu_ms=round(grid_ms, 2))
    frame = Path(path).name.split(".")[0]
    header = FF.frame_header(frame, "live", len(raw), 1, cells, stats=stats, timing_ms=timing,
                             device=f"{device} (segmentation) + CPU (grid engine)")
    sections = FF.build_sections(raw[:, :3], raw[:, 3], labels, conf, cells)
    data = gzip.compress(FF.encode_frame(header, sections), compresslevel=1)   # fastest; ~7% bigger than level 6
    timing["encode_ms"] = round((time.perf_counter() - t_enc) * 1000, 2)
    timing["server_total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)

    out = tempfile.NamedTemporaryFile(prefix=f"{frame}_", suffix=".lgf.gz", delete=False)
    out.write(data)
    out.close()
    summary = dict(frame=frame, points=int(len(raw)), cells=int(len(cells)),
                   splits=int(stats["n_splits"]), merges=int(stats["n_merges"]),
                   device=device, fp16=USE_FP16, bytes=len(data), timing_ms=timing, note=BENCHMARK_NOTE)
    return out.name, summary


def ping():
    """Cheap call the dashboard makes on page load so a sleeping Space starts waking up."""
    return "ok"


with gr.Blocks(title="AS2.5LM live frame") as demo:
    gr.Markdown("## AS2.5LM: live single frame\n"
                "Upload a raw KITTI velodyne `.bin` scan. Returns the segmented 2.5D grid as an "
                "LGF1 frame (`.lgf.gz`) for the dashboard, plus a summary.")
    with gr.Row():
        scan = gr.File(label="KITTI .bin scan", file_types=[".bin"], type="filepath")
        with gr.Column():
            frame_out = gr.File(label="Frame (.lgf.gz)")
            summary_out = gr.JSON(label="Summary")
    run_btn = gr.Button("Run pipeline", variant="primary")
    run_btn.click(infer, inputs=scan, outputs=[frame_out, summary_out], api_name="infer")
    ping_out = gr.Textbox(visible=False)
    ping_btn = gr.Button("ping", visible=False)
    ping_btn.click(ping, inputs=None, outputs=ping_out, api_name="ping")

if __name__ == "__main__":
    demo.queue(max_size=16).launch()
