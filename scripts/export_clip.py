#!/usr/bin/env python
"""
Export a dashboard clip: LGF1 frames (lite/ + full/) and manifest.json (docs/FORMAT.md),
optionally a top-down preview video and an upload to a public Hugging Face dataset.

Example (frames 001500-002499, the clip used by the dashboard):
  python scripts/export_clip.py --scans .../08/velodyne --labels .../08/labels \
      --weights weights/salsanext_weights.pth --start 1500 --frames 1000 --out build/clip \
      --upload-repo YOUR_HF_USERNAME/sih-lidar-clip          # needs HF_TOKEN in the environment
"""

import os

from as25lm import cli
from as25lm.export import export_clip, upload_to_hf


def main():
    p = cli.parser(__doc__)
    cli.add_data_args(p)
    cli.add_model_args(p)
    p.add_argument("--out", required=True, help="output folder for the clip")
    p.add_argument("--lite-stride", type=int, default=4, help="playback files keep every Nth point")
    p.add_argument("--preview-mp4", action="store_true", help="also render a top-down video (needs ffmpeg)")
    p.add_argument("--upload-repo", default=None, help="Hugging Face dataset repo, e.g. user/sih-lidar-clip")
    args = p.parse_args()

    scans = cli.scan_list(args)
    pipe = cli.make_pipeline(args)
    m = export_clip(pipe, scans, args.out, label_dir=args.labels, lite_stride=args.lite_stride,
                    sequence=args.sequence, preview_mp4=args.preview_mp4)
    s = m["summary"]
    print(f"\nExported {m['frame_count']} frames ({m['first_frame']}-{m['last_frame']}) to {args.out}")
    print(f"  lite: {s['lite_total_mb']:,.1f} MB, full: {s['full_total_mb']:,.1f} MB")
    for b in m["bookmarks"]:
        print(f"  {b['label']:45s} frame {b['frame']}  {b['value']}")

    if args.upload_repo:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("Set HF_TOKEN (a Hugging Face write token) to upload.")
        url = upload_to_hf(args.out, args.upload_repo, token)
        print(f"Uploaded. Manifest URL: {url}")


if __name__ == "__main__":
    main()
