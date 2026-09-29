#!/usr/bin/env python
"""
Assemble the Hugging Face Space folder: space/ + the CPU parts of the as25lm package
+ the weights + third-party notices. Upload the output folder to the Space.

Example:
  python scripts/build_space.py --weights weights/salsanext_weights.pth --out build/space
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_FILES = ["__init__.py", "config.py", "data.py", "frame_format.py",
                 "grid/__init__.py", "grid/reference.py",
                 "segmentation/__init__.py", "segmentation/segmenter.py",
                 "segmentation/salsanext.py", "segmentation/knn.py"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="weights-only SalsaNext file (scripts/convert_weights.py)")
    p.add_argument("--out", default=str(ROOT / "build" / "space"))
    args = p.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in ("app.py", "README.md", "requirements.txt"):
        shutil.copy(ROOT / "space" / name, out / name)
    shutil.copy(ROOT / "THIRD_PARTY_NOTICES.md", out / "THIRD_PARTY_NOTICES.md")
    for rel in PACKAGE_FILES:
        dst = out / "as25lm" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / "src" / "as25lm" / rel, dst)
    shutil.copy(args.weights, out / "salsanext_weights.pth")

    print(f"Space folder ready: {out}")
    for f in sorted(out.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(out)}  ({f.stat().st_size / 1024:,.0f} KB)")
    print("Before uploading: set sdk_version in README.md to the version your Space shows.")


if __name__ == "__main__":
    main()
