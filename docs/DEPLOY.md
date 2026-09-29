# Deployment

Three parts:

1. **Clip data:** frames 001500–002499 exported as LGF1 frames and uploaded to a Hugging Face **dataset**.
2. **Live API:** a Hugging Face **ZeroGPU Space** that runs the pipeline on one uploaded `.bin`.
3. **Dashboard:** React on Vercel, reading both with `dashboard/decodeFrame.ts`.

## 1. Export the clip

On a GPU machine (Kaggle T4: last section of `notebooks/kaggle_run.ipynb`):

```bash
export HF_TOKEN=...            # Hugging Face token with write access
python scripts/export_clip.py --scans $DATA/velodyne --labels $DATA/labels \
    --weights weights/salsanext_weights.pth --start 1500 --frames 1000 \
    --out build/clip --upload-repo YOUR_HF_USERNAME/sih-lidar-clip [--preview-mp4]
```

- Writes `lite/` (~430 KB per frame), `full/` (~810 KB per frame) and `manifest.json`: about 1.2 GB for
  1,000 frames. Takes ~6 minutes on a T4, plus the upload.
- The dataset is created **public** (the browser reads it without a login). The script prints the
  manifest URL: `https://huggingface.co/datasets/<repo>/resolve/main/manifest.json`.

## 2. Live Space

```bash
python scripts/build_space.py --weights weights/salsanext_weights.pth --out build/space
```

1. On huggingface.co: **New → Space**, SDK **Gradio**, hardware **ZeroGPU** (free accounts older than
   30 days with a verified email can host two).
2. Copy the `sdk_version:` from the Space's generated `README.md` into `build/space/README.md`.
3. Upload everything in `build/space/` (including the `as25lm/` folder), e.g.
   ```python
   from huggingface_hub import HfApi
   HfApi(token="...").upload_folder(folder_path="build/space", repo_id="USER/SPACE", repo_type="space")
   ```
4. After the build, test with any KITTI `.bin` on the Space page.

Notes:
- The Space runs SalsaNext on the GPU and the NumPy Grid Engine on the CPU; the cells are identical to the
  CUDA engine's. Response time is ~0.5–2 s; the first call after idle is slower.
- The summary includes `gpu_wait_ms` (ZeroGPU hand-over) and `server_total_ms`.
- The benchmark sentence shown in the summary can be changed with the `BENCHMARK_NOTE` environment variable.

## 3. Dashboard

```bash
npm install @gradio/client
```

- Copy `dashboard/decodeFrame.ts` and `dashboard/drivability.ts` into the React project.
- Clip: `loadManifest(MANIFEST_URL)`, then `loadFrame(frameUrl(m, i, 'lite'))` while playing and
  `'full'` when paused; bookmarks are in `manifest.bookmarks`.
- Live: call `/ping` on page load (wakes the Space), `/infer` on upload.
- Format and rendering notes: [FORMAT.md](FORMAT.md).
