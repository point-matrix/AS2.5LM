# -*- coding: utf-8 -*-
"""
SalsaNext semantic segmentation with the fixed preprocessing and KNN post-processing.

Fixes over the original inference script:
  * SalsaNext.forward() already applies softmax, so its output is used directly
    (the old script applied softmax twice and squashed confidences to <= 0.125).
  * Each range-image pixel keeps its nearest point (ties: lowest index), deterministically;
    plain GPU index assignment with duplicate indices is undefined.
  * Empty pixels are 0 in the network input, as in SalsaNext's training loader.
  * Real KNN post-processing (SalsaNext repo) instead of a plain pixel lookup.
  * Checkpoints load with strict key checking.
"""

import time

import numpy as np
import torch

from .. import config as C
from .knn import KNN
from .salsanext import SalsaNext


def load_state_dict(path, trusted=False):
    """Model weights from a weights-only file or the original SalsaNext checkpoint.

    Weights-only files (scripts/convert_weights.py) load with torch's safe loader.
    The original training checkpoint also stores optimizer/scheduler objects, which
    the safe loader rejects; load it only with trusted=True (full unpickling).
    """
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        if not trusted:
            raise RuntimeError(
                f"{path} is not a weights-only file. Convert it with scripts/convert_weights.py, "
                "or pass trusted=True / --trusted-checkpoint if you trust the source.") from e
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    return {k.replace("module.", ""): v for k, v in state.items()}


def read_arch_cfg(path):
    """Sensor and KNN settings from a SalsaNext arch_cfg.yaml (falls back to the defaults)."""
    import yaml
    with open(path) as f:
        arch = yaml.safe_load(f)
    sensor, knn = dict(C.SENSOR), dict(C.KNN_PARAMS)
    try:
        s = arch["dataset"]["sensor"]
        sensor.update(height=s["img_prop"]["height"], width=s["img_prop"]["width"],
                      fov_up=s["fov_up"], fov_down=s["fov_down"],
                      img_means=s["img_means"], img_stds=s["img_stds"])
    except (KeyError, TypeError):
        pass
    try:
        knn.update(arch["post"]["KNN"]["params"])
    except (KeyError, TypeError):
        pass
    return sensor, knn


class Segmenter:
    """Point-wise semantic labels for one LiDAR scan."""

    def __init__(self, weights, device="cuda", sensor=None, knn_params=None, use_knn=True,
                 trusted_checkpoint=False):
        self.device = torch.device(device)
        self.sensor = dict(sensor or C.SENSOR)
        model = SalsaNext(C.N_CLASSES)
        model.load_state_dict(load_state_dict(weights, trusted=trusted_checkpoint), strict=True)
        self.model = model.to(self.device).eval()
        self.knn = KNN(dict(knn_params or C.KNN_PARAMS), C.N_CLASSES) if use_knn else None
        self.means = torch.tensor(self.sensor["img_means"], dtype=torch.float32, device=self.device).view(5, 1)
        self.stds = torch.tensor(self.sensor["img_stds"], dtype=torch.float32, device=self.device).view(5, 1)

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    def project(self, pts):
        """(N, 4) tensor -> network input (1, 5, H, W), range image (H, W; -1 = empty), px, py, depth."""
        H, W = self.sensor["height"], self.sensor["width"]
        fov_up = self.sensor["fov_up"] / 180.0 * np.pi
        fov_down = self.sensor["fov_down"] / 180.0 * np.pi
        fov = abs(fov_down) + abs(fov_up)
        x, y, z, rem = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
        depth = torch.linalg.norm(pts[:, :3], dim=1)
        yaw = -torch.atan2(y, x)
        pitch = torch.asin(torch.clamp(z / depth.clamp_min(1e-6), -1.0, 1.0))
        px = torch.floor(0.5 * (yaw / np.pi + 1.0) * W).long().clamp_(0, W - 1)
        py = torch.floor((1.0 - (pitch + abs(fov_down)) / fov) * H).long().clamp_(0, H - 1)

        n = pts.shape[0]
        pix = py * W + px
        best = torch.full((H * W,), float("inf"), device=pts.device).scatter_reduce(0, pix, depth, reduce="amin")
        cand = torch.where(depth == best[pix], torch.arange(n, device=pts.device), torch.full_like(pix, n))
        owner = torch.full((H * W,), n, dtype=torch.long, device=pts.device).scatter_reduce(0, pix, cand, reduce="amin")
        valid = owner < n
        o = owner[valid]
        proj = torch.zeros((5, H * W), device=pts.device)               # empty pixels stay 0, as in training
        proj[:, valid] = (torch.stack([depth, x, y, z, rem])[:, o] - self.means) / self.stds
        proj_range = torch.full((H * W,), -1.0, device=pts.device)
        proj_range[valid] = depth[o]
        return proj.view(1, 5, H, W), proj_range.view(H, W), px, py, depth

    @torch.no_grad()
    def probabilities(self, proj, fp16=True):
        """(20, H, W) float32 class probabilities (FP16 convolutions on CUDA, softmax in FP32)."""
        use_fp16 = bool(fp16) and self.device.type == "cuda"
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
            out = self.model(proj)
        return out[0].float()

    @torch.no_grad()
    def segment(self, pts, fp16=True, timing=None):
        """pts (N, 4) tensor on self.device -> labels, confidence, pixel-lookup labels (all (N,)).

        If `timing` is a dict, stage times in ms are added as salsa_project,
        salsa_model and salsa_post (the GPU is synchronised at each boundary).
        """
        t = self._sync() if timing is not None else 0.0
        proj, proj_range, px, py, depth = self.project(pts)
        if timing is not None:
            t1 = self._sync(); timing["salsa_project"] = (t1 - t) * 1000; t = t1
        probs = self.probabilities(proj, fp16)
        if timing is not None:
            t1 = self._sync(); timing["salsa_model"] = (t1 - t) * 1000; t = t1
        pred2d = probs.argmax(0)
        pix_labels = pred2d[py, px]
        labels = self.knn(proj_range, depth, pred2d, px, py) if self.knn is not None else pix_labels
        conf = probs[labels, py, px]
        if timing is not None:
            t1 = self._sync(); timing["salsa_post"] = (t1 - t) * 1000
        return labels, conf, pix_labels
