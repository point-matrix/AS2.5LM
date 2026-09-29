"""Range projection and the segmentation wrapper (random weights; no checkpoint needed)."""

import argparse

import numpy as np
import pytest
import torch

from as25lm import config as C
from as25lm.segmentation import segmenter as S


@pytest.fixture(scope="module")
def seg(tmp_path_factory):
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("w") / "random.pth"
    torch.save(S.SalsaNext(C.N_CLASSES).state_dict(), path)
    return S.Segmenter(path, device="cpu")


def test_projection_keeps_nearest_point_per_pixel(seg, scene):
    raw, _, _ = scene
    pts = torch.from_numpy(raw)
    _, proj_range, px, py, depth = seg.project(pts)
    H, W = C.SENSOR["height"], C.SENSOR["width"]
    ref = np.full(H * W, np.inf)
    np.minimum.at(ref, (py * W + px).numpy(), depth.numpy())
    ref[np.isinf(ref)] = -1
    assert np.array_equal(proj_range.numpy().ravel(), ref.astype(np.float32))


def test_projection_deterministic(seg, scene):
    pts = torch.from_numpy(scene[0])
    a = seg.project(pts)[1]
    assert all(torch.equal(seg.project(pts)[1], a) for _ in range(3))


def test_empty_pixels_are_zero_in_input(seg, scene):
    proj, proj_range, *_ = seg.project(torch.from_numpy(scene[0]))
    empty = (proj_range < 0).flatten()
    assert torch.all(proj[0].reshape(5, -1)[:, empty] == 0)


def test_segment_outputs(seg, scene):
    raw = scene[0]
    timing = {}
    labels, conf, pix = seg.segment(torch.from_numpy(raw), fp16=False, timing=timing)
    assert labels.shape == conf.shape == pix.shape == (len(raw),)
    assert int(labels.min()) >= 1 and int(labels.max()) <= 19          # KNN never outputs "unlabeled"
    assert float(conf.min()) >= 0 and float(conf.max()) <= 1
    assert set(timing) == {"salsa_project", "salsa_model", "salsa_post"}


def test_model_output_is_softmax(seg, scene):
    proj = seg.project(torch.from_numpy(scene[0]))[0]
    s = seg.probabilities(proj, fp16=False).sum(0)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-4)


def test_rejects_non_weights_file_without_trust(tmp_path):
    bad = tmp_path / "ckpt.pth"
    torch.save({"state_dict": {}, "optimizer": argparse.Namespace(lr=0.05)}, bad)
    with pytest.raises(RuntimeError, match="weights-only"):
        S.load_state_dict(bad)
