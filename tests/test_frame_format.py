"""LGF1 frame format: round trip, alignment and quantisation limits."""

import gzip

import numpy as np

from as25lm import frame_format as FF
from as25lm.grid.reference import run_reference, to_structured


def _frame(scene, stride=1):
    raw, labels, conf = scene
    cells = to_structured(run_reference(raw[:, :3], labels, conf)[0])
    header = FF.frame_header("000042", "precomputed", len(raw), stride, cells, sequence="08")
    sections = FF.build_sections(raw[:, :3], raw[:, 3], labels, conf, cells, points_gt=labels, point_stride=stride)
    return raw, cells, header, sections


def test_round_trip_exact(scene):
    _, _, header, sections = _frame(scene)
    for data in (FF.encode_frame(header, sections), gzip.compress(FF.encode_frame(header, sections))):
        h, a = FF.decode_frame(data)
        assert h["frame_id"] == "000042" and h["format"] == "LGF1"
        for k, v in sections.items():
            assert np.array_equal(a[k], v), k


def test_sections_aligned(scene):
    _, _, header, sections = _frame(scene)
    h, _ = FF.decode_frame(FF.encode_frame(header, sections))
    assert all(s["offset"] % 8 == 0 for s in h["sections"])


def test_quantisation_limits(scene):
    raw, cells, header, sections = _frame(scene)
    _, a = FF.decode_frame(FF.encode_frame(header, sections))
    assert np.abs(a["points.xyz_cm"] / 100.0 - raw[:, :3]).max() <= 0.005 + 1e-6
    assert np.abs(a["cells.z_cm"][:, 1] / 100.0 - cells["max_height"]).max() <= 0.005 + 1e-6
    assert np.abs(a["cells.traversability"] / 255.0 - cells["traversability"]).max() <= 0.5 / 255 + 1e-9
    size = np.array(FF.frame_header("x", "live", 0, 1, cells)["level_sizes_m"])[a["cells.level"]]
    assert np.allclose(a["cells.ixy"][:, 0] * size, cells["x_min"])


def test_point_stride(scene):
    raw, _, _, sections = _frame(scene, stride=4)
    assert len(sections["points.class"]) == len(raw[::4])
