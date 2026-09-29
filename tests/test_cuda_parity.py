"""CUDA Grid Engine vs NumPy reference (skipped without a GPU and the CUDA toolkit)."""

import os

import numpy as np
import pytest
import torch

from as25lm import config as C
from as25lm.grid.reference import run_reference, to_structured

try:
    from torch.utils.cpp_extension import CUDA_HOME
except Exception:  # pragma: no cover
    CUDA_HOME = None

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or CUDA_HOME is None,
                                reason="needs a CUDA GPU and the CUDA toolkit (nvcc)")


def test_bit_identical(scene):
    from as25lm.grid.cuda_engine import GridEngineCUDA
    raw, labels, conf = scene
    eng = GridEngineCUDA(build_dir=os.environ.get("AS25LM_BUILD_DIR"))    # reuse the scripts' build if set
    res = eng.run(torch.from_numpy(raw[:, :3]).cuda(), torch.from_numpy(labels).cuda(), torch.from_numpy(conf).cuda())
    m_ref, st_ref, _ = run_reference(raw[:, :3], labels, conf)
    assert res.stats == st_ref
    assert np.array_equal(res.matrix_numpy(), m_ref)
    gpu, ref = res.to_numpy(), to_structured(m_ref)
    assert all(np.array_equal(gpu[n], ref[n]) for n, _ in C.FINAL_FIELDS)
