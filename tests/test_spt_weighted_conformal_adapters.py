"""Unit tests for covariate-weighted split conformal helpers (SPT benchmarks)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_SPT_DIR = REPO_ROOT.parent / "torchregress-research" / "benchmarks" / "neurips" / "spt"


def _load_spt_synthetic() -> ModuleType:
    path = RESEARCH_SPT_DIR / "spt_reg_synthetic_comparison.py"
    if not path.is_file():
        pytest.skip("torchregress-research SPT benchmarks not checked out beside torchregress")
    spec = importlib.util.spec_from_file_location("_spt_synth_weighted", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    sys.path.insert(0, str(RESEARCH_SPT_DIR))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod


def test_covariate_density_ratio_weights_shape_and_positive() -> None:
    pytest.importorskip("sklearn")
    mod = _load_spt_synthetic()
    rng = np.random.default_rng(0)
    xs = rng.standard_normal((50, 4))
    xt = rng.standard_normal((60, 4)) + 0.5
    xq = rng.standard_normal((12, 4))
    w = mod._covariate_density_ratio_weights(xs, xt, xq, seed=1)
    assert w.shape == (12,)
    assert np.all(w > 0)


def test_weighted_split_conformal_produces_intervals() -> None:
    mod = _load_spt_synthetic()
    from torchregress.prediction import PredictiveBatch

    n = 20
    mean = np.zeros(n, dtype=np.float32)
    std = np.ones(n, dtype=np.float32)
    batch_cal = PredictiveBatch(mean=mean, std=std)
    batch_test = PredictiveBatch(mean=mean[:10], std=std[:10])
    y_cal = np.zeros(n, dtype=np.float32)
    w = np.ones(n, dtype=np.float64)
    out = mod._weighted_split_conformal(batch_cal, y_cal, batch_test, 0.1, w)
    assert out.extra is not None
    assert "interval_lower" in out.extra
    assert "interval_upper" in out.extra
    assert out.extra.get("conformal_method") == "weighted_split"
