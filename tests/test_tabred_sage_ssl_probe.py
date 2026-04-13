"""Smoke test for TabReD-layout loader + SAGE bundle (tiny synthetic tensors)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

pytest.importorskip("polars")

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
BENCH_DIR = EXAMPLES_DIR / "benchmarks"


def _load_tabred_probe() -> ModuleType:
    path = BENCH_DIR / "tabred_sage_ssl_probe.py"
    spec = importlib.util.spec_from_file_location("_tabred_sage_ssl_probe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    sys.path.insert(0, str(EXAMPLES_DIR))
    sys.path.insert(0, str(BENCH_DIR))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
        sys.path.pop(0)
    return mod


def _write_minimal_tabred_dataset(root: Path) -> None:
    d = root / "smoke_tabred"
    d.mkdir(parents=True)
    (d / "info.json").write_text(
        json.dumps({"name": "smoke_tabred", "task_type": "regression"}),
        encoding="utf-8",
    )
    rng = np.random.default_rng(0)
    n = 800
    x = rng.standard_normal((n, 5)).astype(np.float32)
    y = (x[:, 0] * 0.5 + rng.standard_normal(n) * 0.1).astype(np.float32).reshape(-1, 1)
    np.save(d / "X_num.npy", x)
    np.save(d / "Y.npy", y)
    sp = d / "split-default"
    sp.mkdir()
    np.save(sp / "train_idx.npy", np.arange(600, dtype=np.int64))
    np.save(sp / "val_idx.npy", np.arange(600, 640, dtype=np.int64))
    np.save(sp / "test_idx.npy", np.arange(640, 800, dtype=np.int64))


def test_tabred_sage_ssl_probe_smoke(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    _write_minimal_tabred_dataset(tmp_path)

    mod = _load_tabred_probe()
    TabRedProbeConfig = mod.TabRedProbeConfig
    run_bundle = mod.run_bundle

    pc = TabRedProbeConfig(
        tabred_data_root=str(tmp_path),
        datasets=("smoke_tabred",),
        split_name="default",
        seed=1,
        n_labeled=128,
        n_unlabeled=256,
        max_train_pool=400,
        max_test_rows=100,
        include_x_meta=False,
        teacher_epochs=1,
        student_epochs=1,
        hidden=32,
        batch_size=64,
        lr=1e-3,
        unlabeled_fractions=(1.0,),
        quick=False,
    )
    out = tmp_path / "out"
    rows, bundle_path, notes = run_bundle(pc, out)
    assert bundle_path.is_file()
    assert rows
    assert any(r["Method"] == "SAGE-Reg" for r in rows)
    assert (out / "results_long.csv").is_file()
    assert notes
