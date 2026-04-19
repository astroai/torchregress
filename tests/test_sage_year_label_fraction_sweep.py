from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


def _load_benchmark(stem: str) -> ModuleType:
    path = EXAMPLES_DIR / "benchmarks" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"_bench_{stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load benchmark module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(EXAMPLES_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _write_year_like_csv(path: Path, *, n_rows: int = 320, n_features: int = 8) -> None:
    rng = np.random.default_rng(91)
    x = rng.normal(size=(n_rows, n_features)).astype("float32")
    weights = rng.normal(size=(n_features, 1)).astype("float32")
    y = x @ weights + 0.1 * rng.normal(size=(n_rows, 1)).astype("float32")
    frame = pd.DataFrame(x, columns=[f"feat_{idx}" for idx in range(n_features)])
    frame["target"] = y[:, 0]
    frame.to_csv(path, index=False)


def test_label_fraction_split_sizes_and_shift_order(tmp_path: Path) -> None:
    year = _load_benchmark("self_agreement_realdata_year")
    dataset_path = tmp_path / "year_like.csv"
    _write_year_like_csv(dataset_path)

    cfg = year.YearRealDataConfig(
        dataset_path=str(dataset_path),
        allow_download=False,
        n_labeled=16,
        n_unlabeled=128,
        n_test=48,
        seed=123,
    )
    split_rand = year.make_year_split_label_pool_fraction(
        cfg, label_pool_percent=25.0, shift_mode="none", min_unlabeled=32
    )
    split_cov = year.make_year_split_label_pool_fraction(
        cfg, label_pool_percent=25.0, shift_mode="covariate", min_unlabeled=32
    )
    assert split_rand.x_labeled.shape[0] + split_rand.x_unlabeled.shape[0] == 320 - 48
    assert split_rand.x_labeled.shape[0] == split_cov.x_labeled.shape[0]


def test_label_fraction_sweep_smoke(tmp_path: Path) -> None:
    sweep = _load_benchmark("sage_year_label_fraction_sweep")
    year = _load_benchmark("self_agreement_realdata_year")
    dataset_path = tmp_path / "year_like.csv"
    _write_year_like_csv(dataset_path)

    out_csv = tmp_path / "sweep.csv"
    base_cfg = year.YearRealDataConfig(
        dataset_path=str(dataset_path),
        allow_download=False,
        n_labeled=16,
        n_unlabeled=128,
        n_test=48,
        hidden=24,
        teacher_epochs=1,
        student_epochs=1,
        batch_size=64,
        unlabeled_fractions=(1.0,),
        seed=7,
    )
    rows = sweep.run_sweep(
        base_cfg,
        seeds=[7],
        label_percents=[10.0, 40.0],
        shift_modes=["none"],
        min_unlabeled=32,
        catboost_iterations=0,
        out_csv=out_csv,
        summary_json=tmp_path / "summary.json",
    )
    assert out_csv.exists()
    assert rows
    with out_csv.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
    assert "LabelPoolPercent_requested" in fieldnames
    assert "N_labeled" in fieldnames
    methods = {str(r["Method"]) for r in rows}
    assert "SAGE-Reg" in methods


def test_run_benchmark_on_split_matches_run_benchmark(tmp_path: Path) -> None:
    year = _load_benchmark("self_agreement_realdata_year")
    dataset_path = tmp_path / "year_like.csv"
    _write_year_like_csv(dataset_path, n_rows=400)

    cfg = year.YearRealDataConfig(
        dataset_path=str(dataset_path),
        allow_download=False,
        n_labeled=24,
        n_unlabeled=120,
        n_test=40,
        hidden=16,
        teacher_epochs=1,
        student_epochs=1,
        batch_size=40,
        unlabeled_fractions=(1.0,),
        seed=55,
    )
    split = year._make_split(cfg)
    a = year.run_benchmark_on_split(cfg, split)
    b = year.run_benchmark(cfg)
    assert len(a) == len(b)
    for ra, rb in zip(a, b, strict=True):
        assert ra["Method"] == rb["Method"]
        assert pytest.approx(float(ra["RMSE"]), rel=0, abs=1e-5) == float(rb["RMSE"])
