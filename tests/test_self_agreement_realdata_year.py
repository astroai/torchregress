from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


def _load_module(stem: str) -> ModuleType:
    path = EXAMPLES_DIR / "benchmarks" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"_benchmark_{stem}", path)
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


def _write_year_like_csv(path: Path, *, n_rows: int = 320, n_features: int = 12) -> None:
    rng = np.random.default_rng(17)
    x = rng.normal(size=(n_rows, n_features)).astype("float32")
    weights = rng.normal(size=(n_features, 1)).astype("float32")
    y = x @ weights + 0.3 * np.sin(x[:, :1]) + 0.1 * rng.normal(size=(n_rows, 1)).astype("float32")
    frame = pd.DataFrame(x, columns=[f"feat_{idx}" for idx in range(n_features)])
    frame["target"] = y[:, 0]
    frame.to_csv(path, index=False)


def test_self_agreement_realdata_year_smoke(tmp_path: Path) -> None:
    mod = _load_module("self_agreement_realdata_year")
    dataset_path = tmp_path / "year_like.csv"
    _write_year_like_csv(dataset_path)

    csv_path = tmp_path / "realdata_rows.csv"
    perf_path = tmp_path / "realdata_perf.png"
    calib_path = tmp_path / "realdata_calib.png"
    diag_path = tmp_path / "realdata_diag.png"
    summary_path = tmp_path / "realdata_summary.json"

    cfg = mod.YearRealDataConfig(
        dataset_path=str(dataset_path),
        allow_download=False,
        n_labeled=32,
        n_unlabeled=96,
        n_test=48,
        hidden=24,
        teacher_epochs=2,
        student_epochs=2,
        batch_size=32,
        unlabeled_fractions=(0.5, 1.0),
    )
    rows = mod.main(
        cfg,
        output_csv=str(csv_path),
        performance_figure_path=str(perf_path),
        calibration_figure_path=str(calib_path),
        diagnostic_figure_path=str(diag_path),
        summary_json_path=str(summary_path),
    )
    assert rows
    assert csv_path.exists()
    assert perf_path.exists()
    assert calib_path.exists()
    assert diag_path.exists()
    assert summary_path.exists()
    assert len(rows) == 14

    methods = {str(row["Method"]) for row in rows}
    assert methods == {
        "SupervisedOnly",
        "MeanTeacher",
        "PiModelConsistency",
        "ConfidenceWeightedPseudoLabel",
        "RankUp",
        "PabLOPseudo",
        "SAGE-Reg",
    }

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["artifact"] == "comparison_example_summary"
    assert "yearpredictionmsd" in payload["task"].lower()
    first = payload["rows"][0]
    assert "Seed" in first
    for key in (
        "Dataset",
        "UnlabeledFraction",
        "RMSE",
        "NLL",
        "CRPS",
        "Cov90",
        "Width90",
        "CalibMAE",
        "MeanWeight",
        "MeanDisagreement",
    ):
        assert key in first


def test_realdata_year_parquet_cache_falls_back_to_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_module("self_agreement_realdata_year")

    class _Bunch:
        def __init__(self, data: pd.DataFrame, target: pd.Series) -> None:
            self.data = data
            self.target = target

    frame = pd.DataFrame({"feat_0": [0.1, 0.2], "feat_1": [1.0, 2.0]})
    target = pd.Series([3.0, 4.0])

    def _fake_fetch_openml(*args, **kwargs):
        return _Bunch(frame, target)

    def _raise_parquet(self, path, index=False):
        raise ImportError("missing parquet engine")

    monkeypatch.setattr(mod, "fetch_openml", _fake_fetch_openml)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", _raise_parquet)

    cache_path = tmp_path / "year_cache.parquet"
    loaded, dataset_name = mod._load_dataset_frame(
        mod.YearRealDataConfig(cache_path=str(cache_path), allow_download=True)
    )

    csv_fallback = cache_path.with_suffix(".csv")
    assert csv_fallback.exists()
    assert dataset_name == str(csv_fallback)
    assert list(loaded.columns) == ["feat_0", "feat_1", "target"]


def test_self_agreement_realdata_year_repeatable_metrics(tmp_path: Path) -> None:
    mod = _load_module("self_agreement_realdata_year")
    dataset_path = tmp_path / "year_like_repeatable.csv"
    _write_year_like_csv(dataset_path, n_rows=256, n_features=8)

    cfg = mod.YearRealDataConfig(
        dataset_path=str(dataset_path),
        allow_download=False,
        n_labeled=32,
        n_unlabeled=96,
        n_test=48,
        hidden=24,
        teacher_epochs=2,
        student_epochs=2,
        batch_size=32,
        unlabeled_fractions=(1.0,),
    )
    rows_a = mod.run_benchmark(cfg)
    rows_b = mod.run_benchmark(cfg)

    def _metrics(rows):
        return {
            row["Method"]: (
                row["RMSE"],
                row["NLL"],
                row["CRPS"],
                row["Cov90"],
                row["CalibMAE"],
                row["MeanWeight"],
                row["MeanDisagreement"],
            )
            for row in rows
        }

    assert _metrics(rows_a) == _metrics(rows_b)


def test_subsample_pair_respects_seed() -> None:
    mod = _load_module("self_agreement_realdata_year")
    import torch

    x = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    y = torch.arange(10, dtype=torch.float32).reshape(10, 1)
    a_x, a_y = mod._subsample_pair(x, y, 0.5, subsample_seed=12345)
    b_x, b_y = mod._subsample_pair(x, y, 0.5, subsample_seed=12345)
    c_x, c_y = mod._subsample_pair(x, y, 0.5, subsample_seed=99999)
    assert torch.equal(a_x, b_x) and torch.equal(a_y, b_y)
    assert not torch.equal(a_x, c_x)
