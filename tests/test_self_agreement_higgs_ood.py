from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

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


def test_self_agreement_higgs_ood_smoke(tmp_path: Path) -> None:
    mod = _load_module("self_agreement_higgs_ood")
    csv_path = tmp_path / "higgs_ood.csv"
    perf_path = tmp_path / "higgs_ood_perf.png"
    calib_path = tmp_path / "higgs_ood_calib.png"
    summary_path = tmp_path / "higgs_ood_summary.json"

    cfg = mod.HiggsOODConfig(
        n_train=64,
        n_unlabeled_id=96,
        n_unlabeled_ood=96,
        n_id_test=48,
        n_ood_test=48,
        proxy_dim=12,
        hidden=24,
        teacher_epochs=2,
        student_epochs=2,
        batch_size=32,
    )
    rows = mod.main(
        cfg,
        output_csv=str(csv_path),
        performance_figure_path=str(perf_path),
        calibration_figure_path=str(calib_path),
        summary_json_path=str(summary_path),
    )
    assert rows
    assert csv_path.exists()
    assert perf_path.exists()
    assert calib_path.exists()
    assert summary_path.exists()
    assert len(rows) == 4

    methods = {str(row["Method"]) for row in rows}
    assert methods == {
        "SupervisedOnly",
        "MeanTeacher",
        "ConfidenceWeightedPseudoLabel",
        "SAGE-Reg",
    }

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["artifact"] == "comparison_example_summary"
    assert "higgs" in payload["task"].lower()
    first = payload["rows"][0]
    for key in (
        "Dataset",
        "RMSE_ID",
        "RMSE_OOD",
        "NLL_ID",
        "NLL_OOD",
        "Cov90_ID",
        "Cov90_OOD",
        "OODUncGap",
        "MeanWeightID",
        "MeanWeightOOD",
        "MeanDisagreementID",
        "MeanDisagreementOOD",
    ):
        assert key in first


def test_self_agreement_higgs_ood_local_parquet_sampling(tmp_path: Path) -> None:
    pytest.importorskip("polars")
    import pandas as pd

    mod = _load_module("self_agreement_higgs_ood")
    parquet_path = tmp_path / "higgs_local.parquet"

    n_rows = 512
    frame = pd.DataFrame(
        {
            "feature_0": [float(i % 17) for i in range(n_rows)],
            "PRI_met": [float(i % 29) for i in range(n_rows)],
            "weights": [0.1 + 0.001 * i for i in range(n_rows)],
            "detailed_labels": ["ztautau"] * n_rows,
            "labels": [float(i % 2) for i in range(n_rows)],
        }
    )
    frame.to_parquet(parquet_path, index=False)

    cfg = mod.HiggsOODConfig(
        dataset_path=str(parquet_path),
        target_column="labels",
        ood_score_column="PRI_met",
        drop_columns=("weights", "detailed_labels"),
        n_train=64,
        n_unlabeled_id=96,
        n_unlabeled_ood=96,
        n_id_test=48,
        n_ood_test=48,
        parquet_sample_factor=2,
        parquet_max_sample_rows=300,
        hidden=24,
        teacher_epochs=2,
        student_epochs=2,
        batch_size=32,
    )
    rows = mod.run_benchmark(cfg)
    assert rows
    assert len(rows) == 4


def test_higgs_scale_split_sizes_and_budget() -> None:
    mod = _load_module("self_agreement_higgs_ood")
    base = mod.HiggsOODConfig()
    scaled = mod.higgs_scale_split_sizes(base, 10)
    assert scaled.n_train == base.n_train * 10
    assert scaled.n_unlabeled_id == base.n_unlabeled_id * 10
    assert mod.higgs_split_row_budget(scaled) == mod.higgs_split_row_budget(base) * 10
    assert mod.higgs_scale_split_sizes(base, 1) is base


def test_higgs_parquet_full_read_row_guard(tmp_path: Path) -> None:
    pytest.importorskip("polars")
    import pandas as pd

    mod = _load_module("self_agreement_higgs_ood")
    parquet_path = tmp_path / "tiny.parquet"
    n_rows = 12
    pd.DataFrame(
        {
            "f0": [float(i) for i in range(n_rows)],
            "PRI_met": [float(i % 5) for i in range(n_rows)],
            "labels": [float(i % 2) for i in range(n_rows)],
        }
    ).to_parquet(parquet_path, index=False)

    cfg = mod.HiggsOODConfig(
        dataset_path=str(parquet_path),
        target_column="labels",
        ood_score_column="PRI_met",
        n_train=2,
        n_unlabeled_id=2,
        n_unlabeled_ood=2,
        n_id_test=2,
        n_ood_test=2,
        parquet_max_sample_rows=500,
        parquet_sample_factor=2,
        parquet_full_read_row_limit=8,
    )
    with pytest.raises(ValueError, match="Refusing to read all"):
        mod._load_local_frame(cfg)


def test_self_agreement_higgs_ood_repeatable_metrics() -> None:
    mod = _load_module("self_agreement_higgs_ood")
    cfg = mod.HiggsOODConfig(
        n_train=64,
        n_unlabeled_id=96,
        n_unlabeled_ood=96,
        n_id_test=48,
        n_ood_test=48,
        proxy_dim=12,
        hidden=24,
        teacher_epochs=2,
        student_epochs=2,
        batch_size=32,
    )
    rows_a = mod.run_benchmark(cfg)
    rows_b = mod.run_benchmark(cfg)

    def _metrics(rows):
        return {
            row["Method"]: (
                row["RMSE_ID"],
                row["RMSE_OOD"],
                row["NLL_ID"],
                row["NLL_OOD"],
                row["Cov90_ID"],
                row["Cov90_OOD"],
                row["OODUncGap"],
                row["MeanWeightID"],
                row["MeanWeightOOD"],
                row["MeanDisagreementID"],
                row["MeanDisagreementOOD"],
            )
            for row in rows
        }

    assert _metrics(rows_a) == _metrics(rows_b)
