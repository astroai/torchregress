from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


def _load_benchmark_module(stem: str) -> ModuleType:
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


def test_self_agreement_synthetic_benchmark_smoke(tmp_path: Path) -> None:
    mod = _load_benchmark_module("self_agreement_synthetic")
    csv_path = tmp_path / "stage1.csv"
    perf_path = tmp_path / "stage1_perf.png"
    calib_path = tmp_path / "stage1_calib.png"
    diag_path = tmp_path / "stage1_diag.png"
    summary_path = tmp_path / "stage1.json"
    cfg = mod.SelfAgreementSyntheticConfig(
        data=mod.SyntheticRegressionGeneratorConfig(
            n_labeled=24,
            n_unlabeled=64,
            n_test=40,
            multimodal_prob=0.25,
            imbalance_strength=0.35,
            input_noise_std=0.05,
        ),
        hidden=12,
        teacher_epochs=2,
        student_epochs=2,
        batch_size=16,
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
    assert csv_path.exists()
    assert perf_path.exists()
    assert calib_path.exists()
    assert diag_path.exists()
    assert summary_path.exists()
    assert rows
    methods = {str(row["Method"]) for row in rows}
    assert methods == {
        "SupervisedOnly",
        "PointPseudoLabel",
        "ConfidenceWeightedPseudoLabel",
        "SAGE-Reg",
        "SAGE-Reg (No Weighting)",
        "SAGE-Reg (Single View)",
        "SAGE-Reg (No EMA)",
    }
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["artifact"] == "comparison_example_summary"
    assert "self-agreement synthetic" in payload["task"].lower()
    first = payload["rows"][0]
    for key in ("RMSE", "NLL", "CRPS", "Cov90", "Width90", "CalibMAE", "UnlabeledFraction"):
        assert key in first
