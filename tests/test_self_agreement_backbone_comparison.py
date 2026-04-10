from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

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


def test_self_agreement_backbone_comparison_smoke(tmp_path: Path) -> None:
    mod = _load_module("self_agreement_backbone_comparison")
    probe = mod.QuantileModel(hidden=8, dropout=0.0)
    quantiles = probe(mod.torch.randn(5, 1))
    assert mod.torch.all(quantiles[:, 1:] >= quantiles[:, :-1])

    csv_path = tmp_path / "backbone_rows.csv"
    performance_figure = tmp_path / "backbone_performance.png"
    calibration_figure = tmp_path / "backbone_calibration.png"
    summary_path = tmp_path / "backbone_summary.json"
    cfg = mod.BackboneComparisonConfig(
        data=mod.SyntheticRegressionGeneratorConfig(
            n_labeled=24,
            n_unlabeled=48,
            n_test=32,
            multimodal_prob=0.2,
            imbalance_strength=0.25,
            input_noise_std=0.04,
        ),
        hidden=12,
        epochs=2,
        batch_size=16,
        n_bins=10,
        n_views=3,
        unlabeled_fractions=(0.5, 1.0),
    )
    rows = mod.main(
        cfg,
        output_csv=str(csv_path),
        performance_figure_path=str(performance_figure),
        calibration_figure_path=str(calibration_figure),
        summary_json_path=str(summary_path),
    )
    assert csv_path.exists()
    assert performance_figure.exists()
    assert calibration_figure.exists()
    assert summary_path.exists()
    assert rows
    assert len(rows) == 12
    methods = {str(row["Method"]) for row in rows}
    assert methods == {
        "Gaussian Supervised",
        "Gaussian SAGE-Reg",
        "Quantile Supervised",
        "Quantile SAGE-Reg",
        "Bar Supervised",
        "Bar SAGE-Reg",
    }
    fractions = {float(row["UnlabeledFraction"]) for row in rows}
    assert fractions == {0.5, 1.0}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["artifact"] == "comparison_example_summary"
    assert "backbone comparison" in payload["task"].lower()
    first = payload["rows"][0]
    for key in (
        "Backbone",
        "Regime",
        "UnlabeledFraction",
        "RMSE",
        "NLL",
        "CRPS",
        "Cov90",
        "CoverageGap90",
        "Width90",
        "PITChi2",
        "MeanWeight",
        "MeanDisagreement",
    ):
        assert key in first
