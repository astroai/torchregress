"""Schema and policy guardrails for the native PyTorch leverage matrix artifact."""

from __future__ import annotations

import json
from pathlib import Path

ALLOWED_DECISIONS = {"Keep custom", "Wrap native", "Replace with native", "Hybrid"}
REQUIRED_AREAS = {
    "standard_point_losses": "Hybrid",
    "gaussian_nll_diagonal": "Hybrid",
    "point_metrics_baseline": "Wrap native",
    "calibration_metrics_regression": "Keep custom",
    "ood_metrics": "Hybrid",
    "ensemble_decomposition": "Keep custom",
    "conformal_prediction": "Keep custom",
    "mdn_and_flows": "Hybrid",
    "eiv_losses": "Keep custom",
    "scaling_helpers": "Wrap native",
}

REPO_ROOT = Path(__file__).parents[2]
JSON_PATH = REPO_ROOT / "reports" / "native_pytorch_leverage_matrix_2026-02-26.json"
MD_PATH = REPO_ROOT / "docs" / "audits" / "native_pytorch_leverage_matrix_2026-02-26.md"


@pytest.mark.skipif(not JSON_PATH.exists(), reason="Leverage matrix report not found.")
def test_native_pytorch_leverage_matrix_schema_and_required_areas() -> None:
    path = JSON_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["artifact"] == "native_pytorch_leverage_matrix"
    assert payload["version"] == 1
    decisions = payload["decisions"]
    assert isinstance(decisions, list) and decisions

    seen: dict[str, str] = {}
    for row in decisions:
        assert isinstance(row, dict)
        for key in ("area", "surface", "native_candidate", "decision", "rationale", "action"):
            assert key in row, key
        assert isinstance(row["area"], str) and row["area"]
        assert isinstance(row["surface"], list)
        assert isinstance(row["native_candidate"], list)
        assert row["decision"] in ALLOWED_DECISIONS
        assert isinstance(row["rationale"], str) and row["rationale"].strip()
        assert isinstance(row["action"], str) and row["action"].strip()
        coverage = row.get("coverage_evidence")
        assert isinstance(coverage, dict), row["area"]
        assert isinstance(coverage.get("parity_tests"), list) and coverage["parity_tests"], row[
            "area"
        ]
        assert (
            isinstance(coverage.get("known_divergences"), list) and coverage["known_divergences"]
        ), row["area"]
        seen[row["area"]] = row["decision"]

    for area, expected_decision in REQUIRED_AREAS.items():
        assert area in seen, area
        assert seen[area] == expected_decision, (area, seen[area], expected_decision)


def test_native_strategy_guardrails_have_backing_tests_for_key_wraps() -> None:
    native_parity = Path("tests/test_native_parity.py").read_text(encoding="utf-8")
    verify_scaling = Path("tests/test_verify_scaling.py").read_text(encoding="utf-8")

    # Wrap/hybrid areas that should have concrete regression guards.
    assert "WeightedMSELoss" in native_parity
    assert "WeightedL1Loss" in native_parity
    assert "WeightedHuberLoss" in native_parity
    assert "WeightedLossWrapper" in native_parity
    assert "WeightedCrossEntropyLoss" in native_parity
    assert "WeightedNLLLoss" in native_parity
    assert "WeightedGaussianNLLLoss" in native_parity
    assert "gaussian_nll" in native_parity
    assert "mean_squared_error" in native_parity
    assert "mean_absolute_error" in native_parity
    assert "r2_score" in native_parity
    assert "rmse" in native_parity

    # Native wrapper strategy for scaling helpers should remain test-backed.
    assert "compile_model" in verify_scaling
