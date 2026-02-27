from __future__ import annotations

import json
from pathlib import Path

from tools import example_summary_thresholds


def _write_payload(path: Path, *, methods: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "artifact": "comparison_example_summary",
                "version": 1,
                "example": "examples/dummy.py",
                "task": "dummy task",
                "config": {"epochs": 10},
                "rows": methods,
                "notes": [],
            }
        ),
        encoding="utf-8",
    )


def test_derive_and_evaluate_thresholds_fixture() -> None:
    base_dir = Path("reports/example_summaries")
    if not base_dir.exists():
        import pytest
        pytest.skip("reports/example_summaries missing")

    thresholds = example_summary_thresholds.derive_thresholds_from_artifacts(
        base_dir,
        profile="full",
    )
    assert thresholds["artifact"] == "example_summary_thresholds"
    assert thresholds["target_profile"] == "full"
    assert thresholds["threshold_profile"] == "ci_conservative"
    assert thresholds["n_artifacts"] >= 8
    assert thresholds["n_limits"] > 0

    verdict = example_summary_thresholds.evaluate_artifacts_against_thresholds(
        base_dir,
        profile="full",
        thresholds=thresholds,
    )
    assert verdict["ok"] is True
    assert verdict["failed_limits"] == 0
    assert verdict["missing_limits"] == 0


def test_threshold_evaluation_detects_regression(tmp_path: Path) -> None:
    _write_payload(
        tmp_path / "toy_full.json",
        methods=[{"Method": "A", "MSE": 1.0, "ConformalCov90": 0.9, "train_s": 0.2}],
    )
    thresholds = example_summary_thresholds.derive_thresholds_from_artifacts(
        tmp_path,
        profile="full",
    )
    assert thresholds["n_limits"] == 3

    _write_payload(
        tmp_path / "toy_full.json",
        methods=[{"Method": "A", "MSE": 100.0, "ConformalCov90": 1.5, "train_s": 9.0}],
    )
    verdict = example_summary_thresholds.evaluate_artifacts_against_thresholds(
        tmp_path,
        profile="full",
        thresholds=thresholds,
    )
    assert verdict["ok"] is False
    assert verdict["failed_limits"] >= 2


def test_threshold_derivation_ignores_non_summary_full_json(tmp_path: Path) -> None:
    _write_payload(
        tmp_path / "toy_full.json",
        methods=[{"Method": "A", "MSE": 1.0}],
    )
    # Matches *_full.json naming but is not a comparison summary artifact.
    (tmp_path / "thresholds_full.json").write_text(
        json.dumps({"artifact": "example_summary_thresholds", "limits": {}}),
        encoding="utf-8",
    )
    thresholds = example_summary_thresholds.derive_thresholds_from_artifacts(
        tmp_path,
        profile="full",
    )
    assert thresholds["n_artifacts"] == 1
    assert thresholds["n_limits"] == 1


def test_committed_example_summary_threshold_baseline_schema_and_pass() -> None:
    thresholds_path = Path("reports/example_summaries/thresholds_full.json")
    if not thresholds_path.exists():
        import pytest
        pytest.skip("thresholds_full.json missing")

    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    assert thresholds["artifact"] == "example_summary_thresholds"
    assert thresholds["target_profile"] == "full"
    assert thresholds["threshold_profile"] in {"ci_conservative", "review_strict"}
    assert isinstance(thresholds["limits"], dict) and thresholds["limits"]

    verdict = example_summary_thresholds.evaluate_artifacts_against_thresholds(
        Path("reports/example_summaries"),
        profile="full",
        thresholds=thresholds,
    )
    assert verdict["ok"] is True
