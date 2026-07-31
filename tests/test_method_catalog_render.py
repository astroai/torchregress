from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools import render_method_catalog

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATED_MD = REPO_ROOT / "docs" / "reports" / "method_catalog_generated.md"
GENERATED_JSON = REPO_ROOT / "reports" / "method_catalog_latest.json"
METHOD_MATRIX_MD = REPO_ROOT / "docs" / "guide" / "method-selection.md"
EVIDENCE_MD = REPO_ROOT / "docs" / "reports" / "comparative_evidence_matrix.md"
EVIDENCE_JSON = REPO_ROOT / "reports" / "comparative_evidence_matrix_latest.json"


def test_render_markdown_contains_peer_methods_and_no_experimental_default() -> None:
    rows = render_method_catalog.load_rows()
    md = render_method_catalog.render_markdown(rows)

    assert "# Generated Method Catalog Snapshot" in md
    assert "`SWAG`" in md
    assert "`BayesianNeuralNetwork`" in md
    assert "`MDNLoss`" in md
    assert "Experimental" not in md


def test_render_method_matrix_generated_section_contains_hard_task_shortlists() -> None:
    rows = render_method_catalog.load_rows()
    section = render_method_catalog.render_method_matrix_generated_section(rows)

    assert "### Catalog-Backed Peer Method Snapshot (Generated)" in section
    assert "### Generated Family Capability Matrix" in section
    assert "### Generated Hard-Task Shortlists" in section
    assert "OOD + epistemic signals" in section
    assert "`swag`" in section
    assert "`SWAG`" in section
    assert "`MDNLoss`" in section
    assert "`HeteroscedasticBatchEnsembleModel`" in section


def test_render_task_matrix_table_contains_peer_rows() -> None:
    rows = render_method_catalog.load_task_recommendations()
    table = render_method_catalog.render_task_matrix_table(rows)

    assert "| Task / Constraint | Recommended Start | Strong Alternatives | Notes |" in table
    assert "_Generated provenance_:" in table
    assert "_Source artifacts_:" in table
    assert "_Generated date_:" in table
    assert "OOD scoring / selective prediction" in table
    assert "`HeteroscedasticBatchEnsembleModel + OOD metrics`" in table
    assert "`SWAG + OOD metrics`" in table
    assert "`BayesianNeuralNetwork + OOD metrics`" in table
    assert "Calibrated intervals with coverage guarantees" in table


def test_render_decision_workflow_and_evidence_markdown() -> None:
    workflow = render_method_catalog.load_decision_workflow_steps()
    workflow_md = render_method_catalog.render_decision_workflow(workflow)
    assert "_Generated provenance_:" in workflow_md
    assert "Need coverage guarantees?" in workflow_md
    assert "`ConformalLoss (split / CQR / ACI patterns)`" in workflow_md
    assert "Need OOD scoring / selective prediction under a latency budget?" in workflow_md
    assert "`HeteroscedasticBatchEnsembleModel + OOD metrics`" in workflow_md
    assert "`SWAG + OOD metrics`" in workflow_md

    evidence_rows = render_method_catalog.load_comparative_evidence_rows()
    evidence_md = render_method_catalog.render_comparative_evidence_markdown(evidence_rows)
    assert "# Comparative Evidence Matrix" in evidence_md
    assert "_Generated provenance_:" in evidence_md
    assert "OOD robustness / selective prediction" in evidence_md
    assert "Noisy labels / corruption" in evidence_md
    assert "`Strong`" in evidence_md
    grades = {row["comparison_grade"] for row in evidence_rows}
    assert "Missing" not in grades
    assert "Demo-only" not in grades


def test_build_report_contains_family_and_maturity_counts() -> None:
    rows = render_method_catalog.load_rows()
    report = render_method_catalog.build_report(rows)

    assert report["artifact"] == "method_catalog"
    assert report["summary"]["total_methods"] >= 10
    assert report["summary"]["families"]["swag"] >= 1
    assert report["summary"]["peer_uq_methods_present"]["SWAG"] is True
    assert report["summary"]["peer_uq_methods_present"]["BayesianNeuralNetwork"] is True
    assert report["summary"]["peer_uq_methods_present"]["MDNLoss"] is True


def test_committed_method_catalog_artifacts_are_in_sync() -> None:
    if not GENERATED_MD.exists() or not GENERATED_JSON.exists():
        pytest.skip("Method catalog artifacts (ignored) are missing")

    rows = render_method_catalog.load_rows()
    expected_md = render_method_catalog.render_markdown(rows)
    expected_json = render_method_catalog.build_report(rows)

    assert GENERATED_MD.read_text(encoding="utf-8") == expected_md
    actual_json = json.loads(GENERATED_JSON.read_text(encoding="utf-8"))
    assert actual_json == expected_json


def test_method_selection_matrix_generated_section_is_in_sync() -> None:
    rows = render_method_catalog.load_rows()
    text = METHOD_MATRIX_MD.read_text(encoding="utf-8")
    expected = render_method_catalog._replace_marked_section(  # noqa: SLF001
        text,
        render_method_catalog.render_task_matrix_table(
            render_method_catalog.load_task_recommendations()
        ),
        start_marker=render_method_catalog.TASK_MATRIX_SECTION_START,  # noqa: SLF001
        end_marker=render_method_catalog.TASK_MATRIX_SECTION_END,  # noqa: SLF001
    )
    expected = render_method_catalog._replace_marked_section(  # noqa: SLF001
        expected,
        render_method_catalog.render_family_capability_matrix_table(rows),
        start_marker=render_method_catalog.FAMILY_MATRIX_SECTION_START,  # noqa: SLF001
        end_marker=render_method_catalog.FAMILY_MATRIX_SECTION_END,  # noqa: SLF001
    )
    expected = render_method_catalog._replace_marked_section(  # noqa: SLF001
        expected,
        render_method_catalog.render_decision_workflow(
            render_method_catalog.load_decision_workflow_steps()
        ),
        start_marker=render_method_catalog.DECISION_WORKFLOW_SECTION_START,  # noqa: SLF001
        end_marker=render_method_catalog.DECISION_WORKFLOW_SECTION_END,  # noqa: SLF001
    )
    expected = render_method_catalog._replace_marked_section(  # noqa: SLF001
        expected,
        render_method_catalog.render_method_matrix_generated_section(rows),
    )
    assert text == expected


def test_comparative_evidence_artifacts_are_in_sync() -> None:
    if not EVIDENCE_MD.exists() or not EVIDENCE_JSON.exists():
        pytest.skip("Comparative evidence artifacts (ignored) are missing")

    rows = render_method_catalog.load_comparative_evidence_rows()
    expected_md = render_method_catalog.render_comparative_evidence_markdown(rows)
    expected_json = render_method_catalog.build_comparative_evidence_report(rows)

    assert EVIDENCE_MD.read_text(encoding="utf-8") == expected_md
    actual_json = json.loads(EVIDENCE_JSON.read_text(encoding="utf-8"))
    assert actual_json == expected_json
