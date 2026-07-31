from __future__ import annotations

from pathlib import Path

import pytest
from tools import render_realdata_recommendation_guide

REPO_ROOT = Path(__file__).resolve().parents[1]
GUIDE_DOC = REPO_ROOT / "docs" / "reports" / "real_data_recommendation_guide.md"
COMPARATIVE_JSON = REPO_ROOT / "reports" / "comparative_evidence_matrix_latest.json"


def test_evidence_band_classification_split() -> None:
    synthetic_row = {
        "examples": ["examples/noisy_label_comparison.py"],
        "comparison_grade": "Strong",
        "gaps": "Needs more real data.",
    }
    real_proxy_row = {
        "examples": ["examples/ood_selective_prediction_realdata_comparison.py"],
        "comparison_grade": "Strong",
        "gaps": "Needs additional external datasets.",
    }
    decision_grade_row = {
        "examples": ["examples/eiv_method_realdata_comparison.py"],
        "comparison_grade": "Decision-grade",
        "gaps": "",
    }

    assert render_realdata_recommendation_guide._evidence_band(synthetic_row) == "Synthetic only"
    assert render_realdata_recommendation_guide._evidence_band(real_proxy_row) == "Real proxy"
    assert (
        render_realdata_recommendation_guide._evidence_band(decision_grade_row)
        == "Decision-grade real-data"
    )


def test_render_generated_section_contains_required_headers() -> None:
    if not COMPARATIVE_JSON.exists():
        pytest.skip("Comparative evidence artifact (ignored) is missing")

    payload = render_realdata_recommendation_guide._load_json(COMPARATIVE_JSON)
    section = render_realdata_recommendation_guide.render_generated_section(
        payload,
        source_json_path=COMPARATIVE_JSON,
    )

    assert "## Evidence Band Summary" in section
    assert "## Claim Policy" in section
    assert "## Task-to-Method Recommendations" in section
    assert "| Task | Start Methods | Evidence Band | Claim Boundary | Next Data Step |" in section


def test_committed_realdata_guide_is_in_sync() -> None:
    if not GUIDE_DOC.exists() or not COMPARATIVE_JSON.exists():
        pytest.skip("Guide or comparative evidence artifact (ignored) is missing")

    text = GUIDE_DOC.read_text(encoding="utf-8")
    payload = render_realdata_recommendation_guide._load_json(COMPARATIVE_JSON)

    # Extract date from existing doc to avoid timezone/date-flip issues in CI
    import re

    date_match = re.search(r"_Generated date_: `(\d{4}-\d{2}-\d{2})`", text)
    committed_date = date_match.group(1) if date_match else None

    expected = render_realdata_recommendation_guide.render_recommendation_guide(
        text,
        comparative_payload=payload,
        source_json_path=COMPARATIVE_JSON,
        generated_date=committed_date,
    )
    assert text == expected
