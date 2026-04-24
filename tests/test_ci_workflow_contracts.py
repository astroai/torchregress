from __future__ import annotations

from pathlib import Path

CI_WORKFLOW = Path(".github/workflows/ci.yml")


def test_ci_workflow_uses_uv_and_pre_commit() -> None:
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    assert "astral-sh/setup-uv" in text
    assert "pre-commit/action" in text
    assert "uv run pytest" in text


def test_ci_workflow_has_single_test_job() -> None:
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    assert "jobs:" in text
    assert "lint-test:" in text
