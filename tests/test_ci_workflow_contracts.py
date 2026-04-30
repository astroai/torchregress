from __future__ import annotations

from pathlib import Path

CI_WORKFLOW = Path(".github/workflows/ci.yml")


def test_ci_workflow_uses_uv_and_pre_commit() -> None:
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    assert "astral-sh/setup-uv" in text
    assert "pip install" in text and "pre-commit" in text
    assert "pre-commit run" in text
    assert "uv run pytest" in text


def test_ci_workflow_has_core_jobs() -> None:
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    assert "jobs:" in text
    assert "lint-test:" in text
    assert "uv run pytest" in text
    assert "tools.benchmark_smoke" in text
    assert "benchmark-smoke:" not in text
