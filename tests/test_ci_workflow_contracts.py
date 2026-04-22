from __future__ import annotations

from pathlib import Path

CI_WORKFLOW = Path(".github/workflows/ci.yml")


def test_ci_workflow_uses_pixi_and_pre_commit() -> None:
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    assert "prefix-dev/setup-pixi" in text
    assert "pre-commit/action" in text
    assert "pixi run test" in text


def test_ci_workflow_has_single_test_job() -> None:
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    assert "jobs:" in text
    assert "lint-test:" in text
