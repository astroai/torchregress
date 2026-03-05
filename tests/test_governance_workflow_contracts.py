from __future__ import annotations

from pathlib import Path

WORKFLOW = Path(".github/workflows/governance-refresh.yml")


def test_governance_workflow_exists_and_is_schedule_or_dispatch_only() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "name: Governance Refresh" in text
    assert "workflow_dispatch:" in text
    assert "schedule:" in text
    assert "pull_request:" not in text
    assert "push:" not in text


def test_governance_workflow_runs_expected_governance_commands() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    required_snippets = [
        "tools/render_example_summaries.py --profile smoke",
        "tools/render_example_summaries.py --profile audit",
        "tools/compare_example_summary_profiles.py",
        "tools/example_summary_thresholds.py",
        "python -m tools.benchmark_smoke",
        "python -m tools.benchmark_report_summary",
        "tools/render_method_catalog.py",
        "python -m tools.render_realdata_recommendation_guide",
        "tools/adoption_audit.py",
        "tools/render_review_packet.py",
        "governance-refresh-artifacts",
    ]
    for snippet in required_snippets:
        assert snippet in text
