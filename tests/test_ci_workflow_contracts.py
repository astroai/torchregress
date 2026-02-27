from __future__ import annotations

from pathlib import Path

CI_WORKFLOW = Path(".github/workflows/ci.yml")


def _slice_job(text: str, job_name: str, next_job_name: str | None = None) -> str:
    start_token = f"  {job_name}:"
    start = text.find(start_token)
    assert start != -1, f"Missing CI job: {job_name}"
    if next_job_name is None:
        return text[start:]
    end_token = f"  {next_job_name}:"
    end = text.find(end_token, start + len(start_token))
    assert end != -1, f"Missing next CI job: {next_job_name}"
    return text[start:end]


def test_review_packet_artifact_is_rendered_in_always_on_cpu_benchmark_job() -> None:
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    cpu_job = _slice_job(text, "benchmark-thresholds-cpu", "flow-examples-optional")

    assert "Run benchmark smoke threshold check (CPU)" in cpu_job
    assert "Run benchmark sweep threshold check (CPU)" in cpu_job
    assert "Render review readiness packet" in cpu_job
    assert "review-readiness-packet" in cpu_job
    assert "python tools/render_review_packet.py" in cpu_job


def test_review_packet_artifact_is_not_only_in_optional_flow_job() -> None:
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    flow_job = _slice_job(text, "flow-examples-optional")

    # Review packet artifact publication should be available on normal PR/push CI too.
    assert "review-readiness-packet" not in flow_job
    assert "Render review readiness packet" not in flow_job


def test_review_packet_artifact_name_is_unique_in_ci_workflow() -> None:
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    assert text.count("review-readiness-packet") == 1
