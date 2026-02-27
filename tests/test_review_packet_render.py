from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import render_review_packet

REPO_ROOT = Path(__file__).resolve().parents[1]
REVIEW_PACKET_MD = REPO_ROOT / "docs" / "audits" / "review_readiness_packet_2026-02-26.md"
REVIEW_PACKET_JSON = REPO_ROOT / "reports" / "review_readiness_packet_latest.json"


def test_build_review_packet_contains_expected_governance_fields() -> None:
    packet = render_review_packet.build_review_packet()

    assert packet["artifact"] == "review_readiness_packet"
    assert packet["audit_v1_status"]["closed_v1"] is True
    assert isinstance(packet["audit_v1_status"]["closed_actionables"], list)
    assert packet["audit_v1_status"]["closed_actionables"]
    assert packet["adoption_audit"]["baseline_score"] is not None
    assert packet["adoption_audit"]["provisional_score"] is not None
    if (
        packet["adoption_audit"]["baseline_score"] > 0
        and packet["adoption_audit"]["provisional_score"] > 0
    ):
        assert (
            packet["adoption_audit"]["baseline_score"]
            < packet["adoption_audit"]["provisional_score"]
        )
    assert packet["benchmark_governance"]["cpu_smoke_threshold_limits"] is not None
    assert packet["benchmark_governance"]["cpu_smoke_threshold_limits"] >= 1
    assert packet["benchmark_governance"]["cpu_sweep_threshold_limits"] is not None
    assert packet["benchmark_governance"]["cpu_sweep_threshold_limits"] >= 1
    profile_compare_ok = packet["example_summary_governance"]["profile_compare_ok"]
    if profile_compare_ok is not None:
        assert profile_compare_ok is True
    threshold_ok = packet["example_summary_governance"]["threshold_ok"]
    if threshold_ok is not None:
        assert threshold_ok is True
    assert packet["example_summary_governance"]["ci_threshold_profile"] == "ci_conservative"
    review_profile = packet["example_summary_governance"]["review_threshold_profile"]
    if review_profile is not None:
        assert review_profile == "review_strict"


def test_render_markdown_contains_review_focus_and_counts() -> None:
    packet = render_review_packet.build_review_packet()
    md = render_review_packet.render_markdown(packet)

    assert "# Review Readiness Packet" in md
    assert "## Audit v1 Status" in md
    assert "Audit v1 closed: `True`" in md
    if packet["adoption_audit"]["docs_drift_counts"].get("invalid_attr_refs") is not None:
        assert "Docs/example drift checks: `attr=0`" in md
    assert "## Review Focus Files" in md
    assert "`docs/guides/method_selection_matrix.md`" in md


def test_committed_review_packet_artifacts_are_in_sync() -> None:
    if not REVIEW_PACKET_MD.exists() or not REVIEW_PACKET_JSON.exists():
        pytest.skip("Review packet artifacts (ignored) are missing")

    packet = render_review_packet.build_review_packet()
    expected_md = render_review_packet.render_markdown(packet)

    assert REVIEW_PACKET_MD.read_text(encoding="utf-8") == expected_md
    actual_json = json.loads(REVIEW_PACKET_JSON.read_text(encoding="utf-8"))
    assert actual_json == packet


def test_require_adoption_scores_raises_on_missing_values() -> None:
    with pytest.raises(
        ValueError,
        match="Could not parse baseline/provisional adoption audit scores",
    ):
        render_review_packet._require_adoption_scores({"baseline": 59.6})  # noqa: SLF001
