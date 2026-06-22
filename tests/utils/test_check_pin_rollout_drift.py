"""Unit tests for ``scripts/check_pin_rollout_drift.py``.

The doctor script detects stale ``# noqa: TORxxx`` opt-outs -- opt-outs
that no longer silence a real ``torch.eye`` / ``torch.diag`` /
``torch.diag_embed`` violation because a pin rollout has resolved it.
The tests exercise the AST drift classifier by feeding it fixture
source strings, mirroring the harness layout of
``test_check_test_fixture_pin_discipline.py`` so the pattern is
recognisable to a future contributor.

Severity taxonomy under test:
    - **error**: orphan noqa (no ``torch.<func>(...)`` call on the line).
    - **warning**: redundant noqa (call passes the rule on its own
      merits -- direct ``device=`` / ``dtype=`` kwarg or chained
      ``.to(device=..., dtype=...)``).
    - **hint**: cross-rule (rule ID cited in noqa does not match the
      file's path-resolved rule ID) or unknown rule ID.
    - **silent**: legitimate noqa (silences a real violation -- either
      a plain un-pinned call OR a ``**kwargs`` splat call that the
      rule cannot statically verify).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from check_pin_rollout_drift import (  # noqa: E402
    _severity_of,
    check_source,
)


def _diagnostics(src: str, filename: str = "tests/utils/test_x.py") -> list[str]:
    """Run the doctor against a fixture source string."""
    return check_source(src, filename=filename)


# ═══════════════════════════════════════════════════════════════════════════════
# Legitimate opt-outs: not flagged by the doctor
# ═══════════════════════════════════════════════════════════════════════════════


class TestLegitimateNoqa:
    """A ``# noqa: TORxxx`` on a line that still silences a real violation
    must NOT trip the doctor."""

    def test_noqa_on_unpinned_eye_is_legitimate(self) -> None:
        src = "torch.eye(3)  # noqa: TOR001\n"
        assert _diagnostics(src) == []

    def test_noqa_on_unpinned_diag_is_legitimate(self) -> None:
        src = "torch.diag(matrix)  # noqa: TOR001\n"
        assert _diagnostics(src) == []

    def test_noqa_on_unpinned_diag_embed_is_legitimate(self) -> None:
        src = "torch.diag_embed(cov)  # noqa: TOR001\n"
        assert _diagnostics(src) == []

    def test_noqa_in_examples_routes_to_tor002(self) -> None:
        src = "torch.eye(3)  # noqa: TOR002\n"
        assert _diagnostics(src, filename="examples/foo.py") == []

    def test_noqa_in_src_routes_to_tor003(self) -> None:
        src = "torch.diag(matrix)  # noqa: TOR003\n"
        assert _diagnostics(src, filename="src/torchregress/losses/eiv.py") == []

    def test_noqa_splat_with_noqa_still_legitimate(self) -> None:
        """A splat call CANNOT be statically verified; the noqa legitimately
        suppresses an unverifiable call.  Splat is treated as "violating"
        by the doctor, so the noqa is classified silent/legitimate.
        """
        src = "torch.eye(3, **{})  # noqa: TOR001\n"
        assert _diagnostics(src) == []


# ═══════════════════════════════════════════════════════════════════════════════
# ORPHAN error: noqa on a line with no matching torch.<func>() Call
# ═══════════════════════════════════════════════════════════════════════════════


class TestOrphanNoqa:
    def test_noqa_on_pure_text_line_reports_error(self) -> None:
        src = "x = 5  # noqa: TOR001\n"
        diagnostics = _diagnostics(src)
        assert len(diagnostics) == 1
        assert "[error]" in diagnostics[0]
        assert "orphaned" in diagnostics[0]
        assert "TOR001" in diagnostics[0]

    def test_noqa_on_blank_line_reports_error(self) -> None:
        src = "# noqa: TOR001\n"
        diagnostics = _diagnostics(src)
        assert len(diagnostics) == 1
        assert "[error]" in diagnostics[0]

    def test_noqa_on_import_line_reports_error(self) -> None:
        src = "import torch  # noqa: TOR001\n"
        diagnostics = _diagnostics(src)
        assert len(diagnostics) == 1
        assert "[error]" in diagnostics[0]

    def test_string_literal_with_noqa_is_not_orphan(self) -> None:
        """The tokenize-aware scan ignores ``# noqa:`` text inside string
        literals, so a test fixture that demos the noqa pattern does NOT
        trip the orphan detector.
        """
        src = 'src = "torch.eye(3)  # noqa: TOR001\\n"\n'
        assert _diagnostics(src) == []


# ═══════════════════════════════════════════════════════════════════════════════
# REDUNDANT warning: noqa on a line whose call passes the rule on its own
# ═══════════════════════════════════════════════════════════════════════════════


class TestRedundantNoqa:
    def test_noqa_on_already_pinned_eye_reports_warning(self) -> None:
        src = "torch.eye(3, device=A.device, dtype=A.dtype)  # noqa: TOR001\n"
        diagnostics = _diagnostics(src)
        assert len(diagnostics) == 1
        assert "[warning]" in diagnostics[0]
        assert "redundant" in diagnostics[0]

    def test_noqa_on_already_chained_diag_reports_warning(self) -> None:
        src = "torch.diag(matrix).to(device=matrix.device, dtype=matrix.dtype) # noqa: TOR001\n"
        diagnostics = _diagnostics(src)
        assert len(diagnostics) == 1
        assert "[warning]" in diagnostics[0]
        assert "redundant" in diagnostics[0]

    def test_noqa_on_already_chained_diag_embed_reports_warning(self) -> None:
        src = "torch.diag_embed(cov).to(device=cov.device, dtype=cov.dtype) # noqa: TOR001\n"
        diagnostics = _diagnostics(src)
        assert len(diagnostics) == 1
        assert "[warning]" in diagnostics[0]

    def test_noqa_on_chain_closing_line_is_orphan(self) -> None:
        """Documented contract: the inline ``# noqa: TORxxx`` MUST sit on
        the same line as the ``torch.<func>(...)`` call it silences.
        A noqa on the *closing* line of a multi-line chained ``.to()``
        is therefore an orphan -- even though the chain is a passing
        call.  This locks the inline-only contract the sister script
        already enforces for the rule's third branch.
        """
        src = (
            "cov = (\n"
            "    cov_factor @ cov_factor.transpose(-1, -2)\n"
            "    + torch.diag_embed(cov_diag.clamp(min=1e-6)).to(\n"
            "        device=cov_factor.device,\n"
            "        dtype=cov_factor.dtype,\n"
            "    )\n"
            ")  # noqa: TOR001\n"
        )
        diagnostics = _diagnostics(src)
        # The noqa is on line 6 (the closing of the `cov = (`
        # assignment), which does not contain a
        # ``torch.<func>(...)`` Call node at lineno 6.  ORPHAN, not
        # REDUNDANT, per the inline-only contract.
        assert len(diagnostics) == 1
        assert "[error]" in diagnostics[0]


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS_RULE hint: noqa's cited rule ID doesn't match path's rule ID
# ═══════════════════════════════════════════════════════════════════════════════


class TestCrossRuleNoqa:
    def test_noqa_tor001_in_examples_path_is_hint(self) -> None:
        """``examples/`` resolves to TOR002; citing TOR001 is cross-rule."""
        src = "torch.eye(3)  # noqa: TOR001\n"
        diagnostics = _diagnostics(src, filename="examples/foo.py")
        assert len(diagnostics) == 1
        assert "[hint]" in diagnostics[0]
        assert "TOR001" in diagnostics[0]
        assert "TOR002" in diagnostics[0]

    def test_noqa_tor001_in_src_path_is_hint(self) -> None:
        """``src/`` resolves to TOR003; citing TOR001 is cross-rule."""
        src = "torch.eye(3)  # noqa: TOR001\n"
        diagnostics = _diagnostics(src, filename="src/torchregress/foo.py")
        assert len(diagnostics) == 1
        assert "[hint]" in diagnostics[0]
        assert "TOR003" in diagnostics[0]

    def test_noqa_tor002_in_tests_path_is_hint(self) -> None:
        src = "torch.eye(3)  # noqa: TOR002\n"
        diagnostics = _diagnostics(src, filename="tests/foo.py")
        assert len(diagnostics) == 1
        assert "[hint]" in diagnostics[0]


# ═══════════════════════════════════════════════════════════════════════════════
# Unknown rule ID (catches typos in the rule-ID literal that the scanner reads)
# ═══════════════════════════════════════════════════════════════════════════════


class TestUnknownRuleId:
    def test_noqa_unknown_rule_id_reports_hint(self) -> None:
        src = "torch.eye(3)  # noqa: TOR099\n"
        diagnostics = _diagnostics(src)
        assert len(diagnostics) == 1
        assert "[hint]" in diagnostics[0]
        assert "unknown" in diagnostics[0]
        assert "TOR099" in diagnostics[0]

    def test_noqa_tor009_typo_reports_hint(self) -> None:
        src = "torch.eye(3)  # noqa: TOR009\n"
        diagnostics = _diagnostics(src)
        assert len(diagnostics) == 1
        assert "[hint]" in diagnostics[0]


# ═══════════════════════════════════════════════════════════════════════════════
# Severity parser (helper used by ``main()``)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSeverityParser:
    def test_error_tag(self) -> None:
        assert _severity_of("a.py:1:[error] foo") == "error"

    def test_warning_tag(self) -> None:
        assert _severity_of("a.py:1:[warning] foo") == "warning"

    def test_hint_tag(self) -> None:
        assert _severity_of("a.py:1:[hint] foo") == "hint"


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-line / multi-call files
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultiCall:
    def test_mixed_legitimate_and_redundant_reports_each(self) -> None:
        src = (
            "torch.eye(3)  # noqa: TOR001\n"
            "torch.eye(5, device=A.device, dtype=A.dtype)  # noqa: TOR001\n"
            "torch.diag(m)  # noqa: TOR001\n"
            "torch.diag(m).to(device=m.device, dtype=m.dtype)  # noqa: TOR001\n"
        )
        diagnostics = _diagnostics(src)
        assert len(diagnostics) == 2
        assert all("[warning]" in d for d in diagnostics)
        assert all("redundant" in d for d in diagnostics)

    def test_helpers_do_not_trip_when_call_on_other_line(self) -> None:
        """An ``# noqa`` on a line without a call must NOT suppress an
        actual call two lines below.  The doctor should report the
        noqa as an orphan error even though the file violates the rule
        elsewhere."""
        src = "import torch\nx = 5  # noqa: TOR001\ntorch.eye(3)\n"
        diagnostics = _diagnostics(src)
        # Only the orphan; the unpinned call at L3 is not the doctor's
        # concern (the sister script flags that).
        assert len(diagnostics) == 1
        assert "[error]" in diagnostics[0]
        assert "2" in diagnostics[0]
