"""Unit tests for ``scripts/check_test_fixture_pin_discipline.py``.

The tests exercise the AST checker by feeding it fixture *source strings*,
not by spawning subprocesses. The test file itself contains zero
``torch.eye`` / ``torch.diag`` / ``torch.diag_embed`` literal calls, so it
never trips the very rule it is testing.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from check_test_fixture_pin_discipline import (  # noqa: E402
    CHECK_RULE_ID,
    TARGET_FUNCS,
    check_source,
)


def _violations(src: str) -> list[str]:
    """Run the checker against a fixture source string."""
    return check_source(src, filename="<test>")


# ═══════════════════════════════════════════════════════════════════════════════
# Direct-kwarg pin: ``torch.eye`` path (which natively supports device=/dtype=)
# ═══════════════════════════════════════════════════════════════════════════════


class TestDirectEyePin:
    def test_eye_without_kwarg_violates(self) -> None:
        src = "torch.eye(3)"
        assert _violations(src) == [
            "<test>:1: torch.eye() missing device= or dtype= kwarg "
            "(Coverage invariants rule per docs/loss_test_coverage.md; "
            "or wrap the call in `.to(device=..., dtype=...)` as a chain)"
        ]

    def test_eye_with_only_device_passes(self) -> None:
        src = "torch.eye(3, device=x.device)"
        assert _violations(src) == []

    def test_eye_with_only_dtype_passes(self) -> None:
        src = "torch.eye(3, dtype=x.dtype)"
        assert _violations(src) == []

    def test_eye_with_both_kwarg_passes(self) -> None:
        src = "torch.eye(3, device=x.device, dtype=x.dtype)"
        assert _violations(src) == []

    def test_eye_multiline_kwarg_passes(self) -> None:
        src = "base = A @ A.T + torch.eye(\n    dim,\n    device=A.device,\n    dtype=A.dtype,\n)"
        assert _violations(src) == []

    def test_eye_with_kwargs_splat_violates_with_splat_message(self) -> None:
        src = "torch.eye(3, **{'device': x.device, 'dtype': x.dtype})"
        v = _violations(src)
        assert len(v) == 1
        assert "**kwargs splat" in v[0]
        assert "torch.eye" in v[0]

    def test_eye_with_splat_and_noqa_passes(self) -> None:
        src = "torch.eye(3, **{})  # noqa: TOR001"
        assert _violations(src) == []


# ═══════════════════════════════════════════════════════════════════════════════
# Chained ``.to(...)`` pin: ``torch.diag`` / ``torch.diag_embed`` path
# (these do NOT accept device=/dtype= kwargs natively — fall back to ``.to()``)
# ═══════════════════════════════════════════════════════════════════════════════


class TestChainedToPin:
    def test_diag_without_to_violates(self) -> None:
        src = "torch.diag(matrix)"
        v = _violations(src)
        assert len(v) == 1
        assert "torch.diag()" in v[0]

    def test_diag_with_to_passes(self) -> None:
        # Keyword-only canonical form (matches doc). Positional form is NOT
        # accepted; see test_diag_to_positional_rejected below.
        src = "torch.diag(matrix).to(device=matrix.device, dtype=matrix.dtype)"
        assert _violations(src) == []

    def test_diag_to_positional_rejected(self) -> None:
        """Positional `.to(device, dtype)` is NOT accepted; the rule is keyword-only.

        PyTorch's ``.to()`` signature is ``to(device, dtype, non_blocking,
        memory_format)`` so positional args are ambiguous; the documented
        canonical pattern is keyword-form ``to(device=..., dtype=...)``.
        This test locks that boundary so future contributors aren't surprised.
        """
        src = "torch.diag(matrix).to(matrix.device, matrix.dtype)"
        v = _violations(src)
        assert len(v) == 1
        assert "torch.diag()" in v[0]

    def test_diag_with_to_only_device_passes(self) -> None:
        src = "torch.diag(matrix).to(device=matrix.device)"
        assert _violations(src) == []

    def test_diag_with_to_only_dtype_passes(self) -> None:
        src = "torch.diag(matrix).to(dtype=matrix.dtype)"
        assert _violations(src) == []

    def test_diag_embed_with_multiline_chain(self) -> None:
        src = (
            "cov = cov_factor @ cov_factor.transpose(-1, -2)\n"
            "    + torch.diag_embed(cov_diag.clamp(min=1e-6)).to(\n"
            "        device=cov_factor.device,\n"
            "        dtype=cov_factor.dtype,\n"
            "    )\n"
        )
        assert _violations(src) == []

    def test_diag_with_to_but_neither_kwarg_still_violates(self) -> None:
        src = "torch.diag(matrix).to('cpu')"
        v = _violations(src)
        assert len(v) == 1
        assert "torch.diag()" in v[0]


# ═══════════════════════════════════════════════════════════════════════════════
# No-op / scope filters
# ═══════════════════════════════════════════════════════════════════════════════


class TestScopeFilters:
    def test_noqa_inline_suppresses(self) -> None:
        src = "torch.eye(3)  # noqa: TOR001"
        assert _violations(src) == []

    def test_noqa_different_line_does_not_suppress(self) -> None:
        src = "torch.eye(3)\n# noqa: TOR001  ← wrong line\n"
        assert len(_violations(src)) == 1

    def test_string_literal_with_torch_eye_ignored(self) -> None:
        src = 's = "torch.eye(3)"\nfunc = "torch.diag(x)"\n'
        assert _violations(src) == []

    def test_comment_with_torch_eye_ignored(self) -> None:
        src = "# torch.eye(3) — discussed, not enforced\n"
        assert _violations(src) == []

    def test_aliased_eye_skipped(self) -> None:
        src = "import torch as t\nt.eye(3)\n"
        assert _violations(src) == []

    def test_from_torch_eyeing_skipped(self) -> None:
        src = "from torch import eye\nx = eye(3)\n"
        assert _violations(src) == []

    def test_np_eye_skipped(self) -> None:
        src = "import numpy as np\nnp.eye(3)\n"
        assert _violations(src) == []

    def test_torch_eye_reference_not_call_skipped(self) -> None:
        src = "func = torch.eye\nfunc(3)\n"
        assert _violations(src) == []


# ═══════════════════════════════════════════════════════════════════════════════
# Walk depth: nested defs / loops / conditionals / comprehensions
# ═══════════════════════════════════════════════════════════════════════════════


class TestWalkDepth:
    def test_call_inside_nested_def_found(self) -> None:
        src = "def outer():\n    def inner():\n        torch.eye(3)\n"
        assert len(_violations(src)) == 1

    def test_call_in_conditional(self) -> None:
        src = "x = torch.eye(3) if cond else torch.eye(4, device=A.device)\n"
        assert len(_violations(src)) == 1  # only the un-pinned one

    def test_call_in_lambda(self) -> None:
        src = "fn = lambda: torch.eye(3)\n"
        assert len(_violations(src)) == 1

    def test_multiple_unpinned_each_reported_once(self) -> None:
        src = "a = torch.eye(3)\nb = torch.eye(5)\nc = torch.eye(7, device=A.device)\n"
        v = _violations(src)
        assert len(v) == 2
        assert "<test>:1" in v[0]
        assert "<test>:2" in v[1]


# ═══════════════════════════════════════════════════════════════════════════════
# Syntax-error handling and rule constants
# ═══════════════════════════════════════════════════════════════════════════════


def test_syntax_error_returns_empty_no_crash() -> None:
    bad_src = "def foo(:\n    pass\n"
    assert _violations(bad_src) == []


def test_rule_id_and_target_set_are_stable() -> None:
    """Rule ID is the public contract for `# noqa: TOR001` opt-outs."""
    assert CHECK_RULE_ID == "TOR001"
    assert TARGET_FUNCS == frozenset({"eye", "diag", "diag_embed"})
