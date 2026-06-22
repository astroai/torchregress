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
    _RULE_PREFIXES,
    DEFAULT_RULE_ID,
    TARGET_FUNCS,
    TOR001_RULE_ID,
    TOR002_RULE_ID,
    _rule_id_for_path,
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
            "(Coverage invariants rule per docs/loss_test_coverage.md, TOR001; "
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


# ═══════════════════════════════════════════════════════════════════════════════
# Rule ID resolution: TOR001 vs TOR002 per file path prefix
# ═══════════════════════════════════════════════════════════════════════════════


class TestRuleIdResolution:
    """Per-file rule ID is determined by POSIX path-prefix matching."""

    def test_tests_path_resolves_to_tor001(self) -> None:
        assert _rule_id_for_path("tests/utils/test_augment.py") == TOR001_RULE_ID

    def test_tests_absolute_path_resolves_to_tor001(self) -> None:
        # Absolute path WITHOUT a ``/src/``-named component in front of
        # ``tests/`` -- earlier ``src/`` components in the absolute path
        # now route the file to TOR003 (path-component match, first hit).
        assert _rule_id_for_path("/home/user/projects/tests/test_x.py") == TOR001_RULE_ID

    def test_examples_path_resolves_to_tor002(self) -> None:
        assert _rule_id_for_path("examples/basic_usage.py") == TOR002_RULE_ID

    def test_notebooks_path_resolves_to_tor002(self) -> None:
        assert _rule_id_for_path("notebooks/demo.py") == TOR002_RULE_ID

    def test_examples_absolute_path_resolves_to_tor002(self) -> None:
        # Absolute path WITHOUT a ``/src/``-named component in front of
        # ``examples/`` -- earlier ``src/`` components in the absolute path
        # now route the file to TOR003 (path-component match, first hit).
        assert (
            _rule_id_for_path("/home/user/projects/examples/gaussian_low_rank.py") == TOR002_RULE_ID
        )

    def test_unrouted_path_falls_back_to_default(self) -> None:
        # ``src/...`` is now a routed prefix (TOR003); use a clearly
        # un-routed directory (e.g. ``notebooks/`` is TOR002, ``docs/`` is
        # unrouted) to keep the fallback semantic under test.
        assert _rule_id_for_path("docs/foo.py") == DEFAULT_RULE_ID
        assert _rule_id_for_path("random/file.py") == DEFAULT_RULE_ID

    def test_absolute_path_with_early_src_routes_to_tor003(self) -> None:
        """Documented contract: an absolute path whose early Path component
        is ``src`` (e.g. ``/home/user/src/torchregress/tests/foo.py``) is
        routed to TOR003 because path-component matching returns the FIRST
        match in path-component order. Callers that need the project-relative
        scope must normalize absolute paths beforehand; the docs § Mechanical
        enforcement flag this corner case explicitly.
        """
        from scripts.check_test_fixture_pin_discipline import TOR003_RULE_ID

        # /home/user/src/torchregress/tests/foo.py — parts = ['', 'home',
        # 'user', 'src', 'torchregress', 'tests', 'foo.py']; first match is
        # 'src' → TOR003, NOT 'tests' → TOR001.
        assert (
            _rule_id_for_path("/home/user/src/torchregress/tests/foo.py")
            == TOR003_RULE_ID
        )
        assert (
            _rule_id_for_path("/home/user/src/torchregress/examples/foo.py")
            == TOR003_RULE_ID
        )

    def test_violation_message_cites_resolved_rule_id(self) -> None:
        """The message text embeds the path-resolved rule ID, not a hardcoded one."""
        src = "torch.eye(3)"
        v_tests = check_source(src, filename="tests/test_x.py")
        assert len(v_tests) == 1
        assert TOR001_RULE_ID in v_tests[0]

        v_examples = check_source(src, filename="examples/demo.py")
        assert len(v_examples) == 1
        assert TOR002_RULE_ID in v_examples[0]

    def test_tor002_noqa_suppresses_examples(self) -> None:
        src = "torch.eye(3)  # noqa: TOR002"
        assert check_source(src, filename="examples/demo.py") == []

    def test_tor001_noqa_does_not_suppress_tor002(self) -> None:
        """A `# noqa: TOR001` opt-out does NOT carry over into TOR002-scoped paths."""
        src = "torch.eye(3)  # noqa: TOR001"
        v = check_source(src, filename="examples/demo.py")
        assert len(v) == 1
        assert TOR002_RULE_ID in v[0]

    def test_tor002_noqa_does_not_suppress_tor001(self) -> None:
        """Vice versa: a TOR002 opt-out does NOT carry over into TOR001-scoped paths."""
        src = "torch.eye(3)  # noqa: TOR002"
        v = check_source(src, filename="tests/utils/test_x.py")
        assert len(v) == 1
        assert TOR001_RULE_ID in v[0]  # noqa: TOR001  -> just a comment, no spillover


def test_rule_id_constants_and_prefix_map_are_stable() -> None:
    """Rule IDs are the public contract for `# noqa: TORxxx` opt-outs."""
    assert TOR001_RULE_ID == "TOR001"
    assert TOR002_RULE_ID == "TOR002"
    assert DEFAULT_RULE_ID == TOR001_RULE_ID
    assert TARGET_FUNCS == frozenset({"eye", "diag", "diag_embed"})
    # The prefix map is part of the contract: scripts/check_test_fixture_pin_discipline.py
    # advertises the supported directory scope via _RULE_PREFIXES (path-component
    # keys without trailing slashes; _rule_id_for_path does the component match).
    assert ("tests", TOR001_RULE_ID) in _RULE_PREFIXES
    assert ("examples", TOR002_RULE_ID) in _RULE_PREFIXES
    assert ("notebooks", TOR002_RULE_ID) in _RULE_PREFIXES
