#!/usr/bin/env python3
"""Pre-commit hook: enforce ``torch.eye`` / ``torch.diag`` pin discipline in
``tests/``, ``examples/``, ``notebooks/``, and ``src/``.

The Coverage invariants rule (see ``docs/loss_test_coverage.md``) requires every
ad-hoc ``torch.eye(...)`` / ``torch.diag(...)`` / ``torch.diag_embed(...)``
literal in ``tests/**/*.py``, ``examples/**/*.py``, and ``notebooks/**/*.py``
to pin at least one of ``device=`` / ``dtype=`` so the fixture does not
silently rely on the loss module handling dtype / device of input fixtures
internally. Three rule IDs:

- ``TOR001`` — scoped to ``tests/**/*.py``.
- ``TOR002`` — scoped to ``examples/**/*.py`` and ``notebooks/**/*.py``.
- ``TOR003`` — scoped to ``src/**/*.py`` (production loss fixtures plus
  algorithms/, metrics/, ensemble/, utils/ helpers that emit the same
  fixtures internally).

The rule ID is resolved per-file via POSIX path-component matching
(``_rule_id_for_path`` below).  Inline opt-out: trailing ``# noqa:
TOR001`` / ``# noqa: TOR002`` / ``# noqa: TOR003`` comment on the
offending line.

Why this hook (vs a custom ``ruff`` plugin): ``ruff`` is a Rust CLI and is not
importable as a Python module for plugin code; ``ast``-based static analysis
covers the rule completely without spawning an extra Rust tool. The hook is
invoked as a ``local`` pre-commit hook. ``.pre-commit-config.yaml`` registers
three entries (one per rule ID), each with its own ``files:`` regex.

PyTorch API wrinkle: ``torch.eye`` accepts native ``device=`` / ``dtype=``
kwargs, but ``torch.diag`` and ``torch.diag_embed`` do NOT — they inherit
dtype / device from the input tensor. The documented Coverage invariants
fallback for those is a chained ``.to(device=..., dtype=...)`` cast on the
result. The checker accepts that chain via parent-tracking (see
``_is_in_to_pinning_chain``).

Alias-scope: the checker ONLY recognises ``torch.<func>(...)`` direct
attribute access. Aliases (``t.eye(...)``) and ``from torch import eye`` are
flagged as unverifiable — opt-out with ``# noqa: TOR001`` / ``# noqa:
TOR002`` / ``# noqa: TOR003`` if this matters.

Usage (pre-commit calls this as a local hook):

    python scripts/check_test_fixture_pin_discipline.py FILE [FILE ...]

Each FILE is parsed with ``ast.parse``; the script prints one violation per
line to stderr in the form ``filename:lineno: message`` and exits with
status 1 if any violation is found (0 otherwise).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Optional

TOR001_RULE_ID = "TOR001"
TOR002_RULE_ID = "TOR002"
TOR003_RULE_ID = "TOR003"
DEFAULT_RULE_ID = TOR001_RULE_ID
TARGET_FUNCS = frozenset({"eye", "diag", "diag_embed"})
REQUIRED_KWARGS = frozenset({"device", "dtype"})

# Path-component -> rule ID.  The scope list is iterated for every
# ``Path.parts`` entry; the FIRST matching prefix wins.  Order matters
# when a single filename contains two scope names (rare); tests/ is
# listed first to keep the historical designation.
_RULE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("tests", TOR001_RULE_ID),
    ("examples", TOR002_RULE_ID),
    ("notebooks", TOR002_RULE_ID),
    # TOR003 covers the source-tree loss fixtures — ``src/torchregress/losses``
    # is the canonical site of internal helpers like ``_make_spd_cov``, but
    # the rule is keyed on the broader ``src`` directory so all production
    # fixtures (algorithms/, metrics/, ensemble/, utils/, losses/) are in
    # scope under a single rule ID.
    ("src", TOR003_RULE_ID),
)


def _rule_id_for_path(filename: str) -> str:
    """Map a file path to its fixture-pin-discipline rule ID.

    Walks the directory components of ``filename`` (POSIX-normalised) and
    returns the rule ID associated with the first component whose name
    matches one of ``_RULE_PREFIXES``.  This is path-component matching,
    not substring matching — a file at ``examples/tests/helper.py`` is
    resolved against the ``examples`` component (TOR002), not against a
    nested ``tests`` substring.  Paths that match no prefix fall back to
    ``DEFAULT_RULE_ID`` (``TOR001``) as a safety default for direct CLI
    invocations that bypass the pre-commit ``files:`` regex.
    """
    parts = filename.replace("\\", "/").split("/")
    for part in parts:
        for prefix, rule_id in _RULE_PREFIXES:
            if part == prefix:
                return rule_id
    return DEFAULT_RULE_ID


def _build_parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    """Map ``id(child)`` -> parent node for the entire AST.

    Required because ``ast.walk`` does not surface parent context, and the
    ``.to(device=, dtype=)`` chain exemption needs to walk up from an inner
    ``torch.<func>(...)`` Call to its enclosing ``.to(...)`` Call.
    """
    parent_map: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_map[id(child)] = parent
    return parent_map


def _is_torch_attr(node: ast.AST) -> Optional[str]:
    """Return the attr name iff ``node`` is ``torch.<func>`` (direct access)."""
    if not isinstance(node, ast.Attribute):
        return None
    if node.attr not in TARGET_FUNCS:
        return None
    # Only the canonical ``torch.eye(...)`` form. Aliases (``t.eye(...)``) and
    # chains (``foo.bar.eye(...)``) are out of scope.
    if isinstance(node.value, ast.Name) and node.value.id == "torch":
        return node.attr
    return None


def _is_in_to_pinning_chain(node: ast.Call, parent_map: dict[int, ast.AST]) -> bool:
    """``True`` iff ``torch.<func>(...)`` is wrapped in a chained
    ``.to(device=..., dtype=...)`` pin.

    PyTorch's ``torch.diag`` and ``torch.diag_embed`` do NOT accept ``device=``
    / ``dtype=`` keyword arguments — they inherit dtype / device from the
    input tensor. The documented Coverage invariants fallback for those is a
    chained ``.to(device=..., dtype=...)`` cast on the result. We recognise
    that chain by walking up the AST: the inner Call's parent must be the
    ``Attribute(value=<inner>, attr='to')`` of an outer Call whose kwargs
    include at least one of ``device=`` / ``dtype=``.
    """
    parent_attr = parent_map.get(id(node))
    if not isinstance(parent_attr, ast.Attribute) or parent_attr.attr != "to":
        return False
    parent_call = parent_map.get(id(parent_attr))
    if not isinstance(parent_call, ast.Call):
        return False
    if parent_call.func is not parent_attr:
        return False
    return _has_required_kwarg(parent_call)


def _has_required_kwarg(call: ast.Call) -> bool:
    """``True`` iff the Call has at least one of ``device=`` / ``dtype=`` kwargs."""
    return any(kw.arg in REQUIRED_KWARGS for kw in call.keywords if kw.arg is not None)


def _has_kwargs_splat(call: ast.Call) -> bool:
    """``True`` iff the Call has ``**kwargs`` (statically unverifiable)."""
    return any(kw.arg is None for kw in call.keywords)


def _line_has_noqa(source_lines: list[str], lineno: int, rule_id: str) -> bool:
    """``True`` iff ``source_lines[lineno - 1]`` contains ``# noqa: <rule_id>``."""
    if 1 <= lineno <= len(source_lines):
        return f"noqa: {rule_id}" in source_lines[lineno - 1]
    return False


def check_source(source: str, filename: str) -> list[str]:
    """Return one violation message per offending call in ``source``.

    Returns an empty list if no violations, or if the source cannot be parsed
    (syntax errors are deferred to other tooling).  Violation messages cite
    the path-resolved rule ID (TOR001 for ``tests/``, TOR002 for
    ``examples/`` / ``notebooks/``); noqa opt-outs must match the cited ID.
    """
    return [msg for _rule_id, msg in check_source_with_rule_id(source, filename)]


def check_source_with_rule_id(source: str, filename: str) -> list[tuple[str, str]]:
    """Same as :func:`check_source` but yields ``(rule_id, message)`` tuples.

    Splitting the rule ID from the message lets the caller aggregate by
    rule ID without re-running :func:`_rule_id_for_path` on
    already-resolved paths — the rule ID is the source of truth here.
    """
    out: list[tuple[str, str]] = []
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return out
    parent_map = _build_parent_map(tree)
    source_lines = source.splitlines()
    rule_id = _rule_id_for_path(filename)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        attr_name = _is_torch_attr(node.func)
        if attr_name is None:
            continue
        # 1. Direct kwarg pin: works for `torch.eye` (which accepts device=/dtype=).
        if _has_required_kwarg(node):
            continue
        # 2. Chained `.to(device=, dtype=)` pin: documented fallback for `torch.diag`
        #    and `torch.diag_embed` which do NOT accept those kwargs natively.
        if _is_in_to_pinning_chain(node, parent_map):
            continue
        # 3. Inline opt-out.
        if _line_has_noqa(source_lines, node.lineno, rule_id):
            continue
        # 4. **kwargs splat: can't statically verify.
        if _has_kwargs_splat(node):
            out.append(
                (
                    rule_id,
                    f"{filename}:{node.lineno}: "
                    f"torch.{attr_name}() with **kwargs splat pin device=/dtype= "
                    f"explicitly (or add `# noqa: {rule_id}` if splat carries them)",
                )
            )
            continue
        out.append(
            (
                rule_id,
                f"{filename}:{node.lineno}: "
                f"torch.{attr_name}() missing device= or dtype= kwarg "
                f"(Coverage invariants rule per docs/loss_test_coverage.md, {rule_id}; "
                f"or wrap the call in `.to(device=..., dtype=...)` as a chain)",
            )
        )
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(f"usage: {Path(argv[0]).name} FILES...", file=sys.stderr)
        return 2
    all_errors: list[tuple[str, str]] = []
    for arg in argv[1:]:
        path = Path(arg)
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"{path}: cannot read: {exc}", file=sys.stderr)
            return 2
        all_errors.extend(check_source_with_rule_id(source, str(path)))
    for _rule_id, msg in all_errors:
        print(msg, file=sys.stderr)
    if all_errors:
        # Aggregate distinct rule IDs from the already-resolved tuples —
        # no need to re-run _rule_id_for_path on the message filenames.
        rule_ids = sorted({rid for rid, _msg in all_errors})
        print(
            f"\n{len(all_errors)} violation(s) of fixture pin discipline "
            f"({', '.join(rule_ids)}). Pin device=/dtype= on every "
            "torch.eye/torch.diag/torch.diag_embed literal under tests/, "
            "examples/, notebooks/, or src/. See docs/loss_test_coverage.md "
            "§ Coverage invariants. Inline opt-out: `# noqa: TOR001` "
            "(tests/), `# noqa: TOR002` (examples/, notebooks/), or "
            "`# noqa: TOR003` (src/).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
