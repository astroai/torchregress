#!/usr/bin/env python3
"""Pre-commit hook: detect ``# noqa: TORxxx`` opt-out **drift**.

Pairs with :mod:`scripts.check_test_fixture_pin_discipline.py` to catch
stale opt-outs after a pin rollout.  The sister script enforces "every
ad-hoc ``torch.eye`` / ``torch.diag`` / ``torch.diag_embed`` literal must
pin ``device=`` / ``dtype=``"; this script enforces "every ``# noqa:
TORxxx`` comment still silences a real violation it claims to silence".

Three drift categories the doctor catches:

- **ORPHAN** (severity: error).  The ``# noqa: TORxxx`` comment lives on
  a source line that has **no** matching ``torch.eye`` / ``torch.diag``
  / ``torch.diag_embed`` call at all -- the call site was removed or
  moved away from the noqa line, leaving the opt-out attached to
  nothing.  An orphaned noqa will silently drift across refactors
  because nothing references it.
- **REDUNDANT** (severity: warning).  The ``# noqa: TORxxx`` comment is
  attached to a ``torch.<func>(...)`` call that already passes the rule
  on its own merits -- direct ``device=`` / ``dtype=`` kwarg pin or
  chained ``.to(device=..., dtype=...)`` cast.  This typically follows
  a pin rollout that fixed the violation upstream -- the noqa is no-op
  defensive code that misleads future contributors.
- **CROSS_RULE / UNKNOWN_RULE** (severity: hint).  The rule ID cited
  in the ``# noqa`` comment does not match the path-resolved rule ID
  for the file (cross-rule) or is not a known rule ID at all (unknown
  rule -- typically a typo).  The sister script silently ignores
  cross-rule noqa; the doctor surfaces hints so the contributor can
  confirm the file's scope or the cited rule ID.

Classification priority (the first match wins):

1. Unknown rule ID (not in ``_RULE_PREFIXES`` union ``DEFAULT_RULE_ID``)
   -> ``[hint]``.
2. Cited rule ID != path-resolved rule ID -> ``[hint]``.
3. noqa line has no ``torch.<func>(...)`` call -> ``[error]`` (orphan).
4. noqa line has a VIOLATING call (splat, no device/dtype/chain) ->
   silent (legitimate).
5. noqa line has a passing call (device=/dtype= kwarg, or chained
   ``.to()`` pin) -> ``[warning]`` (redundant).

Public API mirrors the sister script:

    python scripts/check_pin_rollout_drift.py FILE [FILE ...]

Each FILE is parsed with ``ast.parse``; the script prints one diagnostic
per drift in the form ``filename:lineno:[severity] message`` and exits
with status 1 if any error-or-warning drift is found.  Cross-rule and
unknown-rule hints do NOT cause a non-zero exit (they're advisory only).
"""

from __future__ import annotations

import ast
import io
import re
import sys
import tokenize
from pathlib import Path

# Reuse the rule definition so the doctor and the enforcer agree.
sys.path.insert(0, str(Path(__file__).parent))
from check_test_fixture_pin_discipline import (  # noqa: E402
    _RULE_PREFIXES,
    DEFAULT_RULE_ID,
    _build_parent_map,
    _has_required_kwarg,
    _is_in_to_pinning_chain,
    _is_torch_attr,
    _rule_id_for_path,
)

# Bound the regex to 3-digit rule IDs (TOR001..TOR009 today; future
# TOR010+ follows the same 3-digit shape).  Unbounded ``TOR\d+`` would
# silently accept garbage strings; bounded ``TOR\d{3}`` makes the
# regex's contract explicit and keeps new rule IDs a deliberate update.
NOQA_PATTERN = re.compile(r"#\s*noqa:\s*(TOR\d{3})")


def _scan_noqa_lines(source: str) -> list[tuple[int, str]]:
    """Return ``[(lineno, cited_rule_id), ...]`` for every ``# noqa: TOR<3-digits>`` comment token.

    Uses :mod:`tokenize` so ``# noqa:`` text that lives inside a string
    literal (e.g. inside a test fixture ``src = 'torch.eye(3)  # noqa: TOR001'``)
    is correctly excluded -- only **real** Python comments at column 0+
    that contain the noqa directive are reported.
    """
    out: list[tuple[int, str]] = []
    try:
        tokens = list(tokenize.tokenize(io.BytesIO(source.encode("utf-8")).readline))
    except (tokenize.TokenizeError, SyntaxError):
        # Unparseable source: fall back to empty (matches the sister's
        # ``ast.parse`` ``SyntaxError`` graceful-degradation contract).
        return out
    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        m = NOQA_PATTERN.search(tok.string)
        if m is not None:
            out.append((tok.start[0], m.group(1)))
    return out


def _classify_torch_calls(
    tree: ast.AST,
) -> tuple[
    dict[int, tuple[str, ast.Call]],
    dict[int, tuple[str, ast.Call]],
]:
    """Walk the AST once and partition ``torch.<func>(...)`` calls into
    ``(call_lines_all, call_lines_violating)``.

    - ``call_lines_all``: every ``torch.<func>(...)`` Call regardless
      of pin status (used to detect ORPHAN -- noqa on a line with no
      call).
    - ``call_lines_violating``: subset of calls that would VIOLATE the
      rule (used to distinguish LEGITIMATE suppressions from REDUNDANT
      ones).  Splat calls (``**kwargs``) are deliberately counted as
      violating: they cannot be statically verified, so the matching
      noqa legitimately suppresses a real ambiguity rather than
      slapping on a no-op defensive comment.
    """
    all_calls: dict[int, tuple[str, ast.Call]] = {}
    violating: dict[int, tuple[str, ast.Call]] = {}
    parent_map = _build_parent_map(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        attr_name = _is_torch_attr(node.func)
        if attr_name is None:
            continue
        all_calls[node.lineno] = (attr_name, node)
        if _has_required_kwarg(node):
            # Direct kwarg pin: passes the rule, not violating.
            continue
        if _is_in_to_pinning_chain(node, parent_map):
            # Chained .to() pin: passes the rule, not violating.
            continue
        # Splat OR plain unpinned: both are "violating" so a matching
        # noqa is classified LEGITIMATE rather than REDUNDANT.
        violating[node.lineno] = (attr_name, node)
    return all_calls, violating


_SEVERITY_TAG_TO_NAME = {
    "[error]": "error",
    "[warning]": "warning",
    "[hint]": "hint",
}


def _severity_of(diagnostic: str) -> str | None:
    """Return ``"error"`` / ``"warning"`` / ``"hint"`` from the diagnostic line.

    Diagnostic format: ``filename:lineno:[severity] message``.  Returns
    None if the bracket tag isn't recognised (treated as a hard fail so
    a future contributor adding a new severity is flagged immediately).
    """
    for tag, name in _SEVERITY_TAG_TO_NAME.items():
        if tag in diagnostic:
            return name
    return None


def check_source(source: str, filename: str) -> list[str]:
    """Return one diagnostic per drift in ``source``.

    Diagnostics are textual, ``filename:lineno:[severity] message``
    strings.  Empty list means no drift.
    """
    out: list[str] = []
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return out

    call_lines_all, call_lines_violating = _classify_torch_calls(tree)
    path_rule = _rule_id_for_path(filename)
    noqa_lines = _scan_noqa_lines(source)

    valid_cited_rules = {rid for (_, rid) in _RULE_PREFIXES} | {DEFAULT_RULE_ID}

    for line, cited_rule in noqa_lines:
        # First, the outer hints: any cross-rule issue is reported
        # regardless of whether the line has a real call attached.
        if cited_rule not in valid_cited_rules:
            out.append(
                f"{filename}:{line}:[hint] "
                f"`# noqa: {cited_rule}` cites an unknown rule ID; "
                f"valid IDs are {sorted(valid_cited_rules)}"
            )
            continue

        if cited_rule != path_rule:
            out.append(
                f"{filename}:{line}:[hint] "
                f"`# noqa: {cited_rule}` cites rule ID {cited_rule} but the "
                f"file path is routed to {path_rule}; the opt-out is "
                f"harmless (no violation will be silenced) but indicates "
                f"a stale assumption about the file's scope."
            )
            continue

        # Inner classification: orphan / legitimate / redundant.
        if line not in call_lines_all:
            out.append(
                f"{filename}:{line}:[error] "
                f"`# noqa: {cited_rule}` is orphaned: no "
                f"`torch.eye` / `torch.diag` / `torch.diag_embed` call "
                f"on this line.  Remove the stale opt-out comment."
            )
            continue

        if line in call_lines_violating:
            # The noqa silences a real violation -> legitimate.
            continue

        # The call at the noqa line passes the rule on its own merits
        # -> the noqa is redundant drift from a pin rollout.
        attr_name, _ = call_lines_all[line]
        out.append(
            f"{filename}:{line}:[warning] "
            f"`# noqa: {cited_rule}` is redundant: the "
            f"`torch.{attr_name}()` call on this line already passes the "
            f"rule (device=/dtype= kwarg or chained "
            f"`.to(device=..., dtype=...)`).  The pin rollout has "
            f"resolved this violation -- remove the now-defunct "
            f"opt-out comment to keep the source clean."
        )

    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(f"usage: {Path(argv[0]).name} FILES...", file=sys.stderr)
        return 2
    all_diagnostics: list[str] = []
    for arg in argv[1:]:
        path = Path(arg)
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"{path}: cannot read: {exc}", file=sys.stderr)
            return 2
        all_diagnostics.extend(check_source(source, str(path)))
    for diagnostic in all_diagnostics:
        print(diagnostic, file=sys.stderr)
    if not all_diagnostics:
        return 0
    severities = [_severity_of(d) for d in all_diagnostics]
    has_fail_severity = any(sev in ("error", "warning") for sev in severities)
    if has_fail_severity:
        n_err = sum(1 for sev in severities if sev == "error")
        n_warn = sum(1 for sev in severities if sev == "warning")
        n_hint = sum(1 for sev in severities if sev == "hint")
        print(
            f"\n{len(all_diagnostics)} # noqa drift diagnostic(s) "
            f"({n_err} error(s), {n_warn} warning(s), {n_hint} hint(s)). "
            f"Errors and warnings signal stale opt-outs from a rolled-out "
            f"pin fix -- review and remove them.  Hints are advisory only.",
            file=sys.stderr,
        )
        return 1
    n_hint = sum(1 for sev in severities if sev == "hint")
    if n_hint:
        print(
            f"\n{n_hint} # noqa drift hint(s).  These do not block the "
            f"commit but point to stale scope assumptions worth a glance.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
