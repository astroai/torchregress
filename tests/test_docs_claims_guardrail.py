from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_CLAIM_RE = re.compile(r"\bSOTA\b|state[- ]of[- ]the[- ]art", re.IGNORECASE)


def _markdown_files() -> list[Path]:
    files = [REPO_ROOT / "README.md"]
    files.extend(
        sorted(
            path
            for path in (REPO_ROOT / "docs").rglob("*.md")
            if "docs/audits/" not in path.as_posix()
        )
    )
    return files


def test_docs_avoid_unqualified_sota_claims() -> None:
    offenders: list[str] = []
    for path in _markdown_files():
        text = path.read_text(encoding="utf-8")
        for match in FORBIDDEN_CLAIM_RE.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(REPO_ROOT)}:{line}: {match.group(0)!r}")

    assert not offenders, "Unqualified SOTA/state-of-the-art claims found:\n" + "\n".join(offenders)
