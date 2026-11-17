"""Run the examples of the sequence-loss guides (SSIM and linear-nuisance marginalisation)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[2] / "docs" / "losses"


@pytest.mark.parametrize("page", ["structural_similarity.md", "linear_nuisance.md"])
def test_guide_examples_run(page: str) -> None:
    path = DOCS / page
    blocks = re.findall(
        r"```python\s*\n(.*?)```", path.read_text(encoding="utf-8"), flags=re.DOTALL
    )
    assert blocks
    for block in blocks:
        exec(compile(block, str(path), "exec"), {})
