from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_neurips_paper_bundle_rejects_run_root_equals_form(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts/run_neurips_paper_bundle.sh"),
            f"--run-root={tmp_path / 'shared'}",
            "--quick",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "does not support --run-root" in result.stderr


def test_compile_tex_rejects_unknown_paper_before_tool_checks() -> None:
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "papers/compile_tex.sh"), "../README.md"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "unknown paper" in result.stderr
