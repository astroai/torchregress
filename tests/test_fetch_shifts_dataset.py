from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_fetch_shifts_dataset_dry_run() -> None:
    r = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/fetch_shifts_dataset.py"),
            "--dry-run",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "Shifts-Project" in r.stdout


@pytest.mark.parametrize("extra", ([], ["--dataset", "customkey"]))
def test_fetch_shifts_dataset_materialize_writes_readme(tmp_path: Path, extra: list[str]) -> None:
    out_root = tmp_path / "shifts_out"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools/fetch_shifts_dataset.py"),
        "--out-root",
        str(out_root),
        *extra,
    ]
    r = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    ds = "customkey" if extra else "solar"
    readme = out_root / ds / "README.txt"
    assert readme.is_file()
    body = readme.read_text(encoding="utf-8")
    assert "Shifts-Project" in body
