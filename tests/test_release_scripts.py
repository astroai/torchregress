"""Tests for release helper scripts."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BUMP_SCRIPT = REPO_ROOT / "scripts" / "release" / "bump_version.py"
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "release" / "verify_version.py"
BUILD_SCRIPT = REPO_ROOT / "scripts" / "release" / "build_package.sh"


def _write_pyproject(path: Path, version: str) -> None:
    path.write_text(
        "\n".join(
            [
                "[project]",
                'name = "torchregress"',
                f'version = "{version}"',
                "",
            ]
        ),
        encoding="utf-8",
    )


def _run_bump(pyproject: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(BUMP_SCRIPT), *args, "--pyproject", str(pyproject), "--force"],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def _run_verify(pyproject: Path, tag: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT), "--tag", tag, "--pyproject", str(pyproject)],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


class TestBumpVersion:
    def test_patch_bump(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        _write_pyproject(pyproject, "1.2.3")

        result = _run_bump(pyproject, "patch")
        assert result.returncode == 0
        assert "1.2.3 -> 1.2.4" in result.stdout
        assert 'version = "1.2.4"' in pyproject.read_text(encoding="utf-8")

    def test_minor_bump(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        _write_pyproject(pyproject, "1.2.3")

        result = _run_bump(pyproject, "minor")
        assert result.returncode == 0
        assert "1.2.3 -> 1.3.0" in result.stdout

    def test_major_bump(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        _write_pyproject(pyproject, "1.2.3")

        result = _run_bump(pyproject, "major")
        assert result.returncode == 0
        assert "1.2.3 -> 2.0.0" in result.stdout

    def test_explicit_version(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        _write_pyproject(pyproject, "0.1.0")

        result = _run_bump(pyproject, "--version", "0.2.0")
        assert result.returncode == 0
        assert 'version = "0.2.0"' in pyproject.read_text(encoding="utf-8")

    def test_dry_run_does_not_write(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        _write_pyproject(pyproject, "0.1.0")

        result = subprocess.run(
            [
                sys.executable,
                str(BUMP_SCRIPT),
                "patch",
                "--dry-run",
                "--pyproject",
                str(pyproject),
                "--force",
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert result.returncode == 0
        assert "0.1.0 -> 0.1.1" in result.stdout
        assert 'version = "0.1.0"' in pyproject.read_text(encoding="utf-8")


class TestVerifyVersion:
    def test_matching_tag_passes(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        _write_pyproject(pyproject, "0.1.0")

        result = _run_verify(pyproject, "v0.1.0")
        assert result.returncode == 0
        assert "OK:" in result.stdout

    def test_mismatched_tag_fails(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        _write_pyproject(pyproject, "0.1.0")

        result = _run_verify(pyproject, "v0.1.1")
        assert result.returncode == 1
        assert "Version mismatch" in result.stderr

    def test_invalid_tag_format_fails(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        _write_pyproject(pyproject, "0.1.0")

        result = _run_verify(pyproject, "0.1.0")
        assert result.returncode == 1
        assert "Invalid release tag" in result.stderr


def test_build_package_script() -> None:
    if shutil.which("pixi") is None:
        pytest.skip("pixi not found — build_package.sh requires pixi")
    result = subprocess.run(
        [str(BUILD_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (REPO_ROOT / "dist").exists()
    assert list((REPO_ROOT / "dist").glob("*.whl"))
    assert list((REPO_ROOT / "dist").glob("*.tar.gz"))

    check_only = subprocess.run(
        [str(BUILD_SCRIPT), "--check-only"],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert check_only.returncode == 0, check_only.stdout + check_only.stderr
