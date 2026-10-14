#!/usr/bin/env python3
"""Bump the project version in pyproject.toml."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

VERSION_PATTERN = re.compile(r"^version\s*=\s*\"(\d+)\.(\d+)\.(\d+)(?:rc(\d+))?\"$", re.MULTILINE)
EXPLICIT_PATTERN = re.compile(r"(\d+)\.(\d+)\.(\d+)(rc\d+)?")
SEMVER_PARTS = ("major", "minor", "patch")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_full_version(pyproject: Path) -> str:
    """Return the version string as written, including any ``rcN`` suffix."""
    text = pyproject.read_text(encoding="utf-8")
    match = VERSION_PATTERN.search(text)
    if match is None:
        raise SystemExit(f"Could not find semver version in {pyproject}")
    rc = f"rc{match.group(4)}" if match.group(4) is not None else ""
    return f"{match.group(1)}.{match.group(2)}.{match.group(3)}{rc}"


def read_version(pyproject: Path) -> tuple[int, int, int]:
    text = pyproject.read_text(encoding="utf-8")
    match = VERSION_PATTERN.search(text)
    if match is None:
        raise SystemExit(f"Could not find semver version in {pyproject}")
    if match.group(4) is not None:
        raise SystemExit(
            "Current version is a release candidate; set the next version explicitly "
            "with --version X.Y.Z (or X.Y.ZrcN)."
        )
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def format_version(parts: tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in parts)


def parse_explicit_version(value: str) -> str:
    """Validate an explicit ``X.Y.Z`` or ``X.Y.ZrcN`` version and return it."""
    if EXPLICIT_PATTERN.fullmatch(value) is None:
        raise SystemExit(f"Invalid explicit version {value!r}; expected X.Y.Z or X.Y.ZrcN")
    return value


def bump(parts: tuple[int, int, int], part: str) -> tuple[int, int, int]:
    major, minor, patch = parts
    if part == "major":
        return major + 1, 0, 0
    if part == "minor":
        return major, minor + 1, 0
    if part == "patch":
        return major, minor, patch + 1
    raise SystemExit(f"Unknown bump part {part!r}")


def write_version(pyproject: Path, new_version: str) -> None:
    text = pyproject.read_text(encoding="utf-8")
    updated, count = VERSION_PATTERN.subn(f'version = "{new_version}"', text, count=1)
    if count != 1:
        raise SystemExit(f"Failed to update version in {pyproject}")
    pyproject.write_text(updated, encoding="utf-8")


def git_is_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root(),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    return bool(result.stdout.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bump the semver version in pyproject.toml.",
    )
    parser.add_argument(
        "target",
        nargs="?",
        choices=[*SEMVER_PARTS, "explicit"],
        help="Semver part to bump, or use --version for an explicit target.",
    )
    parser.add_argument(
        "--version",
        dest="explicit_version",
        help="Set an explicit X.Y.Z (or X.Y.ZrcN) version instead of bumping.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow bumping when the git working tree is dirty.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the new version without modifying pyproject.toml.",
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=repo_root() / "pyproject.toml",
        help="Path to pyproject.toml (default: repository root).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.explicit_version is not None:
        new_version = parse_explicit_version(args.explicit_version)
    elif args.target in SEMVER_PARTS:
        new_version = format_version(bump(read_version(args.pyproject), args.target))
    else:
        parser.error("Provide patch/minor/major or --version X.Y.Z")

    if not args.force and not args.dry_run and git_is_dirty():
        raise SystemExit(
            "Refusing to bump version on a dirty git tree. Commit/stash first or pass --force."
        )

    current_version = read_full_version(args.pyproject)

    if new_version == current_version:
        print(f"Version unchanged: {current_version}")
        return 0

    if args.dry_run:
        print(f"{current_version} -> {new_version}")
        return 0

    write_version(args.pyproject, new_version)
    print(f"{current_version} -> {new_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
