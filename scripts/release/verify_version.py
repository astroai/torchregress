#!/usr/bin/env python3
"""Verify that a git tag matches the project version in pyproject.toml."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

VERSION_PATTERN = re.compile(r"^version\s*=\s*\"(\d+\.\d+\.\d+)\"$", re.MULTILINE)
TAG_PATTERN = re.compile(r"^v(\d+\.\d+\.\d+)$")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_project_version(pyproject: Path) -> str:
    text = pyproject.read_text(encoding="utf-8")
    match = VERSION_PATTERN.search(text)
    if match is None:
        raise SystemExit(f"Could not find semver version in {pyproject}")
    return match.group(1)


def normalize_tag(tag: str) -> tuple[str, str]:
    match = TAG_PATTERN.fullmatch(tag)
    if match is None:
        raise SystemExit(f"Invalid release tag {tag!r}; expected format vX.Y.Z")
    return tag, match.group(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify that vX.Y.Z matches project.version in pyproject.toml.",
    )
    parser.add_argument(
        "--tag",
        default=os.environ.get("GITHUB_REF_NAME", ""),
        help="Release tag to verify (default: GITHUB_REF_NAME).",
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

    if not args.tag:
        parser.error("Provide --tag vX.Y.Z or set GITHUB_REF_NAME")

    _, tag_version = normalize_tag(args.tag)
    project_version = read_project_version(args.pyproject)

    if tag_version != project_version:
        print(
            f"Version mismatch: tag {args.tag} ({tag_version}) "
            f"!= pyproject.toml ({project_version})",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {args.tag} matches pyproject.toml version {project_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
