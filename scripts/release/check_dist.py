#!/usr/bin/env python3
"""Check the contents of built torchregress distributions before upload.

Usage::

    python scripts/release/check_dist.py dist/

Fails (exit 1) when:

- there is not exactly one wheel and one sdist;
- the wheel holds anything outside ``torchregress/`` and its ``.dist-info``,
  or misses ``py.typed``, ``LICENSE`` or a top-level module;
- the sdist holds build or tooling output (``site/``, ``htmlcov/``, ``dist/``,
  ``.pixi/``, ``coverage.xml``, caches) or any file above 5 MB;
- wheel and sdist versions differ.

``twine check --strict`` (metadata and README rendering) runs separately in
``build_package.sh``.
"""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path

MAX_FILE_BYTES = 5 * 1024 * 1024
FORBIDDEN_SDIST_PARTS = {
    "site",
    "htmlcov",
    "dist",
    "build",
    ".pixi",
    ".venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".hypothesis",
}
FORBIDDEN_SDIST_NAMES = {"coverage.xml", ".coverage"}
REQUIRED_MODULES = ("__init__.py", "py.typed", "losses/__init__.py", "metrics/__init__.py")


def check_wheel(path: Path) -> list[str]:
    errors: list[str] = []
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        bad = zf.testzip()
        if bad is not None:
            errors.append(f"{path.name}: corrupt member {bad}")
        for info in zf.infolist():
            if info.file_size > MAX_FILE_BYTES:
                errors.append(f"{path.name}: {info.filename} is {info.file_size} bytes")
    dist_info = [n for n in names if ".dist-info/" in n]
    package = [n for n in names if n.startswith("torchregress/")]
    stray = sorted(set(names) - set(dist_info) - set(package))
    if stray:
        errors.append(f"{path.name}: files outside torchregress/: {stray[:10]}")
    for module in REQUIRED_MODULES:
        if f"torchregress/{module}" not in names:
            errors.append(f"{path.name}: missing torchregress/{module}")
    if not any(n.endswith(".dist-info/licenses/LICENSE") for n in names):
        errors.append(f"{path.name}: LICENSE not in .dist-info/licenses/")
    if any("__pycache__" in n or n.endswith(".pyc") for n in names):
        errors.append(f"{path.name}: contains bytecode")
    return errors


def check_sdist(path: Path) -> list[str]:
    errors: list[str] = []
    with tarfile.open(path) as tf:
        members = tf.getmembers()
    for member in members:
        parts = Path(member.name).parts[1:]  # drop the torchregress-X.Y.Z/ root
        if not parts:
            continue
        if FORBIDDEN_SDIST_PARTS & set(parts) or parts[-1] in FORBIDDEN_SDIST_NAMES:
            errors.append(f"{path.name}: build/tooling output {member.name}")
        if member.isfile() and member.size > MAX_FILE_BYTES:
            errors.append(f"{path.name}: {member.name} is {member.size} bytes")
    names = {"/".join(Path(m.name).parts[1:]) for m in members}
    for required in ("pyproject.toml", "README.md", "LICENSE", "src/torchregress/__init__.py"):
        if required not in names:
            errors.append(f"{path.name}: missing {required}")
    return errors


def version_of(name: str) -> str:
    stem = name.removesuffix(".tar.gz").removesuffix(".whl")
    return stem.split("-")[1]


def main(argv: list[str]) -> int:
    dist = Path(argv[1] if len(argv) > 1 else "dist")
    wheels = sorted(dist.glob("torchregress-*.whl"))
    sdists = sorted(dist.glob("torchregress-*.tar.gz"))
    errors: list[str] = []
    if len(wheels) != 1 or len(sdists) != 1:
        errors.append(f"expected one wheel and one sdist in {dist}, got {wheels + sdists}")
    for wheel in wheels:
        errors += check_wheel(wheel)
    for sdist in sdists:
        errors += check_sdist(sdist)
    if wheels and sdists and version_of(wheels[0].name) != version_of(sdists[0].name):
        errors.append(f"version mismatch: {wheels[0].name} vs {sdists[0].name}")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if not errors:
        print(f"OK: {', '.join(p.name for p in wheels + sdists)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
