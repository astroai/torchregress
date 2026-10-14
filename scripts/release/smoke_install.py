#!/usr/bin/env python3
"""Smoke-test an installed torchregress with only its core dependencies.

Run in a fresh environment after ``pip install torchregress`` (no extras)::

    python scripts/release/smoke_install.py

Checks that every public submodule imports with the core dependencies only
(``viz`` must instead fail with an ImportError naming its extra), and runs the
packaged ``torchregress-health`` check.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Submodules whose import needs an optional extra, and the extra's name.
NEEDS_EXTRA = {"viz": "viz"}


def main() -> int:
    import torchregress

    print(f"torchregress {torchregress.__version__} from {torchregress.__file__}")
    failures: list[str] = []
    for name in sorted(torchregress._LAZY_SUBMODULES):
        try:
            importlib.import_module(f"torchregress.{name}")
        except ImportError as exc:
            extra = NEEDS_EXTRA.get(name)
            if extra is not None and f"torchregress[{extra}]" in str(exc):
                print(f"  {name}: needs [{extra}] (expected)")
                continue
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
        else:
            print(f"  {name}: ok")
    # The CI venv runs this file by path and does not put the venv's bin on
    # PATH, so which() alone misses the console script installed next to
    # sys.executable.
    bindir = str(Path(sys.executable).parent)
    health = shutil.which(
        "torchregress-health",
        path=os.pathsep.join([bindir, os.environ.get("PATH", "")]),
    )
    if health is None:
        failures.append("torchregress-health entry point not installed")
    else:
        result = subprocess.run([health], capture_output=True, text=True)
        if result.returncode != 0 or "[SUCCESS]" not in result.stdout:
            failures.append(f"torchregress-health failed:\n{result.stdout}\n{result.stderr}")
        else:
            print("  torchregress-health: ok")
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
