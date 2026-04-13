"""Shifts in Weather / Shifts-Project regression hook (placeholder).

The Shifts datasets are large and hosted outside this repo. By default (no
``--dry-run``), this tool **materializes** ``<out-root>/<dataset>/README.txt`` so
automation has a stable provenance stub before real adapters land.

``--dry-run`` prints the README text only and writes nothing.

See https://github.com/Shifts-Project/shifts for dataset documentation.

Planned layout (when wired)::

    data/shifts/<dataset_name>/
      README.txt   # provenance + citation
      (raw files downloaded by the user or future automation)
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "data" / "shifts"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory for Shifts assets (default: data/shifts)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="solar",
        help="Symbolic dataset key for future adapters (default: solar)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create README only; do not download (default behavior for now).",
    )
    args = parser.parse_args()

    target = (args.out_root / args.dataset).resolve()
    target.mkdir(parents=True, exist_ok=True)
    readme = target / "README.txt"
    text = (
        "Shifts-Project placeholder.\n\n"
        "Source project: https://github.com/Shifts-Project/shifts\n"
        f"Dataset key: {args.dataset}\n\n"
        "Download assets manually or extend this tool with a stable URL + checksum.\n"
        "torchregress adapters should read from this directory once populated.\n"
    )
    if args.dry_run:
        print(f"[dry-run] would write:\n  {readme}\n")
        print(text)
        return
    readme.write_text(text, encoding="utf-8")
    print(f"Wrote {readme}")


if __name__ == "__main__":
    main()
