#!/usr/bin/env python3
"""Concatenate CSV files matching a glob (same header). Skips missing paths."""

from __future__ import annotations

import argparse
import csv
import glob as glob_module
import sys
from pathlib import Path


def collate_paths(paths: list[Path], out: Path) -> int:
    paths = [x for x in paths if x.is_file()]
    print(f"[collate_csv_glob] collate_paths: {len(paths)} existing files -> {out}", flush=True)
    if not paths:
        print(
            "[collate_csv_glob] ERROR: no input CSV paths after filtering; nothing written.",
            file=sys.stderr,
            flush=True,
        )
        return 1
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            for row in reader:
                rows.append({k: row.get(k, "") for k in fieldnames})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=fieldnames or [])
        w.writeheader()
        w.writerows(rows)
    print(
        f"[collate_csv_glob] wrote {out} ({len(rows)} rows from {len(paths)} files)",
        flush=True,
    )
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--glob", required=True, help="Glob of CSV files (quote for shell).")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    raw = sorted(glob_module.glob(args.glob))
    print(
        f"[collate_csv_glob] glob expanded to {len(raw)} paths (before exists filter)", flush=True
    )
    paths = [Path(s) for s in raw]
    raise SystemExit(collate_paths(paths, args.out))


if __name__ == "__main__":
    main()
