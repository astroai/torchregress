#!/usr/bin/env python3
"""Delete bulky, non-portable files under ``docs/research/sage_reg_results/``.

Typical accidents (same as bloating ``site/research`` from MkDocs before ``exclude_docs``):

- ``*.zip`` under ``sage_reg_results`` (e.g. raw Higgs download; easy to re-fetch)
- ``*.parquet`` **except** ``FAIR_Universe_HiggsML_data.parquet`` (that file is expensive to
  replace; omit with ``--delete-fair-universe-higgs-parquet`` only after you have moved or
  re-downloaded it to ``data/neurips_inputs/`` or VOS)
- ``openml_year.csv`` copies under ``sage_reg_results/`` (Year cache belongs in ``data/paper/``)

Metrics the repo keeps are mostly ``.json``, ``.csv``, ``.md``, ``.png`` (see ``.gitignore``). This
tool does **not** remove those by default.

Usage::

    uv run python tools/clean_sage_reg_results_heavy_artifacts.py --dry-run
    uv run python tools/clean_sage_reg_results_heavy_artifacts.py
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print paths only; do not delete.",
    )
    p.add_argument(
        "--delete-fair-universe-higgs-parquet",
        action="store_true",
        help=(
            "Also delete FAIR_Universe_HiggsML_data.parquet under sage_reg_results (large). "
            "Only use after the file exists elsewhere (e.g. data/neurips_inputs/ or VOS)."
        ),
    )
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    root = repo / "docs" / "research" / "sage_reg_results"
    if not root.is_dir():
        print(f"Nothing to do (missing): {root}")
        return 0

    removed = 0
    bytes_freed = 0

    for path in root.rglob("*.zip"):
        if not path.is_file():
            continue
        try:
            sz = path.stat().st_size
        except OSError:
            continue
        rel = path.relative_to(repo)
        sz_mb = sz / (1024 * 1024)
        tag = "[dry-run] " if args.dry_run else ""
        print(f"{tag}{rel} ({sz_mb:.1f} MiB)", flush=True)
        if not args.dry_run:
            path.unlink()
            removed += 1
            bytes_freed += sz

    for path in root.rglob("*.parquet"):
        if not path.is_file():
            continue
        if (
            path.name == "FAIR_Universe_HiggsML_data.parquet"
            and not args.delete_fair_universe_higgs_parquet
        ):
            print(
                f"[skip] {path.relative_to(repo)} (FAIR Universe parquet; "
                "use --delete-fair-universe-higgs-parquet only after it is safe)",
                flush=True,
            )
            continue
        try:
            sz = path.stat().st_size
        except OSError:
            continue
        rel = path.relative_to(repo)
        sz_mb = sz / (1024 * 1024)
        tag = "[dry-run] " if args.dry_run else ""
        print(f"{tag}{rel} ({sz_mb:.1f} MiB)", flush=True)
        if not args.dry_run:
            path.unlink()
            removed += 1
            bytes_freed += sz

    for path in root.rglob("openml_year.csv"):
        if not path.is_file():
            continue
        try:
            sz = path.stat().st_size
        except OSError:
            continue
        rel = path.relative_to(repo)
        tag = "[dry-run] " if args.dry_run else ""
        print(f"{tag}{rel} ({sz / (1024 * 1024):.1f} MiB) [misplaced Year cache]", flush=True)
        if not args.dry_run:
            path.unlink()
            removed += 1
            bytes_freed += sz

    if args.dry_run:
        print("\nDry-run only. Re-run without --dry-run to delete.", flush=True)
    else:
        print(f"\nRemoved {removed} file(s), ~{bytes_freed / (1024**3):.2f} GiB.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
