#!/usr/bin/env python3
"""One-shot SPT-Reg NeurIPS evidence: real Year, full + audit renders, Stage-A sweep, extras.

Usage (repo root)::

    uv run python scripts/run_neurips_spt_reg_full.py
    uv run python scripts/run_neurips_spt_reg_full.py --quick

Defaults use ``data/paper/openml_year.csv`` (materialized via OpenML when allowed).

**Default (non-quick):** extra-large OpenML regression track (**diamonds**, OpenML **42225**)
and **Shifts** placeholder layout are **on** (photo-z stays **off**). ``--quick`` skips
that track; use ``--skip-large-tabular`` (alias ``--skip-yolanda``) / ``--skip-shifts``
to opt out of full runs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

YEAR_CACHE_DEFAULT = REPO_ROOT / "data" / "paper" / "openml_year.csv"
LARGE_TABULAR_CACHE_DEFAULT = REPO_ROOT / "data" / "paper" / "openml_large_tabular_diamonds.parquet"
LARGE_TABULAR_OPENML_ID_DEFAULT = 42225  # ggplot2 diamonds, ~54k rows; stable fetch_openml
LARGE_TABULAR_MAX_ROWS_DEFAULT = 250_000
SHIFTS_OUT_ROOT_DEFAULT = REPO_ROOT / "data" / "shifts"
SHIFTS_DATASET_DEFAULT = "solar"


def _uv_run(parts: list[str], *, cwd: Path | None = None, check: bool = True) -> int:
    cmd = [sys.executable, *parts]
    print("==", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(cwd or REPO_ROOT), env=os.environ.copy(), check=False)
    if check and r.returncode != 0:
        raise subprocess.CalledProcessError(r.returncode, cmd, None, None)
    return int(r.returncode)


def _ensure_year_cache(cache: Path, *, allow_download: bool) -> None:
    if cache.is_file():
        return
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not allow_download:
        raise SystemExit(f"Year cache missing and --no-year-download: {cache}")
    _uv_run(
        [
            str(REPO_ROOT / "tools/materialize_openml_year.py"),
            "--cache-path",
            str(cache),
        ]
    )


def _render(
    *,
    profile: str,
    run_subdir: Path,
    year_cache: Path,
    year_allow_download: bool,
    include_photoz: bool,
    large_tabular_openml: bool,
    large_tabular_cache: Path,
    large_tabular_openml_id: int,
    large_tabular_max_rows: int,
) -> dict[str, Any]:
    run_subdir.mkdir(parents=True, exist_ok=True)
    report_path = run_subdir / "artifact_manifest.json"
    parts = [
        str(REPO_ROOT / "tools/render_spt_reg_paper_artifacts.py"),
        "--profile",
        profile,
        "--output-dir",
        str(run_subdir),
        "--report",
        str(report_path),
    ]
    if include_photoz:
        parts.append("--include-photoz")
    if large_tabular_openml:
        parts += [
            "--year-openml-data-id",
            str(large_tabular_openml_id),
            "--year-max-dataset-rows",
            str(large_tabular_max_rows),
            "--year-cache-path",
            str(large_tabular_cache),
        ]
        if year_allow_download:
            parts.append("--year-allow-download")
    else:
        parts += ["--year-cache-path", str(year_cache)]
        if year_allow_download:
            parts.append("--year-allow-download")
    _uv_run(parts)
    return json.loads(report_path.read_text(encoding="utf-8"))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help=(
            "Output root (default: reports/neurips_spt_reg/runs/<UTC-date>/neurips_spt_reg_full)."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke budgets (renderer profile smoke)",
    )
    parser.add_argument("--no-year-download", action="store_true")
    parser.add_argument("--year-cache", type=Path, default=YEAR_CACHE_DEFAULT)
    parser.add_argument("--skip-stage-a-sweep", action="store_true")
    parser.add_argument("--include-photoz", action="store_true")
    parser.add_argument(
        "--skip-large-tabular",
        "--skip-yolanda",
        action="store_true",
        dest="skip_large_tabular",
        help="Skip extra-large OpenML regression track (default: OpenML 42225 diamonds).",
    )
    parser.add_argument(
        "--large-tabular-cache",
        "--yolanda-cache",
        type=Path,
        default=LARGE_TABULAR_CACHE_DEFAULT,
        dest="large_tabular_cache",
        help="Cache path for the large-tabular OpenML track (.parquet or .csv).",
    )
    parser.add_argument(
        "--large-tabular-openml-id",
        "--yolanda-openml-id",
        type=int,
        default=LARGE_TABULAR_OPENML_ID_DEFAULT,
        dest="large_tabular_openml_id",
        help="OpenML data_id for the large-tabular track (default: 42225 diamonds).",
    )
    parser.add_argument(
        "--large-tabular-max-rows",
        "--yolanda-max-rows",
        type=int,
        default=LARGE_TABULAR_MAX_ROWS_DEFAULT,
        dest="large_tabular_max_rows",
        help="Cap rows after fetch (subsample) for the large-tabular track.",
    )
    parser.add_argument(
        "--skip-shifts",
        action="store_true",
        help="Skip Shifts dataset placeholder materialization under data/shifts/.",
    )
    parser.add_argument(
        "--shifts-out-root",
        type=Path,
        default=SHIFTS_OUT_ROOT_DEFAULT,
        help="Root for Shifts placeholder README (default: data/shifts).",
    )
    parser.add_argument(
        "--shifts-dataset",
        type=str,
        default=SHIFTS_DATASET_DEFAULT,
        help="Symbolic Shifts dataset key for placeholder layout (default: solar).",
    )
    args = parser.parse_args()

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run_root = args.run_root
    if run_root is None:
        run_root = (
            REPO_ROOT / "reports" / "neurips_spt_reg" / "runs" / date_str / "neurips_spt_reg_full"
        )
    run_root = run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    year_cache = args.year_cache.resolve()
    allow_y = not args.no_year_download
    _ensure_year_cache(year_cache, allow_download=allow_y)

    quick = args.quick
    profile_main = "smoke" if quick else "full"
    profile_audit = "smoke" if quick else "audit"
    lt_rows = min(4096, args.large_tabular_max_rows) if quick else args.large_tabular_max_rows
    include_large_tabular = (not args.skip_large_tabular) and (not quick)
    include_shifts = not args.skip_shifts

    manifest: dict[str, Any] = {
        "artifact": "neurips_spt_reg_full_manifest",
        "version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "quick": quick,
        "year_cache": str(year_cache),
        "defaults": {
            "large_tabular_openml": include_large_tabular,
            "large_tabular_openml_id": int(args.large_tabular_openml_id),
            "shifts": include_shifts,
            "photoz": bool(args.include_photoz),
        },
        "phases": {},
    }

    # 1) Core: full + audit layouts (directory names fixed for aggregate_spt_paper_report)
    full_dir = run_root / "full"
    manifest["phases"]["render_full"] = str(full_dir)
    manifest["render_full_report"] = _render(
        profile=profile_main,
        run_subdir=full_dir,
        year_cache=year_cache,
        year_allow_download=allow_y,
        include_photoz=False,
        large_tabular_openml=False,
        large_tabular_cache=args.large_tabular_cache,
        large_tabular_openml_id=args.large_tabular_openml_id,
        large_tabular_max_rows=lt_rows,
    )

    audit_dir = run_root / "audit"
    manifest["phases"]["render_audit"] = str(audit_dir)
    manifest["render_audit_report"] = _render(
        profile=profile_audit,
        run_subdir=audit_dir,
        year_cache=year_cache,
        year_allow_download=allow_y,
        include_photoz=False,
        large_tabular_openml=False,
        large_tabular_cache=args.large_tabular_cache,
        large_tabular_openml_id=args.large_tabular_openml_id,
        large_tabular_max_rows=lt_rows,
    )

    # 2) Optional photo-z (full renderer profile, separate tree)
    if args.include_photoz:
        pz = run_root / "photoz"
        manifest["phases"]["render_photoz"] = str(pz)
        manifest["render_photoz_report"] = _render(
            profile=profile_main,
            run_subdir=pz,
            year_cache=year_cache,
            year_allow_download=allow_y,
            include_photoz=True,
            large_tabular_openml=False,
            large_tabular_cache=args.large_tabular_cache,
            large_tabular_openml_id=args.large_tabular_openml_id,
            large_tabular_max_rows=lt_rows,
        )
    else:
        manifest["phases"]["render_photoz"] = "skipped"

    # 3) Extra-large OpenML regression (default: diamonds / 42225)
    if include_large_tabular:
        ltdir = run_root / "large_tabular"
        manifest["phases"]["render_large_tabular"] = str(ltdir)
        args.large_tabular_cache.parent.mkdir(parents=True, exist_ok=True)
        manifest["render_large_tabular_report"] = _render(
            profile=profile_main,
            run_subdir=ltdir,
            year_cache=year_cache,
            year_allow_download=allow_y,
            include_photoz=False,
            large_tabular_openml=True,
            large_tabular_cache=args.large_tabular_cache.resolve(),
            large_tabular_openml_id=args.large_tabular_openml_id,
            large_tabular_max_rows=lt_rows,
        )
    else:
        manifest["phases"]["render_large_tabular"] = "skipped_quick" if quick else "skipped_opt_out"

    # 4) Stage-A prior-ratio clip sweep
    stage_dir = run_root / "stage_a_sweep"
    stage_json = stage_dir / "stage_a_sweep.json"
    if not args.skip_stage_a_sweep:
        stage_dir.mkdir(parents=True, exist_ok=True)
        clips = ["2.0"] if quick else ["1.5", "2.0", "3.0", "5.0"]
        sweep_rc = _uv_run(
            [
                str(REPO_ROOT / "tools/sweep_spt_year_stage_a.py"),
                "--cache-path",
                str(year_cache),
                "--output-json",
                str(stage_json),
                "--clips",
                *clips,
            ]
            + (["--quick"] if quick else [])
            + (["--allow-download"] if allow_y else []),
            check=False,
        )
        manifest["phases"]["stage_a_sweep"] = str(stage_json) if sweep_rc == 0 else "failed"
    else:
        manifest["phases"]["stage_a_sweep"] = "skipped"

    # 5) Shifts helper (Phase-4 hook; does not gate aggregation)
    if include_shifts:
        shifts_out = args.shifts_out_root.resolve()
        sh_cmd = [
            str(REPO_ROOT / "tools/fetch_shifts_dataset.py"),
            "--out-root",
            str(shifts_out),
            "--dataset",
            str(args.shifts_dataset),
        ]
        sh_rc = _uv_run(sh_cmd, check=False)
        readme = shifts_out / args.shifts_dataset / "README.txt"
        if sh_rc == 0 and readme.is_file():
            manifest["phases"]["shifts"] = str(readme)
        elif sh_rc != 0:
            manifest["phases"]["shifts"] = "helper_failed"
        else:
            manifest["phases"]["shifts"] = "incomplete"
    else:
        manifest["phases"]["shifts"] = "skipped"

    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    (run_root / "neurips_spt_reg_full_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    _uv_run(
        [
            str(REPO_ROOT / "tools/aggregate_spt_paper_report.py"),
            "--run-root",
            str(run_root),
            "--write-markdown",
        ]
    )
    print(f"\nDone. Run root:\n  {run_root}", flush=True)


if __name__ == "__main__":
    main()
