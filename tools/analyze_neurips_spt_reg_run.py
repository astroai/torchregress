"""Summarize a completed ``run_neurips_spt_reg_full.py`` tree (full / audit JSONs).

Usage::

    uv run python tools/analyze_neurips_spt_reg_run.py \\
      --run-root reports/neurips_spt_reg/runs/2026-04-11/neurips_spt_reg_full

Prints key Gaussian rows (NLL, CRPS, Cov90, Width90) so you can compare
``RawSplitConformalGaussian`` vs. ``WeightedSplitConformalGaussian`` vs. transport
rows without opening large JSON by hand. If ``large_tabular/year_competing_methods_full.json`` (or legacy ``yolanda/``)
exists under ``--run-root``, prints the same block for that track.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_year_rows(run_root: Path, sub: str) -> tuple[list[dict[str, Any]], str | None]:
    manifest_path = run_root / sub / "artifact_manifest.json"
    if not manifest_path.is_file():
        return [], f"missing {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile = str(manifest.get("profile") or sub)
    year_path = run_root / sub / f"year_competing_methods_{profile}.json"
    if not year_path.is_file():
        alt = run_root / sub / "year_competing_methods_full.json"
        year_path = alt if alt.is_file() else year_path
    if not year_path.is_file():
        return [], f"missing year summary under {run_root / sub}"
    payload = json.loads(year_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        return [], f"bad rows in {year_path}"
    return rows, None


def _row_by_method(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        if isinstance(r, dict) and r.get("Method"):
            out[str(r["Method"])] = r
    return out


def _print_block(title: str, rows: list[dict[str, Any]]) -> None:
    by_m = _row_by_method(rows)
    keys = (
        "SourceGaussian",
        "RawSplitConformalGaussian",
        "WeightedSplitConformalGaussian",
        "PriorTransportGaussian",
        "SPTTransportGaussian",
        "SPTRegGaussian",
        "TargetRefitSmallGaussian",
    )
    print(f"\n=== {title} ===")
    print(f"{'Method':<34} {'NLL':>10} {'CRPS':>10} {'Cov90':>8} {'Width90':>10}")
    for name in keys:
        r = by_m.get(name)
        if not r:
            continue
        print(
            f"{name:<34} "
            f"{r.get('NLL', ''):>10} "
            f"{r.get('CRPS', ''):>10} "
            f"{r.get('Cov90', ''):>8} "
            f"{r.get('Width90', ''):>10}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Output root from run_neurips_spt_reg_full.py",
    )
    args = p.parse_args()
    run_root = args.run_root.resolve()
    if not run_root.is_dir():
        raise SystemExit(f"Not a directory: {run_root}")

    for sub, label in (("full", "Profile full"), ("audit", "Profile audit")):
        rows, err = _load_year_rows(run_root, sub)
        if err:
            print(f"[{sub}] skip: {err}")
            continue
        _print_block(label, rows)

    stage = run_root / "stage_a_sweep" / "stage_a_sweep.json"
    if stage.is_file():
        payload = json.loads(stage.read_text(encoding="utf-8"))
        print("\n=== Stage-A sweep (prior_ratio_clip) ===")
        for row in payload.get("rows", []):
            print(row)

    ypath = run_root / "large_tabular" / "year_competing_methods_full.json"
    if not ypath.is_file():
        ypath = run_root / "yolanda" / "year_competing_methods_full.json"
    if ypath.is_file():
        payload = json.loads(ypath.read_text(encoding="utf-8"))
        rows = payload.get("rows", [])
        if isinstance(rows, list) and rows:
            _print_block("Extra-large OpenML track (profile full)", rows)


if __name__ == "__main__":
    main()
