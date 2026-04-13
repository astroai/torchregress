"""Merge ``year_direct_nl*_summary.json`` files into one table for paper tables / plots."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def _nl_from_path(path: Path) -> int | None:
    m = re.search(r"nl(\d+)_summary\.json$", path.name)
    return int(m.group(1)) if m else None


def _row_for_nl(payload: dict[str, Any], nl: int) -> dict[str, Any]:
    rows = payload.get("rows", [])
    by_method: dict[str, dict[str, Any]] = {}
    for row in rows:
        if float(row.get("UnlabeledFraction", 0.0)) != 1.0:
            continue
        m = str(row.get("Method", ""))
        if m:
            by_method[m] = row
    sup = by_method.get("SupervisedOnly", {})
    sage = by_method.get("SAGE-Reg", {})
    conf = by_method.get("ConfidenceWeightedPseudoLabel", {})
    return {
        "n_labeled": nl,
        "n_unlabeled": (payload.get("config") or {}).get("n_unlabeled"),
        "n_test": (payload.get("config") or {}).get("n_test"),
        "NLL_SupervisedOnly": sup.get("NLL"),
        "NLL_SAGE-Reg": sage.get("NLL"),
        "NLL_ConfidenceWeighted": conf.get("NLL"),
        "NLL_SAGEMinusSupervised": _delta(sage.get("NLL"), sup.get("NLL")),
        "CRPS_SupervisedOnly": sup.get("CRPS"),
        "CRPS_SAGE-Reg": sage.get("CRPS"),
        "Cov90_SupervisedOnly": sup.get("Cov90"),
        "Cov90_SAGE-Reg": sage.get("Cov90"),
        "CalibMAE_SupervisedOnly": sup.get("CalibMAE"),
        "CalibMAE_SAGE-Reg": sage.get("CalibMAE"),
        "RMSE_SupervisedOnly": sup.get("RMSE"),
        "RMSE_SAGE-Reg": sage.get("RMSE"),
    }


def _delta(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def collate(paths: list[Path]) -> dict[str, Any]:
    keyed: list[tuple[int, Path]] = []
    for p in paths:
        nl = _nl_from_path(p)
        if nl is None:
            continue
        keyed.append((nl, p))
    keyed.sort(key=lambda t: t[0])
    table: list[dict[str, Any]] = []
    for nl, p in keyed:
        payload = json.loads(p.read_text(encoding="utf-8"))
        table.append(_row_for_nl(payload, nl))
    return {
        "artifact": "sage_year_labeled_sweep_collated",
        "version": 1,
        "source_files": [str(p) for _, p in keyed],
        "rows": table,
    }


def _write_csv(table: list[dict[str, Any]], path: Path) -> None:
    if not table:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(table[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing year_direct_nl*_summary.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Write merged machine-readable table",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV with the same rows",
    )
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("year_direct_nl*_summary.json"))
    if not paths:
        raise SystemExit(f"no year_direct_nl*_summary.json under {args.input_dir}")
    report = collate(paths)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_json} ({len(report['rows'])} rows)")
    if args.output_csv:
        _write_csv(report["rows"], args.output_csv)
        print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
