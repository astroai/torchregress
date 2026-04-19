#!/usr/bin/env python3
"""Merge torchregress Year SSL CSV rows with user-supplied official RankUp / PabLO metrics.

Designed for experiments: run in-repo benchmarks, run official ``pm25/semi-supervised-regression``
(or paste eval numbers from their logs), write a JSON sidecar, then merge for tables.

Official JSON schema (``--official-json``) — list of entries, each optional fields::

    [
      {
        "track": "rankup_official",
        "method": "RankUp",
        "seed": 260410,
        "RMSE": 8.912,
        "NLL": 2.341,
        "CRPS": null,
        "notes": "eval.py last line / wandb export"
      }
    ]

``--ours-csv`` should be rows from ``self_agreement_realdata_year`` or label-fraction sweep
(containing ``Method``, ``RMSE``, ``NLL``, …). We match on ``Method`` + ``Seed`` when present,
else first row per ``Method``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_official(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "entries" in payload:
        entries = payload["entries"]
    elif isinstance(payload, list):
        entries = payload
    else:
        raise ValueError('official JSON must be a list or {"entries": [...]} ')
    if not isinstance(entries, list):
        raise ValueError("official entries must be a list")
    return [e for e in entries if isinstance(e, dict)]


def _f(x: Any) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _pick_ours_rows(
    rows: list[dict[str, Any]],
    *,
    methods: set[str] | None,
    seed: int | None,
    unlabeled_fraction: float | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        m = str(row.get("Method", ""))
        if methods is not None and m not in methods:
            continue
        if seed is not None and "Seed" in row and str(row["Seed"]).strip() != "":
            if int(float(row["Seed"])) != int(seed):
                continue
        if unlabeled_fraction is not None and "UnlabeledFraction" in row:
            uf = str(row.get("UnlabeledFraction", "")).strip()
            if uf != "" and abs(float(uf) - float(unlabeled_fraction)) > 1e-6:
                continue
        out.append(row)
    return out


def merge(
    *,
    ours_csv: Path,
    official_json: Path,
    methods: set[str] | None,
    seed: int | None,
    unlabeled_fraction: float | None,
    out_json: Path,
    out_csv: Path | None,
) -> dict[str, Any]:
    ours = _pick_ours_rows(
        _load_csv_rows(ours_csv),
        methods=methods,
        seed=seed,
        unlabeled_fraction=unlabeled_fraction,
    )
    official = _load_official(official_json)

    merged_rows: list[dict[str, Any]] = []
    for o in official:
        track = str(o.get("track", "official"))
        method_key = str(o.get("method", ""))
        matched = [r for r in ours if str(r.get("Method", "")) == method_key]
        base = matched[0] if matched else {}
        row: dict[str, Any] = {
            "Track": track,
            "Method": method_key,
            "Seed": o.get("Seed", base.get("Seed", "")),
            "RMSE_ours": _f(base.get("RMSE")),
            "NLL_ours": _f(base.get("NLL")),
            "CRPS_ours": _f(base.get("CRPS")),
            "RMSE_official": _f(o.get("RMSE")),
            "NLL_official": _f(o.get("NLL")),
            "CRPS_official": _f(o.get("CRPS")),
            "notes": o.get("notes", ""),
        }
        if row["RMSE_ours"] is not None and row["RMSE_official"] is not None:
            row["delta_RMSE_ours_minus_official"] = row["RMSE_ours"] - row["RMSE_official"]
        if row["NLL_ours"] is not None and row["NLL_official"] is not None:
            row["delta_NLL_ours_minus_official"] = row["NLL_ours"] - row["NLL_official"]
        merged_rows.append(row)

    payload = {
        "ours_csv": str(ours_csv),
        "official_json": str(official_json),
        "methods_filter": sorted(methods) if methods is not None else None,
        "seed_filter": seed,
        "ours_unlabeled_fraction": unlabeled_fraction,
        "rows": merged_rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[merge_ssl_official_metrics] wrote {out_json} ({len(merged_rows)} merged rows)",
        flush=True,
    )
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        if not merged_rows:
            out_csv.write_text("", encoding="utf-8")
        else:
            keys = list(merged_rows[0].keys())
            with out_csv.open("w", newline="", encoding="utf-8") as handle:
                w = csv.DictWriter(handle, fieldnames=keys)
                w.writeheader()
                w.writerows(merged_rows)
            print(f"[merge_ssl_official_metrics] wrote {out_csv}", flush=True)
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ours-csv", type=Path, required=True)
    p.add_argument("--official-json", type=Path, required=True)
    p.add_argument(
        "--methods",
        type=str,
        default="RankUp,PabLOPseudo",
        help="Comma-separated Method names to pull from ours CSV (default: RankUp,PabLOPseudo).",
    )
    p.add_argument("--seed", type=int, default=None, help="If set, match ours rows with this Seed.")
    p.add_argument(
        "--ours-unlabeled-fraction",
        type=float,
        default=None,
        help="If set, only ours rows whose UnlabeledFraction matches (e.g. 1.0).",
    )
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, default=None)
    args = p.parse_args()
    methods = {m.strip() for m in str(args.methods).split(",") if m.strip()}
    merge(
        ours_csv=args.ours_csv,
        official_json=args.official_json,
        methods=methods,
        seed=args.seed,
        unlabeled_fraction=args.ours_unlabeled_fraction,
        out_json=args.out_json,
        out_csv=args.out_csv,
    )
    print(f"Wrote {args.out_json}", flush=True)
    if args.out_csv:
        print(f"Wrote {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
