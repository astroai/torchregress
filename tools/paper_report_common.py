"""Shared helpers for SAGE / SPT / joint tabular paper JSON digests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SAGE_YEAR_METRICS = (
    "UnlabeledFraction",
    "RMSE",
    "NLL",
    "CRPS",
    "Cov90",
    "CalibMAE",
    "MeanWeight",
    "MeanDisagreement",
    "train_s",
    "eval_s",
)

SPT_YEAR_METRICS = (
    "MSE",
    "MAE",
    "NLL",
    "CRPS",
    "Cov90",
    "Width90",
    "TailRMSE90",
    "train_s",
    "eval_s",
)

MULTISEED_AGG_KEYS = (
    "Benchmark",
    "Seeds",
    "SAGEMinusSupervisedMean",
    "SAGEMinusSupervisedMedian",
    "SAGEMinusSupervisedStd",
    "ConfidenceMinusSupervisedMean",
    "ConfidenceMinusSupervisedMedian",
    "ConfidenceMinusSupervisedStd",
    "SAGEMeanWeightMean",
    "SAGEMeanDisagreementMean",
)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def rows_by_method(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        m = str(row.get("Method", ""))
        if m:
            out[m] = row
    return out


def summarize_sage_year_direct(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows", [])
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        method = str(row.get("Method", ""))
        if not method:
            continue
        slim = {k: row.get(k) for k in SAGE_YEAR_METRICS if k in row}
        by_method.setdefault(method, []).append(slim)
    return {
        "task": payload.get("task"),
        "example": payload.get("example"),
        "config": payload.get("config"),
        "by_method": by_method,
    }


def summarize_multiseed(payload: dict[str, Any]) -> dict[str, Any]:
    aggregates = payload.get("aggregate_rows", [])
    agg_out = []
    for row in aggregates:
        agg_out.append({k: row.get(k) for k in MULTISEED_AGG_KEYS if k in row})
    seed_rows = payload.get("seed_rows", [])
    per_seed = []
    for row in seed_rows:
        per_seed.append(
            {
                "Benchmark": row.get("Benchmark"),
                "Seed": row.get("Seed"),
                "SAGEMinusSupervised": row.get("SAGEMinusSupervised"),
                "ConfidenceMinusSupervised": row.get("ConfidenceMinusSupervised"),
                "SupervisedObjective": row.get("SupervisedObjective"),
                "SAGEObjective": row.get("SAGEObjective"),
                "ConfidenceObjective": row.get("ConfidenceObjective"),
                "SupervisedExtra": row.get("SupervisedExtra"),
                "SAGEExtra": row.get("SAGEExtra"),
                "ConfidenceExtra": row.get("ConfidenceExtra"),
            }
        )
    return {
        "tuning_csv_path": payload.get("tuning_csv_path"),
        "seeds": payload.get("seeds"),
        "aggregate": agg_out,
        "per_seed": per_seed,
    }


def summarize_spt_year(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows", [])
    by_m = rows_by_method(rows)
    slim: dict[str, dict[str, Any]] = {}
    for method, row in by_m.items():
        slim[method] = {k: row.get(k) for k in SPT_YEAR_METRICS if k in row}
    return {
        "task": payload.get("task"),
        "config": payload.get("config"),
        "methods": slim,
    }
