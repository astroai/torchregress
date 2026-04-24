"""Shared helpers for SAGE / SPT / joint tabular paper JSON digests."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any


def _bootstrap_mean_ci(
    values: list[float],
    *,
    n_boot: int,
    rng: random.Random,
) -> tuple[float, float]:
    """Percentile CI (2.5 / 97.5) on the bootstrap distribution of the sample mean."""
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        v = values[0]
        return v, v
    n = len(values)
    means: list[float] = []
    for _ in range(n_boot):
        s = 0.0
        for _i in range(n):
            s += values[rng.randint(0, n - 1)]
        means.append(s / n)
    means.sort()
    lo_i = max(0, int(0.025 * n_boot))
    hi_i = min(n_boot - 1, int(0.975 * n_boot))
    return means[lo_i], means[hi_i]


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _multiseed_gap_bootstrap_table(
    seed_rows: list[dict[str, Any]],
    *,
    n_boot: int,
    rng_seed: int,
) -> list[dict[str, Any]]:
    if n_boot <= 0 or not seed_rows:
        return []
    grouped: dict[str, tuple[list[float], list[float]]] = {}
    for row in seed_rows:
        benchmark = str(row.get("Benchmark", ""))
        if not benchmark:
            continue
        sage_gap = _finite_float(row.get("SAGEMinusSupervised"))
        conf_gap = _finite_float(row.get("ConfidenceMinusSupervised"))
        if sage_gap is None or conf_gap is None:
            continue
        if benchmark not in grouped:
            grouped[benchmark] = ([], [])
        grouped[benchmark][0].append(sage_gap)
        grouped[benchmark][1].append(conf_gap)
    rng = random.Random(rng_seed)
    out: list[dict[str, Any]] = []
    for benchmark in sorted(grouped.keys()):
        sage_gaps, conf_gaps = grouped[benchmark]
        s_lo, s_hi = _bootstrap_mean_ci(sage_gaps, n_boot=n_boot, rng=rng)
        c_lo, c_hi = _bootstrap_mean_ci(conf_gaps, n_boot=n_boot, rng=rng)
        out.append(
            {
                "Benchmark": benchmark,
                "SAGEMeanGapBoot95Low": s_lo,
                "SAGEMeanGapBoot95High": s_hi,
                "ConfidenceMeanGapBoot95Low": c_lo,
                "ConfidenceMeanGapBoot95High": c_hi,
                "BootstrapSamples": n_boot,
            }
        )
    return out


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


def summarize_multiseed(
    payload: dict[str, Any],
    *,
    multiseed_bootstrap_n: int = 2_000,
    multiseed_bootstrap_seed: int = 42,
) -> dict[str, Any]:
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
    gap_boot = _multiseed_gap_bootstrap_table(
        seed_rows,
        n_boot=multiseed_bootstrap_n,
        rng_seed=multiseed_bootstrap_seed,
    )
    return {
        "tuning_csv_path": payload.get("tuning_csv_path"),
        "seeds": payload.get("seeds"),
        "aggregate": agg_out,
        "per_seed": per_seed,
        "gap_bootstrap_95": gap_boot,
        "gap_bootstrap_meta": {
            "n": multiseed_bootstrap_n,
            "seed": multiseed_bootstrap_seed,
            "method": "nonparametric_bootstrap_resample_seeds_mean_gap_percentile_95",
        },
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
