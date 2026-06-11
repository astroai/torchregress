"""Shared helpers for reproducible comparison examples and summary artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import torch

from torchregress.utils.pytorch_compat import set_all_seeds

__all__ = [
    "compute_point_metrics",
    "print_comparison_summary",
    "print_fairness_notes",
    "set_comparison_seed",
    "timed_call",
    "write_comparison_summary_json",
]


def set_comparison_seed(seed: int) -> None:
    """Set Python/NumPy/PyTorch RNGs for reproducible comparisons."""
    set_all_seeds(seed)


def timed_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Run ``fn`` and return ``(result, elapsed_seconds)``."""
    start = perf_counter()
    result = fn(*args, **kwargs)
    elapsed = perf_counter() - start
    return result, elapsed


def compute_point_metrics(y_pred: torch.Tensor, y_true: torch.Tensor) -> dict[str, float]:
    """Compute common point-prediction regression metrics."""
    y_pred_t = y_pred.detach()
    y_true_t = y_true.detach().to(y_pred_t.device)

    mse = torch.mean((y_pred_t - y_true_t) ** 2).item()
    mae = torch.mean(torch.abs(y_pred_t - y_true_t)).item()

    y_true_mean = torch.mean(y_true_t)
    ss_res = torch.sum((y_true_t - y_pred_t) ** 2)
    ss_tot = torch.sum((y_true_t - y_true_mean) ** 2)
    r2 = (1.0 - ss_res / ss_tot).item() if ss_tot.item() > 0 else float("nan")

    return {"MSE": mse, "MAE": mae, "R2": r2}


def print_fairness_notes(
    *,
    title: str,
    seed_policy: str,
    train_budget: str,
    metric_policy: str,
) -> None:
    """Print a compact comparability statement for example outputs."""
    print(f"\n[{title} | comparison fairness]")
    print(f"  seeds:   {seed_policy}")
    print(f"  budget:  {train_budget}")
    print(f"  metrics: {metric_policy}")


def print_comparison_summary(
    title: str,
    rows: list[dict[str, Any]],
    metric_order: list[str] | None = None,
) -> None:
    """Print an aligned summary table for comparison examples."""
    if not rows:
        print(f"\n{title}\n(no rows)")
        return

    metric_order = metric_order or ["MSE", "MAE", "R2", "CRPS", "ECE", "train_s", "eval_s"]
    present_metrics = [m for m in metric_order if any(m in row for row in rows)]
    headers = ["Method", *present_metrics]
    include_notes = any(row.get("Notes") for row in rows)
    if include_notes:
        headers.append("Notes")

    table_rows: list[list[str]] = []
    for row in rows:
        vals = [str(row.get("Method", ""))]
        for key in present_metrics:
            value = row.get(key)
            if value is None:
                vals.append("-")
            elif isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        if include_notes:
            vals.append(str(row.get("Notes", "")))
        table_rows.append(vals)

    widths = [len(h) for h in headers]
    for r in table_rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def _fmt(cells: list[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print(f"\n{title}")
    print(_fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for r in table_rows:
        print(_fmt(r))


def write_comparison_summary_json(
    path: str | Path,
    *,
    example: str,
    task: str,
    config: Any,
    rows: list[dict[str, Any]],
    notes: list[str] | None = None,
) -> Path:
    """Write a machine-readable summary artifact for comparison examples."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    config_payload: Any
    if is_dataclass(config) and not isinstance(config, type):
        config_payload = asdict(config)
    elif hasattr(config, "__dict__"):
        config_payload = dict(vars(config))
    else:
        config_payload = config

    payload = {
        "artifact": "comparison_example_summary",
        "version": 1,
        "example": example,
        "task": task,
        "config": config_payload,
        "rows": rows,
        "notes": notes or [],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
