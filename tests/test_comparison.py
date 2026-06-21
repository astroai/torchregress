"""Tests for torchregress.comparison — reproducible comparison helpers."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from torchregress.comparison import (
    compute_point_metrics,
    print_comparison_summary,
    print_fairness_notes,
    timed_call,
    write_comparison_summary_json,
)

# ── timed_call ───────────────────────────────────────────────────────


def test_timed_call_returns_result_and_elapsed():
    def add(a: float, b: float) -> float:
        return a + b

    result, elapsed = timed_call(add, 3.0, 4.0)
    assert result == 7.0
    assert elapsed >= 0.0
    assert isinstance(elapsed, float)


def test_timed_call_with_kwargs():
    def greet(name: str, greeting: str = "Hello") -> str:
        return f"{greeting}, {name}"

    result, elapsed = timed_call(greet, "World", greeting="Hi")
    assert result == "Hi, World"
    assert elapsed >= 0.0


def test_timed_call_measures_non_zero_for_slow_fn():
    def slow() -> None:
        time.sleep(0.01)

    _, elapsed_slow = timed_call(slow)
    assert elapsed_slow > 0.0


def test_timed_call_returns_none_result():
    def noop() -> None:
        return None

    result, elapsed = timed_call(noop)
    assert result is None
    assert elapsed >= 0.0


# ── compute_point_metrics ────────────────────────────────────────────


def test_compute_point_metrics_mse_mae_r2():
    y_pred = torch.tensor([[3.0], [5.0], [7.0]])
    y_true = torch.tensor([[2.0], [5.0], [8.0]])

    metrics = compute_point_metrics(y_pred, y_true)
    # MSE = ((1)^2 + 0^2 + (-1)^2) / 3 = 2/3
    assert metrics["MSE"] == pytest.approx(2.0 / 3.0)
    # MAE = (|1| + 0 + |1|) / 3 = 2/3
    assert metrics["MAE"] == pytest.approx(2.0 / 3.0)
    # R2 = 1 - SS_res / SS_tot
    # y_true_mean = 5, SS_res = 2, SS_tot = (3^2 + 0^2 + 3^2) = 18
    # R2 = 1 - 2/18 = 16/18 ≈ 0.8889
    assert metrics["R2"] == pytest.approx(1.0 - 2.0 / 18.0)


def test_compute_point_metrics_perfect_prediction():
    y = torch.randn(10, 3)
    metrics = compute_point_metrics(y, y)
    assert metrics["MSE"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["MAE"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["R2"] == pytest.approx(1.0)


def test_compute_point_metrics_constant_target_gives_nan_r2():
    y_pred = torch.tensor([[1.0], [2.0], [3.0]])
    y_true = torch.ones(3, 1)  # constant → SS_tot = 0
    metrics = compute_point_metrics(y_pred, y_true)
    assert metrics["MSE"] >= 0
    assert metrics["MAE"] >= 0
    assert math.isnan(metrics["R2"])


def test_compute_point_metrics_detaches_gradients():
    y_pred = torch.tensor([1.0, 2.0], requires_grad=True)
    y_true = torch.tensor([1.5, 2.5])
    metrics = compute_point_metrics(y_pred, y_true)
    assert isinstance(metrics["MSE"], float)


def test_compute_point_metrics_different_device():
    y_pred = torch.randn(4, 2)
    y_true = torch.randn(4, 2).to(torch.float64)
    metrics = compute_point_metrics(y_pred, y_true)
    assert isinstance(metrics["MSE"], float)


# ── print_fairness_notes ─────────────────────────────────────────────


def test_print_fairness_notes_output(capsys):
    print_fairness_notes(
        title="ExampleA",
        seed_policy="fixed(42)",
        train_budget="100 epochs",
        metric_policy="mean ± std (5 runs)",
    )
    captured = capsys.readouterr().out
    assert "[ExampleA | comparison fairness]" in captured
    assert "seeds:" in captured
    assert "fixed(42)" in captured
    assert "budget:" in captured
    assert "100 epochs" in captured
    assert "metrics:" in captured
    assert "mean ± std" in captured


# ── print_comparison_summary ─────────────────────────────────────────


def test_print_comparison_summary_empty(capsys):
    print_comparison_summary("Empty Table", [])
    captured = capsys.readouterr().out
    assert "Empty Table" in captured
    assert "no rows" in captured


def test_print_comparison_summary_basic(capsys):
    rows = [
        {"Method": "Linear", "MSE": 0.1234, "MAE": 0.2500, "eval_s": 0.01},
        {"Method": "MLP", "MSE": 0.0500, "MAE": 0.1800, "eval_s": 0.15},
    ]
    print_comparison_summary("Basic", rows)
    captured = capsys.readouterr().out
    assert "Basic" in captured
    assert "Method" in captured
    assert "MSE" in captured
    assert "0.1234" in captured
    assert "0.0500" in captured


def test_print_comparison_summary_with_notes(capsys):
    rows = [
        {"Method": "ModelA", "MSE": 0.1, "Notes": "baseline"},
        {"Method": "ModelB", "MSE": 0.05, "Notes": ""},
    ]
    print_comparison_summary("Notes", rows)
    captured = capsys.readouterr().out
    assert "Notes" in captured
    assert "baseline" in captured


def test_print_comparison_summary_custom_metric_order(capsys):
    rows = [{"Method": "X", "MAE": 0.1, "R2": 0.9}]
    print_comparison_summary("Custom", rows, metric_order=["R2", "MAE"])
    captured = capsys.readouterr().out
    r2_pos = captured.index("R2")
    mae_pos = captured.index("MAE")
    assert r2_pos < mae_pos  # R2 appears before MAE


def test_print_comparison_summary_missing_metric(capsys):
    rows = [{"Method": "A", "MSE": 0.1}]
    print_comparison_summary("Missing", rows)
    captured = capsys.readouterr().out
    # MAE is in default metric_order but absent from all rows — excluded
    assert "MAE" not in captured


def test_print_comparison_summary_none_value(capsys):
    rows = [{"Method": "NullModel", "MSE": None, "MAE": 0.1}]
    print_comparison_summary("Nulls", rows)
    captured = capsys.readouterr().out
    assert "-" in captured  # None → dash


def test_print_comparison_summary_string_value(capsys):
    rows = [{"Method": "S", "CRPS": "0.05 ± 0.01"}]
    print_comparison_summary("StringVal", rows)
    captured = capsys.readouterr().out
    assert "0.05 ± 0.01" in captured


def test_print_comparison_summary_omits_unrecognized_metrics(capsys):
    """Fields not in metric_order don't appear as columns."""
    rows: list[dict[str, Any]] = [{"Method": "T", "Foo": 1.0, "Bar": 2.0}]
    print_comparison_summary("Test", rows)
    captured = capsys.readouterr().out
    assert "Foo" not in captured
    assert "Bar" not in captured


# ── write_comparison_summary_json ────────────────────────────────────


def test_write_comparison_summary_json_structure(tmp_path):
    out = tmp_path / "summary.json"
    rows = [{"Method": "Linear", "MSE": 0.1}]
    result = write_comparison_summary_json(
        out, example="ex1", task="regression", config={"lr": 0.01}, rows=rows
    )
    assert result == out
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["artifact"] == "comparison_example_summary"
    assert data["version"] == 1
    assert data["example"] == "ex1"
    assert data["task"] == "regression"
    assert data["config"] == {"lr": 0.01}
    assert data["rows"] == rows
    assert data["notes"] == []


def test_write_comparison_summary_json_with_notes(tmp_path):
    out = tmp_path / "summary.json"
    rows: list[dict[str, Any]] = []
    write_comparison_summary_json(
        out,
        example="ex2",
        task="regression",
        config={},
        rows=rows,
        notes=["note1", "note2"],
    )
    data = json.loads(out.read_text())
    assert data["notes"] == ["note1", "note2"]


def test_write_comparison_summary_json_dataclass_config(tmp_path):
    @dataclass
    class Config:
        lr: float = 0.001
        epochs: int = 50

    out = tmp_path / "summary.json"
    write_comparison_summary_json(out, example="ex3", task="regression", config=Config(), rows=[])
    data = json.loads(out.read_text())
    assert data["config"] == {"lr": 0.001, "epochs": 50}


def test_write_comparison_summary_json_object_config(tmp_path):
    class SimpleConfig:
        def __init__(self) -> None:
            self.alpha = 0.5
            self.beta = 1.0

    out = tmp_path / "summary.json"
    write_comparison_summary_json(
        out, example="ex4", task="regression", config=SimpleConfig(), rows=[]
    )
    data = json.loads(out.read_text())
    assert data["config"] == {"alpha": 0.5, "beta": 1.0}


def test_write_comparison_summary_json_creates_parent_dirs(tmp_path):
    out = tmp_path / "deep" / "nested" / "summary.json"
    write_comparison_summary_json(out, example="e", task="t", config={}, rows=[])
    assert out.exists()


def test_write_comparison_summary_json_returns_path(tmp_path):
    path = Path(tmp_path / "out.json")
    result = write_comparison_summary_json(path, example="e", task="t", config={}, rows=[])
    assert isinstance(result, Path)
    assert result == path
