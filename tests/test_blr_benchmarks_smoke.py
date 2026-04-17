"""Smoke tests for small benchmark scripts under examples/benchmarks/."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_ROOT / "examples" / "benchmarks"


def _load_benchmark_module(stem: str) -> ModuleType:
    path = BENCH_DIR / f"{stem}.py"
    module_name = f"_bench_smoke_{stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load benchmark module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lowshot_benchmark_main_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_benchmark_module("bayesian_linear_head_lowshot_adaptation")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bayesian_linear_head_lowshot_adaptation.py",
            "--dim",
            "6",
            "--n-test",
            "200",
            "--shots",
            "5,10",
            "--noise",
            "0.3",
            "--ridge-alpha",
            "0.2",
            "--seed",
            "3",
        ],
    )
    mod.main()


def test_ot_conformal_score_shift_benchmark_main_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_benchmark_module("ot_conformal_score_shift_benchmark")
    monkeypatch.setattr(sys, "argv", ["ot_conformal_score_shift_benchmark.py", "--seed", "2"])
    mod.main()


def test_online_drift_benchmark_main_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_benchmark_module("bayesian_linear_head_online_drift")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bayesian_linear_head_online_drift.py",
            "--dim",
            "5",
            "--n-phase-a",
            "80",
            "--n-phase-b",
            "80",
            "--batch-size",
            "20",
            "--n-test",
            "150",
            "--noise",
            "0.25",
            "--prior-precision",
            "0.1",
            "--seed",
            "4",
        ],
    )
    mod.main()
