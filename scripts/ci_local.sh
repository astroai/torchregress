#!/usr/bin/env bash
# Local parity with `.github/workflows/ci.yml` (`lint-test`: pytest + benchmark_smoke)
# plus ruff on package code. Run before pushing: `./scripts/ci_local.sh`
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

uv sync --extra test --extra flows --extra dev --extra docs

echo "== ruff (src/torchregress, tests, tools) =="
uv run ruff check src/torchregress tests tools

echo "== ruff format --check (src/torchregress, tests, tools) =="
uv run ruff format --check src/torchregress tests tools

echo "== pytest + coverage (CI test job) =="
uv run python -m pytest --cov=torchregress --cov-report=xml --cov-report=term

echo "== mypy (CI lint-test job) =="
uv run mypy src/torchregress

echo "== zensical build (CI lint-test job) =="
uv run zensical build --strict

echo "== benchmark smoke thresholds (CI benchmark job) =="
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
uv run python -m tools.benchmark_smoke \
  --mode smoke \
  --iterations 2 \
  --warmup 1 \
  --device cpu \
  --thresholds reports/benchmark_thresholds/cpu/smoke.json \
  --fail-on-thresholds

echo "== benchmark sweep thresholds (CI benchmark job) =="
uv run python -m tools.benchmark_smoke \
  --mode sweep \
  --iterations 2 \
  --warmup 1 \
  --device cpu \
  --thresholds reports/benchmark_thresholds/cpu/sweep.json \
  --fail-on-thresholds

echo "OK: local checks matched CI test + benchmark jobs (plus ruff)."
