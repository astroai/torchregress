#!/bin/bash
set -e

echo "--- Run Ruff ---"
uv run ruff check .

echo "--- Run Black ---"
uv run black --check .

echo "--- Run Mypy ---"
uv run mypy torchregress tests

echo "--- Run Unit Tests ---"
uv run pytest

echo "--- Run Smoke Benchmarks ---"
uv run python -m tools.benchmark_smoke --mode smoke --iterations 1 --warmup 0 --device cpu --thresholds reports/benchmark_thresholds/cpu/smoke.json --fail-on-thresholds

echo "--- All checks passed locally! ---"
