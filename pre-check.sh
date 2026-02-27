#!/bin/bash
set -e

echo "--- Run Ruff ---"
pixi run ruff check .

echo "--- Run Black ---"
pixi run black --check .

echo "--- Run Mypy ---"
pixi run mypy torchregress tests

echo "--- Run Unit Tests ---"
pixi run pytest

echo "--- Run Smoke Benchmarks ---"
pixi run python -m tools.benchmark_smoke --mode smoke --iterations 1 --warmup 0 --device cpu --thresholds reports/benchmark_thresholds/cpu/smoke.json --fail-on-thresholds

echo "--- All checks passed locally! ---"
