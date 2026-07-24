#!/usr/bin/env bash
# Pre-push gate: CI lint-test job without benchmarks/docs (verify tier).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
uv sync --all-extras --dev
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
uv run python -m pytest --cov=torchregress --cov-report=term
uv run mypy src/torchregress
