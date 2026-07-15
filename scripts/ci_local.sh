#!/usr/bin/env bash
# Full local gate (verify_full): CI test + docs + benchmark smoke.
# For pre-push use scripts/ci_test_only.sh; for fast edits use scripts/preflight_push.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

"$ROOT/scripts/preflight_push.sh"
"$ROOT/scripts/ci_test_only.sh"

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

echo "OK: full local checks (verify_full)."
