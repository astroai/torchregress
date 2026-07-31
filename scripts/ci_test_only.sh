#!/usr/bin/env bash
# Pre-push gate: lint + typecheck + pytest (verify tier).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
pixi run lint
pixi run typecheck
pixi run test
