#!/usr/bin/env bash
# Fast gate: ruff + syntax only (verify_fast tier).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
pixi run lint
pixi run python -m compileall -q src/torchregress tests tools
