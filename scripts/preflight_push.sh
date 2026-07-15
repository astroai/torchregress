#!/usr/bin/env bash
# Fast gate: ruff + syntax only (verify_fast tier).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export UV_NO_SYNC=1
uv run ruff check src/torchregress tests tools
uv run ruff format --check src/torchregress tests tools
uv run python -m compileall -q src/torchregress tests tools
