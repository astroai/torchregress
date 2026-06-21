#!/usr/bin/env bash
# Thin wrapper: same as `uv run python scripts/canfar_vcp_prepare.py` from repo root.
#
#   ./scripts/canfar_vcp_push_local.sh
#   ./scripts/canfar_vcp_push_local.sh --dry-run
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
exec uv run python "${REPO}/scripts/canfar_vcp_prepare.py" --repo "${REPO}" "$@"
