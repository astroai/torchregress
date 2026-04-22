#!/usr/bin/env bash
# Same as: uv run python scripts/canfar_vcp_prepare.py (from repo root).
#
#   ./scripts/canfar_vcp_push_local.sh
#   ./scripts/canfar_vcp_push_local.sh --dry-run
#   ./scripts/canfar_vcp_push_local.sh --vos-base vos:OTHER/torchregress
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
exec uv run python "${REPO}/scripts/canfar_vcp_prepare.py" --repo "${REPO}" "$@"
