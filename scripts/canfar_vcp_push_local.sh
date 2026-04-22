#!/usr/bin/env bash
# Print or run local → VOS ``vcp`` for NeurIPS/CANFAR assets (same layout as headless pull).
#
# Usage (repo root):
#   ./scripts/canfar_vcp_push_local.sh
#   ./scripts/canfar_vcp_push_local.sh --vos-base vos:YOUR_USER/torchregress --execute
#   ./scripts/canfar_vcp_push_local.sh --with-higgs --write-vcp-specs ./vcp_specs.txt
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
exec uv run python "${REPO}/tools/canfar_vcp_prepare.py" --repo "${REPO}" "$@"
