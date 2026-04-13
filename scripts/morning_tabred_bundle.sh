#!/usr/bin/env bash
# TabReD × SAGE-Reg morning bundle — fetch (Kaggle + upstream preprocessing) then SSL probe.
#
# 1) Materializes TabReD tensors under ``data/tabred/`` by default (clone yandex-research/tabred,
#    patch DATA_DIR, run ``preprocessing/*.py``). Requires ``~/.kaggle/kaggle.json`` and:
#        uv pip install polars kaggle loguru scikit-learn
# 2) Runs ``tabred_sage_ssl_probe.py`` and writes results under docs/research/.../tabred_sage_bundle.
#
# Env:
#   TABRED_DATA_ROOT   — output root for TabReD .npy layout (default: <repo>/data/tabred)
#   TABRED_BUNDLE_OUT  — probe output directory override
#   SKIP_TABRED_FETCH  — set to 1 if data already exists and you only want the probe
#   TABRED_FETCH_ONLY  — set to 1 to only run fetch/preprocess (no SSL probe)
#
# Usage (repo root):
#   ./scripts/morning_tabred_bundle.sh
#   ./scripts/morning_tabred_bundle.sh --quick --seed 42
#   SKIP_TABRED_FETCH=1 ./scripts/morning_tabred_bundle.sh
#
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TABRED_DATA_ROOT="${TABRED_DATA_ROOT:-$REPO_ROOT/data/tabred}"

if [[ "${SKIP_TABRED_FETCH:-}" != "1" ]]; then
  echo "== TabReD fetch -> $TABRED_DATA_ROOT"
  echo "    (SKIP_TABRED_FETCH=1 to skip; needs Kaggle credentials + polars/kaggle/loguru/sklearn)"
  uv run python "$REPO_ROOT/tools/fetch_tabred_data.py" \
    --out-dir "$TABRED_DATA_ROOT" \
    --skip-if-present
else
  echo "== Skipping TabReD fetch (SKIP_TABRED_FETCH=1)"
fi

if [[ "${TABRED_FETCH_ONLY:-}" == "1" ]]; then
  echo "== TABRED_FETCH_ONLY=1 — done after fetch."
  exit 0
fi

OUT="${TABRED_BUNDLE_OUT:-$REPO_ROOT/docs/research/sage_reg_results/$(date -u +%Y-%m-%d)/tabred_sage_bundle}"
mkdir -p "$OUT"

echo "== TabReD data root: $TABRED_DATA_ROOT"
echo "== Probe out dir:   $OUT"
uv run python "$REPO_ROOT/examples/benchmarks/tabred_sage_ssl_probe.py" \
  --tabred-data-root "$TABRED_DATA_ROOT" \
  --out-dir "$OUT" \
  "$@"
