#!/usr/bin/env bash
# Export the Year SSL split (same protocol as self_agreement_realdata_year) for running
# official RankUp (pm25/semi-supervised-regression) or other code, then merge metrics with:
#   uv run python tools/merge_ssl_official_metrics.py --ours-csv ... --official-json ...
#
# Usage:
#   OUT_DIR=/tmp/year_export CACHE_PATH=/path/to/year.csv bash scripts/prepare_year_split_for_official_compare.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-${ROOT}/docs/research/sage_reg_results/$(date +%F)/year_split_export}"
mkdir -p "${OUT_DIR}"

CACHE_PATH="${CACHE_PATH:-}"
DATASET_PATH="${DATASET_PATH:-}"
SEED="${SEED:-260410}"
SPLIT_MODE="${SPLIT_MODE:-default}"

CMD=(uv run python "${ROOT}/tools/export_year_ssl_split_for_external.py" --out-dir "${OUT_DIR}" --seed "${SEED}" --split-mode "${SPLIT_MODE}")
if [[ -n "${CACHE_PATH}" ]]; then
  CMD+=(--cache-path "${CACHE_PATH}" --no-download)
fi
if [[ -n "${DATASET_PATH}" ]]; then
  CMD+=(--dataset-path "${DATASET_PATH}" --no-download)
fi
if [[ "${SPLIT_MODE}" == "label_pool_fraction" ]]; then
  CMD+=(--label-pool-percent "${LABEL_POOL_PERCENT:?}" --shift-mode "${SHIFT_MODE:-none}" --min-unlabeled "${MIN_UNLABELED:-2048}")
fi

"${CMD[@]}"
echo "Copy ${OUT_DIR} into your official RankUp checkout (or point a loader at the .npy files)."
echo "Fill metrics from official eval into a JSON file (see tools/official_metrics_examples/rankup_pablo_official.template.json)."
echo "Then merge:"
echo "  uv run python ${ROOT}/tools/merge_ssl_official_metrics.py \\"
echo "    --ours-csv /path/to/torchregress_rows.csv \\"
echo "    --official-json /path/to/filled_official.json \\"
echo "    --seed ${SEED} --ours-unlabeled-fraction 1.0 \\"
echo "    --out-json ${OUT_DIR}/compare.json --out-csv ${OUT_DIR}/compare.csv"
