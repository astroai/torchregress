#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-${ROOT}/docs/research/sage_reg_results/$(date +%F)/year_label_fraction_sweep}"
mkdir -p "${OUT_DIR}"

# Large-run defaults: bump batch size and workers on a beefy host.
BATCH="${BATCH:-4096}"
WORKERS="${WORKERS:-4}"
SEEDS="${SEEDS:-260410 260411 260412 260413 260414 260415}"

uv run python "${ROOT}/examples/benchmarks/sage_year_label_fraction_sweep.py" \
  --out-csv "${OUT_DIR}/year_label_fraction_sweep.csv" \
  --summary-json "${OUT_DIR}/summary.json" \
  --seeds ${SEEDS} \
  --label-percents 0.1 1 2 5 10 25 50 100 \
  --shift-modes none covariate label \
  --min-unlabeled 2048 \
  --batch-size "${BATCH}" \
  --dataloader-num-workers "${WORKERS}" \
  --catboost-iterations "${CATBOOST_ITERS:-2000}" \
  "$@"
