#!/usr/bin/env bash
# CANFAR Skaha headless entrypoint: stage Year CSV from VOSpace to /scratch, run one sweep shard, write to ARC.
#
# Required env:
#   ARC_RUN_ROOT   Durable output root (e.g. /arc/projects/ots/torchregress/runs/<run_id>)
#   VOS_BASE       VOSpace directory prefix (e.g. vos:sfabbro/torchregress) — no trailing slash required
#   SHARD_ID       0-based shard index (string or int)
#   SHARD_COUNT    Number of shards (e.g. 40)
#
# Optional:
#   TORCHREGRESS_REPO  Code checkout (default: /arc/home/sfabbro/src/torchregress)
#   RUN_ID             Tag for scratch layout (default: basename of ARC_RUN_ROOT)
#   YEAR_VOS_REL       Path under VOS_BASE to openml_year.csv (default: data/paper/openml_year.csv)
#   SCRATCH_PARENT     Default /scratch/torchregress
#   SWEEP_SEEDS        Space-separated seeds (default: 10 NeurIPS seeds)
#   BATCH_SIZE         Default 2048
#   DATALOADER_WORKERS Default 0
#   CATBOOST_ITERS     Default 0
#   TEACHER_EPOCHS / STUDENT_EPOCHS / N_TEST / MIN_UNLABELED — override sweep if set
#
set -euo pipefail

: "${ARC_RUN_ROOT:?}"
: "${VOS_BASE:?}"
: "${SHARD_COUNT:?}"
: "${SHARD_ID:?}"

TORCHREGRESS_REPO="${TORCHREGRESS_REPO:-/arc/home/sfabbro/src/torchregress}"
RUN_ID="${RUN_ID:-$(basename "${ARC_RUN_ROOT}")}"
SCRATCH_PARENT="${SCRATCH_PARENT:-/scratch/torchregress}"
SCRATCH_ROOT="${SCRATCH_ROOT:-${SCRATCH_PARENT}/${RUN_ID}/shard_${SHARD_ID}}"
YEAR_VOS_REL="${YEAR_VOS_REL:-data/paper/openml_year.csv}"

# shellcheck disable=SC2155
export TMPDIR="${TMPDIR:-${SCRATCH_ROOT}/tmp}"

SWEEP_SEEDS="${SWEEP_SEEDS:-260410 260411 260412 260413 260414 260415 260416 260417 260418 260419}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-0}"
CATBOOST_ITERS="${CATBOOST_ITERS:-0}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-32}"
STUDENT_EPOCHS="${STUDENT_EPOCHS:-32}"
N_TEST="${N_TEST:-32768}"
MIN_UNLABELED="${MIN_UNLABELED:-2048}"

mkdir -p "${SCRATCH_ROOT}/data/paper" "${TMPDIR}" "${ARC_RUN_ROOT}/shards" "${ARC_RUN_ROOT}/logs"

# Trim one trailing slash from VOS_BASE for safe joins
VOS_TRIM="${VOS_BASE%/}"
YEAR_VOS="${VOS_TRIM}/${YEAR_VOS_REL#./}"
CACHE_CSV="${SCRATCH_ROOT}/data/paper/openml_year.csv"

if command -v vcp >/dev/null 2>&1; then
  echo "[canfar_headless_job] vcp ${YEAR_VOS} -> ${CACHE_CSV}"
  vcp "${YEAR_VOS}" "${CACHE_CSV}"
else
  echo "[canfar_headless_job] WARN: vcp not in PATH; set YEAR_CACHE_PATH or place openml_year.csv at ${CACHE_CSV}" >&2
fi

if [[ ! -f "${CACHE_CSV}" ]]; then
  echo "[canfar_headless_job] ERROR: missing ${CACHE_CSV} (VOS pull failed or vcp absent)" >&2
  exit 1
fi

SHARD_PAD="$(printf '%02d' "$((10#${SHARD_ID}))")"
OUT_CSV="${ARC_RUN_ROOT}/shards/shard_${SHARD_PAD}.csv"
SUMMARY_JSON="${ARC_RUN_ROOT}/shards/shard_${SHARD_PAD}_summary.json"
LOG_FILE="${ARC_RUN_ROOT}/logs/shard_${SHARD_PAD}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[canfar_headless_job] SHARD_ID=${SHARD_ID} SHARD_COUNT=${SHARD_COUNT} OUT_CSV=${OUT_CSV}"

cd "${TORCHREGRESS_REPO}"
exec python "${TORCHREGRESS_REPO}/examples/benchmarks/sage_year_label_fraction_sweep.py" \
  --out-csv "${OUT_CSV}" \
  --summary-json "${SUMMARY_JSON}" \
  --cache-path "${CACHE_CSV}" \
  --no-download \
  --seeds ${SWEEP_SEEDS} \
  --label-percents 0.1 1 2 5 10 25 50 100 \
  --shift-modes none covariate label \
  --min-unlabeled "${MIN_UNLABELED}" \
  --n-test "${N_TEST}" \
  --batch-size "${BATCH_SIZE}" \
  --dataloader-num-workers "${DATALOADER_WORKERS}" \
  --teacher-epochs "${TEACHER_EPOCHS}" \
  --student-epochs "${STUDENT_EPOCHS}" \
  --catboost-iterations "${CATBOOST_ITERS}" \
  --shard-id "$((10#${SHARD_ID}))" \
  --shard-count "${SHARD_COUNT}"
