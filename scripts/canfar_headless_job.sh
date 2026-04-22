#!/usr/bin/env bash
# CANFAR Skaha headless entrypoint: stage inputs from VOSpace, dispatch by CANFAR_JOB_KIND.
#
# Required env:
#   ARC_RUN_ROOT   Durable output root (e.g. /arc/projects/ots/torchregress/runs/<run_id>)
#   VOS_BASE       Full VOSpace URI prefix (must start with vos:, e.g. vos:sfabbro/torchregress)
#
# Kind-specific:
#   CANFAR_JOB_KIND  year_label_shard (default) | neurips_phase | overnight_year_multiseed |
#                    export_split
#
# year_label_shard (same as legacy):
#   SHARD_ID, SHARD_COUNT
#
# neurips_phase:
#   NEURIPS_ONLY_PHASES   comma list for run_neurips_sage_reg_full.py --only-phases
#   Optional: NEURIPS_FLAGS  extra CLI tokens (word-split; use with care)
#   Optional: YEAR_CACHE_PATH  default ${SCRATCH_ROOT}/data/paper/openml_year.csv
#   Optional: NEURIPS_TUNING_CSV, NEURIPS_HIGGS_PARQUET, NEURIPS_DIAMONDS_CACHE, NEURIPS_TABRED_ROOT,
#             NEURIPS_SHIFTS_OUT_ROOT — host paths under scratch after vcp
#
# overnight_year_multiseed:
#   Writes under ${ARC_RUN_ROOT}/extras/year_multiseed/ (override with YEAR_MULTI_OUT_DIR)
#
# export_split:
#   Writes under ${ARC_RUN_ROOT}/extras/split_export/ (override with EXPORT_SPLIT_OUT_DIR)
#
# Multi-file VOS pulls (all kinds):
#   VCP_SPECS  newline-separated rows:  vos_path_suffix|scratch_path_suffix
#              Left: appended to VOS_BASE (must not include the vos: prefix again).
#              Right: path under SCRATCH_ROOT for the pulled file.
#
set -euo pipefail

: "${ARC_RUN_ROOT:?}"
: "${VOS_BASE:?}"

TORCHREGRESS_REPO="${TORCHREGRESS_REPO:-/arc/home/sfabbro/src/torchregress}"
RUN_ID="${RUN_ID:-$(basename "${ARC_RUN_ROOT}")}"
SCRATCH_PARENT="${SCRATCH_PARENT:-/scratch/torchregress}"
CANFAR_JOB_KIND="${CANFAR_JOB_KIND:-year_label_shard}"
SHARD_ID="${SHARD_ID:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"

if [[ -n "${SCRATCH_ROOT:-}" ]]; then
  :
elif [[ "${CANFAR_JOB_KIND}" == "year_label_shard" ]]; then
  SCRATCH_ROOT="${SCRATCH_PARENT}/${RUN_ID}/shard_${SHARD_ID}"
else
  SCRATCH_ROOT="${SCRATCH_PARENT}/${RUN_ID}/job_${CANFAR_JOB_KIND}_${SHARD_ID}"
fi
# shellcheck disable=SC2155
export TMPDIR="${TMPDIR:-${SCRATCH_ROOT}/tmp}"
mkdir -p "${SCRATCH_ROOT}/data/paper" "${TMPDIR}" "${ARC_RUN_ROOT}/logs"

VOS_TRIM="${VOS_BASE%/}"

vcp_safe() {
  local src="$1"
  local dst="$2"
  if command -v vcp >/dev/null 2>&1; then
    mkdir -p "$(dirname "${dst}")"
    echo "[canfar_headless_job] vcp ${src} -> ${dst}"
    vcp "${src}" "${dst}"
  else
    echo "[canfar_headless_job] WARN: vcp not in PATH; expected ${dst}" >&2
  fi
}

apply_vcp_specs() {
  # Each non-empty line: vos_path_suffix|scratch_path_suffix (see header above)
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line// }" ]] && continue
    [[ "${line}" =~ ^# ]] && continue
    local vos_rel="${line%%|*}"
    local scr_rel="${line#*|}"
    if [[ "${vos_rel}" == "${line}" ]] || [[ -z "${scr_rel}" ]]; then
      echo "[canfar_headless_job] WARN: bad VCP_SPECS line (need a|b): ${line}" >&2
      continue
    fi
    vos_rel="${vos_rel#./}"
    scr_rel="${scr_rel#./}"
    vcp_safe "${VOS_TRIM}/${vos_rel}" "${SCRATCH_ROOT}/${scr_rel}"
  done <<< "${VCP_SPECS:-}"
}

default_year_pull() {
  if [[ "${SKIP_DEFAULT_YEAR_VCP:-}" == "1" ]]; then
    echo "[canfar_headless_job] SKIP_DEFAULT_YEAR_VCP=1 — skip default Year vcp"
    return 0
  fi
  local rel="${YEAR_VOS_REL:-data/paper/openml_year.csv}"
  vcp_safe "${VOS_TRIM}/${rel#./}" "${SCRATCH_ROOT}/data/paper/openml_year.csv"
}

job_year_label_shard() {
  : "${SHARD_COUNT:?}"
  : "${SHARD_ID:?}"
  default_year_pull
  apply_vcp_specs

  local CACHE_CSV="${YEAR_CACHE_PATH:-${SCRATCH_ROOT}/data/paper/openml_year.csv}"
  if [[ ! -f "${CACHE_CSV}" ]]; then
    echo "[canfar_headless_job] ERROR: missing Year cache at ${CACHE_CSV}" >&2
    exit 1
  fi

  mkdir -p "${ARC_RUN_ROOT}/shards"
  local SHARD_PAD
  SHARD_PAD="$(printf '%02d' "$((10#${SHARD_ID}))")"
  local OUT_CSV="${ARC_RUN_ROOT}/shards/shard_${SHARD_PAD}.csv"
  local SUMMARY_JSON="${ARC_RUN_ROOT}/shards/shard_${SHARD_PAD}_summary.json"
  local LOG_FILE="${ARC_RUN_ROOT}/logs/shard_${SHARD_PAD}.log"
  exec > >(tee -a "${LOG_FILE}") 2>&1

  echo "[canfar_headless_job] year_label_shard SHARD_ID=${SHARD_ID} SHARD_COUNT=${SHARD_COUNT}"

  SWEEP_SEEDS="${SWEEP_SEEDS:-260410 260411 260412 260413 260414 260415 260416 260417 260418 260419}"
  BATCH_SIZE="${BATCH_SIZE:-2048}"
  DATALOADER_WORKERS="${DATALOADER_WORKERS:-0}"
  CATBOOST_ITERS="${CATBOOST_ITERS:-0}"
  TEACHER_EPOCHS="${TEACHER_EPOCHS:-32}"
  STUDENT_EPOCHS="${STUDENT_EPOCHS:-32}"
  N_TEST="${N_TEST:-32768}"
  MIN_UNLABELED="${MIN_UNLABELED:-2048}"

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
}

job_neurips_phase() {
  : "${NEURIPS_ONLY_PHASES:?set NEURIPS_ONLY_PHASES for neurips_phase jobs}"

  default_year_pull
  apply_vcp_specs

  local CACHE_CSV="${YEAR_CACHE_PATH:-${SCRATCH_ROOT}/data/paper/openml_year.csv}"
  if [[ ! -f "${CACHE_CSV}" ]]; then
    echo "[canfar_headless_job] ERROR: missing Year cache at ${CACHE_CSV}" >&2
    exit 1
  fi

  local LOG_FILE="${ARC_RUN_ROOT}/logs/neurips_${NEURIPS_ONLY_PHASES//,/_}.log"
  mkdir -p "$(dirname "${LOG_FILE}")"
  exec > >(tee -a "${LOG_FILE}") 2>&1

  echo "[canfar_headless_job] neurips_phase only_phases=${NEURIPS_ONLY_PHASES}"

  cd "${TORCHREGRESS_REPO}"
  local -a cmd=(
    python "${TORCHREGRESS_REPO}/scripts/run_neurips_sage_reg_full.py"
    --run-root "${ARC_RUN_ROOT}"
    --year-cache "${CACHE_CSV}"
    --no-year-download
    --only-phases "${NEURIPS_ONLY_PHASES}"
  )
  if [[ -n "${NEURIPS_TUNING_CSV:-}" ]]; then
    cmd+=(--tuning-csv "${NEURIPS_TUNING_CSV}")
  fi
  if [[ -n "${NEURIPS_HIGGS_PARQUET:-}" ]]; then
    cmd+=(--higgs-parquet "${NEURIPS_HIGGS_PARQUET}")
  fi
  if [[ -n "${NEURIPS_DIAMONDS_CACHE:-}" ]]; then
    cmd+=(--openml-diamonds-cache "${NEURIPS_DIAMONDS_CACHE}")
  fi
  if [[ -n "${NEURIPS_TABRED_ROOT:-}" ]]; then
    cmd+=(--tabred-data-root "${NEURIPS_TABRED_ROOT}")
  fi
  if [[ -n "${NEURIPS_SHIFTS_OUT_ROOT:-}" ]]; then
    cmd+=(--shifts-out-root "${NEURIPS_SHIFTS_OUT_ROOT}")
  fi
  if [[ -n "${NEURIPS_FLAGS:-}" ]]; then
    # shellcheck disable=SC2206
    cmd+=(${NEURIPS_FLAGS})
  fi
  exec "${cmd[@]}"
}

job_overnight_year_multiseed() {
  default_year_pull
  apply_vcp_specs

  local CACHE_CSV="${YEAR_CACHE_PATH:-${SCRATCH_ROOT}/data/paper/openml_year.csv}"
  if [[ ! -f "${CACHE_CSV}" ]]; then
    echo "[canfar_headless_job] ERROR: missing Year cache at ${CACHE_CSV}" >&2
    exit 1
  fi

  local OUT_MULT="${YEAR_MULTI_OUT_DIR:-${ARC_RUN_ROOT}/extras/year_multiseed}"
  mkdir -p "${OUT_MULT}" "${ARC_RUN_ROOT}/logs"
  local LOG_FILE="${ARC_RUN_ROOT}/logs/overnight_year_multiseed.log"
  exec > >(tee -a "${LOG_FILE}") 2>&1

  local SEEDS=(260410 260411 260412 260413 260414 260415 260416 260417 260418 260419)
  if [[ -n "${YEAR_MULTI_SEEDS:-}" ]]; then
    read -r -a SEEDS <<< "${YEAR_MULTI_SEEDS}"
  fi
  local NL="${YEAR_MULTI_N_LABELED:-4096}"
  local NU="${YEAR_MULTI_N_UNLABELED:-131072}"
  local NT="${YEAR_MULTI_N_TEST:-32768}"
  local TEACH="${YEAR_MULTI_TEACHER_EPOCHS:-32}"
  local STUD="${YEAR_MULTI_STUDENT_EPOCHS:-32}"
  local BATCH="${YEAR_MULTI_BATCH:-4096}"
  local DW="${YEAR_MULTI_DATALOADER_WORKERS:-0}"
  local UFRAC=(0.25 0.5 1.0)
  if [[ -n "${YEAR_MULTI_UFRAC:-}" ]]; then
    read -r -a UFRAC <<< "${YEAR_MULTI_UFRAC}"
  fi

  cd "${TORCHREGRESS_REPO}"
  for s in "${SEEDS[@]}"; do
    echo "[canfar_headless_job] overnight_year_multiseed seed=${s}"
    python "${TORCHREGRESS_REPO}/examples/benchmarks/self_agreement_realdata_year.py" \
      --cache-path "${CACHE_CSV}" \
      --no-download \
      --seed "${s}" \
      --n-labeled "${NL}" \
      --n-unlabeled "${NU}" \
      --n-test "${NT}" \
      --teacher-epochs "${TEACH}" \
      --student-epochs "${STUD}" \
      --batch-size "${BATCH}" \
      --dataloader-num-workers "${DW}" \
      --unlabeled-fractions "${UFRAC[@]}" \
      --output-csv "${OUT_MULT}/year_seed_${s}.csv" \
      --summary-json-path "${OUT_MULT}/year_seed_${s}_summary.json" \
      --performance-figure-path "${OUT_MULT}/year_seed_${s}_perf.png" \
      --calibration-figure-path "${OUT_MULT}/year_seed_${s}_calib.png"
  done
  python "${TORCHREGRESS_REPO}/tools/collate_csv_glob.py" \
    --glob "${OUT_MULT}/year_seed_*.csv" \
    --out "${OUT_MULT}/all_seeds_collated.csv"
  exit 0
}

job_export_split() {
  default_year_pull
  apply_vcp_specs

  local CACHE_CSV="${YEAR_CACHE_PATH:-${SCRATCH_ROOT}/data/paper/openml_year.csv}"
  if [[ ! -f "${CACHE_CSV}" ]]; then
    echo "[canfar_headless_job] ERROR: missing Year cache at ${CACHE_CSV}" >&2
    exit 1
  fi

  local SEED="${EXPORT_SPLIT_SEED:-260410}"
  local OUT_DIR="${EXPORT_SPLIT_OUT_DIR:-${ARC_RUN_ROOT}/extras/split_export/seed_${SEED}_default_split}"
  mkdir -p "${OUT_DIR}" "${ARC_RUN_ROOT}/logs"
  local LOG_FILE="${ARC_RUN_ROOT}/logs/export_split.log"
  exec > >(tee -a "${LOG_FILE}") 2>&1

  local NL="${EXPORT_SPLIT_N_LABELED:-4096}"
  local NU="${EXPORT_SPLIT_N_UNLABELED:-131072}"
  local NT="${EXPORT_SPLIT_N_TEST:-32768}"

  cd "${TORCHREGRESS_REPO}"
  python "${TORCHREGRESS_REPO}/tools/export_year_ssl_split_for_external.py" \
    --out-dir "${OUT_DIR}" \
    --split-mode default \
    --cache-path "${CACHE_CSV}" \
    --no-download \
    --seed "${SEED}" \
    --n-labeled "${NL}" \
    --n-unlabeled "${NU}" \
    --n-test "${NT}"

  local tmpl="${TORCHREGRESS_REPO}/tools/official_metrics_examples/rankup_pablo_official.template.json"
  if [[ -f "${tmpl}" ]]; then
    cp -f "${tmpl}" "${ARC_RUN_ROOT}/extras/split_export/official_metrics_TEMPLATE_fill_and_merge.json" || true
  fi
  exit 0
}

case "${CANFAR_JOB_KIND}" in
  year_label_shard) job_year_label_shard ;;
  neurips_phase) job_neurips_phase ;;
  overnight_year_multiseed) job_overnight_year_multiseed ;;
  export_split) job_export_split ;;
  *)
    echo "[canfar_headless_job] ERROR: unknown CANFAR_JOB_KIND=${CANFAR_JOB_KIND}" >&2
    exit 2
    ;;
esac
