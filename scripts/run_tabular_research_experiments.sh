#!/usr/bin/env bash
# Large-tabular experiment entrypoints (SAGE-Reg + SPT-Reg).
#
# For a single end-to-end bundle + combined metric report, use:
#   ./scripts/run_tabular_paper_bundle.sh
#   ./scripts/run_tabular_paper_bundle.sh --smoke   # fast CI-style dry run
#
# This file defines optional shell functions for ad-hoc sweeps (source it).
#
# From repo root:
#   bash scripts/run_tabular_research_experiments.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# Defaults (override with export). Higgs path = FAIR Universe Higgs ML extract per reproducibility doc.
TUNING_CSV="${TUNING_CSV:-${ROOT}/docs/research/sage_reg_results/2026-04-10/supervised_gap_tuning_v3/sweep.csv}"
YEAR_CACHE="${YEAR_CACHE:-${ROOT}/docs/research/sage_reg_results/2026-04-10/openml_year.csv}"
OUT_BASE="${OUT_BASE:-${ROOT}/docs/research/sage_reg_results/$(date +%Y-%m-%d)/tabular_runs}"
if [[ -f "${ROOT}/data/neurips_inputs/FAIR_Universe_HiggsML_data.parquet" ]]; then
  _HIGGS_DEFAULT="${ROOT}/data/neurips_inputs/FAIR_Universe_HiggsML_data.parquet"
else
  _HIGGS_DEFAULT="${ROOT}/docs/research/sage_reg_results/2026-04-09/higgs_public/extracted/FAIR_Universe_HiggsML_data.parquet"
fi
HIGGS_PARQUET="${HIGGS_PARQUET:-${_HIGGS_DEFAULT}}"
HIGGS_SPLIT_SCALE="${HIGGS_SPLIT_SCALE:-10}"
HIGGS_PARQUET_MAX_SAMPLE_ROWS="${HIGGS_PARQUET_MAX_SAMPLE_ROWS:-8000000}"

mkdir -p "${OUT_BASE}"

echo "== Repo: ${ROOT}"
echo "== OUT_BASE: ${OUT_BASE}"

# -----------------------------------------------------------------------------
# 1) SAGE — Year-class OpenML table, default large splits (n_labeled / n_unlabeled / n_test).
#    Adds unlabeled-pool fraction curve (same split sizes; varies how much unlabeled SAGE sees).
# -----------------------------------------------------------------------------
run_sage_year_full_direct() {
  uv run python examples/benchmarks/self_agreement_realdata_year.py \
    --cache-path "${YEAR_CACHE}" \
    --n-labeled 4096 \
    --n-unlabeled 131072 \
    --n-test 32768 \
    --teacher-epochs 32 \
    --student-epochs 32 \
    --unlabeled-fractions 0.25 0.5 1.0 \
    --output-csv "${OUT_BASE}/year_direct_nl4096_nu131k_fractions.csv" \
    --summary-json-path "${OUT_BASE}/year_direct_nl4096_nu131k_summary.json"
}

# -----------------------------------------------------------------------------
# 2) SAGE — Same, but scan labeled count at fixed unlabeled + test (label / unlabeled ratio axis).
#    One CSV per labeled budget. Afterward merge summaries:
#      uv run python tools/collate_sage_year_labeled_sweep.py \
#        --input-dir "${OUT_BASE}" --output-json "${OUT_BASE}/year_labeled_sweep_collated.json" \
#        --output-csv "${OUT_BASE}/year_labeled_sweep_collated.csv"
# -----------------------------------------------------------------------------
run_sage_year_labeled_sweep() {
  local nu=131072 nt=32768
  for nl in 2048 4096 8192 16384 32768; do
    echo "--- n_labeled=${nl} n_unlabeled=${nu} n_test=${nt}"
    uv run python examples/benchmarks/self_agreement_realdata_year.py \
      --cache-path "${YEAR_CACHE}" \
      --n-labeled "${nl}" \
      --n-unlabeled "${nu}" \
      --n-test "${nt}" \
      --teacher-epochs 32 \
      --student-epochs 32 \
      --unlabeled-fractions 1.0 \
      --output-csv "${OUT_BASE}/year_direct_nl${nl}_nu${nu}_ufrac1.0.csv" \
      --summary-json-path "${OUT_BASE}/year_direct_nl${nl}_summary.json"
  done
}

# -----------------------------------------------------------------------------
# 3) SAGE — Multi-seed confirm using tuned hyperparameters (tuning CSV + optional sweep.json).
#    Use --year-n-labeled / --year-n-unlabeled / --year-n-test to align splits across seeds.
# -----------------------------------------------------------------------------
run_sage_year_multiseed_fixed_hyper() {
  uv run python examples/benchmarks/self_agreement_supervised_gap_multiseed.py \
    --tuning-csv "${TUNING_CSV}" \
    --year-cache-path "${YEAR_CACHE}" \
    --skip-higgs \
    --year-n-labeled 4096 \
    --year-n-unlabeled 131072 \
    --year-n-test 32768 \
    --year-teacher-epochs 32 \
    --year-student-epochs 32 \
    --seeds 260410 260411 260412 \
    --out-dir "${OUT_BASE}/multiseed_year_nl4096_nu131k"
}

# -----------------------------------------------------------------------------
# 4) SAGE — Multi-seed × labeled budget (nested dirs); good for tables: gap vs n_labeled.
# -----------------------------------------------------------------------------
run_sage_year_multiseed_labeled_sweep() {
  local nu=131072 nt=32768
  for nl in 2048 4096 8192 16384; do
    uv run python examples/benchmarks/self_agreement_supervised_gap_multiseed.py \
      --tuning-csv "${TUNING_CSV}" \
      --year-cache-path "${YEAR_CACHE}" \
      --skip-higgs \
      --year-n-labeled "${nl}" \
      --year-n-unlabeled "${nu}" \
      --year-n-test "${nt}" \
      --year-teacher-epochs 32 \
      --year-student-epochs 32 \
      --seeds 260410 260411 260412 \
      --out-dir "${OUT_BASE}/multiseed_year_nl${nl}_nu${nu}"
  done
}

# -----------------------------------------------------------------------------
# 5) SAGE — Large OpenML regression (default diamonds id 42225): materialize cache once, then same as (1).
#    Verify data_id on OpenML before citing.
# -----------------------------------------------------------------------------
run_sage_large_tabular_openml_bench() {
  local cache="${OUT_BASE}/openml_large_tabular_diamonds.parquet"
  uv run python examples/benchmarks/self_agreement_realdata_year.py \
    --openml-data-id 42225 \
    --max-dataset-rows 54000 \
    --cache-path "${cache}" \
    --n-labeled 4096 \
    --n-unlabeled 40000 \
    --n-test 8192 \
    --teacher-epochs 32 \
    --student-epochs 32 \
    --unlabeled-fractions 1.0 \
    --output-csv "${OUT_BASE}/large_tabular_openml_direct.csv" \
    --summary-json-path "${OUT_BASE}/large_tabular_openml_direct_summary.json"
}

# -----------------------------------------------------------------------------
# 6) SAGE — Higgs parquet at large split scale (needs HIGGS_PARQUET).
# -----------------------------------------------------------------------------
run_sage_higgs_multiseed_large() {
  if [[ ! -f "${HIGGS_PARQUET}" ]]; then
    echo "Skip Higgs: file not found: ${HIGGS_PARQUET}" >&2
    return 0
  fi
  uv run python examples/benchmarks/self_agreement_supervised_gap_multiseed.py \
    --tuning-csv "${TUNING_CSV}" \
    --skip-year \
    --higgs-dataset-path "${HIGGS_PARQUET}" \
    --higgs-split-scale-factor 10 \
    --higgs-parquet-max-sample-rows 8000000 \
    --higgs-teacher-epochs 32 \
    --higgs-student-epochs 32 \
    --seeds 260410 260411 260412 \
    --out-dir "${OUT_BASE}/multiseed_higgs_10x"
}

# -----------------------------------------------------------------------------
# 7) SPT — Large-tabular track with paper full split budgets (needs real Year cache).
# -----------------------------------------------------------------------------
run_spt_year_full_summary() {
  uv run python tools/render_spt_reg_paper_artifacts.py \
    --profile full \
    --year-cache-path "${YEAR_CACHE}" \
    --output-dir reports/neurips_spt_reg
}

# -----------------------------------------------------------------------------
# Default: run nothing destructive; print usage. Uncomment one line to execute.
# -----------------------------------------------------------------------------
cat <<'EOF'

Defined functions (source this file or call from a subshell after copying):
  run_sage_year_full_direct
  run_sage_year_labeled_sweep
  run_sage_year_multiseed_fixed_hyper
  run_sage_year_multiseed_labeled_sweep
  run_sage_large_tabular_openml_bench
  run_sage_higgs_multiseed_large   # needs HIGGS_PARQUET
  run_spt_year_full_summary

Example one-shot:
  YEAR_CACHE=... TUNING_CSV=... OUT_BASE=... bash -c 'source scripts/run_tabular_research_experiments.sh && run_sage_year_multiseed_fixed_hyper'

EOF
