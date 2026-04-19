#!/usr/bin/env bash
# Full supervised-gap evidence: complete default hyperparameter grid + 10-seed multiseed confirm.
#
# Phase A — full tuning (648 SAGE configs per benchmark × Year; +648 Higgs if parquet exists):
#   tools/launch_full_sage_gap_tuning.py
#
# Phase B — multiseed confirmation on best CSV rows (Year + optional Higgs), 10 seeds:
#   examples/benchmarks/self_agreement_supervised_gap_multiseed.py
#
# Phase C (optional) — refresh NeurIPS digest JSON:
#   tools/aggregate_sage_paper_report.py --run-root <path>
#
# After Phase A completes (1296 CSV rows if Year+Higgs, else 648), run Phase B, e.g.:
#   nohup env FULL_GAP_OUT_ROOT="$OUT" ./scripts/run_full_sage_gap_evidence.sh multiseed \
#     >>"$OUT/multiseed.nohup.log" 2>&1 &
#
# NeurIPS driver now uses ten seeds (260410–260419) for all non-quick multiseed phases.
#
# Environment:
#   FULL_GAP_OUT_ROOT   Output root (default: docs/research/sage_reg_results/<UTC-date>/full_sage_gap_evidence)
#   FULL_GAP_YEAR_CACHE OpenML Year CSV (default: data/paper/openml_year.csv)
#   FULL_GAP_HIGGS_PARQUET  Override Higgs path (see launcher --help)
#   FULL_GAP_SKIP_MULTISEED  set to 1 to only run tuning
#   FULL_GAP_SKIP_AGGREGATE set to 1 to skip aggregate
#   FULL_GAP_NEURIPS_ROOT    Run root for aggregate (default: same date neurips_sage_reg_full)
#
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

DATE_UTC=$(date -u +%Y-%m-%d)
OUT_ROOT="${FULL_GAP_OUT_ROOT:-$REPO_ROOT/docs/research/sage_reg_results/$DATE_UTC/full_sage_gap_evidence}"
YEAR_CACHE="${FULL_GAP_YEAR_CACHE:-$REPO_ROOT/data/paper/openml_year.csv}"
TUNE_DIR="$OUT_ROOT/full_grid_tune"
MULTI_DIR="$OUT_ROOT/multiseed_10"
PY="${FULL_GAP_PYTHON:-.venv/bin/python}"
SEEDS=(260410 260411 260412 260413 260414 260415 260416 260417 260418 260419)

if [[ ! -x "$PY" ]]; then PY="uv run python"; fi

usage() {
  cat <<'EOF'
run_full_sage_gap_evidence.sh <command>

  all              Run full tuning, then 10-seed multiseed, then aggregate NeurIPS digest.
  tune             Full grid tuning only (resumable via sweep.csv).
  multiseed        Run multiseed using existing TUNE_DIR/sweep.csv (expects tune done).
  aggregate        Only regenerate sage_paper_report.json + METRICS.md.

Uses paper-scale Year splits (4096 / 131072 / 32768), 32/32 epochs, cosine LR on Year.
EOF
}

cmd_tune() {
  mkdir -p "$TUNE_DIR"
  if [[ "${FULL_GAP_SKIP_HIGGS:-0}" == "1" ]]; then
    $PY "$REPO_ROOT/tools/launch_full_sage_gap_tuning.py" \
      --out-dir "$TUNE_DIR" \
      --year-cache "$YEAR_CACHE" \
      --skip-higgs
  elif [[ -n "${FULL_GAP_HIGGS_PARQUET:-}" ]]; then
    $PY "$REPO_ROOT/tools/launch_full_sage_gap_tuning.py" \
      --out-dir "$TUNE_DIR" \
      --year-cache "$YEAR_CACHE" \
      --higgs-parquet "$FULL_GAP_HIGGS_PARQUET"
  else
    $PY "$REPO_ROOT/tools/launch_full_sage_gap_tuning.py" \
      --out-dir "$TUNE_DIR" \
      --year-cache "$YEAR_CACHE"
  fi
}

cmd_multiseed() {
  local csv="$TUNE_DIR/sweep.csv"
  if [[ ! -f "$csv" ]]; then
    echo "Missing $csv — run tune first." >&2
    exit 1
  fi
  MULTI_CMD=(
    "$PY" "$REPO_ROOT/examples/benchmarks/self_agreement_supervised_gap_multiseed.py"
    --tuning-csv "$csv"
    --year-cache-path "$YEAR_CACHE"
    --no-year-download
    --year-n-labeled 4096
    --year-n-unlabeled 131072
    --year-n-test 32768
    --year-teacher-epochs 32
    --year-student-epochs 32
    --year-lr-schedule cosine
    --year-lr-min 1e-5
    --out-dir "$MULTI_DIR"
    --seeds "${SEEDS[@]}"
  )
  HIGGS_DEFAULT="$REPO_ROOT/docs/research/sage_reg_results/2026-04-09/higgs_public/extracted/FAIR_Universe_HiggsML_data.parquet"
  HP="${FULL_GAP_HIGGS_PARQUET:-$HIGGS_DEFAULT}"
  if [[ -f "$HP" && "${FULL_GAP_SKIP_HIGGS:-0}" != "1" ]]; then
    MULTI_CMD+=(--higgs-dataset-path "$HP" --higgs-split-scale-factor 10 --higgs-parquet-max-sample-rows 8000000 --higgs-teacher-epochs 32 --higgs-student-epochs 32)
  else
    MULTI_CMD+=(--skip-higgs)
  fi
  "${MULTI_CMD[@]}"
}

cmd_aggregate() {
  local nr="${FULL_GAP_NEURIPS_ROOT:-$REPO_ROOT/docs/research/sage_reg_results/$DATE_UTC/neurips_sage_reg_full}"
  if [[ -d "$nr" ]]; then
    $PY "$REPO_ROOT/tools/aggregate_sage_paper_report.py" --run-root "$nr" --write-markdown
  else
    echo "Skip aggregate: NeurIPS run root not found: $nr" >&2
  fi
}

case "${1:-}" in
  tune) cmd_tune ;;
  multiseed) cmd_multiseed ;;
  aggregate) cmd_aggregate ;;
  all)
    cmd_tune
    if [[ "${FULL_GAP_SKIP_MULTISEED:-0}" != "1" ]]; then
      cmd_multiseed
    fi
    if [[ "${FULL_GAP_SKIP_AGGREGATE:-0}" != "1" ]]; then
      cmd_aggregate
    fi
    ;;
  ""|-h|--help|help) usage ;;
  *) echo "Unknown: $1" >&2; usage; exit 2 ;;
esac
