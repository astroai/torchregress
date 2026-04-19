#!/usr/bin/env bash
# Paper-aligned Year supervised-gap experiments (offline cache + tuning + optional multiseed).
#
# Environment overrides:
#   WAVE2_YEAR_CACHE   CSV path (default: data/paper/openml_year.csv)
#   WAVE2_OUT_DIR      Output root (default: docs/research/sage_reg_results/<UTC-date>/wave2_paper_year)
#   WAVE2_SEEDS        Space-separated seeds (default: 260410 260411 260412 260413 260414 260415)
#
# Usage (from repo root):
#   ./scripts/wave2_year_gap_experiments.sh cache
#   ./scripts/wave2_year_gap_experiments.sh tune-medium
#   ./scripts/wave2_year_gap_experiments.sh multiseed [TUNING_CSV] [MULTISEED_OUT_DIR]
#   ./scripts/wave2_year_gap_experiments.sh aggregate [RUN_ROOT]
#   ./scripts/wave2_year_gap_experiments.sh all    # cache + tune-medium only (multiseed is long; run separately)

set -euo pipefail
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

YEAR_CACHE="${WAVE2_YEAR_CACHE:-$REPO_ROOT/data/paper/openml_year.csv}"
DATE_UTC=$(date -u +%Y-%m-%d)
OUT_ROOT="${WAVE2_OUT_DIR:-$REPO_ROOT/docs/research/sage_reg_results/$DATE_UTC/wave2_paper_year}"
TUNE_DIR="$OUT_ROOT/gap_tune_medium"
SEEDS_DEFAULT=(260410 260411 260412 260413 260414 260415)
# shellcheck disable=SC2206
SEEDS=(${WAVE2_SEEDS:-${SEEDS_DEFAULT[@]}})

PY="${WAVE2_PYTHON:-.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="uv run python"
fi

usage() {
  cat <<'EOF'
wave2_year_gap_experiments.sh <command>

  cache          Materialize OpenML Year CSV cache (network if missing).
  tune-medium    Year-only gap tuning: nl=2048 nu=131072 nt=32768, 32/32 epochs,
                 cosine LR, 12-point grid (3 tau × 2 noise × 2 hard threshold;
                 agreement=0.5, mix=0, drop=0.1, power=2), pseudo_weight=0.8 only.
  multiseed [CSV] [OUT]   Confirm best year row across WAVE2_SEEDS (Year only, --skip-higgs).
                 Defaults: CSV=$TUNE_DIR/sweep.csv, OUT=$OUT_ROOT/multiseed_medium
  aggregate [RUN_ROOT]    Regenerate sage_paper_report.json + METRICS.md (default: neurips 2026-04-17 bundle).
  all            cache + tune-medium
EOF
}

ensure_venv() {
  if [[ ! -f .venv/bin/python ]]; then
    echo "Hint: uv sync or uv pip install -e '.[all]' so .venv exists, or set WAVE2_PYTHON." >&2
  fi
}

cmd_cache() {
  ensure_venv
  if [[ -f "$YEAR_CACHE" ]]; then
    echo "Year cache already present: $YEAR_CACHE"
    return 0
  fi
  $PY "$REPO_ROOT/tools/materialize_openml_year.py" --cache-path "$YEAR_CACHE"
}

cmd_tune_medium() {
  ensure_venv
  mkdir -p "$TUNE_DIR"
  $PY "$REPO_ROOT/examples/benchmarks/self_agreement_supervised_gap_tuning.py" \
    --skip-higgs \
    --year-cache-path "$YEAR_CACHE" \
    --no-year-download \
    --year-n-labeled 2048 \
    --year-n-unlabeled 131072 \
    --year-n-test 32768 \
    --year-teacher-epochs 32 \
    --year-student-epochs 32 \
    --year-lr-schedule cosine \
    --year-lr-min 1e-5 \
    --pseudo-weight-values 0.8 \
    --tau-values 0.12 0.18 0.28 \
    --unlabeled-noise-values 0.05 0.10 \
    --feature-drop-prob-values 0.1 \
    --feature-mix-prob-values 0.0 \
    --agreement-weight-values 0.5 \
    --weight-power-values 2.0 \
    --hard-weight-threshold-values none 0.85 \
    --out-dir "$TUNE_DIR"

  mkdir -p "$OUT_ROOT"
  export WAVE2_MANIFEST_OUT="$OUT_ROOT/manifest.json"
  export WAVE2_MANIFEST_YEAR_CACHE="$YEAR_CACHE"
  export WAVE2_MANIFEST_TUNE="$TUNE_DIR"
  export WAVE2_MANIFEST_MULTI="$OUT_ROOT/multiseed_medium"
  $PY -c "import json, os; from pathlib import Path
out = Path(os.environ['WAVE2_MANIFEST_OUT'])
year_cache = os.environ['WAVE2_MANIFEST_YEAR_CACHE']
tune_dir = os.environ['WAVE2_MANIFEST_TUNE']
multi_out = os.environ['WAVE2_MANIFEST_MULTI']
td = Path(tune_dir)
payload = {
  'artifact': 'wave2_year_gap_experiment_manifest',
  'version': 1,
  'year_cache': year_cache,
  'tuning_out_dir': tune_dir,
  'splits': {'year_n_labeled': 2048, 'year_n_unlabeled': 131072, 'year_n_test': 32768},
  'epochs': {'teacher': 32, 'student': 32},
  'lr': {'schedule': 'cosine', 'min': 1e-5},
  'grid_note': (
    '12 combos: 3 tau × 2 noise × 2 hard threshold; agreement=0.5, '
    'mix=0, drop=0.1, power=2, pseudo=0.8'
  ),
  'next_multiseed': {
    'tuning_csv': str(td / 'sweep.csv'),
    'out_dir': multi_out,
    'seeds': [260410, 260411, 260412, 260413, 260414, 260415],
    'shell': (
      './scripts/wave2_year_gap_experiments.sh multiseed '
      + str(td / 'sweep.csv') + ' ' + multi_out
    ),
  },
}
out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
"
  echo "Wrote $OUT_ROOT/manifest.json"
}

cmd_multiseed() {
  ensure_venv
  local csv="${1:-$TUNE_DIR/sweep.csv}"
  local mout="${2:-$OUT_ROOT/multiseed_medium}"
  $PY "$REPO_ROOT/examples/benchmarks/self_agreement_supervised_gap_multiseed.py" \
    --tuning-csv "$csv" \
    --year-cache-path "$YEAR_CACHE" \
    --no-year-download \
    --year-n-labeled 2048 \
    --year-n-unlabeled 131072 \
    --year-n-test 32768 \
    --year-teacher-epochs 32 \
    --year-student-epochs 32 \
    --year-lr-schedule cosine \
    --year-lr-min 1e-5 \
    --skip-higgs \
    --out-dir "$mout" \
    --seeds "${SEEDS[@]}"
}

cmd_aggregate() {
  ensure_venv
  local root="${1:-$REPO_ROOT/docs/research/sage_reg_results/2026-04-17/neurips_sage_reg_full}"
  $PY "$REPO_ROOT/tools/aggregate_sage_paper_report.py" --run-root "$root" --write-markdown
}

case "${1:-}" in
  cache) cmd_cache ;;
  tune-medium) cmd_cache; cmd_tune_medium ;;
  multiseed) shift || true; cmd_multiseed "${1:-}" "${2:-}" ;;
  aggregate) shift || true; cmd_aggregate "${1:-}" ;;
  all) cmd_cache; cmd_tune_medium ;;
  ""|-h|--help|help) usage ;;
  *) echo "Unknown command: $1" >&2; usage; exit 2 ;;
esac
