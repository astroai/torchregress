#!/usr/bin/env bash
# One-shot SAGE + SPT tabular experiment bundle for paper assessment.
#
# Produces under TABULAR_BUNDLE_OUT:
#   sage/year_direct/summary.json (+ csv, figures)
#   sage/multiseed/multiseed_*.json|csv
#   spt/full/year_competing_methods_<profile>.json + synthetic/realdata summaries + artifact_manifest.json
#   tabular_paper_bundle_report.json (+ METRICS.md) from tools/aggregate_tabular_paper_bundle_report.py
#
# Usage (repo root):
#   ./scripts/run_tabular_paper_bundle.sh              # full budgets
#   ./scripts/run_tabular_paper_bundle.sh --smoke      # tiny synthetic year + SPT smoke (~minutes)
#
# All defaults are set below (export VAR=value before running to override).
#
set -euo pipefail

SMOKE=false
for arg in "$@"; do
  if [[ "${arg}" == "--smoke" ]]; then
    SMOKE=true
  fi
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# Defaults (same layout as papers/neurips_sage_reg/reproducibility.md). Override with export.
TABULAR_BUNDLE_OUT="${TABULAR_BUNDLE_OUT:-${ROOT}/docs/research/sage_reg_results/$(date +%Y-%m-%d)/tabular_paper_bundle}"
YEAR_CACHE="${YEAR_CACHE:-${ROOT}/docs/research/sage_reg_results/2026-04-10/openml_year.csv}"
TUNING_CSV="${TUNING_CSV:-${ROOT}/docs/research/sage_reg_results/2026-04-10/supervised_gap_tuning_v3/sweep.csv}"
# FAIR Universe Higgs ML tabular parquet (download / extract per reproducibility doc; not in git by default).
HIGGS_PARQUET="${HIGGS_PARQUET:-${ROOT}/docs/research/sage_reg_results/2026-04-09/higgs_public/extracted/FAIR_Universe_HiggsML_data.parquet}"
HIGGS_SPLIT_SCALE="${HIGGS_SPLIT_SCALE:-10}"
HIGGS_PARQUET_MAX_SAMPLE_ROWS="${HIGGS_PARQUET_MAX_SAMPLE_ROWS:-8000000}"

OUT="${TABULAR_BUNDLE_OUT}"
mkdir -p "${OUT}/sage/year_direct" "${OUT}/sage/multiseed" "${OUT}/spt/full"

STARTED="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

if ${SMOKE}; then
  YEAR_EFF="${OUT}/_smoke_year.csv"
  echo "== Smoke: materialize ${YEAR_EFF}"
  uv run python -c "
import pathlib
import numpy as np
import pandas as pd
p = pathlib.Path('${YEAR_EFF}')
rng = np.random.default_rng(42)
n = 12000
cols = {f'f{i}': rng.standard_normal(n).astype('float32') for i in range(8)}
cols['target'] = rng.standard_normal(n).astype('float32')
pd.DataFrame(cols).to_csv(p, index=False)
print('wrote', p)
"
  YEAR_USE="${YEAR_EFF}"
  NL=512
  NU=4096
  NT=1024
  YTEACH=1
  YSTU=1
  SEEDS=(260410)
  UFRAC=(0.25 0.5 1.0)
  SPT_PROFILE="smoke"
  SKIP_HIGGS=true
else
  YEAR_USE="${YEAR_CACHE}"
  if [[ ! -f "${YEAR_USE}" ]]; then
    echo "error: YEAR_CACHE not found: ${YEAR_USE}" >&2
    echo "  Set YEAR_CACHE to your OpenML Year table, or run with --smoke." >&2
    exit 1
  fi
  NL=4096
  NU=131072
  NT=32768
  YTEACH=32
  YSTU=32
  SEEDS=(260410 260411 260412)
  UFRAC=(0.25 0.5 1.0)
  SPT_PROFILE="full"
  SKIP_HIGGS=true
  if [[ -f "${HIGGS_PARQUET}" ]]; then
    SKIP_HIGGS=false
  else
    echo "== Note: Higgs parquet not found at default path (skipping Higgs in multiseed):" >&2
    echo "    ${HIGGS_PARQUET}" >&2
    echo "    Download/extract per papers/neurips_sage_reg/reproducibility.md or set HIGGS_PARQUET." >&2
  fi
fi

if [[ ! -f "${TUNING_CSV}" ]]; then
  echo "error: TUNING_CSV not found: ${TUNING_CSV}" >&2
  exit 1
fi

echo "== Bundle output: ${OUT}"
echo "== Year table:   ${YEAR_USE}"
echo "== Tuning CSV:   ${TUNING_CSV}"
echo "== Mode:         $(${SMOKE} && echo smoke || echo full)"
if ! ${SKIP_HIGGS}; then
  echo "== Higgs:        ${HIGGS_PARQUET}"
fi

echo "== 1/3 SAGE year direct (methods × unlabeled fraction)"
uv run python examples/benchmarks/self_agreement_realdata_year.py \
  --cache-path "${YEAR_USE}" \
  --no-download \
  --n-labeled "${NL}" \
  --n-unlabeled "${NU}" \
  --n-test "${NT}" \
  --teacher-epochs "${YTEACH}" \
  --student-epochs "${YSTU}" \
  --unlabeled-fractions "${UFRAC[@]}" \
  --output-csv "${OUT}/sage/year_direct/metrics.csv" \
  --performance-figure-path "${OUT}/sage/year_direct/performance.png" \
  --calibration-figure-path "${OUT}/sage/year_direct/calibration.png" \
  --summary-json-path "${OUT}/sage/year_direct/summary.json"

echo "== 2/3 SAGE supervised-gap multiseed (tuned hypers from CSV)"
MULTI_CMD=(
  uv run python examples/benchmarks/self_agreement_supervised_gap_multiseed.py
  --tuning-csv "${TUNING_CSV}"
  --year-cache-path "${YEAR_USE}"
  --no-year-download
  --year-n-labeled "${NL}"
  --year-n-unlabeled "${NU}"
  --year-n-test "${NT}"
  --year-teacher-epochs "${YTEACH}"
  --year-student-epochs "${YSTU}"
  --out-dir "${OUT}/sage/multiseed"
  --seeds "${SEEDS[@]}"
)

if ${SKIP_HIGGS}; then
  "${MULTI_CMD[@]}" --skip-higgs
else
  "${MULTI_CMD[@]}" \
    --higgs-dataset-path "${HIGGS_PARQUET}" \
    --higgs-split-scale-factor "${HIGGS_SPLIT_SCALE}" \
    --higgs-parquet-max-sample-rows "${HIGGS_PARQUET_MAX_SAMPLE_ROWS}" \
    --higgs-teacher-epochs "${YTEACH}" \
    --higgs-student-epochs "${YSTU}"
fi

echo "== 3/3 SPT paper render (profile=${SPT_PROFILE})"
uv run python tools/render_spt_reg_paper_artifacts.py \
  --profile "${SPT_PROFILE}" \
  --year-cache-path "${YEAR_USE}" \
  --output-dir "${OUT}/spt/full" \
  --report "${OUT}/spt/full/artifact_manifest.json"

FINISHED="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

export BUNDLE_META_OUT="${OUT}/bundle_meta.json"
export BUNDLE_STARTED="${STARTED}"
export BUNDLE_FINISHED="${FINISHED}"
export BUNDLE_SMOKE="$(${SMOKE} && echo true || echo false)"
export BUNDLE_YEAR_USE="${YEAR_USE}"
export BUNDLE_TUNING_CSV="${TUNING_CSV}"
export BUNDLE_NL="${NL}"
export BUNDLE_NU="${NU}"
export BUNDLE_NT="${NT}"
export BUNDLE_YTEACH="${YTEACH}"
export BUNDLE_YSTU="${YSTU}"
export BUNDLE_SEEDS="${SEEDS[*]}"
export BUNDLE_SPT_PROFILE="${SPT_PROFILE}"
export BUNDLE_HIGGS="${HIGGS_PARQUET:-}"
export BUNDLE_SKIP_HIGGS="$(${SKIP_HIGGS} && echo true || echo false)"

uv run python -c "
import json, os, pathlib
out = pathlib.Path(os.environ['BUNDLE_META_OUT'])
seeds = [int(x) for x in os.environ['BUNDLE_SEEDS'].split()]
meta = {
    'started_at_utc': os.environ['BUNDLE_STARTED'],
    'finished_at_utc': os.environ['BUNDLE_FINISHED'],
    'smoke': os.environ['BUNDLE_SMOKE'] == 'true',
    'year_cache': os.environ['BUNDLE_YEAR_USE'],
    'tuning_csv': os.environ['BUNDLE_TUNING_CSV'],
    'splits': {
        'n_labeled': int(os.environ['BUNDLE_NL']),
        'n_unlabeled': int(os.environ['BUNDLE_NU']),
        'n_test': int(os.environ['BUNDLE_NT']),
        'teacher_epochs': int(os.environ['BUNDLE_YTEACH']),
        'student_epochs': int(os.environ['BUNDLE_YSTU']),
    },
    'seeds': seeds,
    'spt_profile': os.environ['BUNDLE_SPT_PROFILE'],
    'higgs_parquet': os.environ.get('BUNDLE_HIGGS', ''),
    'skip_higgs': os.environ['BUNDLE_SKIP_HIGGS'] == 'true',
}
out.write_text(json.dumps(meta, indent=2), encoding='utf-8')
print('Wrote', out)
"

echo "== Aggregate report"
uv run python tools/aggregate_tabular_paper_bundle_report.py \
  --bundle-dir "${OUT}" \
  --write-markdown

echo ""
echo "Done. Primary artifacts:"
echo "  ${OUT}/tabular_paper_bundle_report.json"
echo "  ${OUT}/METRICS.md"
echo "  ${OUT}/spt/full/year_competing_methods_${SPT_PROFILE}.json"
