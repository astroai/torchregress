#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

today="$(date +%F)"
out_dir="$ROOT_DIR/docs/research/sage_reg_results/$today/higgs_public"
dataset_path=""
run_tests=1

usage() {
  cat <<'EOF'
Usage:
  tools/run_sage_reg_higgs_public.sh [options]

Options:
  --dataset-path PATH   Use an already-downloaded FAIR Universe parquet/CSV instead of downloading.
  --out-dir PATH        Output directory. Default: docs/research/sage_reg_results/<today>/higgs_public
  --skip-tests          Do not run the focused pytest slice first.
  --help                Show this message.

Behavior:
  - Downloads the official FAIR Universe HiggsML public dataset from Zenodo if needed.
  - Extracts FAIR_Universe_HiggsML_data.parquet from the official zip.
  - Runs the torchregress Higgs OOD benchmark on that table using:
      target column: labels
      excluded feature columns: weights, detailed_labels
      OOD ranking column: PRI_met

Important:
  This does NOT reproduce the official challenge ingestion/scoring protocol.
  It runs our torchregress OOD benchmark on the public event table.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-path)
      dataset_path="$2"
      shift 2
      ;;
    --out-dir)
      out_dir="$2"
      shift 2
      ;;
    --skip-tests)
      run_tests=0
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

cmd=(
  bash tools/run_sage_reg_higgs_ood.sh
  --out-dir "$out_dir"
)

if [[ "$run_tests" == "0" ]]; then
  cmd+=(--skip-tests)
fi

if [[ -n "$dataset_path" ]]; then
  cmd+=(
    --dataset-path "$dataset_path"
  )
else
  cmd+=(
    --dataset-url "https://zenodo.org/records/15131565/files/FAIR_Universe_HiggsML_data.zip?download=1"
    --archive-member "FAIR_Universe_HiggsML_data.parquet"
  )
fi

cmd+=(
  --target-column "labels"
  --ood-score-column "PRI_met"
  --drop-column "weights"
  --drop-column "detailed_labels"
)

printf 'Running official public Higgs benchmark wrapper:\n'
printf '  %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
