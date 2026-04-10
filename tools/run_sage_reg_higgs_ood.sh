#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

today="$(date +%F)"
out_dir="$ROOT_DIR/docs/research/sage_reg_results/$today"
dataset_path=""
dataset_url=""
archive_member=""
target_column="target"
ood_score_column=""
shift_feature_idx="0"
drop_columns=()
run_tests=1

n_train=4096
n_unlabeled_id=16384
n_unlabeled_ood=16384
n_id_test=8192
n_ood_test=8192
teacher_epochs=24
student_epochs=24

usage() {
  cat <<'EOF'
Usage:
  tools/run_sage_reg_higgs_ood.sh [options]

Options:
  --dataset-path PATH        Local CSV/Parquet table for the Higgs benchmark.
  --dataset-url URL          Optional direct download URL for a prepared CSV/Parquet table.
  --archive-member PATH      Optional CSV/Parquet member path inside a downloaded .zip/.tar(.gz) archive.
  --target-column NAME       Target column name in the local table. Default: target
  --ood-score-column NAME    Optional column used to rank OOD severity.
  --drop-column NAME         Column to exclude from features. Repeatable.
  --shift-feature-idx INT    Numeric feature index used for OOD ranking if no score column is given. Default: 0
  --out-dir PATH             Output directory. Default: docs/research/sage_reg_results/<today>
  --n-train INT              ID labeled train size. Default: 4096
  --n-unlabeled-id INT       ID unlabeled pool size. Default: 16384
  --n-unlabeled-ood INT      OOD unlabeled pool size. Default: 16384
  --n-id-test INT            ID test size. Default: 8192
  --n-ood-test INT           OOD test size. Default: 8192
  --teacher-epochs INT       Teacher epochs. Default: 24
  --student-epochs INT       Student epochs. Default: 24
  --skip-tests               Do not run the focused pytest slice first.
  --help                     Show this message.

Behavior:
  - If --dataset-path is omitted and --dataset-url is omitted, the script runs
    the built-in Higgs-like OOD proxy benchmark.
  - If --dataset-url is provided and --dataset-path is omitted, the file is
    downloaded into <out-dir>/raw/. CSV/Parquet files are used directly.
  - For downloaded .zip/.tar/.tar.gz archives, the script extracts either:
      1. the file named by --archive-member, or
      2. the first CSV/Parquet file found in the archive.
  - Results are written into <out-dir>.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-path)
      dataset_path="$2"
      shift 2
      ;;
    --dataset-url)
      dataset_url="$2"
      shift 2
      ;;
    --archive-member)
      archive_member="$2"
      shift 2
      ;;
    --target-column)
      target_column="$2"
      shift 2
      ;;
    --ood-score-column)
      ood_score_column="$2"
      shift 2
      ;;
    --drop-column)
      drop_columns+=("$2")
      shift 2
      ;;
    --shift-feature-idx)
      shift_feature_idx="$2"
      shift 2
      ;;
    --out-dir)
      out_dir="$2"
      shift 2
      ;;
    --n-train)
      n_train="$2"
      shift 2
      ;;
    --n-unlabeled-id)
      n_unlabeled_id="$2"
      shift 2
      ;;
    --n-unlabeled-ood)
      n_unlabeled_ood="$2"
      shift 2
      ;;
    --n-id-test)
      n_id_test="$2"
      shift 2
      ;;
    --n-ood-test)
      n_ood_test="$2"
      shift 2
      ;;
    --teacher-epochs)
      teacher_epochs="$2"
      shift 2
      ;;
    --student-epochs)
      student_epochs="$2"
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

mkdir -p "$out_dir"

resolve_archive_dataset_path() {
  local archive_path="$1"
  local extract_dir="$2"
  local selected_member=""

  mkdir -p "$extract_dir"

  case "$archive_path" in
    *.zip)
      if [[ -n "$archive_member" ]]; then
        selected_member="$extract_dir/$(basename "$archive_member")"
        if [[ ! -f "$selected_member" ]]; then
          unzip -j -o "$archive_path" "$archive_member" -d "$extract_dir" >/dev/null
        fi
      else
        selected_member="$(zipinfo -1 "$archive_path" | rg -m1 '\.(csv|parquet)$' || true)"
        if [[ -z "$selected_member" ]]; then
          echo "No CSV/Parquet member found in archive: $archive_path" >&2
          exit 1
        fi
        local member_name="$selected_member"
        selected_member="$extract_dir/$(basename "$member_name")"
        if [[ ! -f "$selected_member" ]]; then
          unzip -j -o "$archive_path" "$member_name" -d "$extract_dir" >/dev/null
        fi
      fi
      ;;
    *.tar|*.tar.gz|*.tgz)
      if [[ -n "$archive_member" ]]; then
        selected_member="$extract_dir/$(basename "$archive_member")"
        if [[ ! -f "$selected_member" ]]; then
          tar -xOf "$archive_path" "$archive_member" >"$selected_member"
        fi
      else
        selected_member="$(tar -tf "$archive_path" | rg -m1 '\.(csv|parquet)$' || true)"
        if [[ -z "$selected_member" ]]; then
          echo "No CSV/Parquet member found in archive: $archive_path" >&2
          exit 1
        fi
        local member_name="$selected_member"
        selected_member="$extract_dir/$(basename "$member_name")"
        if [[ ! -f "$selected_member" ]]; then
          tar -xOf "$archive_path" "$member_name" >"$selected_member"
        fi
      fi
      ;;
    *)
      echo "Unsupported archive format: $archive_path" >&2
      exit 1
      ;;
  esac

  if [[ ! -f "$selected_member" ]]; then
    echo "Expected extracted dataset not found: $selected_member" >&2
    exit 1
  fi

  printf '%s\n' "$selected_member"
}

if [[ -z "$dataset_path" && -n "$dataset_url" ]]; then
  raw_dir="$out_dir/raw"
  mkdir -p "$raw_dir"
  filename="$(basename "${dataset_url%%\?*}")"
  if [[ -z "$filename" || "$filename" == "/" ]]; then
    echo "Could not infer filename from dataset URL: $dataset_url" >&2
    exit 1
  fi
  dataset_path="$raw_dir/$filename"
  if [[ ! -f "$dataset_path" ]]; then
    echo "Downloading dataset to $dataset_path"
    curl -L --fail --output "$dataset_path" "$dataset_url"
  else
    echo "Using existing downloaded dataset: $dataset_path"
  fi

  case "$dataset_path" in
    *.csv|*.parquet)
      ;;
    *.zip|*.tar|*.tar.gz|*.tgz)
      echo "Extracting dataset archive: $dataset_path"
      dataset_path="$(resolve_archive_dataset_path "$dataset_path" "$out_dir/extracted")"
      ;;
    *)
      echo "Downloaded file is not a supported dataset or archive: $dataset_path" >&2
      exit 1
      ;;
  esac
fi

if [[ "$run_tests" == "1" ]]; then
  uv run pytest -q \
    tests/test_self_agreement_higgs_ood.py \
    tests/test_self_agreement_benchmark.py \
    tests/test_self_agreement_realdata_year.py \
    tests/test_semi_supervised.py \
    tests/test_public_api_contracts.py \
    tests/test_self_agreement_backbone_comparison.py
fi

cmd=(
  uv run python examples/benchmarks/self_agreement_higgs_ood.py
  --n-train "$n_train"
  --n-unlabeled-id "$n_unlabeled_id"
  --n-unlabeled-ood "$n_unlabeled_ood"
  --n-id-test "$n_id_test"
  --n-ood-test "$n_ood_test"
  --teacher-epochs "$teacher_epochs"
  --student-epochs "$student_epochs"
  --output-csv "$out_dir/higgs_ood.csv"
  --performance-figure-path "$out_dir/higgs_ood_perf.png"
  --calibration-figure-path "$out_dir/higgs_ood_calib.png"
  --summary-json-path "$out_dir/higgs_ood_summary.json"
)

if [[ -n "$dataset_path" ]]; then
  cmd+=(--dataset-path "$dataset_path")
  cmd+=(--target-column "$target_column")
  cmd+=(--shift-feature-idx "$shift_feature_idx")
  if [[ -n "$ood_score_column" ]]; then
    cmd+=(--ood-score-column "$ood_score_column")
  fi
  for drop_column in "${drop_columns[@]:-}"; do
    if [[ -n "$drop_column" ]]; then
      cmd+=(--drop-column "$drop_column")
    fi
  done
fi

echo "Running benchmark:"
printf '  %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}" | tee "$out_dir/higgs_ood_run.log"

cat <<EOF

Artifacts written to:
  $out_dir

Suggested next step:
  inspect $out_dir/higgs_ood_summary.json
EOF
