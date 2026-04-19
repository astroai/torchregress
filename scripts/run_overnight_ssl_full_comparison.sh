#!/usr/bin/env bash
# Overnight driver: full SSL benchmark coverage + export for official RankUp/PabLO comparison.
#
# Canonical path (override with REPO=...):
#   /Users/fabbros/src/torchregress/scripts/run_overnight_ssl_full_comparison.sh
#
# Usage:
#   mkdir -p ~/logs/ssl_overnight && nohup env OUT=~/logs/ssl_overnight \
#     /Users/fabbros/src/torchregress/scripts/run_overnight_ssl_full_comparison.sh \
#     >>~/logs/ssl_overnight/nohup.log 2>&1 &
#
# Environment (optional):
#   REPO                 Repo root (default: /Users/fabbros/src/torchregress)
#   OUT                  Output root (default: docs/research/sage_reg_results/<date>/overnight_ssl_full)
#   YEAR_CACHE           Year CSV (default: $REPO/data/paper/openml_year.csv)
#   RUN_NEURIPS_FULL     1=run scripts/run_neurips_sage_reg_full.py (default 1)
#   RUN_YEAR_MULTI       1=10-seed Year all-methods (default 1)
#   RUN_LABEL_SWEEP      1=label-fraction × shifts sweep (default 1)
#   RUN_EXPORT_SPLIT     1=export npy split (default 1)
#   STOP_ON_ERROR        1=exit on first failed subprocess (default 1); 0=log and continue
#   OFFICIAL_JSON        If set and exists, run merge_ssl_official_metrics
#   BATCH, DATALOADER_WORKERS, CATBOOST_ITERS
#
set -euo pipefail

REPO="${REPO:-/Users/fabbros/src/torchregress}"
if [[ ! -d "$REPO" ]]; then
  echo "Expected repo at $REPO — export REPO= to your clone root." >&2
  exit 1
fi
cd "$REPO"
export PYTHONUNBUFFERED=1

DATE_LOCAL=$(date +%F)
OUT="${OUT:-$REPO/docs/research/sage_reg_results/$DATE_LOCAL/overnight_ssl_full}"
mkdir -p "$OUT/logs" "$OUT/year_multiseed" "$OUT/year_label_fraction_sweep" "$OUT/split_export"

YEAR_CACHE="${YEAR_CACHE:-$REPO/data/paper/openml_year.csv}"
RUN_NEURIPS_FULL="${RUN_NEURIPS_FULL:-1}"
RUN_YEAR_MULTI="${RUN_YEAR_MULTI:-1}"
RUN_LABEL_SWEEP="${RUN_LABEL_SWEEP:-1}"
RUN_EXPORT_SPLIT="${RUN_EXPORT_SPLIT:-1}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
BATCH="${BATCH:-4096}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"
CATBOOST_ITERS="${CATBOOST_ITERS:-2000}"

SEEDS=(260410 260411 260412 260413 260414 260415 260416 260417 260418 260419)
NL=4096
NU=131072
NT=32768
TEACH=32
STUD=32
UFRAC=(0.25 0.5 1.0)

log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$OUT/logs/driver.log" >&2; }

# Run "$@" with stdout+stderr copied to logfile; log PIPESTATUS (bash 3.2+).
# First element is the main command (e.g. uv); second is tee when piped.
run_tee() {
  local logf="$1"
  shift
  log "PIPELINE_START log=$logf :: $*"
  set +e
  "$@" 2>&1 | tee -a "$logf"
  # PIPESTATUS length can be 1 under some bash/nounset combos; avoid PIPESTATUS[1] unbound.
  local -a _ps=("${PIPESTATUS[@]}")
  local rc0="${_ps[0]}"
  local rc1="${_ps[1]:-0}"
  set -e
  log "PIPELINE_END rc_main=${rc0} rc_tee=${rc1} (0 expected for tee)"
  if [[ "$STOP_ON_ERROR" == "1" && "$rc0" -ne 0 ]]; then
    log "ABORT: main subprocess exit=${rc0} (STOP_ON_ERROR=1). See: $logf"
    exit "$rc0"
  fi
  if [[ "$STOP_ON_ERROR" != "1" && "$rc0" -ne 0 ]]; then
    log "WARN: main subprocess exit=${rc0} (STOP_ON_ERROR=0, continuing)"
  fi
}

log "==== overnight_ssl_full_comparison START ===="
log "REPO=$REPO OUT=$OUT STOP_ON_ERROR=$STOP_ON_ERROR"
log "Materializing Year cache if missing: $YEAR_CACHE"
if [[ ! -f "$YEAR_CACHE" ]]; then
  mkdir -p "$(dirname "$YEAR_CACHE")"
  run_tee "$OUT/logs/materialize_openml_year.log" \
    uv run python "$REPO/tools/materialize_openml_year.py" --cache-path "$YEAR_CACHE"
fi

if [[ "$RUN_NEURIPS_FULL" == "1" ]]; then
  log "Phase A: NeurIPS full evidence driver -> $OUT/neurips_sage_reg_full"
  run_tee "$OUT/logs/neurips_sage_reg_full.log" \
    uv run python "$REPO/scripts/run_neurips_sage_reg_full.py" \
      --run-root "$OUT/neurips_sage_reg_full" \
      --year-cache "$YEAR_CACHE"
  log "Phase A finished OK"
fi

if [[ "$RUN_YEAR_MULTI" == "1" ]]; then
  log "Phase B: 10-seed Year benchmark (all methods)"
  for s in "${SEEDS[@]}"; do
    log "  seed=$s"
    run_tee "$OUT/logs/year_seed_${s}.log" \
      uv run python "$REPO/examples/benchmarks/self_agreement_realdata_year.py" \
        --cache-path "$YEAR_CACHE" \
        --no-download \
        --seed "$s" \
        --n-labeled "$NL" \
        --n-unlabeled "$NU" \
        --n-test "$NT" \
        --teacher-epochs "$TEACH" \
        --student-epochs "$STUD" \
        --batch-size "$BATCH" \
        --dataloader-num-workers "$DATALOADER_WORKERS" \
        --unlabeled-fractions "${UFRAC[@]}" \
        --output-csv "$OUT/year_multiseed/year_seed_${s}.csv" \
        --summary-json-path "$OUT/year_multiseed/year_seed_${s}_summary.json" \
        --performance-figure-path "$OUT/year_multiseed/year_seed_${s}_perf.png" \
        --calibration-figure-path "$OUT/year_multiseed/year_seed_${s}_calib.png"
  done
  log "Phase B: collating CSVs"
  run_tee "$OUT/logs/collate_multiseed.log" \
    uv run python "$REPO/tools/collate_csv_glob.py" \
      --glob "$OUT/year_multiseed/year_seed_*.csv" \
      --out "$OUT/year_multiseed/all_seeds_collated.csv"
  log "Phase B finished OK"
fi

if [[ "$RUN_LABEL_SWEEP" == "1" ]]; then
  log "Phase C: label-pool fraction sweep (long)"
  run_tee "$OUT/logs/year_label_fraction_sweep.log" \
    uv run python "$REPO/examples/benchmarks/sage_year_label_fraction_sweep.py" \
      --out-csv "$OUT/year_label_fraction_sweep/year_label_fraction_sweep.csv" \
      --summary-json "$OUT/year_label_fraction_sweep/summary.json" \
      --cache-path "$YEAR_CACHE" \
      --no-download \
      --seeds "${SEEDS[@]}" \
      --label-percents 0.1 1 2 5 10 25 50 100 \
      --shift-modes none covariate label \
      --min-unlabeled 2048 \
      --n-test "$NT" \
      --batch-size "$BATCH" \
      --dataloader-num-workers "$DATALOADER_WORKERS" \
      --teacher-epochs "$TEACH" \
      --student-epochs "$STUD" \
      --catboost-iterations "$CATBOOST_ITERS"
  log "Phase C finished OK"
fi

if [[ "$RUN_EXPORT_SPLIT" == "1" ]]; then
  log "Phase D: export split seed=${SEEDS[0]}"
  run_tee "$OUT/logs/export_split.log" \
    uv run python "$REPO/tools/export_year_ssl_split_for_external.py" \
      --out-dir "$OUT/split_export/seed_${SEEDS[0]}_default_split" \
      --split-mode default \
      --cache-path "$YEAR_CACHE" \
      --no-download \
      --seed "${SEEDS[0]}" \
      --n-labeled "$NL" \
      --n-unlabeled "$NU" \
      --n-test "$NT"
  cp -f "$REPO/tools/official_metrics_examples/rankup_pablo_official.template.json" \
    "$OUT/split_export/official_metrics_TEMPLATE_fill_and_merge.json"
  log "Phase D finished OK"
fi

if [[ -n "${OFFICIAL_JSON:-}" && -f "${OFFICIAL_JSON}" ]]; then
  log "Phase E: merge official metrics"
  if [[ ! -f "$OUT/year_multiseed/all_seeds_collated.csv" ]]; then
    log "Skip merge: missing $OUT/year_multiseed/all_seeds_collated.csv"
  else
    run_tee "$OUT/logs/merge_official.log" \
      uv run python "$REPO/tools/merge_ssl_official_metrics.py" \
        --ours-csv "$OUT/year_multiseed/all_seeds_collated.csv" \
        --official-json "$OFFICIAL_JSON" \
        --methods "RankUp,PabLOPseudo" \
        --ours-unlabeled-fraction 1.0 \
        --out-json "$OUT/implementation_compare_rankup_pablo.json" \
        --out-csv "$OUT/implementation_compare_rankup_pablo.csv"
    log "Phase E finished OK"
  fi
fi

log "==== overnight_ssl_full_comparison DONE (all requested phases) ===="
log "Artifacts: $OUT"
log "Tail logs: tail -f $OUT/logs/neurips_sage_reg_full.log"
log "Merge later: OFFICIAL_JSON=/path/to.json RUN_NEURIPS_FULL=0 RUN_YEAR_MULTI=0 RUN_LABEL_SWEEP=0 RUN_EXPORT_SPLIT=0 $0"
