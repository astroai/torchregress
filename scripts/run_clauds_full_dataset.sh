#!/usr/bin/env bash
# Run CLAUDS photo-z comparison on full dataset: supervised + semi-supervised,
# proper 80/10/10 stratified splits, all-bands with missing data (outer join).
#
# Requires: data/clauds_specz/clauds_specz_catalog.parquet (154k+ rows with spec_z)
# Runtime: several hours depending on hardware (large train set).
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CATALOG="${1:-data/clauds_specz/clauds_specz_catalog.parquet}"
# 80% train, 10% cal, 10% test. After all-bands+mask filter we have ~153999 rows (not 154072).
N_TRAIN="${N_TRAIN:-123199}"
N_CAL="${N_CAL:-15400}"
N_TEST="${N_TEST:-15400}"

# Optional: larger batch and more epochs for full data (override via env)
BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-15}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-18}"

echo "CLAUDS full-dataset run: catalog=$CATALOG"
echo "  train=$N_TRAIN cal=$N_CAL test=$N_TEST batch=$BATCH_SIZE epochs=$EPOCHS teacher_epochs=$TEACHER_EPOCHS"
echo "  feature_set=all_bands_missing_ok split=stratified_redshift label_policy=highz_scarce"
echo ""

uv run python examples/photoz_clauds_specz_comparison.py \
  --catalog-path "$CATALOG" \
  --feature-set all_bands_missing_ok \
  --n-train "$N_TRAIN" \
  --n-cal "$N_CAL" \
  --n-test "$N_TEST" \
  --split-policy stratified_redshift \
  --stratify-n-bins 20 \
  --label-policy highz_scarce \
  --labeled-fractions 0.1 0.25 0.5 1.0 \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --teacher-epochs "$TEACHER_EPOCHS" \
  --summary-json-path "reports/example_summaries/photoz_clauds_specz_full_dataset_latest.json" \
  "$@"
