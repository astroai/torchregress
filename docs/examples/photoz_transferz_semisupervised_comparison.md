# TransferZ Semi-Supervised Photo-z Comparison

Script: `examples/photoz_transferz_semisupervised_comparison.py`

This is the dedicated real-data semi-supervised photo-z benchmark for the public `TransferZ` release.

Split semantics are preserved:

- `TRAINING` -> partially labeled student/teacher fit
- `VALIDATION` -> teacher and Gaussian-student variance calibration
- `TESTING` -> final evaluation

The default protocol is intentionally difficult:

- multiple labeled fractions (`10%`, `25%`, `50%`)
- `highz_scarce` label selection by default, so labeled examples underrepresent the rare high-z tail
- pseudo labels are only generated for the unlabeled part of the released training split

## Methods

- `HuberLabeledOnly`
- `GaussianLabeledOnly`
- `PseudoLabelNLL`
- `PseudoLabelConsistency`
- `SelectivePseudoLabelNLL`
- `FeatureAwarePseudoLabelConsistency`
- `EMASelectiveConsistency`

## Reported metrics

- `LabeledFraction`
- `NMAD`
- `CatastrophicRate`
- `HighZ_MAE`
- `PseudoAcceptRate`
- `PseudoMeanConfidence`
- `AcceptedHighZShare`
- `AcceptedLowErrShare`
- `TeacherDisagreement`
- `FeatureStability`
- `LabeledHighZShare`
- `TrainHighZShare`
- `NLL`, `Cov90`, `Width90` for Gaussian rows
- `train_s`, `eval_s`

## Run

If normalized `TransferZ` splits are already present:

```bash
uv run python examples/photoz_transferz_semisupervised_comparison.py \
  --train-dataset-path data/transferz/normalized/transferz_train_photoz.csv \
  --cal-dataset-path data/transferz/normalized/transferz_cal_photoz.csv \
  --test-dataset-path data/transferz/normalized/transferz_test_photoz.csv \
  --require-real-data
```

Or run through the end-to-end pipeline:

```bash
uv run python tools/photoz_transferz_pipeline.py \
  --profile full \
  --download-if-missing
```

## Summary Artifact

The `TransferZ` pipeline writes this example into:

```bash
reports/example_summaries/transferz/photoz_transferz_semisupervised_comparison_full.json
```

## Notes

!!! info
    The benchmark uses `VALIDATION` to calibrate Gaussian teacher uncertainty before pseudo-label confidence thresholding. The stronger SSL rows also use the catalogued feature errors (`*_err`) to reject pseudo labels that are unstable under feature perturbations. `EMASelectiveConsistency` adds a mean-teacher-style consistency path on top of the same selective pseudo-label pool.

!!! warning
    If pseudo-label rows lose to labeled-only baselines here, that is useful evidence. On public tabular photo-z data with real shift and tail scarcity, semi-supervision is not automatically a win.
