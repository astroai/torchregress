# Photo-z Benchmark Comparison (SDSS-Style)

This example provides a standardized comparison benchmark for **photometric redshift estimation** using SDSS-style features (colors plus measurement errors).

- Script: `examples/photoz_benchmark_comparison.py`
- Output: shared-budget comparison table + optional machine-readable summary JSON
- Benchmark focus: robust losses, probabilistic intervals, transforms, pseudo-label semi-supervision, and EIV under a domain-realistic workload

## Why this benchmark matters

Photo-z is a high-value external-validity benchmark for `torchregress` because it combines:

- heteroscedastic noise
- measurement error in inputs (EIV relevance)
- tail/rare-target behavior (high-z degradation)
- calibration-sensitive evaluation (interval coverage/width)
- partial-label realism (spec-z available for only part of the training pool)
- positive-support target skew (transform-loss relevance)

## Methods compared

The benchmark compares a shared-budget set of methods:

- `MSE`
- `Huber`
- `LogTransform`
- `Quantile90` (`MultiQuantileLoss`)
- `GaussianNLL`
- `PseudoLabelNLL`
- `PseudoLabelConsistency`
- `FunctionalEIV`

## Fairness controls

- Fixed seeds
- Shared train/cal/test splits
- Shared epoch/batch-size/lr budget across methods
- Common metrics and runtime reporting

## Metrics reported

- `RMSE`, `MAE`
- `NMAD` (photo-z standard)
- catastrophic outlier rate (`|Δz| > 0.15(1+z)`)
- high-z MAE (tail slice)
- interval coverage/width (`Cov90`, `Width90`) for uncertainty-capable methods
- labeled-fraction and pseudo-label diagnostics for the SSL rows
- runtime (`train_s`, `eval_s`)

## Data source behavior

- Uses cached real SDSS data if available (`data/sdss/sdss_photoz_real.csv`)
- Falls back to deterministic simulated SDSS-style data otherwise

This makes the benchmark runnable in offline/CI environments while preserving a clean path to real-data evaluation.

## Example usage

```python
from examples.photoz_benchmark_comparison import PhotoZBenchmarkConfig, main

cfg = PhotoZBenchmarkConfig(
    n_train=192,
    n_cal=64,
    n_test=64,
    epochs=8,
    labeled_fraction=0.35,
    teacher_epochs=8,
    pseudo_confidence_threshold=0.35,
    force_simulated=True,   # set False to use cached real SDSS if available
    allow_download=False,   # optional; keep False in CI/offline environments
)
main(cfg, summary_json_path="reports/example_summaries/photoz_benchmark_comparison_audit.json")
```

## Notes

- This benchmark improves domain realism and external-validity evidence, but it is still a compact MLP-based benchmark rather than a production astronomy pipeline.
- `PseudoLabelNLL` and `PseudoLabelConsistency` use a partial-spec-z protocol inside the train split: a bootstrap Gaussian teacher is fitted on the labeled subset, then pseudo labels are generated for the remainder.
- `LogTransform` is included as a positive-support skew ablation, not as a claim that log-space is universally best for photo-z.
- For a broader astronomy walkthrough and diagnostics, see `examples/photoz.py` and this docs section’s `Photo-z` page.
