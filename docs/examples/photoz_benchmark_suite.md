# Photo-z Benchmark Suite

Tool: `tools/photoz_benchmark_suite.py`

Use this when you want one reproducible entry point for the main photo-z benchmark tracks instead of running each example by hand.

## What the suite runs

Core local suite:

- `examples/photoz_benchmark_comparison.py`
- `examples/photoz_nnc_crps_rail_comparison.py`
- `examples/ppi_photoz_inference_comparison.py`

Dedicated TransferZ-only extension:

- `examples/photoz_transferz_semisupervised_comparison.py`
- `examples/photoz_transferz_conformal_comparison.py`

Optional external merge:

- `tools/photoz_rail_pipeline.py` on top of the ordered-bin benchmark

## Benchmark map

| Track | Purpose | Main methods | Output artifact |
|---|---|---|---|
| `photoz_benchmark_comparison` | tabular regression benchmark with robust/probabilistic/uncertain-target/imbalance/EIV/SSL methods | `MSE`, `Huber`, `DensityWeightedHuber`, `LogTransform`, `GaussianNLL`, `NoisyTargetGaussianNLL`, `Quantile90`, `PseudoLabelNLL`, `PseudoLabelConsistency`, `FunctionalEIV` | `photoz_benchmark_comparison_<profile>.json` |
| `photoz_nnc_crps_rail_comparison` | ordered-bin regression-as-classification benchmark | `BinnedCE`, `SoftBinnedCE`, `SoftBinnedCE+Pseudo`, `SoftCumulativeLink`, `OrderedBinCRPS`, anchors | `photoz_nnc_crps_rail_comparison_<profile>.json` |
| `ppi_photoz_inference_comparison` | low-label inference benchmark | labeled-only vs `PPI` mean/quantile CIs | `ppi_photoz_inference_comparison_<profile>.json` |
| `photoz_transferz_semisupervised_comparison` | dedicated real-data SSL benchmark on released `TransferZ` splits | labeled-only baselines plus bootstrap and feature-error-aware selective pseudo-label methods across labeled fractions | `transferz/photoz_transferz_semisupervised_comparison_<profile>.json` |
| `photoz_transferz_conformal_comparison` | dedicated real-data conformal benchmark on released `TransferZ` splits | `SplitConformal`, `CQR`, `DensityConformal`, `PrevalenceAdjustedCP`, `MonteCarloConformal`, `R2CConformal` | `transferz/photoz_transferz_conformal_comparison_<profile>.json` |
| `photoz_rail_pipeline` | cross-framework comparison against RAIL baselines | torchregress ordered-bin summary + `flexzboost`, `pzflow`, `delight`, `bpz` | `photoz_rail_baseline_comparison_<profile>.json` |

## Recommended run order

1. Run the local suite first.
2. Inspect the standard regression benchmark.
3. Inspect the ordered-bin benchmark.
4. Only then run the RAIL merge, once you have the external baseline assets or manifest.

## Run the local suite

Smoke:

```bash
uv run python tools/photoz_benchmark_suite.py --profile smoke
```

Audit-sized:

```bash
uv run python tools/photoz_benchmark_suite.py --profile audit
```

Full local suite:

```bash
uv run python tools/photoz_benchmark_suite.py --profile full
```

Real-data-only suite:

```bash
uv run python tools/photoz_benchmark_suite.py --profile full --real-data-only
```

Real-data-only suite on an explicit external dataset:

```bash
uv run python tools/photoz_benchmark_suite.py \
  --profile full \
  --real-data-only \
  --dataset-path data/nnc_crps/nnc_photoz_real.csv
```

Real-data-only suite on explicit released split files:

```bash
uv run python tools/photoz_benchmark_suite.py \
  --profile full \
  --real-data-only \
  --train-dataset-path data/transferz/normalized/transferz_train_photoz.csv \
  --cal-dataset-path data/transferz/normalized/transferz_cal_photoz.csv \
  --test-dataset-path data/transferz/normalized/transferz_test_photoz.csv
```

This writes:

- `reports/example_summaries/photoz_benchmark_comparison_<profile>.json`
- `reports/example_summaries/photoz_nnc_crps_rail_comparison_<profile>.json`
- `reports/example_summaries/ppi_photoz_inference_comparison_<profile>.json`
- `reports/example_summaries/photoz_benchmark_suite_latest.json`
- `reports/example_summaries/photoz_benchmark_suite_latest.md`

When `--real-data-only` is enabled:

- the suite requires real data for `photoz_benchmark_comparison`
- the suite requires real data for `photoz_nnc_crps_rail_comparison`
- the synthetic-only `ppi_photoz_inference_comparison` track is skipped

## Run with RAIL merge

If you already have a manifest and the baseline payloads:

```bash
uv run python tools/photoz_benchmark_suite.py \
  --profile full \
  --include-rail-merge \
  --manifest data/rail/rail_photoz_manifest.json
```

If you want the merge step to materialize missing assets automatically:

```bash
uv run python tools/photoz_benchmark_suite.py \
  --profile full \
  --include-rail-merge \
  --manifest data/rail/rail_photoz_manifest.json \
  --allow-download
```

If the manifest does not exist yet and you want the NNC/RAIL preset bootstrap:

```bash
uv run python tools/photoz_benchmark_suite.py \
  --profile full \
  --include-rail-merge \
  --rail-preset nnc_crps \
  --manifest data/rail/rail_photoz_manifest.json \
  --allow-download
```

## Real-data preparation

DP0.2-style frame for the standard benchmark:

```bash
uv run python -m tools.photoz_collect_real_data dp02 \
  --photoz-output data/sdss/sdss_photoz_real.csv \
  --report reports/example_summaries/photoz_dp02_collection_latest.json
```

NNC-style catalog collection:

```bash
uv run python -m tools.photoz_collect_real_data nnc \
  --report reports/example_summaries/photoz_nnc_catalog_collection_latest.json
```

TransferZ split collection and normalization:

```bash
uv run python -m tools.photoz_collect_real_data transferz \
  --report reports/example_summaries/photoz_transferz_collection_latest.json
```

TransferZ end-to-end tabular benchmark:

```bash
uv run python tools/photoz_transferz_pipeline.py \
  --profile full \
  --download-if-missing
```

After the real data exists locally, the standard benchmark will consume `data/sdss/sdss_photoz_real.csv` automatically unless `--force-simulated` is set.

If you want the benchmark to fail instead of silently simulating, use:

```bash
uv run python examples/photoz_benchmark_comparison.py \
  --require-real-data \
  --summary-json-path reports/example_summaries/photoz_benchmark_comparison_full.json
```

```bash
uv run python examples/photoz_nnc_crps_rail_comparison.py \
  --require-real-data \
  --summary-json-path reports/example_summaries/photoz_nnc_crps_rail_comparison_full.json
```

## How to compare results

Start with these questions:

1. `photoz_benchmark_comparison`: which methods reduce `NMAD`, `CatastrophicRate`, and `HighZ_MAE` without collapsing coverage?
2. `photoz_nnc_crps_rail_comparison`: do soft-bin methods or ordered-bin CRPS improve `CRPS`, `PDF_NLL`, and `PITChi2` over hard-bin CE?
3. `ppi_photoz_inference_comparison`: do prediction-powered intervals shrink `CIWidth` while retaining `CoversTruth`?
4. `photoz_rail_baseline_comparison`: does the best torchregress row beat the best external baseline on the chosen metric?

The suite markdown report is the fastest first pass:

- `reports/example_summaries/photoz_benchmark_suite_latest.md`
- it ranks the standard track by `NMAD`, `CatastrophicRate`, `HighZ_MAE`
- it also calls out the best row on the highest-feature-error slice using the catalogued `*_err` columns
- it also calls out best robust and noisy-label-aware rows inside the standard track
- it ranks the ordered-bin track by `CRPS`, `PDF_NLL`, `PITChi2`
- it ranks the TransferZ semi-supervised track by `LabeledFraction`, `NMAD`, `CatastrophicRate`, and also shows the best SSL-only row at each labeled fraction
- it includes the PPI track when that track was run

For the ordered-bin track specifically:

- `BinnedCE`: hard-bin baseline
- `SoftBinnedCE`: soft bin targets from spectroscopic-redshift uncertainty
- `SoftBinnedCE+Pseudo`: soft bin targets plus teacher soft pseudo labels on a partial-label track
- `SoftCumulativeLink`: core cumulative ordinal objective on soft ordered-bin targets
- `OrderedBinCRPS`: specialized example-local objective

## Decision guidance

!!! tip
    Use `photoz_benchmark_comparison` to choose the general-purpose regression family first.

!!! tip
    Use `photoz_nnc_crps_rail_comparison` only when the PDF itself matters and you want an ordered-bin formulation.

!!! warning
    Do not compare the ordered-bin rows to the point-regression rows on a single metric only. Use the photo-z domain metrics together with `CRPS`, `PDF_NLL`, `PITChi2`, and interval width/coverage.

!!! info
    `TransferZ` should be preferred over the older ad hoc NNC fallback when you want a public tabular benchmark with released train/validation/test/conformal splits and explicit domain-shift motivation.
