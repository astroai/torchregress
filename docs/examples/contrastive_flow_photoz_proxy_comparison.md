# Contrastive Flow Photo-z Proxy Comparison

This example uses the repository's SDSS-style photo-z benchmark loader as a **real-data proxy**
for parameter estimation. Instead of predicting per-object redshift directly, it constructs
catalog-level summaries under:

- a global redshift calibration shift
- a global noise / error-inflation nuisance term

and compares summary-density models on the task of recovering those parameters.

## Why This Exists

The synthetic Higgs-style benchmark is useful, but it does not connect to the domain workloads
already emphasized in `torchregress`. This proxy comparison reuses the photo-z data path so that
the contrastive objective is evaluated on:

- real cached photo-z covariates when available
- deterministic SDSS-style simulated fallback otherwise

## Compared Methods

| Method | Role |
|:--|:--|
| `GaussianSummary` | simple density baseline on catalog summaries |
| `NormalizingFlow` | plain conditional flow likelihood |
| `ContrastiveFlow` | contrastive flow objective for hypothesis ranking |

## Runnable Example

```python
from examples.contrastive_flow_photoz_proxy_comparison import (
    ContrastivePhotoZProxyConfig,
    main,
)

main(
    ContrastivePhotoZProxyConfig(
        force_simulated=True,
        n_train_experiments=256,
        n_test_experiments=80,
        epochs=20,
    )
)
```

## Caveats

!!! warning
    This is a **proxy benchmark**, not a scientifically validated photo-z parameter-inference
    workflow. The catalog summaries and global-shift parameters are synthetic abstractions built
    on top of the photo-z covariates.

!!! tip
    The benchmark is still useful as a realism check: if `ContrastiveFlow` only helps on the toy
    synthetic benchmark but not here, the method may be too brittle for adoption.

!!! info
    In the current smoke benchmark, the plain `NormalizingFlow` baseline is stronger than
    `ContrastiveFlow`. That is still a useful outcome: this example is meant to test whether the
    contrastive objective transfers beyond the toy synthetic setup, not to guarantee a win.

## Related Pages

- See [Photo-z Benchmark Comparison](photoz_benchmark_comparison.md) for the underlying
  per-object regression benchmark path.
- See [Contrastive Flow Parameter Estimation Comparison](contrastive_flow_parameter_estimation_comparison.md)
  for the synthetic shared-budget benchmark.
