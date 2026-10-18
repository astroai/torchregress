# Contrastive Flow Parameter Estimation Comparison

This comparison turns the single-method contrastive-flow demo into a decision artifact.

→ API: [`ContrastiveFlowLoss`](../api/losses.md), [`NeuralRatioLoss`](../api/losses.md),
[`NormalizingFlowLoss`](../api/losses.md), [`SBC/TARP`](../api/metrics.md).
It evaluates five approaches on the same synthetic nuisance-shifted pseudoexperiment task:

- `GaussianSummary`: diagonal Gaussian density over summary vectors
- `NormalizingFlow`: conditional flow trained with plain NLL (forward density, grid scan)
- `ContrastiveFlow`: conditional flow trained with the InfoNCE/contrastive ratio objective
- `NeuralRatio`: classifier-based neural ratio estimation (NRE, joint-vs-product pairs)
- `NormalizingFlowPosterior`: NPE-style conditional flow $p(\theta\mid x)$ trained with NLL,
  evaluated from posterior samples instead of a grid scan

## What It Answers

Use this example when you want to know whether the contrastive objective adds value beyond
ordinary conditional density estimation on the same parameter-estimation workload.

## Shared-Budget Setup

- same pseudoexperiment generator
- same parameter scan grid at evaluation time
- same train/test split and fixed seed
- same MLP backbone width and comparable flow context size

## Metrics

| Metric | Meaning |
|:--|:--|
| `ParamMAE` | Mean absolute error across all recovered parameters |
| `Dim0_MAE` | Mean absolute error for the signal-strength parameter |
| `Dim1_MAE` | Mean absolute error for the nuisance parameter |
| `SBC_min_p` | Smallest per-dimension simulation-based-calibration uniformity p-value (higher is better; stochastic) |
| `TARP_err` | TARP coverage error $\max_\alpha\lvert\widehat{\text{cov}}(\alpha)-\alpha\rvert$ (lower is better) |
| `train_s`, `eval_s` | Runtime for training and evaluation (grid scan, or posterior sampling for NPE) |

## Runnable Example

```python
from examples.contrastive_flow_parameter_estimation_comparison import (
    ContrastiveFlowComparisonConfig,
    main,
)

main(
    ContrastiveFlowComparisonConfig(
        n_train=512,
        n_test=96,
        epochs=30,
        n_negatives=4,
    )
)
```

## Calibration diagnostics

For the grid-scan methods, the scan is turned into a posterior by
`torchregress.inference.posterior_log_weights` and resampled on the grid (uniform grid prior);
for `NormalizingFlowPosterior`, posterior samples are drawn directly from the flow. Both are then
scored with `torchregress.metrics.simulation_based_calibration_ranks` /
`sbc_uniformity_pvalue` and `tarp_coverage` (see
[SBC and TARP](../metrics/calibration.md#simulation-based-calibration-sbc-and-tarp)).
With only 96 test pseudoexperiments the SBC p-values are noisy; treat them as diagnostics, not
decisions.

## Observed results (single seed, synthetic task)

These are observed on the repository defaults (`n_train=512`, `n_test=96`, `batch_size=64`,
`lr=2e-3`); they are a snapshot, not a general ranking.

**30 epochs**

| Method | ParamMAE | SBC_min_p | TARP_err | train_s | eval_s |
|:--|--:|--:|--:|--:|--:|
| `GaussianSummary` | 0.1202 | 0.054 | 0.139 | 0.60 | 0.54 |
| `NormalizingFlow` | 0.1050 | 0.012 | 0.108 | 1.32 | 1.14 |
| `ContrastiveFlow` | 0.1510 | 0.021 | 0.075 | 3.55 | 1.14 |
| `NeuralRatio` | 0.2180 | 0.214 | 0.074 | 0.21 | 0.05 |
| `NormalizingFlowPosterior` | 0.1226 | 0.000 | 0.117 | 1.13 | 0.08 |

**100 epochs**

| Method | ParamMAE | SBC_min_p | TARP_err | train_s | eval_s |
|:--|--:|--:|--:|--:|--:|
| `GaussianSummary` | 0.1226 | 0.006 | 0.149 | 0.85 | 0.55 |
| `NormalizingFlow` | 0.1037 | 0.067 | 0.131 | 4.36 | 1.13 |
| `ContrastiveFlow` | 0.1349 | 0.000 | 0.115 | 11.15 | 1.12 |
| `NeuralRatio` | 0.1304 | 0.125 | 0.073 | 0.66 | 0.05 |
| `NormalizingFlowPosterior` | **0.0842** | 0.166 | 0.112 | 3.62 | 0.08 |

At this small simulation budget (512 training pseudoexperiments) NRE is **data-hungry**: it is by far
the cheapest to train but only approaches the density baselines after more epochs, and this task's
8-dimensional hand-engineered summary favours explicit density models. `NormalizingFlowPosterior`
(posterior NPE) gives the best point accuracy at the longer budget while avoiding a grid scan. The
contrastive flow is the slowest and does not beat plain NLL here; its niche remains genuine
hypothesis ranking/reference-distribution workflows, not generic density estimation.

## Interpretation

No method is uniformly best. Prefer a method only if it improves parameter recovery (and calibration
where relevant) on the same budget, not because it is more specialized. The Gaussian row remains
important: if it is already competitive, the summary representation may be too simple to justify a
heavier method. The NPE row is the natural baseline when posterior samples are the deliverable;
NRE is cheapest but needs enough simulations to train the classifier.

!!! info
    Treat this example as an honest comparison harness, not a showcase. Plain NLL and NPE-style
    posterior flows are often stronger than the contrastive objective on generic density tasks;
    the contrastive/NRE objectives target likelihood-ratio ranking with informative negatives.

!!! tip
    The comparison is most favorable to `ContrastiveFlow` when the task really is hypothesis
    ranking, the negative hypotheses are informative rather than random, and the summary features
    preserve the structure needed for likelihood-ratio discrimination.

## Related Pages

- See [Contrastive Flow Parameter Estimation](contrastive_flow_parameter_estimation.md) for the
  single-method walkthrough.
- See [Neural Ratio Estimation](../losses/nre.md) for the classifier NRE loss and negative-sampling conventions.
- See [Simulation-based calibration (SBC) and TARP](../metrics/calibration.md#simulation-based-calibration-sbc-and-tarp)
  for the calibration diagnostics used above.
- See [Normalizing Flows](../losses/nflows.md) for the base flow APIs.
