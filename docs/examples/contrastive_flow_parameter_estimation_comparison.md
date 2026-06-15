# Contrastive Flow Parameter Estimation Comparison

This comparison turns the single-method contrastive-flow demo into a decision artifact.

→ API: [`ContrastiveFlowLoss`](../api/losses.md#contrastiveflowloss), [`NormalizingFlowLoss`](../api/losses.md#normalizingflowloss).
It evaluates three approaches on the same synthetic nuisance-shifted pseudoexperiment task:

- `GaussianSummary`: diagonal Gaussian density over summary vectors
- `NormalizingFlow`: conditional flow trained with plain NLL
- `ContrastiveFlow`: conditional flow trained with the contrastive likelihood-ratio objective

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
| `train_s`, `eval_s` | Runtime for training and grid-scan evaluation |

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

## Interpretation

`ContrastiveFlow` should only be preferred if it improves parameter recovery on the same scan grid,
not merely because it is more specialized. The Gaussian row remains important: if it is already
competitive, then the summary representation may be too simple to justify a heavier method.

!!! info
    On the repository's smoke profile, `NormalizingFlow` is often stronger than
    `ContrastiveFlow`. Treat this example as an honest comparison harness, not a showcase where
    the contrastive objective is assumed to win by construction.

!!! tip
    The comparison is most favorable to `ContrastiveFlow` when the task really is hypothesis
    ranking, the negative hypotheses are informative rather than random, and the summary features
    preserve the structure needed for likelihood-ratio discrimination.

## Related Pages

- See [Contrastive Flow Parameter Estimation](contrastive_flow_parameter_estimation.md) for the
  single-method walkthrough.
- See [Normalizing Flows](../losses/nflows.md) for the base flow APIs.
