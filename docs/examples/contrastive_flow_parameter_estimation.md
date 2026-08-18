# Contrastive Flow Parameter Estimation

This example shows how to turn the flow machinery in `torchregress` into a lightweight
parameter-estimation workflow for nuisance-shifted data.

→ API: [`ContrastiveFlowLoss`](../api/losses.md), [`NormalizingFlowLoss`](../api/losses.md).

## When To Use This

Use this pattern when you have:

- a low-dimensional parameter of interest to estimate
- nuisance or domain-shift variables that distort the observed distribution
- simulation or augmentation that can generate data under alternate parameters
- a downstream need for parameter ranking or scanning, not just point prediction

Use plain [Normalizing Flows](normalizing_flows_multitarget.md) when you only need conditional
density estimation. Use `ContrastiveFlowLoss` when the main objective is to separate the correct
parameter setting from nearby alternatives.

Do **not** expect the contrastive objective to beat plain flow NLL by default. It is most plausible
when the deployment metric is hypothesis ordering and you can train with realistic alternate
parameter settings. If the real need is calibrated density estimation, ordinary flow NLL should
remain the default baseline.

## Objective

The example uses `ContrastiveFlowLoss`, which compares a pseudoexperiment summary `s` under:

- the positive context `c^+` produced from the generating parameters
- one or more negative contexts `c^-_k` produced from alternative parameters

The loss is:

$$
\mathcal{L}(s, c^+, \{c^-_k\}) =
-\log
\frac{\exp((\log p(s \mid c^+) - m)/T)}
{\exp((\log p(s \mid c^+) - m)/T) + \sum_k \exp(\log p(s \mid c^-_k)/T)}
$$

where $T$ is a temperature and $m$ is an optional positive-class margin.

## Comparison Table

| Approach | Best for | Weak point |
|:--|:--|:--|
| `NormalizingFlowLoss` | Conditional density estimation | Does not explicitly rank correct vs alternate parameters |
| `ContrastiveFlowLoss` | Parameter estimation with simulated alternate hypotheses | Requires negative contexts during training |
| `QuantileLoss` / conformal methods | Interval prediction on scalar targets | Not a natural fit for likelihood-style parameter scans |

## Runnable Example

```python
from examples.contrastive_flow_parameter_estimation import ContrastiveFlowConfig, main

metrics = main(
    ContrastiveFlowConfig(
        n_train=512,
        n_test=48,
        n_epochs=40,
        n_negatives=4,
        make_plot=True,
    )
)
print(metrics)
```

The full script is [`examples/contrastive_flow_parameter_estimation.py`](https://github.com/astroai/torchregress/blob/main/examples/contrastive_flow_parameter_estimation.py).

## What The Example Does

1. Simulate nuisance-shifted pseudoexperiments under a signal-strength parameter and a nuisance term.
2. Collapse each pseudoexperiment into a fixed summary vector containing moments, quantiles, and tail rates.
3. Train a parameter-to-context network together with a conditional flow using `ContrastiveFlowLoss`.
4. Recover parameters on held-out pseudoexperiments by scanning the learned flow likelihood over a small grid.

Negative hypotheses in this example are passed explicitly with shape
`[batch, n_negatives, context_dim]`, which avoids the ambiguous 2-D case where the number of
negative hypotheses accidentally matches the minibatch size.

## Practical Notes

!!! tip
    Keep the parameter scan small and interpretable at first. This pattern is strongest when the
    parameter grid is low-dimensional and simulation under alternate settings is cheap.

!!! warning
    The scan in this example is a simple brute-force grid, not a production-grade profile-likelihood
    backend. For non-toy applications, validate grid resolution, identifiability, and calibration
    before interpreting intervals.

!!! info
    This example is intentionally more useful than HEP-specific: the same pattern applies to domain
    shift, simulator calibration, nuisance-aware causal sensitivity analysis, and other settings
    where one wants to rank structured hypotheses with a learned density model.

!!! info
    The expected win condition here is narrow: `ContrastiveFlowLoss` should help when the model must
    rank the generating hypothesis above nearby alternatives. It is not expected to dominate plain
    flow NLL on every dataset or every summary representation.

## Related Pages

- See [Normalizing Flows](../losses/nflows.md) for the base conditional flow API.
- See [Task-First Method Selection Matrix](../guide/method-selection.md) for when to choose
  flow-based methods over Gaussian, MDN, or conformal alternatives.
