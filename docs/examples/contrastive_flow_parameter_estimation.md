# Contrastive Flow Parameter Estimation

This example shows how to turn the flow machinery in `torchregress` into a lightweight
parameter-estimation workflow for nuisance-shifted data.

→ API: [`ContrastiveFlowLoss`](../api/losses.md), [`NormalizingFlowLoss`](../api/losses.md),
[`NeuralRatioLoss`](../api/losses.md), [`SBC/TARP`](../api/metrics.md).

**Run the script** [`examples/contrastive_flow_parameter_estimation.py`](https://github.com/astroai/torchregress/blob/main/examples/contrastive_flow_parameter_estimation.py)
(needs `zuko`; `--mode all` by default runs both modes):

```bash
pixi run python examples/contrastive_flow_parameter_estimation.py --mode demo     # contrastive flow + likelihood scan
pixi run python examples/contrastive_flow_parameter_estimation.py --mode compare \
    --summary-json-path contrastive_flow_summary.json                            # shared-budget baselines
```

| Mode | Function | What it shows |
|:--|:--|:--|
| `demo` | `run_demo(ContrastiveFlowConfig(...))` | One `ContrastiveFlowLoss` model, held-out scan MAE, likelihood-scan figure (`--no-plot` to skip) |
| `compare` | `run_comparison(ContrastiveFlowComparisonConfig(...), summary_json_path=...)` | Gaussian summary density, flow NLL, contrastive flow, NRE and NPE on the same pseudoexperiments, with SBC/TARP diagnostics |

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

## Runnable Example (`--mode demo`)

```python
from examples.contrastive_flow_parameter_estimation import ContrastiveFlowConfig, run_demo

metrics = run_demo(
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

This is `--mode demo` of [`examples/contrastive_flow_parameter_estimation.py`](https://github.com/astroai/torchregress/blob/main/examples/contrastive_flow_parameter_estimation.py).

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

## See also

- [Comparison against baselines](#comparison-against-baselines) (`--mode compare`) — the same task with NRE, NPE and SBC/TARP baselines.
- [Neural Ratio Estimation](../losses/nre.md) — classifier-based likelihood-ratio estimator.

---

## Comparison Against Baselines

`--mode compare` turns the single-method demo above into a decision artifact. It evaluates five approaches on the same synthetic nuisance-shifted pseudoexperiment task:

- `GaussianSummary`: diagonal Gaussian density over summary vectors
- `NormalizingFlow`: conditional flow trained with plain NLL (forward density, grid scan)
- `ContrastiveFlow`: conditional flow trained with the InfoNCE/contrastive ratio objective
- `NeuralRatio`: classifier-based neural ratio estimation (NRE, joint-vs-product pairs)
- `NormalizingFlowPosterior`: NPE-style conditional flow $p(\theta\mid x)$ trained with NLL,
  evaluated from posterior samples instead of a grid scan

### What It Answers

Use this example when you want to know whether the contrastive objective adds value beyond
ordinary conditional density estimation on the same parameter-estimation workload.

### Shared-Budget Setup

- same pseudoexperiment generator
- same parameter scan grid at evaluation time
- same train/test split and fixed seed
- same MLP backbone width and comparable flow context size

### Metrics

| Metric | Meaning |
|:--|:--|
| `ParamMAE` | Mean absolute error across all recovered parameters |
| `Dim0_MAE` | Mean absolute error for the signal-strength parameter |
| `Dim1_MAE` | Mean absolute error for the nuisance parameter |
| `SBC_min_p` | Smallest per-dimension simulation-based-calibration uniformity p-value (higher is better; stochastic) |
| `TARP_err` | TARP coverage error $\max_\alpha\lvert\widehat{\text{cov}}(\alpha)-\alpha\rvert$ (lower is better) |
| `train_s`, `eval_s` | Runtime for training and evaluation (grid scan, or posterior sampling for NPE) |

### Runnable Comparison (`--mode compare`)

```python
from examples.contrastive_flow_parameter_estimation import (
    ContrastiveFlowComparisonConfig,
    run_comparison,
)

rows = run_comparison(
    ContrastiveFlowComparisonConfig(
        n_train=512,
        n_test=96,
        epochs=30,
        n_negatives=4,
    )
)
```

### Calibration diagnostics

For the grid-scan methods, the scan is turned into a posterior by
`torchregress.inference.posterior_log_weights` and resampled on the grid (uniform grid prior);
for `NormalizingFlowPosterior`, posterior samples are drawn directly from the flow. Both are then
scored with `torchregress.metrics.simulation_based_calibration_ranks` /
`sbc_uniformity_pvalue` and `tarp_coverage` (see
[SBC and TARP](../metrics/calibration.md#simulation-based-calibration-sbc-and-tarp)).
With only 96 test pseudoexperiments the SBC p-values are noisy; treat them as diagnostics, not
decisions.

### Observed results (single seed, synthetic task)

These are observed on the repository defaults (`n_train=512`, `n_test=96`, `batch_size=64`,
`lr=2e-3`); they are a snapshot, not a general ranking.

**30 epochs**

| Method | ParamMAE | SBC_min_p | TARP_err | train_s | eval_s |
|:--|--:|--:|--:|--:|--:|
| `GaussianSummary` | 0.1202 | 0.054 | 0.139 | 0.57 | 0.72 |
| `NormalizingFlow` | 0.1050 | 0.113 | 0.082 | 1.16 | 1.04 |
| `ContrastiveFlow` | 0.1420 | 0.024 | 0.097 | 3.15 | 0.99 |
| `NeuralRatio` | 0.2180 | 0.214 | 0.074 | 0.19 | 0.04 |
| `NormalizingFlowPosterior` | 0.1224 | 0.000 | 0.115 | 0.95 | 0.09 |

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

### Interpretation

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

### Comparison Related Pages

- See [Neural Ratio Estimation](../losses/nre.md) for the classifier NRE loss and negative-sampling conventions.
- See [Simulation-based calibration (SBC) and TARP](../metrics/calibration.md#simulation-based-calibration-sbc-and-tarp)
  for the calibration diagnostics used above.
- See [Normalizing Flows](../losses/nflows.md) for the base flow APIs.
