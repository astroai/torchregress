# Decision Metrics

Decision metrics quantify how a model's **uncertainty scores translate
into selective-prediction decisions** — i.e. "should we trust this
prediction, defer it, or reject it?". They are the *downstream
contract* of any uncertainty-aware model: a model with perfect
calibration but uninformative uncertainty is worse than one with
slightly worse calibration but uncertainty that ranks errors correctly.

The two primitives in this module are

- [RiskCoverageCurve](../api/metrics.md#riskcoveragecurve) —
  the full risk-vs-coverage trajectory, with the **Area Under the
  Risk-Coverage Curve** (AURC) as a scalar summary.
- [RejectionPolicy](../api/metrics.md#rejectionpolicy) —
  the per-decision evaluation at a fixed **threshold** or
  **rejection fraction**.

Both consume three tensors: predictions, targets, and an
uncertainty score (higher = less certain). The risk function is
pluggable; the default is squared error.

---

## 1. Why decision metrics?

A model can pass every calibration diagnostic (reliability diagram
flat, ECE near zero, PICP matches the nominal level) and still be
useless for selective prediction if its **uncertainty is not
rank-correlated with the realised error**. The risk-coverage curve
makes this rank-correlation visible.

A well-designed uncertainty score should produce a curve where
**risk decreases monotonically as coverage decreases** — i.e. the
most-uncertain predictions are exactly the worst ones. A flat curve
means the uncertainty score carries no information about which
predictions to trust.

Formally, let $r_i = \rho(\hat y_i, y_i)$ be the per-sample risk and
$u_i$ the per-sample uncertainty. The risk-coverage curve at
coverage $c \in (0, 1]$ is

$$
R(c) = \frac{1}{\lceil c \cdot n \rceil} \sum_{i \in S(c)} r_i
$$

where $S(c)$ is the set of $\lceil c \cdot n \rceil$ indices with the
smallest uncertainty. The **AURC** is the area under $R(c)$:

$$
\text{AURC} = \int_0^1 R(c) \, dc
$$

A **random** (uninformative) uncertainty score produces a flat curve
$R(c) \approx \mathbb{E}[r]$ for all $c$, yielding
$\text{AURC} \approx \mathbb{E}[r]$. An **oracle** uncertainty score
that ranks by the true per-sample error achieves the minimum
possible AURC, strictly less than $\mathbb{E}[r]$ when errors vary.
The **excess AURC** (E-AURC) = $\text{AURC}_{\text{model}} - \text{AURC}_{\text{oracle}}$
is the canonical headline number — lower is better.

---

## 2. `RiskCoverageCurve`

### 2.1 Class API

```python
from torchregress.metrics import RiskCoverageCurve

rcc = RiskCoverageCurve(n_points=100)
rcc.update(y_pred, y_true, uncertainty)
curve = rcc.compute()
# curve["coverage"]  — Tensor of coverage levels in (0, 1]
# curve["risk"]      — Tensor of mean risk at each coverage level
# curve["aurc"]      — Scalar AURC
```

The stateful class follows the `torchmetrics` convention: call
`update` for each batch, `compute` at the end of the epoch, and
`reset` between epochs. `RiskCoverageCurve` uses
`full_state_update=False`, so `update` is O(batch) and `compute` is
O(n) where n is the total number of samples accumulated.

### 2.2 Functional API

```python
from torchregress.metrics.decision import risk_coverage_curve

curve = risk_coverage_curve(
    y_pred, y_true, uncertainty,
    risk_fn=lambda p, t: (p - t) ** 2,  # default
    n_points=100,
)
```

### 2.3 Custom risk functions

Any callable that takes `(y_pred, y_true)` and returns per-sample
risk (a tensor of shape `[N]`) is accepted:

```python
import torch

# Absolute error
rcc = RiskCoverageCurve(risk_fn=lambda p, t: (p - t).abs())

# Huber-like: L1 outside a delta, L2 inside
def huber(p, t, delta=1.0):
    r = (p - t).abs()
    return torch.where(r <= delta, 0.5 * r ** 2, delta * r - 0.5 * delta ** 2)

rcc = RiskCoverageCurve(risk_fn=huber)
```

For multivariate targets, the risk function should reduce the
per-dimension risk to a scalar (e.g. take the mean across
dimensions) — the curve machinery then operates on per-sample
scalars.

### 2.4 Reading the curve

| Property | Healthy value | Symptom of bad uncertainty |
|:---------|:--------------|:---------------------------|
| $R(1.0)$ (full coverage) | mean risk | n/a (this is the reference) |
| $R(0.5)$ | $\le R(1.0)$ | flat curve = uncertainty ranks errors randomly |
| $R(0.0)$ (limit) | min risk | n/a (this is the best-case risk) |
| AURC | $\le \mathbb{E}[r]$ by construction | depends on the gap to oracle |
| Monotonicity | strictly decreasing (or non-increasing) | any non-monotonicity is a sign of an inconsistent uncertainty score |

The **AURC gap** to the oracle is the single most informative
summary for a research write-up. Compute it by sorting samples by
their *true* error and evaluating the curve as if that ordering
were the uncertainty ranking.

---

## 3. `RejectionPolicy`

The `RejectionPolicy` metric evaluates performance at a single
operating point: a fixed **uncertainty threshold** or a fixed
**rejection fraction**.

### 3.1 Class API

```python
from torchregress.metrics import RejectionPolicy

# Reject the top-10% most uncertain samples
policy = RejectionPolicy(fraction=0.1)
policy.update(y_pred, y_true, uncertainty)
result = policy.compute()
# result["mean_risk"]  — risk on the retained samples
# result["coverage"]   — fraction of samples retained
# result["n_rejected"] — count of rejected samples
```

### 3.2 Threshold-based rejection

```python
# Reject any sample with uncertainty > 0.5
policy = RejectionPolicy(threshold=0.5)
```

`fraction` and `threshold` are mutually exclusive; if both are
provided, `fraction` takes precedence.

### 3.3 Functional vs. class

The class API accumulates state across batches; the threshold is
fixed at construction time. If you need to sweep over multiple
thresholds, either:

- Construct multiple `RejectionPolicy` instances and call
  `update` on each per batch (memory cost: linear in the number of
  thresholds), or
- Use `RiskCoverageCurve` to compute the full curve and read off
  any operating point post hoc.

### 3.4 Choosing an operating point

The choice of `threshold` or `fraction` is a **business decision**,
not a model-evaluation decision. Common framings:

- **Fixed budget:** "I can defer at most 10% of cases to a human."
  Set `fraction=0.1` and report the mean risk on the remaining
  90%.
- **Quality target:** "I need mean risk $\le \tau$ on retained
  cases." Sweep `fraction` until `RejectionPolicy.compute()["mean_risk"]
  \le \tau`.
- **OOD screening:** "Flag any sample with uncertainty above the
  99th percentile of a clean calibration set." Set `threshold`
  to the empirical 99th percentile.

---

## 4. Worked example: ranking uncertainty for selective prediction

```python
import torch
from torchregress.metrics import RiskCoverageCurve, RejectionPolicy

# 1. Trained model produces mean and per-sample uncertainty
y_pred = model(x_test)                  # [N]
uncertainty = (y_pred - y_true).abs()   # placeholder: real uncertainty from the model

# 2. Full risk-coverage curve
rcc = RiskCoverageCurve(n_points=50)
rcc.update(y_pred, y_true, uncertainty)
curve = rcc.compute()
print(f"AURC: {curve['aurc']:.4f}")
# → AURC: 0.0432

# 3. Operating point: reject the 10% most uncertain
policy = RejectionPolicy(fraction=0.1)
policy.update(y_pred, y_true, uncertainty)
op = policy.compute()
print(f"Coverage: {op['coverage']:.2f}, mean risk: {op['mean_risk']:.4f}")
# → Coverage: 0.90, mean risk: 0.0187
```

In a research write-up, pair the AURC with at least two operating
points (e.g. reject 5% and 20%) to characterise the risk-coverage
tradeoff in the deployment-relevant regime.

---

## 5. Common pitfalls

- **Confusing calibration with selectivity.** A model with
  $\text{ECE} = 0$ (perfect calibration) can still have a flat
  risk-coverage curve. The two are independent contracts.
- **Risk function mismatch.** If the deployment risk is MAE but the
  curve is computed with squared error, the AURC optimises the
  wrong objective. Use the same risk function for evaluation and
  the training loss (or a *proper* alternative if the deployment
  risk is non-proper).
- **Threshold leakage.** If `threshold` is fit on the test set, the
  reported coverage is optimistic. Always derive the threshold from
  a held-out calibration split.
- **Tiny `n_points`.** With `n_points=2` the curve is uninformative.
  Use at least 50–100 points; use `n_points=n` (one per sample) for
  the most detailed curve at the cost of compute.
- **Multivariate targets.** The risk function must reduce to a
  per-sample scalar before being passed to the metric. The
  default `(p - t) ** 2` does *not* reduce — pass
  `lambda p, t: ((p - t) ** 2).mean(dim=-1)` for multivariate
  inputs.

---

## 6. Decision rules

| If you need… | Use | Don't |
|:-------------|:----|:------|
| Full risk-vs-coverage trajectory + scalar summary | `RiskCoverageCurve` (or `risk_coverage_curve` functional) | A single threshold-based metric alone |
| AURC for a research write-up | `curve["aurc"]` from `RiskCoverageCurve.compute()` | RMSE (does not capture selectivity) |
| Performance at a single operating point | `RejectionPolicy(fraction=...)` or `RejectionPolicy(threshold=...)` | Manual masking + RMSE |
| Sweep over many operating points | `RiskCoverageCurve` (read off any point) | Multiple `RejectionPolicy` instances |
| OOD screening with a fixed threshold | `RejectionPolicy(threshold=...)` | A `RiskCoverageCurve` (more expensive, no single answer) |
| Per-epoch accumulation across batches | the class API (`update` / `compute` / `reset`) | The functional API (no accumulation) |

---

## References

| # | Reference |
|:-:|:----------|
| 1 | El-Yaniv & Wiener. ["On the Foundations of Noise-free Selective Classification"](https://www.cs.technion.ac.il/~ran/papers/Selective-JAIR-2010.pdf) *JAIR*, 2010. |
| 2 | Geifman & El-Yaniv. ["Selective Classification for Deep Neural Networks"](https://arxiv.org/abs/1705.08500) *NeurIPS*, 2017. |
| 3 | Lakshminarayanan, Pritzel & Blundell. ["Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"](https://arxiv.org/abs/1612.01474) *NeurIPS*, 2017. |
| 4 | Gneiting & Raftery. ["Strictly Proper Scoring Rules, Prediction, and Estimation"](https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf) *JASA*, 2007. |
| 5 | Hendrickx, Perini, Van der Plas, Meert & Davis. ["Machine Learning with a Reject Option: A survey"](https://arxiv.org/abs/2107.00277) *ArXiv*, 2021. |

---

## Next steps

- [Uncertainty Decomposition](../guide/uncertainty-decomposition.md) — full taxonomy of the four uncertainty contracts.
- [Choosing by Constraint](../guide/choosing-by-constraint.md) — when to escalate to selective prediction.
- [Method Selection Matrix](../guide/method-selection.md) — task-first capability matrix.
- [Visualization Methods](../methods/visualization.md) — risk-coverage and selective-prediction plots.
