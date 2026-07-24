# Uncertainty Decomposition

> ← [Choosing by Constraint](choosing-by-constraint.md) | [Guide Overview](index.md) →

Uncertainty language is easy to overstate. The vocabulary used in
torchregress follows a strict **four-contract taxonomy**, with each
contract backed by a different mathematical construction. This page
formalises the taxonomy, states the law that underpins the variance
decomposition, and gives method-by-method semantics so that you can
audit which contract a given API actually delivers.

The guiding principle: **the answer to "what kind of uncertainty does
this method give me?" determines which metric, which diagnostic, and
which downstream decision is appropriate.** Conflating the contracts
is the most common source of miscommunicated uncertainty claims.

---

## 1. The four contracts

| Contract | Definition | What it means | Typical tools |
|:---------|:-----------|:--------------|:--------------|
| **Predictive spread** | The predictive distribution or interval is wide | The model is "uncertain" in the sense of assigning probability mass to a range | [`GaussianNLLLoss`](../api/losses.md), [`MDNLoss`](../api/losses.md), [`NormalizingFlowLoss`](../api/losses.md), [`QuantileLoss`](../api/losses.md) |
| **Coverage guarantee** | A calibrated interval contains future labels at a target rate under exchangeability | The intervals have a *finite-sample* probability guarantee, not a Bayesian one | [`ConformalLoss`](../api/losses.md), [`CQR`](../api/losses.md) / [`UACQR`](../api/losses.md) |
| **Epistemic signal** | Different plausible models disagree | The model would say something different if retrained — a *model-ignorance* signal | `DeepEnsemble`, `PackedEnsembleRegressor`, `MCDropoutWrapper`, `SWAG`, `BayesianNeuralNetwork` |
| **Full variance decomposition** | Total variance is split into model disagreement and expected per-model noise | Aleatoric and epistemic components are reported separately and sum to the total | Heteroscedastic ensembles, heteroscedastic BNNs, ensembles of probabilistic heads |

The contracts are **distinct and not interchangeable**. A method that
satisfies the *predictive spread* contract does **not** satisfy the
*epistemic signal* contract. A method that satisfies the *coverage
guarantee* does **not** satisfy the *decomposition* contract.

---

## 2. The Law of Total Variance (the identity)

For a set of $M$ probabilistic predictors, each producing a
conditional distribution $p_m(y \mid x) = \mathcal{N}(\mu_m(x), \sigma_m^2(x))$ (the identity extends to non-Gaussian members with
the appropriate variance definitions):

$$
\underbrace{\operatorname{Var}_{\text{total}}\!\bigl[y \mid x\bigr]}_{\text{predictive spread}}
= \underbrace{\frac{1}{M} \sum_{m=1}^{M} \sigma_m^2(x)}_{\text{aleatoric}}
+ \underbrace{\frac{1}{M} \sum_{m=1}^{M} \bigl(\mu_m(x) - \bar\mu(x)\bigr)^2}_{\text{epistemic}},
$$

with $\bar\mu(x) = \frac{1}{M} \sum_m \mu_m(x)$.

This is a direct application of the **Law of Total Variance** to
the mixture distribution
$p(y \mid x) = \frac{1}{M} \sum_m p_m(y \mid x)$:

$$
\operatorname{Var}\!\bigl[y \mid x\bigr]
= \mathbb{E}_m\!\bigl[\operatorname{Var}\!\bigl(y \mid x, m\bigr)\bigr]
+ \operatorname{Var}_m\!\bigl(\mathbb{E}\!\bigl[y \mid x, m\bigr]\bigr)
$$

The first term is the **expected per-model noise** (aleatoric); the
second is the **variance of the per-model means** (epistemic).

In torchregress, compute the decomposition with
[`uncertainty_decomposition(means, variances)`](../api/metrics.md) (where `means` is
`[M, B]` and `variances` is `[M, B]`) or with
`ensemble_variance_decomposition(...)` for richer inputs (e.g.
multivariate).

### 2.1 The multivariate identity

For multi-target heads, the Law of Total Covariance gives

$$
\underbrace{\text{Cov}_{\text{total}}\!\bigl[\mathbf{y} \mid \mathbf{x}\bigr]}_{\text{predictive spread}}
= \underbrace{\frac{1}{M} \sum_{m=1}^{M} \mathbf{\Sigma}_m(\mathbf{x})}_{\text{aleatoric}}
+ \underbrace{\frac{1}{M} \sum_{m=1}^{M} (\boldsymbol\mu_m - \bar{\boldsymbol\mu})(\boldsymbol\mu_m - \bar{\boldsymbol\mu})^\top}_{\text{epistemic}},
$$

with $\bar{\boldsymbol\mu} = \frac{1}{M} \sum_m \boldsymbol\mu_m$.
The first term captures the expected per-member noise; the second
captures the disagreement of the per-member means.

---

## 3. The taxonomy: method-by-method semantics

The table below is the **canonical reference** for which uncertainty
contract each method delivers. Read it column-by-column: the
"Epistemic" column says whether the method *models* model ignorance
via disagreement or sampling; the "Aleatoric / spread" column says
whether the method models data noise or only the predictive band;
the "Decomposition status" column says whether the API returns a
*split* into the two components.

For constructor signatures and scoring-rule definitions, see the
[Losses API](../api/losses.md), [Metrics API](../api/metrics.md),
and [Ensemble API](../api/ensemble.md).

| Method / API | Epistemic | Aleatoric / spread | Decomposition status |
|:---|:---:|:---:|:---|
| `HeteroscedasticEnsembleModel`, `HeteroscedasticBatchEnsembleModel` | yes | yes | **Full** variance decomposition; returns `epistemic_variance`, `aleatoric_variance`, and total `variance`. |
| `HeteroscedasticBNN` | yes | yes | **Full** variance decomposition via `predict_with_decomposition()`. |
| `MDNEnsembleModel` | yes | yes | Ensemble disagreement plus mixture predictive spread; decomposition reported as a sum of two well-defined terms. |
| `DeepEnsemble` | yes | partial | **Full only if** members also predict variances or distributions. Plain point ensembles expose *epistemic disagreement* only. |
| `PackedEnsembleRegressor` | yes | partial | **Full** for heteroscedastic heads; homoscedastic heads expose no aleatoric component. |
| `BinnedPDFEnsembleModel`, `CumulativeLinkEnsembleModel` | yes | partial | Ensemble disagreement plus distributional / ordinal spread; decomposition is representation-specific (binned PDF entropy or cumulative-link logits). |
| `MCDropoutWrapper`, `SWAG`, `MultiSWAG`, `BayesianNeuralNetwork` | yes | partial | Weight / sample uncertainty is *epistemic*; *aleatoric* requires an explicit variance head or likelihood model. |
| `EvidentialRegressionLoss` | partial | yes | Analytic NIG-derived uncertainty; **validate calibration** before treating the epistemic term as model uncertainty. See [API](../api/losses.md). |
| [`MDNLoss`](../api/losses.md), [`NormalizingFlowLoss`](../api/losses.md), [`GaussianNLLLoss`](../api/losses.md), [`LowRankGaussianLoss`](../api/losses.md), [`MultivariateGaussianLoss`](../api/losses.md) | no | yes | Single-model predictive distributions model aleatoric or predictive spread, **not** epistemic uncertainty. |
| [`QuantileLoss`](../api/losses.md) | no | yes | Conditional quantile spread / intervals; **no epistemic signal** without an ensemble or sampling mechanism. |
| [`ConformalLoss`](../api/losses.md) and conformal predictors ([`SplitConformal`](../api/losses.md), [`CQR`](../api/losses.md), [`UACQR`](../api/losses.md), `DensityConformal`, `CTI`, `SLSConformal`, `MonteCarloConformal`, `LocalConformal`, `LocalConformalMAD`) | no | no | Coverage guarantees and calibrated intervals, **not** uncertainty decomposition. |

The "no" entries are not failures — they are scope statements. A
single-model `GaussianNLLLoss` is a perfectly correct *aleatoric*
model; it just doesn't claim anything about model ignorance.

---

## 4. Quantile ensembles (a special case)

Yes, you can deep-ensemble quantile regressors. An ensemble of
quantile heads gives useful *epistemic information* through
**disagreement among the predicted quantile functions**. For example,
if independently trained models disagree strongly about the
$0.9$ quantile, that is a model-uncertainty signal.

But this is **not** a clean variance decomposition. Quantile
regression predicts conditional quantiles, not per-member Gaussian
variances. You can:

- Summarise ensemble disagreement across quantiles (epistemic
  signal, well-defined).
- Estimate interval width from the quantile band (predictive
  spread, well-defined).
- **Not** call those two numbers `epistemic_variance` and
  `aleatoric_variance` without extra modelling choices.

Use this framing:

- **Single quantile model:** calibrated / non-Gaussian intervals,
  no epistemic split.
- **Quantile ensemble:** epistemic-style disagreement across
  quantile functions.
- **Quantile ensemble + conformal calibration:** stronger interval
  coverage story.
- **Quantile ensemble + explicit distributional / variance model:**
  only then consider formal variance decomposition.

---

## 5. What "epistemic" actually means

Three distinct mathematical objects share the *epistemic* label, and
the choice of object affects downstream decisions:

1. **Weight-space variance.** The posterior over weights
   $p(\boldsymbol\theta \mid \mathcal{D})$ induces a distribution
   over functions. Methods: `BayesianNeuralNetwork`, `IVON`,
   `HeteroscedasticLaplaceRegressor`. Cheapest in inference (one
   forward pass per weight sample); requires a weight-perturbation
   sampling step.
2. **Function-space disagreement.** A finite sample of models
   $\{f_m\}_{m=1}^M$ produces a sample of predictions. The
   empirical variance of the sample is the *epistemic* estimate.
   Methods: `DeepEnsemble`, `MCDropoutWrapper` (with $N$ dropout
   masks as the sample), `SWAG` (with $N$ weight samples).
3. **Single-pass prior.** A prior over the predictive distribution
   parameters (e.g. Normal-Inverse-Gamma) is updated to a posterior
   in one pass. Methods: `EvidentialRegressionLoss`. The
   "epistemic" component is the variance of the posterior
   over the *likelihood parameters*, not over the *weights* or the
   *predictions*.

These three objects are not interchangeable. A weight-space variance
is small when the posterior over weights is concentrated; a
function-space disagreement is small when the sampled models
agree on the prediction; a single-pass prior's epistemic term is
small when the posterior over likelihood parameters is
concentrated.

In all three cases, the **epistemic term should vanish** as the
training set size grows — but it does so at different rates, and
the calibration diagnostics should be run on each separately.

---

## 6. The catalog contract

When updating `method_catalog.py`, use conservative capability labels:

- `yes`: the component is explicitly modelled and returned, or
  directly computed by a tested helper.
- `partial`: the component exists only for a specific head,
  sampling mode, ensemble construction, or modelling assumption.
- `no`: the method may produce intervals or spread, but not that
  uncertainty component.

Avoid listing single-model MDN, flow, or quantile losses as
epistemic + aleatoric decomposition methods. List their *ensemble*
variants when the intended claim is model-disagreement plus
predictive spread.

The `method_catalog` API exposes `list_methods(capability_filters=...)`
and a per-row `task_tags` field. Use these for capability-flag-based
shortlisting:

```python
import torchregress as tr

# Methods that explicitly deliver the full variance decomposition
rows = tr.method_catalog.list_methods(
    capability_filters={"decomposition": "yes"},
    maturity=("Core", "Strong", "Available"),
)
for row in rows:
    print(row["name"], row["family"], row["maturity"], row["task_tags"])
```

---

## 7. Diagnostics by contract

| Contract | Primary diagnostic | Secondary diagnostic |
|:---------|:-------------------|:---------------------|
| Predictive spread | [reliability diagram](../methods/calibration.md), [PIT histogram](../metrics/distribution.md) | CRPS, energy score |
| Coverage guarantee | empirical PICP vs. nominal level | MPIW, coverage under shift |
| Epistemic signal | [risk-coverage curve](../metrics/decision.md), selective prediction accuracy | Spearman $\rho(\hat\sigma_{\text{epi}}, |\text{error}|)$ |
| Full decomposition | ratio $\sigma_{\text{epi}}^2 / \sigma_{\text{aleatoric}}^2$ on held-out and OOD sets | contribution to total predictive variance |

A **single** diagnostic is not enough to validate any contract.
Combine at least one primary and one secondary diagnostic from
above.

---

## 8. Decision rules

| If you need… | Use | Don't |
|:-------------|:----|:------|
| Predictive spread (a wide or narrow distribution) | [`GaussianNLLLoss`](../api/losses.md), [`MDNLoss`](../api/losses.md), [`NormalizingFlowLoss`](../api/losses.md) | A point loss + post-hoc $\sigma$ estimation |
| Coverage guarantee at level $1 - \alpha$ | [`SplitConformal`](../api/losses.md) / [`CQR`](../api/losses.md) / [`UACQR`](../api/losses.md) / `DensityConformal` | A likelihood head alone (no guarantee) |
| Epistemic signal (model disagreement) | `DeepEnsemble`, `MCDropoutWrapper`, `SWAG`, `IVON` | A single forward pass with no sampling |
| Full aleatoric + epistemic decomposition | `HeteroscedasticEnsembleModel`, `HeteroscedasticBNN`, `HeteroscedasticBatchEnsembleModel` | A plain `DeepEnsemble` of point heads |
| Single-pass decomposition | [`EvidentialRegressionLoss`](../api/losses.md) (with calibration checks) | A heavy ensemble for marginal benefit |
| Multivariate decomposition | Multivariate heteroscedastic ensemble | Stacking $K$ independent univariate decompositions |

---

## References

| # | Reference |
|:-:|:----------|
| 1 | Kendall & Gal. ["What Uncertainties Do We Need in Bayesian Deep Learning?"](https://arxiv.org/abs/1703.04977) *NeurIPS*, 2017. |
| 2 | Lakshminarayanan, Pritzel & Blundell. ["Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"](https://arxiv.org/abs/1612.01474) *NeurIPS*, 2017. |
| 3 | Gneiting & Raftery. ["Strictly Proper Scoring Rules, Prediction, and Estimation"](https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf) *JASA*, 2007. |
| 4 | Maddox, Garipov, Vetrov & Wilson. ["A Simple Baseline for Bayesian Uncertainty Estimation"](https://arxiv.org/abs/1902.02476) (SWAG) *NeurIPS*, 2019. |
| 5 | Sensoy, Kaplan & Kandemir. ["Evidential Deep Learning to Quantify Classification Uncertainty"](https://arxiv.org/abs/1806.01768) (NIG) *NeurIPS*, 2018. |
| 6 | Vovk, Gammerman & Shafer. *Algorithmic Learning in a Random World*. Springer, 2005. |
| 7 | Romano, Patterson & Candès. ["Conformalized Quantile Regression"](https://arxiv.org/abs/1905.03222) *NeurIPS*, 2019. |

---

## Next steps

- [Method Selection Matrix](method-selection.md) — task-first capability matrix.
- [Choosing by Constraint](choosing-by-constraint.md) — latency / coverage / decomposition tradeoffs.
- [Multi-Target Regression](multi-target-regression.md) — joint multi-target modelling and decomposition.
- [Practical Usage](practical-usage.md) — concrete loss recipes.
