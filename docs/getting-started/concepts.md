# Core Concepts

> ← [Quick Start](quickstart.md) | [User Guide](../guide/index.md) →

This page formalises the vocabulary used throughout torchregress. The
treatment is intentionally compact; for derivations see
[Mathematical Foundations](../guide/math/index.md), and for method-by-method
guidance see the [Method Selection Matrix](../guide/method-selection.md).

---

## 1. Regression as probabilistic inference

Given features $x \in \mathbb{R}^D$ and a continuous target $y \in \mathbb{R}$, the goal is to model the **conditional distribution**
$p(y \mid x)$, not a single point $\hat y$. A point estimate is then
recovered as an aspect of that distribution:

$$
\hat y_{\text{central}} = \mathbb{E}[y \mid x], \qquad
\hat y_{\tau} = F^{-1}_{y \mid x}(\tau), \qquad
\hat I_{1-\alpha} = \bigl[F^{-1}_{y \mid x}(\alpha/2),\; F^{-1}_{y \mid x}(1-\alpha/2)\bigr]
$$

for central tendency, $\tau$-quantile, and a $1-\alpha$ prediction
interval respectively.

torchregress losses are organised around this view: a `DistributionLoss`
outputs the parameters of a parametric family (or samples from a
non-parametric head), and a `RegressionLoss` is the special case of a
distribution with fixed $\sigma^2$.

---

## 2. Uncertainty decomposition

Following the [Law of Total Variance](../methods/ensemble/index.md#uncertainty-decomposition)
and Kendall & Gal (2017), we decompose predictive uncertainty into two
fundamentally different sources:

### 2.1 Aleatoric uncertainty (data noise)

The **irreducible** noise in the observation process.

- **Homoscedastic** — $\operatorname{Var}(y \mid x) = \sigma^2$ for all $x$.
- **Heteroscedastic** — $\operatorname{Var}(y \mid x) = \sigma^2(x)$ depends on $x$.
- Cannot be reduced by collecting more data from the same distribution.

Modelled by parametric likelihood heads such as `GaussianNLLLoss`,
`BetaNLLLoss`, `MultivariateGaussianLoss`, and non-parametric heads such as
`MDNLoss` and `NormalizingFlowLoss`.

### 2.2 Epistemic uncertainty (model ignorance)

The **reducible** uncertainty due to finite training data or limited model
capacity in some region of input space.

- High in **out-of-distribution (OOD)** regions where the model has not seen
  similar features.
- High when **multiple parameter settings** are consistent with the training
  data (posterior variance in Bayesian models, member disagreement in
  ensembles, weight-space variance in SWAG).

Modelled by **ensembles of likelihood heads** (`DeepEnsemble`,
`HeteroscedasticEnsembleModel`, `MDNEnsembleModel`), **Bayesian
approximations** (`BayesianNeuralNetwork`, `SWAG`, `MultiSWAG`,
`HeteroscedasticLaplaceRegressor`, `IVON`), or **evidential single-pass
priors** (`EvidentialRegressionLoss`).

### 2.3 Why decomposition matters

Conformal prediction provides **interval coverage guarantees** but does
*not* separate aleatoric and epistemic uncertainty. For risk-aware
decisions (e.g. defer to human review when epistemic uncertainty is high
but accept the prediction when only aleatoric noise is high), the
decomposition is essential. See
[Uncertainty Decomposition](../guide/uncertainty-decomposition.md) for the
full taxonomy and the contracts used in torchregress.

---

## 3. Proper scoring rules

A scoring rule $S(F, y)$ is **proper** if
$\mathbb{E}_{y \sim G}\bigl[S(G, y)\bigr] \leq \mathbb{E}_{y \sim G}\bigl[S(F, y)\bigr]$
for all $F \neq G$ — i.e. it is uniquely minimised in expectation when the
predicted distribution $F$ matches the true distribution $G$.

| Scoring rule | Definition | Use |
|:-------------|:-----------|:----|
| Negative log-likelihood (NLL) | $-\log p(y \mid x)$ | Training and evaluation; requires a density |
| Continuous Ranked Probability Score (CRPS) | $\int_{-\infty}^{\infty} (F(z) - \mathbf{1}_{y \le z})^2 \, dz$ | Evaluation; does not require a closed-form density |
| Energy score (multivariate CRPS) | $\mathbb{E}\lVert X - Y \rVert - \tfrac{1}{2}\mathbb{E}\lVert X - X' \rVert$ | Multivariate evaluation |
| Interval score | $\text{width} + \tfrac{2}{\alpha}\bigl(\text{lower} - y\bigr)_+ + \tfrac{2}{\alpha}\bigl(y - \text{upper}\bigr)_+$ | Evaluation of prediction intervals |
| Brier score (probabilistic) | $\sum_k (f_k - \mathbf{1}_{y=k})^2$ | Probabilistic classification |

torchregress evaluates models with **proper scoring rules** as the
primary accuracy measure; point-error metrics (RMSE, MAE) are reported as
secondary summaries.

---

## 4. Robustness and influence functions

The **influence function** $\psi(r) = \frac{d}{dr}\rho(r)$ quantifies the
gradient contribution of a residual $r$. Standard MSE has $\psi(r) = 2r$ —
unbounded — so a single outlier can dominate the gradient. Robust losses bound
$\psi$ as follows:

- **Huber** — quadratic near zero, linear in the tails.
- **Cauchy** — $\psi(r) = 2r / (c^2 + r^2)$, logarithmic growth.
- **Tukey biweight** — $\psi(r) = 0$ for $|r| > c$ (hard rejection).
- **Barron** — continuous family parametrised by $\alpha \in (-\infty, 2]$
  interpolating between $L_1$ and $L_2$.
- **AdaptiveRobust** — jointly optimises the shape $\alpha$ alongside the
  model weights.

Use the [`Uncertainty vs. error` diagnostic](../methods/visualization.md) to
visualise the empirical relationship between predicted $\sigma$ and realised
absolute error; departures from $y = x$ reveal miscalibration or model
misspecification.

---

## 5. Calibration vs. sharpness

A good probabilistic model must balance two properties:

1. **Calibration** — the empirical coverage of predicted intervals matches
   their nominal level. The
   [reliability diagram](../methods/calibration.md) and the
   [probability integral transform (PIT)](../metrics/distribution.md) are the
   standard diagnostics.
2. **Sharpness** — among all calibrated models, the one with the **narrowest**
   intervals is preferred (Gneiting & Raftery, 2007).

CRPS balances both. The
[`expected_calibration_error`](../metrics/calibration.md) and
[`marginal_calibration_error`](../metrics/calibration.md) decompose the
miscalibration component.

---

## 6. Conformal prediction

[Conformal prediction](../methods/conformal/index.md) is a **post-hoc
framework** that converts any pre-trained model into one with
distribution-free coverage guarantees under exchangeability.

Given a non-conformity score $s(x, y) = |y - \hat y|$ (or a learned
score for quantile-based methods), the conformal interval at level
$1 - \alpha$ is

$$
\hat C_{1-\alpha}(X_{n+1}) = \bigl[\hat y_{n+1} - q_{1-\alpha},\; \hat y_{n+1} + q_{1-\alpha}\bigr]
$$

where $q_{1-\alpha}$ is the $\lceil (n+1)(1-\alpha) \rceil / n$ empirical
quantile of calibration scores. Under exchangeability this guarantees
$P(Y_{n+1} \in \hat C_{1-\alpha}(X_{n+1})) \geq 1 - \alpha$.

**Important:** conformal prediction provides *coverage*, not density
estimation or uncertainty decomposition. It complements, but does not
replace, likelihood-based losses.

---

## 7. When to use which (decision rules)

| If you need… | Use | Don't use |
|:-------------|:----|:----------|
| Plain regression, no missing data | `torch.nn.MSELoss` (or [`WeightedMSELoss`](../api/losses.md) if you may need masks later) | A complex probabilistic model |
| Mask support or sample weights | [`WeightedMSELoss`](../api/losses.md) | `torch.nn.MSELoss` |
| Aleatoric uncertainty (per-sample σ) | [`GaussianNLLLoss`](../api/losses.md) with `learn_variance=True` | Point loss + post-hoc σ estimation |
| Epistemic uncertainty | Ensembles, SWAG, BNN, evidential | Single MC-dropout run with no diagnostics |
| Multimodal conditional | [`MDNLoss`](../api/losses.md) or [`NormalizingFlowLoss`](../api/losses.md) | A single Gaussian head |
| Coverage guarantees | [`SplitConformal`](../api/losses.md) / [`CQR`](../api/losses.md) on top of a strong backbone | Conformal *instead of* likelihood |
| Distribution shift at test time | `BayesianLinearHead` or `ShiftFactoredPredictiveTransport` | A model retrained on a single batch |
| Causal treatment effects | `dr_ate` / `dr_cate` | Naive difference in means |
| OOD detection | Ensemble + multiple OOD scores | A single OOD score in isolation |

---

## References

| # | Reference |
|:-:|:----------|
| 1 | Kendall & Gal. ["What Uncertainties Do We Need in Bayesian Deep Learning?"](https://arxiv.org/abs/1703.04977) *NeurIPS*, 2017. |
| 2 | Gneiting & Raftery. ["Strictly Proper Scoring Rules, Prediction, and Estimation"](https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf) *JASA*, 2007. |
| 3 | Abdar et al. ["A Review of Uncertainty Quantification in Deep Learning."](https://arxiv.org/abs/2011.06225) *Information Fusion*, 2021. |
| 4 | Vovk, Gammerman & Shafer. *Algorithmic Learning in a Random World*. Springer, 2005. |
| 5 | Romano, Patterson & Candès. ["Conformalized Quantile Regression"](https://arxiv.org/abs/1905.03222) *NeurIPS*, 2019. |
| 6 | Lakshminarayanan, Pritzel & Blundell. ["Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"](https://arxiv.org/abs/1612.01474) *NeurIPS*, 2017. |
| 7 | Madry et al. ["From Robustness to Conformal Prediction"](https://arxiv.org/abs/2202.04913) 2022. |

---

## Next steps

- [Mathematical Foundations](../guide/math/index.md) — derivations of NLL, CRPS, interval score, and decomposition formulas
- [Method Selection Matrix](../guide/method-selection.md) — task-first guidance with capability matrices
- [Quick Start](quickstart.md) — end-to-end workflows for the most common tasks
