# Losses API

Complete reference for the `torchregress.losses` module. This page is **standalone** — every class, factory, and helper is listed here. For background, derivations, and decision guides, see [Loss Functions](../losses/index.md).

---

## Module overview

`torchregress.losses` is organised into focused sub-modules by family. All losses share a unified contract:

```python
forward(y_pred, target, mask=None, weights=None) -> Tensor | dict
```

- **`mask`**: boolean tensor marking valid (True) vs. missing (False) targets.
- **`weights`**: per-sample weight tensor broadcastable to `target`.
- **Reduction**: every loss supports `mean` / `sum` / `none` via the `reduction` constructor argument (handled by [`BaseLoss`](#baseloss)).

---

## Base classes

| Symbol | Description |
|:-------|:------------|
| [`BaseLoss`](#baseloss) | Root class: reduction strategies, mask/weight support, common utilities |
| [`RegressionLoss`](#regressionloss) | Subclass for point-prediction losses (MSE, Huber, …) |
| [`DistributionLoss`](#distributionloss) | Subclass for distributional losses (Gaussian NLL, MDN, …) |

### Weighted wrappers (drop-in replacements for `torch.nn`)

| Symbol | PyTorch nearest | Why it exists |
|:-------|:----------------|:--------------|
| `WeightedMSELoss` | `nn.MSELoss` | Adds **mask** and per-sample **weight** support |
| `WeightedL1Loss` | `nn.L1Loss` | Same mask + weight pattern |
| `WeightedHuberLoss` | `nn.HuberLoss` | Same mask + weight pattern |
| `WeightedCrossEntropyLoss` | `nn.CrossEntropyLoss` | Mask + weight for regression-as-classification |
| `WeightedNLLLoss` | `nn.NLLLoss` | Mask + weight for regression-as-classification |
| [`WeightedLossWrapper`](#weightedlosswrapper) | any `nn.Module` | Generic wrapper adding mask/weight to any loss |
| `PseudoHuberLoss` | `nn.HuberLoss` | C²-smooth approximation with mask support |
| `LogCoshLoss` | — | Smooth L1 alternative, mask support |
| `CharbonnierLoss` | — | Differentiable L1 approximation |

```python
from torchregress.losses import WeightedMSELoss, WeightedHuberLoss

loss_fn = WeightedHuberLoss(delta=1.0, reduction="mean")
loss = loss_fn(y_pred, y_true, mask=valid_mask, weights=sample_w)
```

### Class Details

#### `BaseLoss`

The root loss class that handles mask and weight reductions:

$$
\mathcal{L} = \frac{1}{\sum_{i=1}^N w_i m_i} \sum_{i=1}^N w_i m_i \ell(y_{\text{pred}, i}, y_{\text{true}, i})
$$

under `reduction="mean"`.

#### `RegressionLoss`

Base class for point-prediction losses. Subclasses implement per-sample loss $\ell_i$ and inherit mask/weight reduction from [`BaseLoss`](#baseloss):

$$
\mathcal{L} = \frac{1}{\sum_i w_i m_i} \sum_i w_i m_i \, \ell(y_{\text{pred}, i}, y_i)
$$

#### `DistributionLoss`

Base class for probabilistic losses that model $p(y \mid x, \theta)$. Subclasses return negative log-likelihood (or proper scoring-rule) terms aggregated with the same mask/weight contract as [`RegressionLoss`](#regressionloss).

#### `WeightedLossWrapper`

Wraps any standard `nn.Module` to automatically handle boolean masking and sample weighting:

```python
WeightedLossWrapper(loss_module, reduction="mean")
```

#### `WeightedMSELoss`

Homoscedastic mean squared error with mask and weight support:

```python
WeightedMSELoss(reduction="mean")
```

$$
\text{MSE} = \frac{1}{\sum_i w_i m_i} \sum_i w_i m_i (y_i - \hat{y}_i)^2
$$

#### `WeightedHuberLoss`

Huber loss with mask and weight support:

```python
WeightedHuberLoss(delta=1.0, reduction="mean")
```

$$
\text{Huber}(r; \delta) = \begin{cases} \frac{1}{2} r^2 & |r| \le \delta \\ \delta |r| - \frac{1}{2} \delta^2 & |r| > \delta \end{cases}
$$

---

## Gaussian & heteroscedastic

| Symbol | Output | Use case |
|:-------|:-------|:---------|
| [`GaussianNLLLoss`](#gaussiannllloss) | `(mean, log_var)` | Heteroscedastic, independent targets |
| [`FaithfulGaussianLoss`](#faithfulgaussianloss) | `(mean, log_var)` | MSE on μ + NLL on variance with stop-gradient on μ |
| [`GaussianCRPSLoss`](#gaussiancrpsloss) | `(mean, log_var)` | Analytic Gaussian CRPS proper scoring rule |
| [`BetaNLLLoss`](#betanllloss) | `(mean, log_var)` | β-NLL: detached variance rescaling |
| [`GaussianWassersteinBoundLoss`](#gaussianwassersteinboundloss) | mean + cov params | Supervise mean + covariance vs labels / pseudo-labels |

### Class Details

#### `GaussianNLLLoss`

Diagonal Gaussian negative log-likelihood loss for predicting mean $\mu$ and log-variance $\log\sigma^2$:

```python
GaussianNLLLoss(eps=1e-6, reduction="mean")
```

$$
\mathcal{L}_i = \frac{1}{2} \log(2\pi \sigma_i^2) + \frac{(y_i - \mu_i)^2}{2\sigma_i^2}
$$

where $\sigma_i^2 = \exp(\log\sigma_i^2) + \varepsilon$.

#### `FaithfulGaussianLoss`

Implements the faithful heteroscedastic loss, splitting the objective into point prediction MSE on the mean and NLL on the variance, using a stop-gradient on the mean to prevent variance gradients from corrupting mean predictions:

```python
FaithfulGaussianLoss(eps=1e-6, reduction="mean")
```

$$
\mathcal{L}_i = \frac{1}{2} (y_i - \mu_i)^2 + \frac{1}{2} \log(2\pi \sigma_i^2) + \frac{(y_i - \operatorname{sg}(\mu_i))^2}{2\sigma_i^2}
$$

where $\operatorname{sg}$ is the stop-gradient operator.

#### `GaussianCRPSLoss`

Analytic Continuous Ranked Probability Score for diagonal Gaussian predictions (same output formats as [`GaussianNLLLoss`](#gaussiannllloss)):

```python
GaussianCRPSLoss(eps=1e-6, reduction="mean")
```

$$
\text{CRPS}(F, y) = \sigma \left[ z \cdot (2\Phi(z) - 1) + 2\phi(z) - \frac{1}{\sqrt{\pi}} \right], \quad z = \frac{y - \mu}{\sigma}
$$

where $\Phi$ and $\phi$ are the standard normal CDF and PDF.

#### `BetaNLLLoss`

Rescales the diagonal Gaussian NLL by a power of the detached predicted variance:

```python
BetaNLLLoss(beta=0.5, eps=1e-6, reduction="mean")
```

$$
\mathcal{L}_i = (\sigma_i^2 + \varepsilon)^{-\beta} \cdot \left( \frac{1}{2} \log(2\pi \sigma_i^2) + \frac{(y_i - \mu_i)^2}{2\sigma_i^2} \right)
$$

where the prefactor $(\sigma_i^2 + \varepsilon)^{-\beta}$ is computed with a stop-gradient (detached $\sigma_i^2$).

#### `GaussianWassersteinBoundLoss`

Wasserstein distance bound loss supervising mean and covariance:

```python
GaussianWassersteinBoundLoss(covariance_parameterization="diagonal", reduction="mean")
```

$$
\mathcal{L}_i = \|\mu_i - y_i\|_2^2 + \text{Tr}\left(\Sigma_i + \Sigma_{y, i} - 2(\Sigma_i^{1/2} \Sigma_{y, i} \Sigma_i^{1/2})^{1/2}\right)
$$

---

## Robust losses

Bounded-influence losses for data with outliers or heavy-tailed noise.

| Symbol | Influence | Tail behaviour |
|:-------|:----------|:---------------|
| [`PseudoHuberLoss`](#pseudohuberloss) | C² smooth | Linear (controlled by `delta`) |
| [`LogCoshLoss`](#logcoshloss) | `tanh(r)` | Linear |
| [`CharbonnierLoss`](#charbonnierloss) | `r / sqrt(r² + ε²)` | Linear |
| [`CauchyLoss`](#cauchyloss) | `2r / (c² + r²)` | Logarithmic (controlled by `c`) |
| [`TukeyBiweightLoss`](#tukeybiweightloss) | Redescending | **Zero** for `|r| > c` |
| `AdaptiveRobustLoss` | Trainable shape | Jointly optimises Barron shape α and scale |
| `BarronLoss` | `\|r\|²/α` shape | Continuous family spanning L1 ↔ L2 |
| `CVaRLoss` | Tail-focused | Optimises the upper α-fraction of per-sample loss |

### Class Details

#### `PseudoHuberLoss`

A smooth, twice-differentiable approximation of the Huber loss:

```python
PseudoHuberLoss(delta=1.0, reduction="mean")
```

$$
\mathcal{L}_i = \delta^2 \left(\sqrt{1 + \left(\frac{y_i - \hat{y}_i}{\delta}\right)^2} - 1\right)
$$

#### `LogCoshLoss`

A smooth L1 approximation using hyperbolic cosine:

```python
LogCoshLoss(reduction="mean")
```

$$
\mathcal{L}_i = \log\left(\cosh(y_i - \hat{y}_i)\right)
$$

#### `CharbonnierLoss`

A differentiable approximation to L1 loss used in image restoration:

```python
CharbonnierLoss(eps=1e-3, reduction="mean")
```

$$
\mathcal{L}_i = \sqrt{(y_i - \hat{y}_i)^2 + \varepsilon^2} - \varepsilon
$$

#### `CauchyLoss`

A robust loss with logarithmic influence for heavy-tailed noise:

```python
CauchyLoss(c=1.0, reduction="mean")
```

$$
\mathcal{L}_i = \frac{c^2}{2} \log\left(1 + \left(\frac{y_i - \hat{y}_i}{c}\right)^2\right)
$$

#### `TukeyBiweightLoss`

A redescending M-estimator loss that completely rejects extreme outliers:

```python
TukeyBiweightLoss(c=4.685, reduction="mean")
```

$$
\mathcal{L}_i = \begin{cases} \frac{c^2}{6} \left(1 - \left(1 - \left(\frac{y_i - \hat{y}_i}{c}\right)^2\right)^3\right) & \text{if } |y_i - \hat{y}_i| \le c \\ \frac{c^2}{6} & \text{otherwise} \end{cases}
$$

---

## Quantile, expectile, distributional

| Symbol | Output | Description |
|:-------|:-------|:------------|
| [`QuantileLoss`](#quantileloss) | `q_τ` | Pinball loss for one quantile |
| [`MultiQuantileLoss`](#multiquantileloss) | `(q_τ1, …, q_τK)` | Joint pinball for multiple quantiles |
| `QuantileCrossoverLoss` | — | Penalises `q_τ1 > q_τ2` when `τ1 < τ2` |
| [`ExpectileLoss`](#expectileloss) | `e_τ` | Asymmetric squared loss |
| `MultiExpectileLoss` | `(e_τ1, …, e_τK)` | Joint expectile regression |
| `AsymmetricLeastSquaresLoss` | `e_τ` | Lightweight expectile single-head |
| [`MDNLoss`](#mdnloss) | mixture params | Mixture Density Network NLL |
| [`NormalizingFlowLoss`](#normalizingflowloss) | NLL | Change-of-variables (requires `zuko`) |
| `ContrastiveFlowLoss` | NLL | Contrastive likelihood-ratio training |
| [`EvidentialRegressionLoss`](#evidentialregressionloss) | `(γ, ν, α, β)` | Normal-Inverse-Gamma prior evidential regression |

### Class Details

#### `QuantileLoss`

Pinball loss for a single quantile level $\tau \in (0, 1)$:

```python
QuantileLoss(tau=0.5, reduction="mean")
```

$$
\mathcal{L}_i = \max\left(\tau(y_i - \hat{y}_i), (\tau - 1)(y_i - \hat{y}_i)\right)
$$

#### `MultiQuantileLoss`

Joint pinball loss for multiple quantiles:

```python
MultiQuantileLoss(taus=[0.1, 0.5, 0.9], reduction="mean")
```

$$
\mathcal{L}_i = \sum_{\tau \in \text{taus}} \max\left(\tau(y_i - \hat{y}_{i, \tau}), (\tau - 1)(y_i - \hat{y}_{i, \tau})\right)
$$

#### `ExpectileLoss`

Asymmetric squared loss for expectile regression at level $\tau \in (0, 1)$:

```python
ExpectileLoss(tau=0.5, reduction="mean")
```

$$
\mathcal{L}_i = |\tau - \mathbb{I}(y_i < \hat{y}_i)| \cdot (y_i - \hat{y}_i)^2
$$

#### `MDNLoss`

Negative log-likelihood loss for Mixture Density Networks predicting Gaussian mixtures:

```python
MDNLoss(n_components=5, reduction="mean")
```

$$
\mathcal{L}_i = -\log \sum_{k=1}^K \pi_{i, k} \mathcal{N}(y_i \mid \mu_{i, k}, \sigma_{i, k}^2)
$$

#### `NormalizingFlowLoss`

Conditional negative log-likelihood under a Normalizing Flow:

```python
NormalizingFlowLoss(flow=my_flow, reduction="mean")
```

Forward: `loss_fn(y_pred=context_features, target=y)` where `y_pred` is the flow context.

$$
\mathcal{L}_i = -\log p_Z\left(f_\theta(y_i; x_i)\right) - \log \left| \det J_{f_\theta}(y_i; x_i) \right|
$$

#### `EvidentialRegressionLoss`

Implements the evidential regression loss over Normal-Inverse-Gamma parameters $(\gamma, \nu, \alpha, \beta)$ to model both aleatoric and epistemic uncertainty:

```python
EvidentialRegressionLoss(coeff=1e-2, reduction="mean")
```

$$
\mathcal{L}_i = \mathcal{L}_{\text{NIG}, i} + \lambda \cdot \mathcal{L}_{\text{reg}, i}
$$

$$
\mathcal{L}_{\text{NIG}, i} = \frac{1}{2}\log\left(\frac{\pi}{\nu_i}\right) - \alpha_i \log(\Omega_i) + \log\left(\frac{\Gamma(\alpha_i - 0.5)}{\Gamma(\alpha_i)}\right) + (\alpha_i + 0.5)\log\left(\Omega_i + \frac{\nu_i(y_i - \gamma_i)^2}{1 + \nu_i}\right)
$$

where $\Omega_i = 2\beta_i(1+\nu_i)$.

---

## Special target types

### Ordinal

| Symbol | Method | Outputs |
|:-------|:-------|:--------|
| `CumulativeLinkLoss` | Cumulative-threshold model | `K-1` logits |
| `CORALLoss` | Consistent Rank Logits | `K-1` binary logits |
| `OrdinalCrossEntropyLoss` | Standard CE baseline | `K` logits |

### Censored / survival

| Symbol | Censoring model | Use case |
|:-------|:----------------|:---------|
| [`CensoredGaussianNLLLoss`](#censoredgaussiannllloss) | Gaussian + Φ | Clipped sensors, detection limits |
| `CensoredQuantileLoss` | Quantile + censoring | Non-parametric censored |
| `AFTLoss` | Accelerated Failure Time | Survival analysis |

### Count data (Poisson, Tweedie, mixture)

| Symbol | Variance model | Best for |
|:-------|:----------------|:---------|
| [`PoissonDevianceLoss`](#poissondevianceloss) | `Var = μ` | Standard counts |
| [`TweedieLoss`](#tweedieloss) | `Var = φ μ^p` | Compound Poisson-Gamma, gamma, inverse Gaussian |
| [`NegativeBinomialNLLLoss`](#negativebinomialnllloss) | `Var = μ + μ²/r` | Overdispersed counts |
| [`ZeroInflatedPoissonNLLLoss`](#zeroinflatedpoissonnllloss) | ZIP | Counts with excess zeros |
| [`PoissonGaussianMixtureLoss`](#poissongaussianmixtureloss) | Poisson + Gaussian readout | Imaging / low-light sensing |

### Class Details

#### `CensoredGaussianNLLLoss`

Gaussian NLL adapted for left, right, or interval censored targets:

```python
CensoredGaussianNLLLoss(reduction="mean")
```

$$
\mathcal{L}_i = \begin{cases} \frac{1}{2} \log(2\pi \sigma_i^2) + \frac{(y_i - \mu_i)^2}{2\sigma_i^2} & \text{if uncensored } (c_i = 0) \\ -\log \Phi\left(\frac{y_i - \mu_i}{\sigma_i}\right) & \text{if left-censored } (c_i = -1) \\ -\log \left(1 - \Phi\left(\frac{y_i - \mu_i}{\sigma_i}\right)\right) & \text{if right-censored } (c_i = 1) \end{cases}
$$

where $\Phi$ is the standard normal cumulative distribution function.

#### `PoissonDevianceLoss`

Poisson deviance loss for count data modeling:

```python
PoissonDevianceLoss(reduction="mean")
```

$$
\mathcal{L}_i = 2 \left(y_i \log\left(\frac{y_i}{\hat{y}_i}\right) - (y_i - \hat{y}_i)\right)
$$

---

## Data quality issues

| Symbol | Strategy |
|:-------|:---------|
| [`BalancedMSELoss`](#balancedmseloss) | Inverse bin-frequency weighted MSE |
| [`BMCLoss`](#bmcloss) | Balanced MSE with Gaussian-smoothed bin weights |
| `DensityWeightedLoss` | Inverse target-density reweighting |
| `FocalRLoss` | Focus on hard / rare examples |
| `LDSLoss` | Label Distribution Smoothing |
| [`PropensityWeightedLoss`](#propensityweightedloss) | Inverse propensity / density ratio weighting |
| [`NoisyTargetGaussianNLL`](#noisytargetgaussiannll) | Adds known target-noise `σ_y²` to predicted variance |
| [`PseudoLabelNLL`](#pseudolabelnll) | Gaussian NLL with pseudo-label + confidence weighting |
| [`ConsistencyRegLoss`](#consistencyregloss) | Student–teacher MSE consistency |
| [`PseudoLabelConsistencyLoss`](#pseudolabelconsistencyloss) | Single objective for pseudo-label + teacher consistency |

### Class Details

#### `BalancedMSELoss`

Inverse bin-frequency weighted MSE for long-tailed scalar targets. Call `fit(train_targets)` once to estimate per-bin weights $w(y)$:

```python
BalancedMSELoss(bin_edges, count_smoothing=0.0, reduction="mean")
```

$$
\mathcal{L}_{\text{BalancedMSE}}(y, \hat{y}) = w(y) (y - \hat{y})^2, \qquad
w(b) \propto \frac{1}{n_b + \varepsilon}
$$

where $n_b$ is the training count in bin $b$ containing $y$.

→ Guide: [Imbalanced losses](../losses/imbalanced.md). Example: [Balanced MSE](../examples/balanced_mse.md).

#### `BMCLoss`

Balanced MSE with optional Gaussian smoothing over bin frequencies (`noise_sigma`). Same `fit(train_targets)` contract as `BalancedMSELoss`.

```python
BMCLoss(bin_edges, noise_sigma=0.0, reduction="mean")
```

#### `NoisyTargetGaussianNLL`

Gaussian NLL incorporating known target observation variance $\sigma_{y, i}^2$:

```python
NoisyTargetGaussianNLL(reduction="mean")
```

$$
\mathcal{L}_i = \frac{1}{2} \log\left(2\pi (\sigma_i^2 + \sigma_{y, i}^2)\right) + \frac{(y_i - \mu_i)^2}{2(\sigma_i^2 + \sigma_{y, i}^2)}
$$

#### `PseudoLabelConsistencyLoss`

Combines pseudo-label training with teacher-student consistency:

```python
PseudoLabelConsistencyLoss(confidence_threshold=0.9, reduction="mean")
```

$$
\mathcal{L}_i = \mathbb{I}(p_i > \tau) \cdot (y_{\text{pseudo}, i} - \hat{y}_i)^2 + \lambda \cdot (y_{\text{teacher}, i} - \hat{y}_i)^2
$$

---

## Error-in-variables (EIV)

Call pattern: construct with `model=...`, then `loss(x_obs, y_obs, mask=...)` — the model is passed at construction so gradients flow through its parameters.

| Symbol | Model | When to use |
|:-------|:------|:------------|
| [`StructuralEIVLoss`](#structuraleivloss) | Known error ratio λ | Known `σ_x / σ_y` |
| [`FunctionalEIVLoss`](#functionaleivloss) | Per-sample `σ_x` | Heteroscedastic input noise |
| [`OrthogonalDistanceRegressionLoss`](#orthogonaldistanceregressionloss) | Perpendicular distances | General EIV |
| `EnsembleEIVLoss` | Ensemble disagreement | No known error model |

### Class Details

#### `StructuralEIVLoss`

Structural Error-in-Variables loss using a known input/output noise ratio $\lambda = \sigma_{x}^2 / \sigma_{y}^2$:

```python
StructuralEIVLoss(
    model=my_model,
    sigma_x=0.5,
    sigma_y=0.3,
    sigma_xy=torch.zeros(2, 2),
    reduction="mean",
)
```

$$
\mathcal{L}_i = \frac{(y_{\text{obs}, i} - f(x_{\text{true}, i}))^2 + \lambda (x_{\text{obs}, i} - x_{\text{true}, i})^2}{\sigma_{y, i}^2}
$$

#### `FunctionalEIVLoss`

Functional Error-in-Variables loss using known sample-specific input uncertainties $\sigma_{x, i}^2$:

```python
FunctionalEIVLoss(model=my_model, sigma_x=0.5, sigma_y=0.3, reduction="mean")
```

$$
\mathcal{L}_i = \frac{(y_{\text{obs}, i} - f(x_{\text{true}, i}))^2}{\sigma_{y, i}^2} + \frac{(x_{\text{obs}, i} - x_{\text{true}, i})^2}{\sigma_{x, i}^2}
$$

#### `OrthogonalDistanceRegressionLoss`

Loss that penalises perpendicular distance from observation coordinates to the regression curve.

---

## Conformal prediction

Conformal losses implement both the **training** and **calibration** phases of conformal prediction in a single objective.

| Symbol | Strategy |
|:-------|:---------|
| [`ConformalLoss`](#conformalloss) | Training + calibration wrapper (`method="split"`, `"cqr"`, `"uacqr"`, …) |
| [`SplitConformal`](#splitconformal) | Residual-based |
| [`CQR`](#cqr) | Conformalized Quantile Regression |
| [`UACQR`](#uacqr) | Width-normalised CQR |

### Class Details

#### `ConformalLoss`

Loss wrapper that trains a base model and applies conformal calibration on target outputs:

```python
ConformalLoss(method="cqr", alpha=0.1)
```

#### `SplitConformal`

Residual-based split conformal calibration. Nonconformity scores are absolute residuals $s_i = |y_i - \hat{y}_i|$ on a held-out calibration set. The $(1-\alpha)$ quantile $\hat{q}$ of calibration scores yields intervals:

$$
\hat{C}(x) = \left[\hat{y}(x) - \hat{q},\; \hat{y}(x) + \hat{q}\right]
$$

#### `CQR`

Conformalized Quantile Regression. Uses lower/upper quantile predictions $\hat{q}_{\text{lo}}, \hat{q}_{\text{hi}}$ and the score:

$$
s_i = \max\left(\hat{q}_{\text{lo}, i} - y_i,\; y_i - \hat{q}_{\text{hi}, i}\right)
$$

The calibrated interval at test time is $[\hat{q}_{\text{lo}}(x) - \hat{q},\; \hat{q}_{\text{hi}}(x) + \hat{q}]$.

#### `UACQR`

Uncertainty-aware CQR: same construction as [`CQR`](#cqr), but scores are normalised by the predicted quantile band width (clamped):

$$
s_i = \frac{\max\left(\hat{q}_{\text{lo}, i} - y_i,\; y_i - \hat{q}_{\text{hi}, i}\right)}{\max\left(\hat{q}_{\text{hi}, i} - \hat{q}_{\text{lo}, i},\; \varepsilon\right)}
$$

---

## Additional loss reference

### WeightedL1Loss

Standard Mean Absolute Error with mask and weight support:

$$
\text{L1} = \frac{1}{\sum w_i m_i} \sum w_i m_i |y_i - \hat{y}_i|
$$

### BarronLoss

General robust loss function parameterized by shape $\alpha$:

$$
\rho(r; \alpha, c) = \frac{|2-\alpha|}{\alpha} \left( \left( \frac{(r/c)^2}{|2-\alpha|} + 1 \right)^{\alpha/2} - 1 \right)
$$

### AdaptiveRobustLoss

trainable Barron loss where $\alpha$ and scale $c$ are learned parameters.

### CVaRLoss

Conditional Value at Risk loss optimizing the tail $\alpha$-fraction of sample losses.

### create_loss_from_config

Factory method to build a loss function from configuration options.

### LowRankGaussianLoss

Gaussian NLL assuming a low-rank plus diagonal covariance:

$$
\Sigma = W W^T + D
$$

### MultivariateGaussianLoss

Gaussian NLL assuming a full covariance matrix $\Sigma$.

### CumulativeLinkLoss

Loss function for ordinal targets using the cumulative link model:

$$
P(Y \le k) = \sigma(\theta_k - f(x))
$$

### CORALLoss

Consistent Rank Logits loss for ordinal regression.

### OrdinalCrossEntropyLoss

Standard cross-entropy loss applied to ordinal targets.

### CensoredQuantileLoss

Quantile loss adapted for left, right, or interval censored targets.

### AFTLoss

Accelerated Failure Time loss for survival analysis.

### NegativeBinomialNLLLoss

Negative log-likelihood for the Negative Binomial distribution to handle overdispersed count data.

### ZeroInflatedPoissonNLLLoss

Negative log-likelihood for the Zero-Inflated Poisson distribution to handle count data with excess zeros.

### TweedieLoss

Tweedie deviance loss for power-variance relationships $\text{Var}(Y) = \phi \mu^p$:

```python
TweedieLoss(p=1.5, link="log", reduction="mean")
```

$$
D(y, \mu) = 2 \left( y^{2-p} - (2-p) y \mu^{1-p} + (1-p) \mu^{2-p} \right) / \bigl((1-p)(2-p)\bigr)
$$

for $1 < p < 2$ (compound Poisson-Gamma); other $p$ use the standard Tweedie deviance forms.

### PoissonGaussianMixtureLoss

Negative log-likelihood for Poisson counting noise plus Gaussian readout noise:

```python
PoissonGaussianMixtureLoss(log_input=True, initial_variance=1.0, reduction="mean")
```

$$
\mathcal{L}_{\text{mix}} = w_P \mathcal{L}_{\text{Poisson}}(y, \lambda) + w_G \mathcal{L}_{\text{Gaussian}}(y, \lambda, \sigma^2)
$$

### SLSLoss

Super-Level-Set regression loss learning minimum-volume prediction regions with target coverage $\tau$:

```python
SLSLoss(d=2, context_dim=64, K=1, tau=0.9, warmup_steps=500)
```

Balances log-volume of a Mahalanobis frontier with a quantile-network coverage penalty after warmup.

### LogTransformLoss

Applies $\log(\cdot)$ to predictions and targets before a base point loss:

```python
LogTransformLoss(reduction="mean")
```

$$
\mathcal{L}(\hat{y}, y) = \ell\bigl(\log(\hat{y}), \log(y)\bigr)
$$

### BoxCoxTransformLoss

Box–Cox target transform for positive skewed targets:

$$
T_\lambda(y) = \begin{cases} \dfrac{y^\lambda - 1}{\lambda} & \lambda \neq 0 \\ \log y & \lambda = 0 \end{cases}
$$

### SqrtTransformLoss

Square-root target transform for moderate variance growth on non-negative targets:

$$
\mathcal{L}(\hat{y}, y) = \ell(\sqrt{\hat{y}}, \sqrt{y})
$$

### YeoJohnsonTransformLoss

Yeo–Johnson power transform supporting signed skewed targets (generalizes Box–Cox).

### TransformedTargetLoss

Generic wrapper applying any `make_target_transform` mapping before a base loss.

### ConsistencyRegLoss

Student–teacher consistency regularizer:

```python
ConsistencyRegLoss(reduction="mean")
```

$$
\mathcal{L} = \| f_{\text{student}}(x) - \operatorname{sg}(f_{\text{teacher}}(x)) \|_2^2
$$

### PseudoLabelNLL

Gaussian NLL with pseudo-label and confidence weighting for semi-supervised heteroscedastic heads.

### DensityWeightedLoss

Inverse kernel-density sample weights for imbalanced / long-tail targets (`fit_density(y)` required).

$$
w_i \propto \frac{1}{\hat{p}(y_i) + \varepsilon}
$$

### PropensityWeightedLoss

Inverse propensity score weighting for selection-bias correction:

$$
w_i \propto \frac{1}{\hat{e}(x_i) + \varepsilon}
$$

### ContrastiveFlowLoss

Contrastive likelihood-ratio training for conditional normalizing flows — scores the observed target under a positive context vs. one or more negative contexts.
