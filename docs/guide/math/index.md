# Mathematical Foundations

This page provides a rigorous overview of the mathematical principles underlying **torchregress**. We cover notation, the duality between loss functions and likelihoods, uncertainty decomposition, and modern estimation techniques.

---

## Notation

Throughout the documentation and codebase, we adhere to the following conventions:

| Symbol | Meaning | Domain |
|:------:|:--------|:-------|
| $x, \mathbf{x}$ | Input features | $\mathbb{R}^d$ |
| $y$ | True target value (ground truth) | $\mathbb{R}$ or $\mathbb{R}^k$ |
| $\hat{y}$ | Point prediction (expectation) | $\mathbb{R}$ |
| $\mu, \hat{\mu}$ | Predicted distribution mean | $\mathbb{R}$ |
| $\sigma^2, \hat{\sigma}^2$ | Predicted distribution variance | $\mathbb{R}^+$ |
| $\hat{q}_\tau$ | Predicted $\tau$-quantile | $\tau \in (0, 1)$ |
| $\hat{e}_\tau$ | Predicted $\tau$-expectile | $\tau \in (0, 1)$ |
| $\mathcal{L}$ | Loss function to be minimised | $\mathbb{R}$ |
| $n$ | Number of training samples | $\mathbb{Z}^+$ |
| $M$ | Number of ensemble members | $\mathbb{Z}^+$ |
| $\mathbf{1}_A$ | Indicator function ($1$ if $A$ is true, $0$ otherwise) | $\{0, 1\}$ |

---

## Loss Functions as Negative Log-Likelihoods

Most regression losses in **torchregress** are derived from the **Maximum Likelihood Estimation (MLE)** framework. Choosing a loss function $\mathcal{L}$ is mathematically equivalent to assuming a specific probability distribution $p(y \mid x, \theta)$ for the targets and minimising the **Negative Log-Likelihood (NLL)**:

$$\mathcal{L}(\theta) = -\mathbb{E}_{(x, y) \sim \mathcal{D}} \left[ \log p(y \mid x, \theta) \right]$$

### Core Parametric Losses

| Loss | Implied Distribution | Mathematical Form | API Reference |
|:-----|:---------------------|:------------------|:--------------|
| **MSE** | Gaussian (fixed $\sigma$) | $(y - \hat{y})^2$ | [`WeightedMSELoss`](../../api/losses.md#torchregress.losses.base.WeightedMSELoss) |
| **MAE** | Laplace (fixed $b$) | $\lvert y - \hat{y}\rvert$ | [`WeightedL1Loss`](../../api/losses.md#torchregress.losses.base.WeightedL1Loss) |
| **Gaussian NLL** | Gaussian (learned $\sigma$) | $\frac{1}{2}\log(2\pi\sigma^2) + \frac{(y-\mu)^2}{2\sigma^2}$ | [`GaussianNLLLoss`](../../api/losses.md#torchregress.losses.gaussian.GaussianNLLLoss) |
| **Poisson NLL** | Poisson | $\hat{\mu} - y\log\hat{\mu} + \log(y!)$ | [`PoissonDevianceLoss`](../../api/losses.md#torchregress.losses.poisson.PoissonDevianceLoss) |

→ See [Gaussian Losses](../../losses/gaussian.md) for heteroscedastic implementation details.

### Quantile & Expectile Regression

Unlike NLL losses which target the mean, quantile and expectile losses target specific properties of the predictive distribution.

**Quantile Loss (Pinball Loss):**
Used to estimate the $\tau$-th quantile $\hat{q}_\tau$. It is the $L_1$ analogue for distributional modeling [6]. See [`QuantileLoss`](../../api/losses.md#torchregress.losses.quantile.QuantileLoss) for API details.

$$\mathcal{L}_\tau^{\text{quantile}}(y, \hat{q}_\tau) = (y - \hat{q}_\tau) \left( \tau - \mathbf{1}_{y < \hat{q}_\tau} \right)$$

**Expectile Loss:**
Used to estimate the $\tau$-th expectile $\hat{e}_\tau$. It is the $L_2$ analogue of quantile regression, which is often easier to optimise due to its smoothness. See [`ExpectileLoss`](../../api/losses.md#torchregress.losses.expectile.ExpectileLoss).

$$\mathcal{L}_\tau^{\text{expectile}}(y, \hat{e}_\tau) = \lvert\tau - \mathbf{1}_{y < \hat{e}_\tau}\rvert \cdot (y - \hat{e}_\tau)^2$$

---

## Robust M-Estimation

Robust losses are designed to mitigate the influence of outliers by using "redescending" or sub-quadratic influence functions $\psi(r) = \frac{\partial \rho}{\partial r}$.

| Loss | Function $\rho(r)$ | Influence $\psi(r)$ | API Reference |
|:-----|:-------------------|:-------------------|:--------------|
| **Huber** | Quadratic near 0, Linear at tails | Bounded | [`WeightedHuberLoss`](../../api/losses.md#torchregress.losses.base.WeightedHuberLoss) |
| **Log-Cosh** | $\log(\cosh r)$ | $\tanh(r)$ | [`LogCoshLoss`](../../api/losses.md#torchregress.losses.robust.LogCoshLoss) |
| **Tukey Biweight** | Redescending polynomial | $\rightarrow 0$ for large $r$ | [`TukeyBiweightLoss`](../../api/losses.md#torchregress.losses.robust.TukeyBiweightLoss) |
| **Cauchy** | $\log(1 + r^2/c^2)$ | Decreasing | [`CauchyLoss`](../../api/losses.md#torchregress.losses.robust.CauchyLoss) |

→ See [Robust Losses](../../losses/robust.md) for parameter selection guides.

---

## Uncertainty Quantification (UQ)

**torchregress** distinguishes between two fundamental types of uncertainty:

### 1. Aleatoric Uncertainty (Data Noise)

Inherent randomness in the data-generating process. It is **irreducible** even with infinite data.

- **Homoscedastic**: Constant noise level $\sigma^2$ across all inputs.
- **Heteroscedastic**: Input-dependent noise level $\sigma^2(x)$.

### 2. Epistemic Uncertainty (Model Ignorance)

Uncertainty in the model parameters or structure due to limited training data. It is **reducible** as the dataset size increases.

### Deep Ensemble Decomposition

For an ensemble of $M$ models, each predicting a mean $\mu_m$ and variance $\sigma_m^2$, the total predictive uncertainty can be decomposed as [1]:

$$\boxed{\;\sigma_{\text{total}}^2 = \underbrace{\frac{1}{M}\sum_{m=1}^{M}\sigma_m^2}_{\text{Aleatoric}} + \underbrace{\frac{1}{M}\sum_{m=1}^{M}(\mu_m - \bar\mu)^2}_{\text{Epistemic}}\;}$$

where $\bar\mu = \frac{1}{M}\sum \mu_m$.

→ See [Ensemble & UQ](../../methods/ensemble/index.md) for advanced decomposition methods (e.g., [`DeepEnsemble`](../../api/ensemble.md#torchregress.ensemble.DeepEnsemble), SWAG, BNN).

---

## Proper Scoring Rules

A scoring rule $S(F, y)$ is **proper** if the expected score is minimised when the predicted distribution $F$ matches the true distribution $G$ [9].

### Continuous Ranked Probability Score (CRPS)

The CRPS generalizes the MAE to probabilistic forecasts. It measures both **calibration** and **sharpness**.

$$CRPS(F, y) = \int_{-\infty}^{\infty} [F(z) - \mathbf{1}_{z \geq y}]^2 dz$$

For a Gaussian distribution $\mathcal{N}(\mu, \sigma^2)$, this simplifies to a closed form implemented in [`crps_gaussian`](../../api/metrics.md#torchregress.metrics.distribution.crps_gaussian):

$$CRPS(\mu, \sigma, y) = \sigma \left[ \frac{y-\mu}{\sigma} \Phi\left(\frac{y-\mu}{\sigma}\right) + 2\phi\left(\frac{y-\mu}{\sigma}\right) - \frac{1}{\sqrt{\pi}} \right]$$

→ See [Distribution Metrics](../../metrics/distribution.md) for multivariate [`energy_score`](../../api/metrics.md#torchregress.metrics.distribution.energy_score).

---

## Conformal Prediction

Conformal Prediction (CP) provides a framework for generating prediction intervals with **guaranteed coverage** under the sole assumption of exchangeability [4, 5].

Given a non-conformity score $s(x, y)$ (e.g., absolute residual $\lvert y - \hat{y} \rvert$), the conformal interval at level $1-\alpha$ is:

$$\hat{C}(x) = \{ y : s(x, y) \leq \hat{q} \}$$

where $\hat{q}$ is the $\frac{\lceil(n+1)(1-\alpha)\rceil}{n}$ quantile of calibration scores.

**Coverage Guarantee:**

$$P\bigl(Y_{n+1} \in \hat{C}(X_{n+1})\bigr) \;\geq\; 1 - \alpha$$

→ See [Conformal Prediction](../../methods/conformal/index.md) for [`SplitConformal`](../../api/losses.md#torchregress.losses.conformal.SplitConformal), CQR and distributional CP.

---

## Specialized Regression Tasks

### Measurement Error (Errors-in-Variables)

Standard OLS assumes $X$ is measured perfectly. If $X_{\text{obs}} = X^* + \epsilon$, then OLS estimates are biased toward zero (**attenuation bias**) [10]. **torchregress** implements SIMEX and Regression Calibration (RC) to correct this. See [Algorithms](../../methods/algorithms/rc.md).

### Ordinal Regression

For discrete ordered targets, we use the **Cumulative Link Model**:

$$P(Y \leq k \mid x) = \sigma(\theta_k - f(x))$$

where $\theta_1 < \theta_2 < \dots < \theta_{K-1}$ are learned thresholds. See [`CumulativeLinkLoss`](../../api/losses.md#torchregress.losses.ordinal.CumulativeLinkLoss).

---

## References

| # | Reference |
|:-:|:----------|
| 1 | Lakshminarayanan et al. ["Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles."](https://arxiv.org/abs/1612.01474) *NeurIPS*, 2017. |
| 2 | Amini et al. ["Deep Evidential Regression."](https://arxiv.org/abs/1910.02600) *NeurIPS*, 2020. |
| 3 | Bishop, C. M. *Pattern Recognition and Machine Learning*. Springer, 2006. |
| 4 | Vovk et al. *Algorithmic Learning in a Random World*. Springer, 2005. |
| 5 | Romano et al. ["Conformalized Quantile Regression."](https://arxiv.org/abs/1905.03222) *NeurIPS*, 2019. |
| 6 | Koenker, R., & Bassett, G. ["Regression Quantiles."](https://www.jstor.org/stable/1913643) *Econometrica*, 1978. |
| 7 | McCullagh, P. ["Regression Models for Ordinal Data."](https://www.jstor.org/stable/2984952) *JRSS B*, 1980. |
| 8 | Jørgensen, B. *The Theory of Dispersion Models*. Chapman & Hall, 1997. |
| 9 | Gneiting, T., & Raftery, A. E. ["Strictly Proper Scoring Rules, Prediction, and Estimation."](https://www.tandfonline.com/doi/abs/10.1198/016214506000001437) *JASA*, 2007. |
| 10 | Carroll et al. *Measurement Error in Nonlinear Models*. Chapman & Hall, 2006. |
| 11 | Kuleshov et al. ["Accurate Uncertainties for Deep Learning Using Calibrated Regression."](https://arxiv.org/abs/1807.00263) *ICML*, 2018. |
| 12 | Nix, D. A., & Weigend, A. S. "Estimating the Mean and Variance of the Target Probability Distribution." *ICNN*, 1994. |
