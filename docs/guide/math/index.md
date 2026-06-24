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
| **MSE** | Gaussian (fixed $\sigma$) | $(y - \hat{y})^2$ | [WeightedMSELoss](../../api/losses.md#weightedmseloss) |
| **MAE** | Laplace (fixed $b$) | $\lvert y - \hat{y}\rvert$ | [WeightedL1Loss](../../api/losses.md#weightedl1loss) |
| **Gaussian NLL** | Gaussian (learned $\sigma$) | $\frac{1}{2}\log(2\pi\sigma^2) + \frac{(y-\mu)^2}{2\sigma^2}$ | [GaussianNLLLoss](../../api/losses.md#gaussiannllloss) |
| **Poisson NLL** | Poisson | $\hat{\mu} - y\log\hat{\mu} + \log(y!)$ | [PoissonDevianceLoss](../../api/losses.md#poissondevianceloss) |

→ See [Gaussian Losses](../../losses/gaussian.md) for heteroscedastic implementation details.

### Quantile & Expectile Regression

Unlike NLL losses which target the mean, quantile and expectile losses target specific properties of the predictive distribution.

**Quantile Loss (Pinball Loss):**
Used to estimate the $\tau$-th quantile $\hat{q}_\tau$. It is the $L_1$ analogue for distributional modeling \[6\]. See [QuantileLoss](../../api/losses.md#quantileloss) for API details.

$$\mathcal{L}_\tau^{\text{quantile}}(y, \hat{q}_\tau) = (y - \hat{q}_\tau) \left( \tau - \mathbf{1}_{y < \hat{q}_\tau} \right)$$

**Expectile Loss:**
Used to estimate the $\tau$-th expectile $\hat{e}_\tau$. It is the $L_2$ analogue of quantile regression, which is often easier to optimise due to its smoothness. See [ExpectileLoss](../../api/losses.md#expectileloss).

$$\mathcal{L}_\tau^{\text{expectile}}(y, \hat{e}_\tau) = \lvert\tau - \mathbf{1}_{y < \hat{e}_\tau}\rvert \cdot (y - \hat{e}_\tau)^2$$

---

## Robust M-Estimation

Robust losses are designed to mitigate the influence of outliers by using "redescending" or sub-quadratic influence functions $\psi(r) = \frac{\partial \rho}{\partial r}$.

| Loss | Function $\rho(r)$ | Influence $\psi(r)$ | API Reference |
|:-----|:-------------------|:-------------------|:--------------|
| **Huber** | Quadratic near 0, Linear at tails | Bounded | [WeightedHuberLoss](../../api/losses.md#weightedhuberloss) |
| **Log-Cosh** | $\log(\cosh r)$ | $\tanh(r)$ | [LogCoshLoss](../../api/losses.md#logcoshloss) |
| **Tukey Biweight** | Redescending polynomial | $\rightarrow 0$ for large $r$ | [TukeyBiweightLoss](../../api/losses.md#tukeybiweightloss) |
| **Cauchy** | $\log(1 + r^2/c^2)$ | Decreasing | [CauchyLoss](../../api/losses.md#cauchyloss) |

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

### Deep Ensemble Decomposition Derivation

For an ensemble of $M$ models, each outputting a parametric prediction $p(y \mid x, \theta_m) = \mathcal{N}(\mu_m(x), \sigma^2_m(x))$, the mixture distribution of the ensemble is:

$$p(y \mid x) = \frac{1}{M} \sum_{m=1}^{M} p(y \mid x, \theta_m)$$

The total predictive mean is $\bar\mu(x) = \mathbb{E}[y \mid x] = \frac{1}{M}\sum_{m=1}^{M}\mu_m(x)$.
Using the **Law of Total Variance**, the total predictive variance is:

$$\operatorname{Var}(y \mid x) = \mathbb{E}_{\theta_m}\bigl[\operatorname{Var}(y \mid x, \theta_m)\bigr] + \operatorname{Var}_{\theta_m}\bigl(\mathbb{E}[y \mid x, \theta_m]\bigr)$$

Plugging in the ensemble moments:

$$\sigma_{\text{total}}^2 = \underbrace{\frac{1}{M}\sum_{m=1}^{M}\sigma_m^2}_{\text{Aleatoric (Expected Variance)}} + \underbrace{\frac{1}{M}\sum_{m=1}^{M}(\mu_m - \bar\mu)^2}_{\text{Epistemic (Variance of the Means)}}$$

This provides a clean separation: aleatoric uncertainty represents the average data noise estimated across ensemble members, while epistemic uncertainty captures model parameter disagreement.

→ See [Ensembles for Uncertainty](../../methods/ensemble/index.md) for advanced decomposition methods (e.g., [DeepEnsemble](../../api/ensemble.md#deepensemble), SWAG, BNN).

---

## Proper Scoring Rules

A scoring rule $S(F, y)$ is **proper** if the expected score is minimised when the predicted distribution $F$ matches the true distribution $G$:

$$\mathbb{E}_{y \sim G}[S(G, y)] \leq \mathbb{E}_{y \sim G}[S(F, y)]$$

### Continuous Ranked Probability Score (CRPS)

The CRPS generalizes Mean Absolute Error (MAE) to probabilistic forecasts:

$$CRPS(F, y) = \int_{-\infty}^{\infty} [F(z) - \mathbf{1}_{z \geq y}]^2 dz = \mathbb{E}[|X - y|] - \frac{1}{2}\mathbb{E}[|X - X'|]$$

where $X, X' \sim F$ are independent and identically distributed copies.

#### Derivation of Gaussian CRPS
For $F = \mathcal{N}(\mu, \sigma^2)$, let $Z = \frac{X - \mu}{\sigma} \sim \mathcal{N}(0, 1)$ and $z = \frac{y - \mu}{\sigma}$. The evaluation reduces to:

$$CRPS(\mu, \sigma, y) = \sigma \mathbb{E}[|Z - z|] - \frac{1}{2}\sigma \mathbb{E}[|Z - Z'|]$$

1. **First Term**: Evaluating the expectation $\mathbb{E}[|Z - z|]$ yields:
   $$\mathbb{E}[|Z - z|] = z \bigl(2\Phi(z) - 1\bigr) + 2\phi(z)$$
2. **Second Term**: The expected absolute difference between two standard Gaussians $\mathbb{E}[|Z - Z'|]$ evaluates to $\frac{2}{\sqrt{\pi}}$ since $Z - Z' \sim \mathcal{N}(0, 2)$.
3. Combining terms leads to the closed-form implemented in [crps_gaussian](../../api/metrics.md#crps_gaussian):
   $$CRPS(\mu, \sigma, y) = \sigma \left[ z \bigl(2\Phi(z) - 1\bigr) + 2\phi(z) - \frac{1}{\sqrt{\pi}} \right]$$

### Interval Score
For evaluating prediction intervals $[L(x), U(x)]$ at nominal level $(1 - \alpha)$, we use the strictly proper **Interval Score**:

$$S_{\alpha}(L, U; y) = (U - L) + \frac{2}{\alpha}(L - y)\mathbf{1}_{y < L} + \frac{2}{\alpha}(y - U)\mathbf{1}_{y > U}$$

The score penalizes wide intervals (sharpness) and penalizes boundary violations (calibration) with a scale factor of $2/\alpha$.

→ See [Distribution Metrics](../../metrics/distribution.md) for multivariate [energy_score](../../api/metrics.md#energy_score).

---

## Conformal Prediction

Conformal Prediction (CP) provides a framework for generating prediction intervals with **guaranteed coverage** under the sole assumption of exchangeability \[4, 5\].

Given a non-conformity score $s(x, y)$ (e.g., absolute residual $\lvert y - \hat{y} \rvert$), the conformal interval at level $1-\alpha$ is:

$$\hat{C}(x) = \{ y : s(x, y) \leq \hat{q} \}$$

where $\hat{q}$ is the $\frac{\lceil(n+1)(1-\alpha)\rceil}{n}$ quantile of calibration scores.

**Coverage Guarantee:**

$$P\bigl(Y_{n+1} \in \hat{C}(X_{n+1})\bigr) \;\geq\; 1 - \alpha$$

→ See [Conformal Prediction](../../methods/conformal/index.md) for [SplitConformal](../../api/losses.md#splitconformal), CQR and distributional CP.

---

## Specialized Regression Tasks

### Continuous Imbalanced Regression (LDS & FDS)
When continuous targets $y$ are highly imbalanced, empirical estimators in sparse regions are unreliable.

**Label Distribution Smoothing (LDS):**
LDS estimates the empirical label density $p(y)$ and smooths it using a symmetric kernel $k$:
$$\tilde{p}(y) = \int p(z) k(y, z) dz$$
Samples are then reweighted by $w_i \propto 1 / \tilde{p}(y_i)$ to balance the objective function. See [LDSLoss](../../api/losses.md#ldsloss).

**Feature Distribution Smoothing (FDS):**
FDS smooths the running mean $\mu_b$ and covariance $\Sigma_b$ of model features across adjacent target bins $b$:
$$\tilde{\mu}_b = \sum_{b'} \omega(b, b') \mu_{b'} \qquad \tilde{\Sigma}_b = \sum_{b'} \omega(b, b') \Sigma_{b'}$$
Features $\mathbf{z}$ are then calibrated during training via whitening and recoloring:
$$\tilde{\mathbf{z}} = \tilde{\Sigma}_b^{1/2} \Sigma_b^{-1/2} (\mathbf{z} - \mu_b) + \tilde{\mu}_b$$
See [FeatureDistributionSmoother](../../api/losses.md#featuredistributionsmoother).

### Measurement Error (Errors-in-Variables)

Standard OLS assumes $X$ is measured perfectly. If $X_{\text{obs}} = X^* + \epsilon$, then OLS estimates are biased toward zero (**attenuation bias**) \[10\]. **torchregress** implements SIMEX and Regression Calibration (RC) to correct this. See [Algorithms](../../methods/algorithms/rc.md).

### Ordinal Regression

For discrete ordered targets, we use the **Cumulative Link Model**:

$$P(Y \leq k \mid x) = \sigma(\theta_k - f(x))$$

where $\theta_1 < \theta_2 < \dots < \theta_{K-1}$ are learned thresholds. See [CumulativeLinkLoss](../../api/losses.md#cumulativelinkloss).

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
| 12 | Nix, D. A., & Weigend, A. S. ["Estimating the Mean and Variance of the Target Probability Distribution."](https://ieeexplore.ieee.org/document/341257) *ICNN*, 1994. |
