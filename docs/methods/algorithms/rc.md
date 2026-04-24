# Regression Calibration (RC)

Regression Calibration is a classical method for correcting **measurement error in inputs** (errors-in-variables).  When the observed input $W$ is a noisy version of the true input $X$, naive regression on $W$ produces **attenuated** (biased-toward-zero) estimates.

!!! abstract "The idea"
    Estimate $\;\mathbb{E}[X \mid W]\;$ — the most likely true input given the noisy observation — and train on those calibrated inputs instead.

---

## The Attenuation Problem

When inputs have additive Gaussian measurement error:

$$W = X + U, \qquad U \sim \mathcal{N}(0,\, \Sigma_u)$$

the naive OLS estimate of $\beta$ is biased:

$$\hat\beta_{\text{naive}} = \underbrace{\Sigma_x(\Sigma_x + \Sigma_u)^{-1}}_{\Lambda\;\text{(reliability matrix)}} \cdot \beta_{\text{true}}$$

Every entry of the reliability matrix $\Lambda$ satisfies $0 \leq \Lambda_{ij} \leq 1$, so coefficients are **shrunk toward zero**.

---

## The RC Correction

RC estimates calibrated inputs via the conditional expectation:

$$\boxed{\; X_{\text{cal}} = \bar{W} + \Lambda \cdot (W - \bar{W}) \;}$$

where:

| Symbol | Definition |
|:------:|:-----------|
| $\bar{W}$ | Sample mean of observed inputs |
| $\Sigma_w = \text{Cov}(W)$ | Observed covariance |
| $\hat\Sigma_x = \Sigma_w - \Sigma_u$ | Estimated signal covariance (clamped PSD) |
| $\Lambda = \hat\Sigma_x\,(\hat\Sigma_x + \Sigma_u)^{-1}$ | Reliability matrix |

---

## Usage

```python
import torch
from torchregress.algorithms import RegressionCalibration

# Known measurement error (scalar, vector, or covariance matrix)
rc = RegressionCalibration(sigma_u=0.5)

# Fit on observed noisy inputs, then transform
X_calibrated = rc.fit_transform(X_observed)

# Or separate fit / transform steps for new data
rc.fit(X_observed)
X_cal_new = rc.transform(X_new)
```

| `sigma_u` format | Interpretation |
|:-----------------|:---------------|
| `float` | Isotropic noise — same std for all features |
| 1-D `Tensor` | Per-feature standard deviations |
| 2-D `Tensor` | Full noise covariance matrix $\Sigma_u$ |

---

## When to Use

!!! tip "Good fit for RC"
    - **Known** measurement uncertainties on inputs
    - Noise is approximately **Gaussian**
    - You want a **fast, analytical** correction (no retraining)

!!! warning "Limitations"
    - Assumes the relationship between $X$ and $Y$ is approximately **linear** in the correction
    - For nonlinear models, [SIMEX](simex.md) is more flexible

---

## References

| # | Reference |
|:-:|:----------|
| 1 | R.J. Carroll, D. Ruppert, L.A. Stefanski, C.M. Crainiceanu. *Measurement Error in Nonlinear Models*. 2nd ed., Chapman & Hall/CRC, **2006**. |
| 2 | W.A. Fuller. *Measurement Error Models*. Wiley, **1987**. |
