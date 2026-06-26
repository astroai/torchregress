# Regression Calibration (RC)

> ← [SIMEX](simex.md) | [LatentNN](latentnn.md) →

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
    - **PSD Clamping**: If measurement noise exceeds the observed signal ($\Sigma_u > \Sigma_w$), the estimated signal covariance $\hat\Sigma_x = \Sigma_w - \Sigma_u$ becomes negative. The implementation clamps to a small positive value, but this means RC cannot recover when noise dominates signal.
    - **Homoscedastic noise only**: RC assumes a single measurement error covariance $\Sigma_u$ for all samples. It cannot handle heteroscedastic measurement error where different samples have different noise levels.

---

## Limitations

1. **Linear correction only**: RC applies a linear shrinkage $\Lambda \cdot (W - \bar{W})$. For nonlinear $f(X)$, the debiasing is approximate at best. Use [SIMEX](simex.md) or [EIV losses](../../losses/eiv.md) for nonlinear models.
2. **PSD clamping**: If measurement noise exceeds the observed signal ($\Sigma_u > \Sigma_w$), the estimated signal covariance $\hat\Sigma_x = \Sigma_w - \Sigma_u$ becomes negative. RC clamps to a small positive value but cannot recover when noise dominates signal.
3. **Homoscedastic noise only**: RC assumes a single $\Sigma_u$ for all samples. Heteroscedastic measurement error is not supported.
4. **Gaussian noise assumption**: The conditional expectation $\mathbb{E}[X \mid W]$ is exact only under joint Gaussian $(X, W)$. For non-Gaussian noise, RC is a best-linear predictor, not the true conditional expectation.
5. **Known $\Sigma_u$ required**: RC requires the measurement error covariance as a known input. It does not estimate $\Sigma_u$ from data.

## Recommendations

- **Default for linear models**: RC is the fastest, simplest EIV correction. Apply it first, then validate on held-out data before considering more complex methods.
- **For nonlinear models**: If RC's linear correction is insufficient (validate on test set), upgrade to [SIMEX](simex.md) (flexible but expensive) or [LatentNN](latentnn.md) (joint optimisation).
- **Validate the correction**: Compare RC-corrected predictions against a naive baseline on clean held-out data (if available) or via simulation with known ground truth.
- **Combine with robust losses**: RC corrects input noise. Pair with [Robust losses](../../losses/robust.md) to handle outlier targets simultaneously.

## Next steps

- [SIMEX](simex.md) — simulation-extrapolation for nonlinear models where RC's linear correction is insufficient
- [LatentNN](latentnn.md) — joint optimisation of latent clean inputs and model parameters
- [EIV losses](../../losses/eiv.md) — functional and structural error-in-variables losses at the loss level
- [Error-Aware Encoding](error_aware.md) — quality-gated feature engineering from known measurement noise

---

## References

| # | Reference |
|:-:|:----------|
| 1 | R.J. Carroll, D. Ruppert, L.A. Stefanski, C.M. Crainiceanu. *Measurement Error in Nonlinear Models*. 2nd ed., Chapman & Hall/CRC, **2006**. |
| 2 | W.A. Fuller. *Measurement Error Models*. Wiley, **1987**. |
