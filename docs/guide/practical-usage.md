# Practical Usage Guide

This guide maps **real-world data pathologies** to specific torchregress
methods, with quantitative tradeoffs and mathematical justification for each
choice.

For the principled workflow (baseline → probabilistic → robust), see
[Best Practices](best-practices.md). For the full capability matrix, see
[Method Selection Matrix](method-selection.md).

---

## Loss Selection by Data Pathology

### Heteroscedastic Noise (variance depends on $x$)

The canonical approach is to predict both $\mu(x)$ and $\log\sigma^2(x)$,
optimising the Gaussian negative log-likelihood:

$$
\mathcal{L}_{\text{NLL}} = \frac{1}{2}\sum_{i=1}^n \left[\log\sigma^2(x_i) + \frac{(y_i - \mu(x_i))^2}{\sigma^2(x_i)}\right] + \text{const}
$$

This is a **proper scoring rule** — it is uniquely minimised by the true
conditional distribution $p(y \mid x)$ (Gneiting & Raftery 2007).

- Use [`GaussianNLLLoss`](../api/losses.md) for unimodal, heteroscedastic data.
- Use [`BetaNLLLoss`](../api/losses.md) when Gaussian NLL exhibits variance
  collapse early in training (Skafte et al. 2019).
- For heteroscedasticity with outliers, combine a robust mean term with a
  variance head: see [Robust losses](../losses/robust.md).

### Outliers & Heavy Tails

The choice of robust loss is governed by the **influence function**
$\psi(r) = \rho'(r)$, which measures how much a single residual
$r = y - \hat{y}$ affects the parameter estimate (Huber 1964):

| Loss | Influence $\psi(r)$ | Breakdown | Best For | Reference |
|:-----|:--------------------|:----------|:---------|:----------|
| Huber | $\max(-c, \min(c, r))$ | Linear tail | 5–10% outliers | Huber (1964) |
| Cauchy | $2r / (1 + r^2/c^2)$ | Redescending | 10–25% | |
| Tukey | $r(1 - r^2/c^2)^2 \cdot \mathbf{1}_{|r| \leq c}$ | Zero after $c$ | >25% | Tukey (1977) |
| Barron | $\alpha$-parametrized continuum | Tunable | Unknown regime | Barron (2019) |

```python
# Rule of thumb:
# <10% outliers  → WeightedHuberLoss(delta=1.0)
# 10–25%         → CauchyLoss(c=1.0)
# >25%           → TukeyBiweightLoss(c=4.685)
# unknown        → AdaptiveRobustLoss()  # learn alpha
```

For a complete treatment with influence-function plots, see the
[Robust losses guide](../losses/robust.md).

### Count Data & Positive Targets

The mean–variance relationship determines the appropriate loss:

$$
\begin{cases}
\text{Var}(y) = \mu & \text{Poisson} \\
\text{Var}(y) = \mu + \mu^2/r & \text{Negative Binomial} \\
\text{Var}(y) = \phi\mu^p & \text{Tweedie} \quad (0 < p < 2) \\
\text{Var}(y) = \phi\mu^3 & \text{Inverse Gaussian}
\end{cases}
$$

| Data Pattern | Loss | Reference |
|:-------------|:-----|:----------|
| Counts, $\text{Var} \approx \mu$ | [`PoissonDevianceLoss`](../api/losses.md) | Nelder & Wedderburn (1972) |
| Overdispersed counts | [`NegativeBinomialNLLLoss`](../api/losses.md) | |
| Zeros + continuous positives | [`TweedieLoss`](../api/losses.md) | Tweedie (1984) |
| Positive, right-skewed | [`GammaLoss`](../api/losses.md) | |
| Excess zeros | [`ZeroInflatedPoissonNLLLoss`](../api/losses.md) | Lambert (1992) |

See [Poisson & Tweedie guide](../losses/poisson_tweedie.md) for details.

### Censored Targets

When the target is only partially observed — right-censored (sensor clipped),
left-censored (detection limit), or interval-censored — the likelihood is:

$$
\mathcal{L} = \begin{cases}
\log p(y_i \mid x_i) & \text{fully observed} \\
\log \Phi\!\left(\frac{c - \mu_i}{\sigma_i}\right) & \text{right-censored at } c \\
\log \left[1 - \Phi\!\left(\frac{c - \mu_i}{\sigma_i}\right)\right] & \text{left-censored at } c \\
\log \left[\Phi\!\left(\frac{u_i - \mu_i}{\sigma_i}\right) - \Phi\!\left(\frac{l_i - \mu_i}{\sigma_i}\right)\right] & \text{interval-censored } [l_i, u_i]
\end{cases}
$$

where $\Phi$ is the standard Gaussian CDF (Tobin 1958).

- Use [`CensoredGaussianNLLLoss`](../api/losses.md) for Gaussian censored regression.
- Use [`CensoredQuantileLoss`](../api/losses.md) for a non-parametric alternative.
- Use [`AFTLoss`](../api/losses.md) for survival / time-to-event modelling (Cox 1972).

### Multimodal Targets

When a given $x$ can produce multiple distinct $y$ values, Gaussian losses
are misspecified (they average modes). Use mixture models:

$$
p(y \mid x) = \sum_{k=1}^K \pi_k(x)\ \mathcal{N}\!\bigl(y \mid \mu_k(x),\, \sigma_k^2(x)\bigr)
$$

- [`MDNLoss`](../api/losses.md) (Mixture Density Network) for known $K$ (Bishop 1994).
- [`NormalizingFlowLoss`](../api/losses.md) for flexible, non-parametric densities
  (Rezende & Mohamed 2015).

See [Multimodal targets guide](multi-target-regression.md).

### Imbalanced Targets

When the target distribution is heavily skewed, loss reweighting methods
target specific tail-region tradeoffs (Ren et al. 2019):

| Loss | Strategy | Tail Focus |
|:-----|:---------|:-----------|
| `DensityWeightedLoss` | Inverse target-density | Moderate |
| `FocalRLoss` | Hard-example focus | Strong |
| `LDSLoss` | Label Distribution Smoothing | Strong |
| `PropensityWeightedLoss` | Inverse propensity scores | Moderate |

See [Imbalanced regression guide](../losses/imbalanced.md).

### Measurement Error in Inputs

When $x$ is observed with noise $x_{\text{obs}} = x_{\text{true}} + \epsilon$,
standard regression produces attenuated estimates (Carroll et al. 2006):

| Method | Requires | Tradeoff |
|:-------|:---------|:---------|
| `StructuralEIVLoss` | Known $\sigma_x / \sigma_y$ | Simple, assumes constant ratio |
| `FunctionalEIVLoss` | Per-sample $\sigma_x$ | Handles heteroscedastic input noise |
| `OrthogonalDistanceRegressionLoss` | Nothing extra | General, slower |
| Regression Calibration (RC) | Validation data | Bias correction via surrogate |
| SIMEX | Error distribution | Simulation-based, flexible |

See [EIV losses](../losses/eiv.md), [RC](../methods/algorithms/rc.md),
[SIMEX](../methods/algorithms/simex.md).

---

## Prediction Intervals: Distributional vs. Distribution-Free

| Method | Assumes | Guarantee | Width Adaptivity | Reference |
|:-------|:--------|:----------|:-----------------|:----------|
| Gaussian NLL | Gaussian likelihood | Asymptotic | Heteroscedastic $\sigma(x)$ | Nix & Weigend (1994) |
| Quantile regression | None for each $q$ | Asymptotic consistent $q$ | Per-quantile | Koenker & Bassett (1978) |
| Split conformal | Exchangeability | Finite-sample $1-\alpha$ | Via CQR | Vovk et al. (2005), Romano et al. (2019) |
| Ensemble quantiles | None | Heuristic | Decomposable | Lakshminarayanan et al. (2017) |

**Recommendation:** use quantile regression + CQR for robust, adaptive
intervals with *finite-sample coverage*. Use Gaussian NLL when you need
a full density (not just intervals) and trust the Gaussian assumption.

---

## Evaluation Checklist

Before trusting a model's uncertainty estimates, verify:

- **RMSE / NLL / CRPS** all improve on holdout (not just one).
- **Reliability diagram**: 90% intervals contain $\approx 90\%$ of holdout data.
- **PIT histogram**: uniform for well-calibrated densities.
- **OOD sensitivity**: uncertainty increases on out-of-distribution inputs.
- **No variance collapse**: $\sigma(x)$ is not near zero everywhere.
- **Residuals**: no obvious structure (heteroscedasticity, curvature).

For implementation, see [Visualization Methods](../methods/visualization.md)
and [Calibration Metrics](../metrics/calibration.md).

---

## References

| # | Reference |
|:-:|:----------|
| 1 | Gneiting, T. & Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. *JASA*, 102(477), 359–378. |
| 2 | Skafte, N., Jørgensen, M. & Hauberg, S. (2019). Reliable Training and Estimation of Variance Networks. *NeurIPS*. |
| 3 | Huber, P. J. (1964). Robust Estimation of a Location Parameter. *Annals of Mathematical Statistics*, 35(1), 73–101. |
| 4 | Tobin, J. (1958). Estimation of Relationships for Limited Dependent Variables. *Econometrica*, 26(1), 24–36. |
| 5 | Cox, D. R. (1972). Regression Models and Life-Tables. *J. Royal Statistical Society B*, 34(2), 187–202. |
| 6 | Bishop, C. M. (1994). Mixture Density Networks. *Technical Report*, Aston University. |
| 7 | Rezende, D. J. & Mohamed, S. (2015). Variational Inference with Normalizing Flows. *ICML*. |
| 8 | Koenker, R. & Bassett, G. (1978). Regression Quantiles. *Econometrica*, 46(1), 33–50. |
| 9 | Carroll, R. J., Ruppert, D., Stefanski, L. A. & Crainiceanu, C. M. (2006). *Measurement Error in Nonlinear Models* (2nd ed.). Chapman & Hall. |
| 10 | Vovk, V., Gammerman, A. & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer. |
| 11 | Romano, Y., Patterson, E. & Candès, E. (2019). Conformalized Quantile Regression. *NeurIPS*. |
| 12 | Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *NeurIPS*. |
| 13 | Ren, J., Liu, P. J., Fertig, E., Snoek, J., Poplin, R., DePristo, M. A., ... & Angermueller, C. (2019). Likelihood Ratios for Out-of-Distribution Detection. *NeurIPS*. |
| 14 | Nix, D. A. & Weigend, A. S. (1994). Estimating the Mean and Variance of the Target Probability Distribution. *IEEE ICNN*. |
