# Best Practices

Building reliable regression models requires more than picking a loss function.
This guide covers the principled workflow — from baseline to deployment — with
rigorous evaluation and common failure modes.

---

## The 3-Phase Development Workflow

### Phase 1: Baseline (Day 1)

Start with simple point prediction. This establishes a performance floor and
catches data or architecture issues before investing in uncertainty modeling.

```python
loss_fn = WeightedHuberLoss(delta=1.0)  # robust default for mild outliers
```

- **Goal**: Establish a performance floor.
- **Check**: Can the model overfit a tiny subset (5–10 samples)? If not, check
  learning rate, architecture, or data pipeline.
- **Metric**: RMSE and $R^2$.

### Phase 2: Probabilistic Modeling (Day 2–3)

Switch to a heteroscedastic loss to estimate **aleatoric uncertainty** — the
irreducible noise in the data.

```python
loss_fn = GaussianNLLLoss()  # outputs [mean, log_var]
```

- **Goal**: Estimate input-dependent aleatoric uncertainty.
- **Check**: Does $\sigma(x)$ align with actual error magnitudes? Plot a
  [reliability diagram](../methods/calibration.md).
- **Metric**: NLL and CRPS — proper scoring rules that reward both accuracy
  and honest uncertainty [GneitingRaftery2007].

### Phase 3: Robustness & Refinement (Week 1+)

Add robustness (bounded-influence losses), epistemic uncertainty (ensembles),
or distribution-free guarantees (conformal prediction).

- **Goal**: Handle outliers, quantify *model uncertainty*, or guarantee coverage.
- **Check**: Compare losses on held-out data with known contamination.
- **Metric**: CRPS for overall distributional quality; calibration curves for
  interval honesty.

---

## Data Preparation

### Feature Scaling

Neural networks are sensitive to input scale. Always standardize $X$ before
training:

```python
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
```

### Target Scaling

For `GaussianNLLLoss`, scale targets $y$ to zero mean and unit variance so
the model's initial guess ($\mu \approx 0$, $\sigma \approx 1$) is reasonable.

---

## Numerical Stability

### Predict Log-Variance, Not Variance

Output $s = \log \sigma^2$, never $\sigma^2$ directly. The exponential
enforces positivity and prevents NaN gradients [KendallGal2017]:

```python
logvar = model(x)
var = torch.exp(logvar)       # always positive
std = torch.sqrt(var)
```

Initialise the log-variance head to predict $\log \sigma^2 \approx 0$
(i.e. $\sigma \approx 1$) so the model starts with a neutral prior.

### Variance Clamping

Gaussian NLL can diverge when predicted variance approaches zero. Enforce a
floor:

```python
loss_fn = GaussianNLLLoss(min_variance=1e-6)  # stabilises early training
```

---

## Evaluation Strategy

### Never Trust a Single Metric

| Metric | What It Measures | Weakness | Reference |
|:-------|:-----------------|:---------|:----------|
| **RMSE** / **MAE** | Point prediction accuracy | RMSE is outlier-sensitive; neither measures uncertainty quality | |
| **NLL** | Density fit quality | Can be "cheated" by overconfident models (variance collapse) | |
| **CRPS** | Accuracy + uncertainty jointly | Calibrated across all thresholds | [GneitingRaftery2007] |
| **PICP** / **MPIW** | Interval coverage vs. width | PICP alone can be trivially satisfied by wide intervals | |
| **Calibration Error** | Honesty of predicted probabilities | Marginal only; conditional calibration requires more data | [Kuleshov2018] |
| **AURC** | Selective prediction quality | Risk-coverage tradeoff | [GeifmanEl-Yaniv2019] |

**Rule of thumb:** report RMSE/NLL/CRPS + calibration diagram as the minimal
set for any probabilistic regression paper or report.

### Diagnostic Plots

- **Reliability diagram** (calibration curve): bin predicted probabilities
  vs. empirical frequencies. Deviations from the diagonal signal miscalibration.
- **PIT histogram**: probability integral transform values should be uniform
  if the predictive distribution is well-calibrated.

See [Visualization Methods](../methods/visualization.md) for implementation.

---

## Common Pitfalls

| Pitfall | Cause | Solution |
|:--------|:------|:---------|
| **NaN losses** | Variance → 0 or extreme outliers | Set `min_variance`, clamp logits, use robust loss for the mean term |
| **Variance collapse** | Model predicts $\sigma \to 0$ to exploit NLL density peak | Use Beta-NLL, weight-decay on the variance head, or early stopping [Skafte2019] |
| **Overconfident intervals** | Model misspecification or data leakage | Post-hoc calibration, temperature scaling [Guo2017] |
| **Underconfident intervals** | High epistemic uncertainty or poor fit | More data, better architecture, ensemble averaging |
| **Quantile crossing** | Multiple quantile heads unconstrained | Apply `NonCrossingSort` layer or crossover penalty |
| **Coverage ≠ density** | Conformal intervals are wide with poor predictors | Improve base model; conformal guarantees coverage, not sharpness [Vovk2005] |
| **Exchangeability violation** | Distribution shift at test time | Use weighted conformal or test-time adaptation [Tibshirani2019] |

---

## When to Reach for Each Method

| Goal | Method | When It Works | When It Fails |
|:-----|:-------|:--------------|:--------------|
| Point prediction | `WeightedHuberLoss` | Mild outliers (<10%) | Heavy contamination |
| Aleatoric uncertainty | `GaussianNLLLoss` | Heteroscedastic, unimodal data | Multimodal densities |
| Robustness | `CauchyLoss` / `TukeyBiweightLoss` | 10–25% outliers | Severe contamination + small $n$ |
| Coverage guarantees | `SplitConformal` + CQR | Exchangeable calibration set | Covariate shift |
| Epistemic decomposition | Deep ensemble / SWAG | Sufficient compute | Budget-constrained |
| Single-pass uncertainty | Evidential regression | Fast inference needed | Calibration can be poor |
| Multimodal densities | MDN / Normalizing flows | Multi-target, non-Gaussian | Requires more data |
| Measurement error | EIV losses / RC / SIMEX | Known or estimable $\sigma_x$ | Unknown error structure |

---

## References

1. [GneitingRaftery2007] Gneiting, T. & Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. *J. American Statistical Association*, 102(477), 359–378.
2. [KendallGal2017] Kendall, A. & Gal, Y. (2017). What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? *NeurIPS*.
3. [Kuleshov2018] Kuleshov, V., Fenner, N. & Ermon, S. (2018). Accurate Uncertainties for Deep Learning Using Calibrated Regression. *ICML*.
4. [Skafte2019] Skafte, N., Jørgensen, M. & Hauberg, S. (2019). Reliable Training and Estimation of Variance Networks. *NeurIPS*.
5. [Guo2017] Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML*.
6. [Vovk2005] Vovk, V., Gammerman, A. & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
7. [Tibshirani2019] Tibshirani, R. J., Foygel Barber, R., Candès, E. & Ramdas, A. (2019). Conformal Prediction Under Covariate Shift. *NeurIPS*.
8. [GeifmanEl-Yaniv2019] Geifman, Y. & El-Yaniv, R. (2019). SelectiveNet: A Deep Neural Network with an Integrated Reject Option. *ICML*.
9. [Huber1964] Huber, P. J. (1964). Robust Estimation of a Location Parameter. *Annals of Mathematical Statistics*, 35(1), 73–101.
10. [Lakshminarayanan2017] Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *NeurIPS*.
