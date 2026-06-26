# User Guide

This guide is organized by **what you want to achieve**, not by module. Each
section below maps a regression problem to the right tool, with code snippets
and links to deeper references. If you know your problem, jump straight to it.
If you're exploring, read top-to-bottom — the sections build from simple
(heteroscedastic regression) to advanced (measurement error, shift adaptation).

!!! tip "New here?"
    Start with [Core Concepts](../getting-started/concepts.md) for the
    vocabulary (aleatoric vs. epistemic, proper scoring rules, conformal
    coverage), then come back here to find the right tool for your task.

---

## Predicting with Uncertainty

Most regression models produce a single point prediction $\hat{y}$.
But in practice you also need to know **how confident** the model is.
The sections below walk you from simplest (heteroscedastic head) to most
comprehensive (full ensemble decomposition).

### Heteroscedastic variance (simplest)

Predict mean and variance per sample using [`GaussianNLLLoss`](../api/losses.md):

```python
from torchregress.losses import GaussianNLLLoss

loss_fn = GaussianNLLLoss()

# Model outputs 2× target dim: [mean, log_var]
out = model(x)          # (batch, 2)
loss = loss_fn(out, y)  # trains both mean and variance
```

The predicted variance $\sigma^2(x)$ captures **aleatoric** uncertainty — irreducible noise that varies with input.

!!! tip "When is this enough?"
    If you only need to know "how noisy is this region of input space", GaussianNLLLoss is sufficient and cheap.  For model uncertainty (epistemic), add an [ensemble](../methods/ensemble/index.md).

### Ensemble uncertainty (recommended)

Train $M$ independent models and examine disagreement:

```python
from torchregress.ensemble import HeteroscedasticEnsembleModel

ensemble = HeteroscedasticEnsembleModel(base_model=MyModel, ensemble_size=5)
result = ensemble.predict(x_test)
aleatoric = result["aleatoric_variance"]   # data noise
epistemic = result["epistemic_variance"]   # model ignorance
```

→ See [Ensembles for Uncertainty](../methods/ensemble/index.md) for full details.
For terminology and edge cases such as quantile ensembles, see
[Uncertainty Decomposition](uncertainty-decomposition.md).

### Single-pass decomposition

If you can't afford an ensemble, evidential regression provides both uncertainty types in one forward pass — see [Evidential Regression](../losses/advanced.md). This page continues with other common tasks below.

---

## Handling Outliers

Standard MSE is highly sensitive to outliers — a single extreme point can dominate the gradient.

| Robustness | Loss | When to use |
|:-----------|:-----|:------------|
| ⭐⭐ | [`WeightedHuberLoss`](../api/losses.md) | Quick fix, moderate outliers |
| ⭐⭐⭐ | [`CauchyLoss`](../api/losses.md) | Logarithmic suppression |
| ⭐⭐⭐⭐ | [`TukeyBiweightLoss`](../api/losses.md) | Complete rejection beyond threshold |

→ See [Robust Losses](../losses/robust.md) for the full family.

---

## Prediction Intervals

### Quantile regression (training time)

```python
from torchregress.losses import MultiQuantileLoss

loss_fn = MultiQuantileLoss(quantiles=[0.05, 0.5, 0.95])
# Model outputs 3 values per target
```

→ See [MultiQuantileLoss API](../api/losses.md).

### Conformal prediction (post-hoc, guaranteed coverage)

Wrap **any** trained model with a calibration procedure:

```python
from torchregress.losses import CQR

cqr = CQR(alpha=0.1)  # target 90% coverage
cqr.calibrate(y_pred_cal, y_cal)
lower, upper = cqr.predict_interval(y_pred_test)
```

→ See [CQR API](../api/losses.md).

!!! success "Finite-sample guarantee"
    $$P(Y_{n+1} \in [\text{lower}, \text{upper}]) \geq 1 - \alpha$$

→ See [Conformal Prediction](../methods/conformal/index.md).

---

## Dealing with Imbalanced Targets

When some target ranges are under-represented (e.g., rare extreme values):

```python
from torchregress.losses import DensityWeightedLoss, FocalRLoss

# Inverse-density reweighting
loss_fn = DensityWeightedLoss(base_loss=GaussianNLLLoss())

# Focus on hard (rare) examples
loss_fn = FocalRLoss(gamma=2.0)
```

→ See [Imbalanced Regression](../losses/imbalanced.md).

---

## Noisy, Censored, or Ordinal Data

| Problem | Loss | Link |
|:--------|:-----|:-----|
| Known target noise | `NoisyTargetGaussianNLL` | [Noisy Labels](../losses/noisy_labels.md) |
| Censored / survival | `CensoredGaussianNLLLoss` | [Censored](../losses/censored.md) |
| Ordered discrete | `CORALLoss` | [Ordinal](../losses/ordinal.md) |

---

## Measurement Error in Inputs

When features $X$ are measured with known noise $\sigma_u$:

=== "Quick fix: Regression Calibration"

    ```python
    from torchregress.algorithms import RegressionCalibration

    rc = RegressionCalibration(sigma_u=0.3)
    X_clean = rc.fit_transform(X_noisy)
    ```

=== "Nonlinear models: SIMEX"

    ```python
    from torchregress.algorithms import SIMEX

    simex = SIMEX(model_factory, train_func, sigma_u=0.3)
    simex.fit(X_noisy, y)
    ```

=== "Loss-based: EIV losses"

    ```python
    from torchregress.losses import StructuralEIVLoss

    loss_fn = StructuralEIVLoss(error_ratio=0.5)
    ```

→ See [RC](../methods/algorithms/rc.md) · [SIMEX](../methods/algorithms/simex.md) · [EIV Losses](../losses/eiv.md)

---

## Multi-Target Regression

When you need to predict multiple continuous outcomes $\mathbf{y} \in \mathbb{R}^K$ jointly:

- **Parametric (Gaussian)**: Use [`MultivariateGaussianLoss`](../api/losses.md) for full covariance modeling or [`LowRankGaussianLoss`](../api/losses.md) for high-dimensional targets.
- **Non-Parametric**: Use [`NormalizingFlowLoss`](../api/losses.md) or [`MDNLoss`](../api/losses.md) for multimodal joint distributions.
- **Interval Calibration**: Use `MultiTargetConformal` for coordinate-wise coverage guarantees.

→ See [Multi-Target Regression](multi-target-regression.md).

---

## Post-Hoc Calibration

After training, calibrate your model's uncertainty estimates on a held-out set:

```python
from torchregress.calibration import VarianceTemperatureScaler

scaler = VarianceTemperatureScaler()
scaler.fit(pred_mean_cal, pred_var_cal, y_cal)
calibrated_var = scaler.transform(pred_var_test)
```

→ See [Post-Hoc Calibration](../methods/calibration.md).

---

## Common Acronyms

| Acronym | Meaning |
|:--------|:--------|
| NLL | Negative Log-Likelihood |
| CRPS | Continuous Ranked Probability Score |
| OOD | Out-of-Distribution |
| ECE | Expected Calibration Error |
| MPIW | Mean Prediction Interval Width |
| PICP | Prediction Interval Coverage Probability |
| RMSE / MAE | Root Mean Squared Error / Mean Absolute Error |
| MDN | Mixture Density Network |
| SWAG | Stochastic Weight Averaging — Gaussian |
| EIV | Error-in-Variables |
| BNN | Bayesian Neural Network |

---

## What Next?

- [Loss Functions catalogue](../losses/index.md) — every loss with formulas and use cases
- [Examples by topic](../examples/index.md) — runnable comparisons with benchmarks
- [Mathematical Foundations](math/index.md) — derivations of scoring rules and decompositions
- [API Reference](../api/index.md) — complete function signatures
- [Method Selection Matrix](method-selection.md) — task-first capability matrix

If you've read this far top-to-bottom, you've covered the most common
regression tasks. The remaining guide pages dive deeper into:

- [Uncertainty Decomposition](uncertainty-decomposition.md) — contracts and taxonomy
- [Multi-Target Regression](multi-target-regression.md) — correlated outputs
- [Performance Tuning](performance.md) — training speed, mixed precision, profiling
- [Debugging & Diagnostics](debugging.md) — common failure modes and their fixes
