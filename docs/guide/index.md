# User Guide

This guide is organized by **what you want to achieve**, not by module. Find your problem below and follow the links to the right tools.

---

## Predicting with Uncertainty

Most regression models produce a single point prediction $\hat{y}$.
But in practice you also need to know **how confident** the model is.

### Heteroscedastic variance (simplest)

Predict mean and variance per sample using `GaussianNLLLoss`:

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

→ See [Ensemble & UQ](../methods/ensemble/index.md) for full details.

### Single-pass decomposition

If you can't afford an ensemble, evidential regression provides both uncertainty types in one forward pass:

```python
from torchregress.losses import EvidentialRegressionLoss

loss_fn = EvidentialRegressionLoss()
# Model outputs 4 values: [γ, ν, α, β] per target
```

→ See [Evidential Regression](../losses/advanced.md).

---

## Handling Outliers

Standard MSE is highly sensitive to outliers — a single extreme point can dominate the gradient.

| Robustness | Loss | When to use |
|:-----------|:-----|:------------|
| ⭐⭐ | `HuberLoss(delta=1.0)` | Quick fix, moderate outliers |
| ⭐⭐⭐ | `CauchyLoss(scale=1.0)` | Logarithmic suppression |
| ⭐⭐⭐⭐ | `TukeyBiweightLoss(c=4.685)` | Complete rejection beyond threshold |

→ See [Robust Losses](../losses/robust.md) for the full family.

---

## Prediction Intervals

### Quantile regression (training time)

```python
from torchregress.losses import MultiQuantileLoss

loss_fn = MultiQuantileLoss(quantiles=[0.05, 0.5, 0.95])
# Model outputs 3 values per target
```

### Conformal prediction (post-hoc, guaranteed coverage)

Wrap **any** trained model with a calibration procedure:

```python
from torchregress.losses import CQR

cqr = CQR(alpha=0.1)  # target 90% coverage
cqr.calibrate(y_pred_cal, y_cal)
lower, upper = cqr.predict_interval(y_pred_test)
```

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

## What Next?

- Browse the [Loss Functions catalogue](../losses/index.md)
- Explore [Examples](../examples/index.md) by topic
- Check the [Mathematical Foundations](math/index.md) for rigorous derivations
- See the [API Reference](../api/index.md) for complete function signatures
