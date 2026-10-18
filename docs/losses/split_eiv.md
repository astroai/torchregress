# Split-Measurement & Corrected-Score EIV Losses

> ← [EIV losses](eiv.md) | [Noisy labels](noisy_labels.md) → [Forward inversion](../methods/algorithms/forward_inversion.md)

When the input features are measured with error, the ordinary squared error **attenuates** the regression: the model learns a shrunken map because the noisy feature looks like a noisier response than it really is. Two classical errors-in-variables (EIV) constructions remove that bias *inside the loss*, without changing the model:

* **Split-measurement loss** — use two independent noisy measurements of the same object and multiply their residuals. The cross term averages to zero, so no noise covariance is needed.
* **Corrected-score loss** — subtract the measurement-error-induced curvature from the squared residual (Stefanski 1989; Nakamura 1990). This is the *opposite* sign of the usual TLS/ODR-style gradient penalty, and it does not flatten the model.

Both are implemented as losses that take the model as a constructor argument, exactly like the other EIV losses in [`torchregress.losses`](eiv.md).

!!! abstract "The idea"
    For a linear model, the noisy residual is biased by $+\operatorname{tr}(\Sigma_x J^\top J)$. The corrected score subtracts exactly that quantity, and the split loss arranges for the same quantity to cancel in expectation through the product of two independent-errors residuals.

---

## 1. Split-measurement loss

Let $x_i$ be the latent clean input and $x_i^{(1)} = x_i + e_i^{(1)}$, $x_i^{(2)} = x_i + e_i^{(2)}$ two conditionally independent measurements with zero-mean errors and $\mathbb{E}[e^{(1)}(e^{(2)})^\top] = 0$. The per-sample loss is the product of the two-view residuals:

$$\mathcal{L}_i^{\text{split}} = \frac{1}{M}\sum_{m=1}^{M}\bigl(y_{i,m} - f_m(x_i^{(1)})\bigr)\bigl(y_{i,m} - f_m(x_i^{(2)})\bigr).$$

With $r_i = y_i - f(x_i)$ the noiseless residual, the expectation over the two independent errors satisfies, for a model linear in $x$,

$$\mathbb{E}\bigl[\mathcal{L}_i^{\text{split}}\bigr] = \frac{1}{M}\lVert y_i - f(x_i)\rVert^2,$$

so the loss sees the *noiseless* squared error: the $+\sigma_x^2$ variance inflation of the ordinary MSE is cancelled by the cross-product structure. No estimate of $\Sigma_x$ is required — only a second independent measurement (repeat exposures, multi-epoch photometry, split spectra).

!!! warning "Exact only for models linear in the features"
    For non-linear $f$ the cross-product expectation retains the squared *smoothing bias* $\lVert \mathbb{E}[f(x+e)] - f(x)\rVert^2$, so the correction is exact only to first order in the noise. The split loss still has the right gradient direction, but a strongly non-linear model should be paired with a corrected-score or latent-variable treatment. Independent **target** noise adds $+\sigma_y^2$ to the expected loss, exactly as it does for the ordinary squared error.

### API

```python
import torch
import torch.nn as nn
from torchregress.losses import SplitMeasurementLoss

model = nn.Linear(4, 1)
loss_fn = SplitMeasurementLoss(model)          # no sigma_x needed
loss = loss_fn(x_first, y, y_pred_second=x_second)
```

| Symbol | Signature |
|:-------|:----------|
| `SplitMeasurementLoss` | `(model, reduction="mean", eps=1e-3)`; `forward(y_pred, target, y_pred_second=None, mask=None, weights=None, **kwargs)` |
| `split_measurement_loss` | `(model, x_first, target, x_second, reduction="mean")` |

---

## 2. Corrected-score loss

For a normally distributed measurement error $\varepsilon \sim \mathcal N(0, \Sigma_x)$ and a model $f$ evaluated at the *observed* $w = x + \varepsilon$, the corrected score is

$$\mathcal{L}_i^{\text{corr}} = \lVert y_i - f(w_i)\rVert^2 - \operatorname{tr}\!\left(J_i\,\Sigma_x\,J_i^\top\right), \qquad J_i = \left.\frac{\partial f}{\partial w}\right|_{w_i},$$

which in the diagonal-covariance case is the familiar

$$\mathcal{L}_i^{\text{corr}} = \lVert y_i - f(w_i)\rVert^2 - \sum_{m=1}^{M}\sum_{d=1}^{D}\sigma_d^2\left(\frac{\partial f_m}{\partial w_d}\right)^2.$$

This is an unbiased estimating equation: its expectation equals the noiseless squared error (for the linear model exactly; for non-linear models up to higher-order terms), so the minimiser is not attenuated. Note the sign: TLS/ODR-style losses *add* a gradient penalty, which flattens $f$ and actually increases attenuation when $\sigma_x$ is large. The corrected score **subtracts** the error contribution instead.

!!! warning "The corrected score can be negative"
    The subtraction makes individual corrected losses negative when the measurement noise is large relative to the residual. This is expected for an unbiased estimating equation. Use the default `stabilize="none"` for statistical correctness; `stabilize="clamp"` clamps per-sample losses at zero (biased, but can stabilise training when the noise is extreme). The model must also be `torch.func.jacrev` / `vmap`-safe: no batch-dependent BatchNorm, no stochastic dropout in the forward pass.

### API

```python
from torchregress.losses import CorrectedScoreEIVLoss

loss_fn = CorrectedScoreEIVLoss(
    model,
    sigma_x=0.3,          # scalar, (D,), (D, D), or per-sample (N, D)/(N, D, D)
    stabilize="none",     # statistically unbiased default; "clamp" for stability
)
loss = loss_fn(x_observed, y)
```

| Symbol | Signature |
|:-------|:----------|
| `CorrectedScoreEIVLoss` | `(model, sigma_x, reduction="mean", eps=1e-3, stabilize="none", jacobian_chunk_size=None)`; `forward(y_pred, target, mask=None, weights=None, **kwargs)` |
| `corrected_score_eiv_loss` | `(model, y_pred, target, sigma_x, reduction="mean", eps=1e-3, stabilize="none", jacobian_chunk_size=None)` |

---

## When to use which

| Method | Extra data needed | Noise model needed | Exactness | Cost per step |
|:-------|:------------------|:-------------------|:----------|:--------------|
| `SplitMeasurementLoss` | Two independent measurements per object | None | Exact for linear $f$; first order otherwise | 2 forward passes |
| `CorrectedScoreEIVLoss` | One measurement | $\Sigma_x$ (std/covariance) | Unbiased estimating equation for linear $f$; small bias otherwise | 1 forward pass + Jacobian (≈ $M$ backward/forward-mode passes) |
| Plain MSE | One measurement | None | Biased (attenuated) | 1 forward pass |
| [Regression Calibration](../methods/algorithms/rc.md) | One measurement | $\Sigma_x$ + Gaussian feature model | Approximate, needs a noise model | 1 forward pass after preprocessing |
| [SIMEX](../methods/algorithms/simex.md) | One measurement | $\Sigma_x$ | Extrapolation-dependent | $K$ model fits |
| [LatentNN](../methods/algorithms/latentnn.md) | One measurement | $\Sigma_x$ | Exact latent-variable MLE | 1 fit with $N\times D$ latent parameters |
| [Forward inversion](../methods/algorithms/forward_inversion.md) | Accurate labels | $\sigma_x$ (likelihood) | Generative MLE; no attenuation in $y$ | Surrogate fit + inversion |

!!! tip "Practical choice"
    * You have repeat observations (multi-epoch photometry, split spectra, DESI/SDSS repeats) → `SplitMeasurementLoss` is the cheapest and needs no noise calibration.
    * You have one spectrum with well-calibrated per-pixel errors → `CorrectedScoreEIVLoss` with a diagonal $\sigma_x$ vector.
    * Your labels are much more accurate than the features and the label space is low-dimensional → consider [forward generative inversion](../methods/algorithms/forward_inversion.md) instead.

---

## Example

A linear forward model $y = 1.5 + 2x$ observed through noisy features. With the *correct* model, plain MSE on a noisy view is biased high by $b^2\sigma^2$, while both corrected losses recover the noiseless squared error:

```python
import torch
import torch.nn as nn
from torchregress.losses import CorrectedScoreEIVLoss, SplitMeasurementLoss

torch.manual_seed(0)
n, sigma = 4000, 0.5
x = torch.randn(n, 1)
y = 1.5 + 2.0 * x + 0.1 * torch.randn(n, 1)
x1 = x + sigma * torch.randn(n, 1)   # first noisy exposure
x2 = x + sigma * torch.randn(n, 1)   # independent second exposure

model = nn.Linear(1, 1)
with torch.no_grad():
    model.weight.fill_(2.0)
    model.bias.fill_(1.5)
    plain = nn.MSELoss()(model(x1), y)
    corrected = CorrectedScoreEIVLoss(model, sigma_x=sigma)(x1, y)
    split = SplitMeasurementLoss(model)(x1, y, y_pred_second=x2)
    clean = nn.MSELoss()(model(x), y)

print(f"noiseless={clean:.4f}  plain={plain:.4f}  "
      f"corrected={corrected:.4f}  split={split:.4f}")
# noiseless ~ 0.01, plain ~ 1.01 (= clean + 4 sigma^2),
# corrected ~ 0.00, split ~ 0.00
```

For training, pass the loss to any optimizer with the model parameters; the EIV losses evaluate the model internally:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = CorrectedScoreEIVLoss(model, sigma_x=sigma)
for _ in range(200):
    optimizer.zero_grad()
    loss = loss_fn(x1, y)
    loss.backward()
    optimizer.step()
# model.weight ~ 2, model.bias ~ 1.5: no attenuation
```

---

## Caveats

!!! warning "Assumptions"
    * **Independent views**: split-measurement correction requires the two measurements to have independent errors. Shared calibration-systematics or a common template error break the cancellation. In practice use genuinely independent exposures or split the spectrum in wavelength.
    * **Target noise is not corrected**: both losses add $+\sigma_y^2$ to the expected risk (same as MSE); they address *input* error only.
    * **Non-linear models**: split loss retains the smoothing bias squared, corrected score retains higher-order terms; both are exact for models linear in the features.
    * **Full covariance**: `CorrectedScoreEIVLoss` accepts a full $\Sigma_x$ and evaluates the trace exactly; the split loss needs no covariance at all.
    * **Batch-dependent layers**: the corrected score differentiates the model with respect to its input, so BatchNorm batch statistics and dropout randomness break the assumptions. Train with them in evaluation mode or avoid them for EIV work.

---

## References

| # | Reference |
|:-:|:----------|
| 1 | Stefanski, L. A. (1989). Unbiased estimation of a nonlinear function of a normal mean with application to measurement error models. *Communications in Statistics — Theory and Methods*, 18(12), 4335–4358. [doi:10.1080/03610928908830159](https://doi.org/10.1080/03610928908830159) |
| 2 | Stefanski, L. A. (1990). Correcting data for measurement error in generalized linear models. *Statistics & Probability Letters*, 9(2), 123–128. [doi:10.1016/0167-7152(90)90071-2](https://doi.org/10.1016/0167-7152(90)90071-2) |
| 3 | Nakamura, T. (1990). Corrected score function for errors-in-variables models: methodology and application to generalized linear models. *Biometrika*, 77(1), 127–137. [doi:10.1093/biomet/77.1.127](https://doi.org/10.1093/biomet/77.1.127) |
| 4 | Fuller, W. A. (1987). *Measurement Error Models*. Wiley. (Repeated-measurement and instrumental-variable EIV.) |
| 5 | Carroll, R. J., Ruppert, D., Stefanski, L. A., & Crainiceanu, C. M. (2006). *Measurement Error in Nonlinear Models: A Modern Perspective* (2nd ed.). Chapman & Hall/CRC. |

## See also

* [EIV losses](eiv.md) — the other measurement-error losses.
* [Forward generative inversion](../methods/algorithms/forward_inversion.md) — the $y \to x$ alternative.
* [Regression Calibration](../methods/algorithms/rc.md) and [SIMEX](../methods/algorithms/simex.md) — algorithm-level corrections.
* [Noisy features / measurement error](../guide/method-selection.md) — method-selection matrix.
