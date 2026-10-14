# Tabular MLP

`torchregress.models` ships a strong-default neural network for medium-size
tabular regression: robust-scaled inputs, periodic numeric embeddings, SiLU
blocks with dropout, AdamW with a one-cycle schedule, early stopping on a
validation split, and target standardisation. It plugs into **any**
torchregress loss, so the same backbone gives point predictions, a
heteroscedastic Gaussian, or quantile intervals.

!!! abstract "What it is"
    Three small pieces:

    - [`TabularPreprocessor`](../api/models.md#tabularpreprocessor) — median/IQR scaling, smooth clipping, NaN imputation with missing-indicator columns.
    - [`TabularMLP`](../api/models.md#tabularmlp) — optional [`PeriodicEmbedding`](../api/models.md#periodicembedding) of every column followed by `Linear → activation → Dropout` blocks and a linear head of width `out_features`.
    - [`fit_tabular`](../api/models.md#fit_tabular) — the training recipe, returning a [`TabularFit`](../api/models.md#tabularfit) with restore-best-weights and a `predict` method that undoes target standardisation.

---

## When to use it

| Situation | Use |
|:----------|:----|
| Tabular data, a few thousand to a few hundred thousand rows, smooth targets | `TabularMLP` (this page) |
| Need `mean + log-variance`, quantiles or any loss with `forward(y_pred, target)` on a tabular input | `TabularMLP` with the matching `out_features` |
| Sharp axis-aligned thresholds, many categorical-like columns, tiny data | Gradient boosting is still the safer baseline; try both |
| Epistemic uncertainty | [`fit_tabular_ensemble`](../api/models.md#fit_tabular_ensemble), or the [ensemble methods](ensemble/index.md) |
| Need coverage guarantees | Wrap the fitted predictor in [conformal prediction](conformal/index.md) |

!!! warning "Evidence"
    The benchmarks behind the defaults are synthetic (Friedman #1, a smooth
    heteroscedastic design, a threshold-heavy design) and scikit-learn's
    diabetes data. No claim is made for real medium-size suites such as
    California housing until they have been run; see the project benchmark
    reports for current numbers.

---

## Mathematical background

### Robust input scaling

For column $j$ with training median $m_j$ and interquartile range
$s_j = q_{75} - q_{25}$ (half the range if the IQR is zero, 1 if the column is
constant):

$$z_j = \frac{x_j - m_j}{s_j}, \qquad
\tilde z_j = \frac{z_j}{\sqrt{1 + (z_j / c)^2}}, \quad c = 3 .$$

The smooth clip $\tilde z_j \in (-c, c)$ is strictly monotone, so it is
exactly invertible (`inverse_transform`) and keeps heavy-tailed columns from
dominating the first layer (RealMLP, Holzmüller et al., 2024).

### Periodic numeric embedding

Each scalar feature $\tilde z_j$ is lifted by $k$ learned frequencies
$c_{j} \in \mathbb{R}^k$, initialised from $\mathcal{N}(0, \sigma^2)$
(`frequency_scale`):

$$v_j = 2\pi\, c_j\, \tilde z_j, \qquad
e_j = \operatorname{ReLU}\!\bigl(W_j\,[\sin v_j \,\|\, \cos v_j] + b_j\bigr)
\in \mathbb{R}^{d},$$

with a separate $W_j \in \mathbb{R}^{d \times 2k}$ per feature (Gorishniy et
al., 2022). The raw $\tilde z_j$ is concatenated to $e_j$, so the MLP sees
$F\,(d+1)$ inputs for $F$ features. Sines and cosines at several frequencies
let an MLP represent sharp, non-monotone responses to a single feature that a
plain ReLU network learns slowly (spectral bias). Set `embedding="none"` to
feed $\tilde z$ directly.

### Training recipe

Targets are standardised, $y' = (y - \mu_y)/\sigma_y$, and the loss is
evaluated on $y'$. Predictions are mapped back by `TabularFit.predict`:

$$\hat y = \mu_y + \sigma_y \hat y' \quad\text{(point, quantile heads)}, \qquad
\hat\ell = \hat\ell' + 2\log\sigma_y \quad\text{(Gaussian log-variance)}.$$

Optimisation is AdamW (decoupled weight decay) with a one-cycle learning-rate
schedule (10 % warm-up, cosine decay), gradient-norm clipping at 1, and early
stopping on the validation loss with the best weights restored.

---

## Loss heads

| Loss | `out_features` | Notes |
|:-----|:--------------:|:------|
| `WeightedMSELoss`, `WeightedHuberLoss`, … | $D$ | Point prediction |
| `GaussianNLLLoss` | $2D$ | `[mean, log_variance]`; `fit_tabular` auto-selects the `"gaussian"` output layout |
| `MultiQuantileLoss(qs)` (one target) | `len(qs)` | Wrap with `NonCrossingSort` if crossing matters |
| Other losses | as the loss requires | Set `standardize_target=False` for heads that are not location/log-variance outputs |

---

## Example

```python
import numpy as np
import torch
from torchregress.losses import GaussianNLLLoss
from torchregress.models import TabularMLP, fit_tabular

rng = np.random.default_rng(0)
X = rng.uniform(-2, 2, size=(4000, 5))
y = np.sin(2 * X[:, 0]) + X[:, 1] * X[:, 2] + (0.1 + 0.5 * np.abs(X[:, 3])) * rng.normal(size=4000)

torch.manual_seed(0)
model = TabularMLP(in_features=5, out_features=2)            # [mean, log_variance]
fit = fit_tabular(model, GaussianNLLLoss(), X[:3000], y[:3000], epochs=60, seed=0)
mean, log_var = fit.predict(X[3000:]).unbind(dim=1)          # original target units
print(fit.best_epoch, float(torch.sqrt(((mean.numpy() - y[3000:]) ** 2).mean())))
```

!!! tip "Missing values"
    NaNs are replaced by the training median and every column that had a NaN at
    fit time gets a 0/1 indicator column, so build the model with
    `in_features=TabularPreprocessor().fit(X).n_features_out_` (or use
    `fit_tabular_ensemble`, which passes the width to a factory function).

!!! info "Determinism"
    `fit_tabular(seed=...)` seeds the split, shuffling and dropout in a forked
    RNG state and leaves the global torch RNG untouched. The initial weights
    come from the RNG at model construction, so call `torch.manual_seed` first.

---

## Limitations

- Full-batch data is moved to the training device once; this targets medium tables, not out-of-core data.
- `fit_tabular_ensemble` averages raw head outputs. That is exact for point and quantile heads; for Gaussian heads it is not the mixture variance (use `predict_members` and build the mixture yourself).
- Early stopping with a one-cycle schedule can stop while the learning rate is still high; the best weights are restored, but a longer `epochs` with `patience` equal to `epochs` follows the full schedule.
- Raw categorical columns must be encoded numerically before use.

---

## Next steps

- [Models API](../api/models.md) — signatures and attributes
- [Gaussian losses](../losses/gaussian.md) and [quantile losses](../losses/quantile_expectile.md) — the heads this backbone is built for
- [Ensemble methods](ensemble/index.md) — epistemic uncertainty on top of a backbone
- [Conformal prediction](conformal/index.md) — coverage guarantees for the fitted predictor

---

## References

| # | Reference |
|:-:|:----------|
| 1 | D. Holzmüller, L. Grinsztajn, I. Steinwart. ["Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data."](https://arxiv.org/abs/2407.04491) *NeurIPS*, **2024**. |
| 2 | Y. Gorishniy, I. Rubachev, A. Babenko. ["On Embeddings for Numerical Features in Tabular Deep Learning."](https://arxiv.org/abs/2203.05556) *NeurIPS*, **2022**. |
| 3 | I. Loshchilov, F. Hutter. ["Decoupled Weight Decay Regularization."](https://arxiv.org/abs/1711.05101) *ICLR*, **2019**. |
| 4 | L. N. Smith, N. Topin. ["Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates."](https://arxiv.org/abs/1708.07120) **2017**. |
