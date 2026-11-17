# Structural similarity for 1-D sequences

> ← [Robust Losses](robust.md) | [Linear-Nuisance Marginal Likelihood](linear_nuisance.md) →

`structural_similarity_1d` and `StructuralSimilarity1dLoss` compute SSIM along the
last dimension of `[..., N]` tensors: spectra, light curves, radial profiles, any
sequence where the *shape* of the signal matters more than its pointwise values.
Unlike image SSIM implementations, they handle **masks** (gaps, bad pixels, telluric
regions) and **per-sample weights** (e.g. inverse variances) inside every local
moment.

| # | Reference |
|:-:|:----------|
| 1 | Wang, Z., Bovik, A. C., Sheikh, H. R. & Simoncelli, E. P. (2004). *Image quality assessment: from error visibility to structural similarity*. IEEE Trans. Image Process. 13(4), 600–612. |
| 2 | Knutsson, H. & Westin, C.-F. (1993). *Normalized and differential convolution*. Proc. IEEE CVPR, 515–523. (masked local moments) |
| 3 | Rybicki, G. B. & Press, W. H. (1992). *Interpolation, realization, and reconstruction of noisy, irregularly sampled data*. ApJ 398, 169–176. (the likelihood alternative, see [below](#when-to-use-which)) |

→ API: [`structural_similarity_1d`, `StructuralSimilarity1dLoss`](../api/losses.md#sequence-shape-losses)

---

## Definition

Let $g$ be a normalised window of odd length $k$ (Gaussian with standard deviation
$\sigma$ samples, or uniform), $m_i\in\{0,1\}$ the mask and $w_i\ge0$ the weights
(both 1 by default). With $a_i = w_i m_i$, the local moments at window position $j$
are **normalised convolutions** over the valid samples only:

$$
\mu_x = \frac{\sum_i g_i a_{j+i} x_{j+i}}{\sum_i g_i a_{j+i}},\qquad
\sigma_x^2 = \frac{\sum_i g_i a_{j+i} x_{j+i}^2}{\sum_i g_i a_{j+i}} - \mu_x^2,\qquad
\sigma_{xy} = \frac{\sum_i g_i a_{j+i} x_{j+i} y_{j+i}}{\sum_i g_i a_{j+i}} - \mu_x\mu_y ,
$$

and the local index is

$$
\mathrm{SSIM}_j = \frac{(2\mu_x\mu_y + C_1)\,(2\sigma_{xy} + C_2)}
{(\mu_x^2 + \mu_y^2 + C_1)\,(\sigma_x^2 + \sigma_y^2 + C_2)},
\qquad C_1 = (k_1 L)^2,\quad C_2 = (k_2 L)^2 .
$$

A row's score is the mean of $\mathrm{SSIM}_j$ over its **valid windows**: windows
whose valid samples carry at least `min_coverage` of the window mass,
$\sum_i g_i \mathbb 1[w_{j+i} m_{j+i} > 0] \ge$ `min_coverage`, with positive
weighted mass. Windows inside
a gap are excluded, not scored as zeros. The loss is $1 - \overline{\mathrm{SSIM}}$
per row. Moments are population moments (no Bessel correction), as in `torchmetrics`.

Masked samples never enter a moment: values there may be `NaN` or `inf`, and their
gradient is exactly zero. A sample with **zero weight** (zero inverse variance at a
bad pixel, say) is a masked sample in every respect: it does not count towards
coverage or the default data range, and non-finite values there are ignored, so a
window holding a single weighted sample is excluded rather than scored on luminance
alone. A weight that is constant over a window cancels, so a
per-row weight changes nothing; weights only re-balance samples *within* windows.

### Edge handling

| `padding` | Window centres | Edge windows | Use |
|:----------|:---------------|:-------------|:----|
| `"valid"` (default) | $N-k+1$ | none: only windows fully inside the sequence | Wang et al. / scikit-image convention |
| `"same"` | $N$ | the sequence is treated as masked beyond its ends: truncated and renormalised like masked windows, subject to `min_coverage` | keep edge information with masks |
| `"reflect"` | $N$ | reflect-padded sequence | exact `torchmetrics` 1.9 parity |

### Dynamic range

`data_range` sets $L$. The default (`None`) uses each row's **target** range
$\max y - \min y$ over its valid samples, so a row's score does not depend on the
predictions' range or on the other rows in the batch. `torchmetrics` instead uses
$\max(\operatorname{range}\hat y, \operatorname{range} y)$ over the *whole batch*.
A constant target row with `data_range=None` has $L=0$ and returns `NaN`; pass a
float or a per-row tensor when rows can be flat or when scores must be comparable
across rows.

---

## Caveats: SSIM is not a likelihood

!!! warning "What SSIM does not do"
    - **No noise model.** SSIM does not know the measurement errors. Weights change
      which samples dominate a window, but the score is not a log-likelihood, gives
      no uncertainties and is not calibrated. Low-S/N regions count as much as
      high-S/N ones unless you weight them, and even then there is no $\chi^2$
      interpretation.
    - **Depends on normalisation.** $C_1$ and $C_2$ scale with $L$: the same pair of
      spectra scores differently after a rescaling of the flux units unless
      `data_range` is rescaled too. Normalise rows consistently (e.g. by the
      continuum) or pass a physically motivated `data_range`.
    - **Insensitive to local gain and offset.** SSIM factors into luminance,
      contrast and structure. The structure term is the windowed correlation
      $(\sigma_{xy} + C_3)/(\sigma_x\sigma_y + C_3)$, unchanged by
      $y \to a y + b$ within a window. Only the luminance term
      $2\mu_x\mu_y/(\mu_x^2+\mu_y^2)$ and the contrast term
      $2\sigma_x\sigma_y/(\sigma_x^2+\sigma_y^2)$ see gain and offset, and both are
      flat near equality: a 10% local gain error costs about $0.5\%$ in contrast.
      That is why SSIM "reconstructs shape" — and also why it will not pin down
      absolute fluxes, equivalent widths or continuum levels.

!!! tip "Likelihood-based alternative"
    If the reason to prefer SSIM is that a smooth baseline, continuum or
    calibration error should not be penalised, model it: write the distortion as
    a linear nuisance and marginalise it analytically with
    [`MarginalizedNuisanceGaussianNLLLoss`](linear_nuisance.md). That loss is a
    proper Gaussian likelihood with your noise model, is *exactly* invariant to
    the distortions you specify (and only those), and returns the fitted baseline.
    A common recipe is a likelihood loss as the main objective with a small SSIM
    term as a shape regulariser.

---

## When to use which

| Objective | Noise model | Invariant to | Missing data | Output | Best for |
|:----------|:------------|:-------------|:-------------|:-------|:---------|
| `WeightedMSELoss` | implicit, homoscedastic | nothing | mask | per-element loss | baselines |
| `GaussianNLLLoss` / `student_t_nll` | known or predicted variance | nothing | mask | log-likelihood | calibrated fits with known errors |
| `StructuralSimilarity1dLoss` | none | approx. local gain/offset (structure term) | mask + coverage rule | similarity index in $[-1, 1]$ | perceptual shape agreement, regularising shape |
| [`MarginalizedNuisanceGaussianNLLLoss`](linear_nuisance.md) | known variance | exactly the span of a chosen basis $U$ | exact (samples drop out) | log-likelihood + fitted nuisance | continuum / baseline / calibration marginalisation |

### Relation to `torchmetrics`

`torchmetrics.functional.image.structural_similarity_index_measure` accepts only 4-D/5-D
images. A 1-D sequence can be passed as `[B, 1, N, 1]` with a `(k, 1)` kernel, and
with `padding="reflect"` this module reproduces its scores to float64 round-off
(tested in `tests/losses/test_ssim.py`). torchregress keeps its own implementation
because the wrapper cannot express masks, weights, coverage-based window exclusion,
per-row dynamic ranges, arbitrary leading batch dimensions or valid-only windows.

### Cost

All local moments of all rows go through **one depthwise (grouped) 1-D
convolution**: $O(R\,N\,k)$ time and a handful of $[R, N]$ moment tensors in
memory, with no $k$-fold `im2col` copy. Bounded CPU run (2 threads, float32,
`kernel_size=11`, `data_range=1.0`, forward + backward, mask and weights):

| Rows $R$ | Length $N$ | Rows / s | Peak extra memory |
|---------:|-----------:|---------:|------------------:|
| 100 | 1 000 | ≈ 5 200 | ≈ 40 MB |
| 10 000 | 1 000 | ≈ 3 700 | ≈ 1.9 GB |
| 1 000 | 10 000 | ≈ 330 | ≈ 1.9 GB |

Memory grows linearly in $R N$ (roughly 190 bytes per sample with mask, weights
and gradients in float32), so for very large batches evaluate the loss in chunks of
rows and accumulate.

---

## Example

A model reconstructs continuum-normalised spectra with a masked telluric band and
per-pixel inverse variances. The SSIM loss uses both; masked pixels may hold `NaN`.

```python
import torch
from torchregress.losses import StructuralSimilarity1dLoss, structural_similarity_1d

torch.manual_seed(0)
wave = torch.linspace(0.0, 1.0, 512)
lines = 1.0 - 0.6 * torch.exp(-0.5 * ((wave[None] - torch.rand(8, 1)) / 0.01) ** 2)
ivar = torch.full((8, 512), 400.0)
observed = lines + torch.randn(8, 512) * ivar.rsqrt()
mask = torch.ones(8, 512, dtype=torch.bool)
mask[:, 300:330] = False                 # telluric band
observed[~mask] = float("nan")          # never read

model = torch.nn.Sequential(torch.nn.Linear(512, 512))
recon = model(torch.where(mask, observed, torch.ones_like(observed)))

loss_fn = StructuralSimilarity1dLoss(kernel_size=11, sigma=1.5, data_range=1.0)
loss = loss_fn(recon, observed, mask=mask, weights=ivar)
loss.backward()
assert torch.isfinite(loss)

# Per-row scores and the local map (NaN at excluded windows).
score, ssim_map = structural_similarity_1d(
    recon.detach(), observed, data_range=1.0, mask=mask, reduction="none", return_map=True
)
assert score.shape == (8,) and ssim_map.shape == (8, 512 - 10)
assert torch.isnan(ssim_map[:, 310]).all()   # windows inside the gap are excluded
```

→ See also [Gaussian losses](gaussian.md), [Robust losses](robust.md) for the
Student-$t$ likelihood with known precisions, and the
[point metrics](../metrics/point.md).
