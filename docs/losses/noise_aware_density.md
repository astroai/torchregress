# Noise-Aware Conditional Densities and Binned CDF Heads

> ← [Noisy Labels](noisy_labels.md) | [Normalizing Flows](nflows.md) →

Standard density losses fit the observed label directly:

$$
\mathcal{L}_{\text{clean}} = -\log p_\theta(\tilde y \mid x).
$$

When $\tilde y$ is a noisy measurement of a latent target $y$, that objective
fits the **convolution** of the physical conditional density with the
measurement-error distribution, not the physical density itself.  At faint
magnitudes, high redshift, or with catalogue cross-matches, this can make a
correct density head look over-confident or biased.  The same distinction
matters for sparse labels: no density model can recover unsupported target
regions without a measurement model, a sampling model, or transferable
representation structure.

## Measurement model

Let $\tilde y$ be the recorded label, $y$ the true target, and $\sigma$ the
reported measurement standard deviation.  Assume

$$
\tilde y \mid y \sim \mathcal{N}(y, \sigma^2).
$$

The noise-aware likelihood is the marginal

$$
p_\theta(\tilde y \mid x, \sigma)
= \int p_\theta(y \mid x)\,\mathcal{N}(\tilde y; y, \sigma^2)\,dy,
\qquad
\mathcal{L}_{\text{noise}} = -\log p_\theta(\tilde y \mid x, \sigma).
$$

For a **diagonal Gaussian mixture** the convolution is analytic: add
$\sigma^2$ to every component variance.  For a **scalar spline flow** the
integral is evaluated with Gaussian–Hermite quadrature:

$$
p(\tilde y \mid x,\sigma)
\approx \frac{1}{\sqrt{\pi}}\sum_{k=1}^{K} w_k\,
p_\theta\!\left(\tilde y + \sqrt{2}\sigma t_k \mid x\right),
$$

where $(t_k, w_k)$ are the Hermite nodes and weights.  For non-scalar flows the
loss falls back to Monte-Carlo label-noise averaging.  A grossly wrong label
(not merely a noisy measurement) can be represented by an optional
contamination component,

$$
p(\tilde y \mid x) = (1-\epsilon)p_{\text{clean}}(\tilde y \mid x)
+ \epsilon\,p_{\text{bad}}(\tilde y),
$$

with $p_{\text{bad}}$ a wide Student-t density.  Keep $\epsilon$ small unless
the contaminating process is known.

!!! warning "Identifiability"
    Without reliable $\sigma$ metadata, repeated measurements, or structural
    assumptions, the clean conditional density and the label-noise distribution
    are not separately identifiable.  `NoiseAwareDensityLoss` uses the supplied
    $\sigma$; it does not estimate it from the residuals.

```python
import torch
from torchregress.losses import (
    MixtureDensityLoss, NoiseAwareDensityLoss, NormalizingFlowLoss, create_flow_model,
)

flow = create_flow_model(n_features=1, context_dim=64, flow_type="nsf",
                         n_transforms=3, hidden_features=[64, 64],
                         bound=8.0)
clean = NormalizingFlowLoss(flow=flow)
loss_fn = NoiseAwareDensityLoss(clean, contamination_fraction=0.01,
                                contamination_scale=2.0)

context = model(x)                       # [B, 64]
sigma = label_sigma(x)                   # [B] or [B, 1]
loss = loss_fn(context, y_observed, sigma=sigma)
```

For an MDN the analytic convolution is selected automatically:

```python
mdn = MixtureDensityLoss(n_components=5, n_features=1)
loss_fn = NoiseAwareDensityLoss(mdn, mode="analytic")
loss = loss_fn(mdn_output, y_observed, sigma=sigma)
```

Joint density losses accept row masks, not partially observed target vectors:
partial-coordinate likelihoods require actual marginal densities. Masked rows
may contain unavailable label/error values and contribute zero. Negative or
nonfinite observed standard deviations are rejected; zero is the clean limit.
Explicit integration modes fail for unsupported density/target combinations
rather than silently switching algorithms.

## Binned histogram / ordinal CDF head

`BinnedDensityLoss` is the corresponding smoothed categorical head.  The model
outputs $K$ logits over fixed bin edges.  A hard label is a one-hot target; with
$\sigma$ the target is replaced by its Gaussian bin masses

$$
q_i^{(k)} = \Phi\!\left(\frac{e_{k+1}-\tilde y_i}{\sigma_i}\right)
- \Phi\!\left(\frac{e_k-\tilde y_i}{\sigma_i}\right),
$$

and the loss is the cross-entropy $-\sum_k q_i^{(k)}\log \mathrm{softmax}(z_i)_k$.
The Gaussian bin masses are normalized over the finite bin range. This is
soft-target cross-entropy, **not** the measurement marginal likelihood above:
putting the label kernel inside an expected log score differs from taking the
log of the convolved density. It does not establish latent-density recovery.
The CDF interpolates linearly within each bin, matching uniform intra-bin
sampling. This is a deliberately strong
baseline for **low-label, scalar** problems: it has no Jacobian, no mode
collapse, and is much cheaper than a flow.  It is not a replacement for a flow
on genuinely multi-dimensional or highly structured targets.

```python
from torchregress.losses import BinnedDensityLoss

head = BinnedDensityLoss(bin_edges=torch.linspace(-3, 3, 33))
loss = head(logits, y_observed, sigma=sigma)
cdf = head.cdf(logits, values)
```

## Choosing an objective

| Problem | Recommended |
|:--|:--|
| Noisy labels, known $\sigma$ | `NoiseAwareDensityLoss` over NSF or MDN |
| Scalar target, very few labels | `BinnedDensityLoss` or `NoiseAwareDensityLoss(mdn)` |
| Multimodal, multi-target | NSF flow / MDN; noise-aware if $\sigma$ is known |
| Gross label contamination | Add the contamination component, or use a robust task loss |
| Population density for inference | Do **not** apply target-density reweighting; use sampling-selection corrections instead |
| Tail-balanced prediction | Use LDS / tempered weighting and report the changed calibration separately |

## Calibration and reporting

Report NLL, CRPS, PIT, interval coverage, and rare-target slices separately.
Noise-aware training changes the likelihood scale: NLL on noisy labels is not
directly comparable to NLL against clean labels.  Evaluate both on a clean,
representative held-out set when one exists, and state the measurement model
used to construct the labels.
