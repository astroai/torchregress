# Marginal Contrastive Discrimination

[`MarginalContrastiveDensity`](../api/models.md#marginalcontrastivedensity) estimates a
conditional density by splitting it into an unconditional density and a
classification problem (Meziani, Ndiaye & Riu 2022, ref. 1):

$$
p(\mathbf{y} \mid \mathbf{x}) = p(\mathbf{y})\, r(\mathbf{x}, \mathbf{y}),
\qquad
r(\mathbf{x}, \mathbf{y}) = \frac{p(\mathbf{x}, \mathbf{y})}{p(\mathbf{x})\, p(\mathbf{y})} .
$$

- $p(\mathbf{y})$ is any unconditional flow over the targets, e.g.
  `create_flow_model(n_features=D, context_dim=0)`.
- $r$ is learned by a classifier that tells joint pairs $(\mathbf{x}_i, \mathbf{y}_i)$
  from shuffled pairs $(\mathbf{x}_j, \mathbf{y}_i)$. It is trained with contrastive
  neural ratio estimation ([`NeuralRatioLoss(criterion="cnre")`](../losses/nre.md), ref. 2),
  whose logit estimates the absolute $\log r$, not $\log r$ up to a constant per
  $\mathbf{x}$.

The training objective `loss(x, y)` is the marginal negative log-likelihood plus
the CNRE loss, where each $\mathbf{y}_i$ is paired with $\mathbf{x}$ from
`n_negatives` other rows of the batch:

$$
\mathcal{L} = -\frac{1}{N}\sum_i \log p_\phi(\mathbf{y}_i)
+ \mathcal{L}_{\text{CNRE}}\bigl(s_\psi;\ \{(\mathbf{x}_i, \mathbf{y}_i)\},\ \{(\mathbf{x}_{i+k}, \mathbf{y}_i)\}_{k=1}^{K}\bigr),
\qquad s_\psi \approx \log r .
$$

## Normalization and sampling

The exact ratio satisfies $Z(\mathbf{x}) = \mathbb{E}_{\mathbf{y} \sim p(\mathbf{y})}[r(\mathbf{x}, \mathbf{y})] = 1$;
a learned one does not. `log_prob` therefore subtracts a Monte-Carlo estimate by
default,

$$
\log \hat p(\mathbf{y} \mid \mathbf{x}) = \log p_\phi(\mathbf{y}) + s_\psi(\mathbf{x}, \mathbf{y})
- \log \frac{1}{M} \sum_{m=1}^{M} e^{s_\psi(\mathbf{x}, \tilde{\mathbf{y}}_m)},
\qquad \tilde{\mathbf{y}}_m \sim p_\phi ,
$$

with $M = 4096$ (`n_samples`). Its error is about
$\operatorname{sd}(r(\mathbf{x}, \cdot)) / \sqrt{M}$, larger for contexts whose
conditional sits in the tail of $p(\mathbf{y})$. The normalizer is computed without
gradients and once per distinct context, so `log_prob` (and `forward`, its negative) is
a score, not a training objective: train with `loss`.

In eval mode the marginal draws are cached (one bank per size) and shared by `log_prob`,
`sample` and `effective_sample_size`. Scores from separate calls, such as a target and
its predictive draws in `model_hpd_level`, then use the same normalizer, and the
effective sample size describes the bank that `sample` resampled. `train()`,
`load_state_dict` and `refresh_bank()` discard the cache.

`sample(x, n)` uses sampling-importance-resampling: a bank of `n_proposals`
marginal draws is resampled with weights $r(\mathbf{x}, \tilde{\mathbf{y}}_m)$.
When $p(\mathbf{y} \mid \mathbf{x})$ is much narrower than $p(\mathbf{y})$, a few draws
carry the weight and are repeated; `effective_sample_size(x)` reports the Kish
effective size of the bank, between 1 and `n_proposals`.

!!! warning "Informative features shrink the effective sample size"
    The effective sample size is roughly the bank size times the ratio of the
    conditional to the marginal volume. With several targets and features that
    pin the targets down, it can fall to a handful of draws; samples and HPD
    levels are then unreliable. Raise `n_proposals`, or use a conditional flow.

## Example

```python
import torch

from torchregress.losses import build_ratio_classifier, create_flow_model
from torchregress.metrics import model_hpd_level
from torchregress.models import MarginalContrastiveDensity

torch.manual_seed(0)
x = torch.randn(2000, 3)
y = torch.stack([x[:, 0] + 0.3 * torch.randn(2000), x[:, 0] * x[:, 1] + 0.3 * torch.randn(2000)], 1)
y = (y - y.mean(0)) / y.std(0)

model = MarginalContrastiveDensity(
    create_flow_model(n_features=2, context_dim=0, n_transforms=3),
    build_ratio_classifier(context_dim=3, observable_dim=2),
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(5):
    for batch in torch.randperm(2000).split(128):
        optimizer.zero_grad()
        model.loss(x[batch], y[batch]).backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    log_density = model.log_prob(x[:5], y[:5])  # normalized by default
    samples = model.sample(x[:5], n_samples=500)  # [5, 500, 2], for corner plots
    ess = model.effective_sample_size(x[:5])  # check before trusting the samples
levels = model_hpd_level(model, x[:200], y[:200])  # joint HPD calibration
```

To score several targets per context (e.g. predictive draws for HPD levels),
compute `log_normalizer(x)` once and call `log_prob(..., normalize=False)`.

## When to use it

| Property | MCD | Conditional flow (NSF, TarFlow) |
|:--|:--|:--|
| Conditional part | any classifier over $(\mathbf{x}, \mathbf{y})$ | conditioner network inside the flow |
| Exact likelihood | up to the Monte-Carlo normalizer | ✅ |
| Sampling | importance resampling (degrades as $p(\mathbf{y}\mid\mathbf{x})$ narrows) | exact, one inverse pass |
| Unconditional structure in $\mathbf{y}$ | modelled once by the marginal flow | relearned for every context |
| Best for | weakly informative features, a reference baseline | informative features, sharp conditionals |

### Harness comparison

On the torchregress-harness `multivariate_intervals` suite (five datasets with 2 or 3
targets, one seed), all rows trained with the same early-stopping recipe: a 15% validation
carve, patience 10, `recommended_tail_bound` and 16 bins. MCD uses an NSF marginal and a
CNRE classifier on the raw features. *HPD90* is the share of test targets inside the 90%
joint highest-density region; *cal. err.* is the mean absolute coverage error over the
levels 0.5, 0.68, 0.9 and 0.95; *ESS* is the mean effective share of the 8192-draw bank.

| Dataset | Test NLL: NSF / 5-flow ensemble / MCD | HPD90: NSF / ensemble / MCD | Cal. err.: NSF / ensemble / MCD | MCD ESS |
|:--|:--|:--|:--|--:|
| `energy_efficiency` | 0.373 / **0.119** / 0.842 | 0.844 / **0.939** / 0.978 | 0.080 / **0.038** / 0.136 | 0.16 |
| synthetic multivariate | 0.009 / **−0.250** / 0.222 | 0.788 / **0.903** / 0.922 | 0.113 / **0.022** / 0.027 | 0.05 |
| synthetic bimodal | 4.517 / **4.412** / 4.629 | 0.855 / 0.902 / **0.905** | 0.045 / 0.008 / **0.006** | 0.51 |
| synthetic Gaussian | 4.018 / **3.911** / 3.964 | 0.835 / **0.882** / 0.850 | 0.071 / **0.027** / 0.053 | 0.38 |
| synthetic non-Gaussian | 3.703 / **3.600** / 4.250 | 0.893 / **0.912** / 0.883 | 0.021 / **0.013** / 0.014 | 0.48 |

MCD's NLL was worse than a single NSF on four of the five datasets. Its joint
calibration was competitive, best of the three on the bimodal set, but it over-covered
on `energy_efficiency`, where only 16% of the bank carried the weight. The ensemble of
early-stopped flows had the best NLL on every dataset.

MCD's appeal is that the conditional part is plain classification, so the ratio
model can be anything that produces a logit. Its weak point is structural:
sampling and normalization both average over $p(\mathbf{y})$, which wastes draws
when the features are informative.

→ See also [normalizing flow losses](../losses/nflows.md),
[neural ratio estimation](../losses/nre.md) and
[joint HPD calibration metrics](../metrics/distribution.md).

## References

| # | Reference |
|:-:|:----------|
| 1 | K. Meziani, A. Ndiaye, B. Riu. ["MCD: Marginal Contrastive Discrimination for conditional density estimation."](https://arxiv.org/abs/2206.01592) *arXiv:2206.01592*, **2022**. |
| 2 | B. K. Miller, C. Weniger, P. Forré. ["Contrastive Neural Ratio Estimation."](https://arxiv.org/abs/2210.06170) *NeurIPS*, **2022**. |
| 3 | D. B. Rubin. "Using the SIR Algorithm to Simulate Posterior Distributions." *Bayesian Statistics 3*, 395–402, **1988**. (Sampling-importance-resampling.) |
