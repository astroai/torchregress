# Photo-z Recipe: CE + CDF + RnC-N

> ← [Ordinal](ordinal.md) | [Imbalanced](imbalanced.md) →

The two-stage photo-z estimator in the companion astronomy paper uses a binned
conditional distribution rather than a flow or MDN.  Its first-stage objective is

$$
L = 0.5\,L_{\rm CE} + 0.5\,L_{\rm CDF} + 0.1\,L_{\rm RnC-N},
$$

$$
L_{\rm CDF} = \bar h \sum_{j=1}^{K}\left[C_j - \mathbf{1}(c_j \ge z)\right]^2,
\qquad C_j = \sum_{\ell\le j}p_\ell,
$$

where $c_j$ are redshift-bin centers (quantiles of the training redshift) and
$\bar h$ is the mean bin separation.  Equal weight per bin is intentional even
when bins have unequal widths; the paper's CRPS evaluation instead integrates
over the actual redshift intervals.

## `PZRecipeLoss`

```python
from torchregress.losses import PZRecipeLoss

loss_fn = PZRecipeLoss(
    bin_centers,          # quantiles of training redshifts, shape [K]
    ce_weight=0.5,
    cdf_weight=0.5,
    rnc_weight=0.1,
    rnc_temperature=2.0,
    rnc_normalize=True,   # RnC-N: L2-normalize learned features
    rnc_dropouts=(0.05, 0.20),  # two augmented views
)
logits, features = model(x)
loss = loss_fn(logits, z, features=features)
```

`RnC-N` follows the paper convention: L2-normalized learned quantities,
absolute redshift differences, temperature 2, and two feature-dropout views
(5% and 20%).  It is a representation-shaping term, not a calibrated density
likelihood.

## Two-stage model

```python
from torchregress.models import PZRecipe

model = PZRecipe(n_features=37, n_bins=100, stage2_bins=200)
stage1_logits = model(x, stage=1)
stage2_logits = model(x, stage=2)
features = model.encode(x)
```

`PZRecipe` is a lean FT-Transformer (per-feature tokenizer + CLS + Transformer)
with a stage-1 bin head and a compact stage-2 MLP classifier.  The paper's
historical pipeline uses a TabM classifier in stage 2; any TabM-like classifier
can be substituted.  Bins and normalization must be fitted from the training
sample, as in the paper.

### Tri-loss variant

The methods paper compares a CE+CRPS+RnC encoder ("tri-loss").  `TriLoss`
uses the same quantile bins but replaces the squared CDF error with an exact
piecewise-linear CRPS:

```python
from torchregress.losses import TriLoss

loss_fn = TriLoss(bin_centers, ce_weight=1.0, crps_weight=0.5, rnc_weight=0.1)
loss = loss_fn(logits, z, features=features)
```

`CumulativeCDFLoss` is also available standalone.  For continuous densities use
[normalizing flows](nflows.md) or [MDNs](mdn.md); the recipe is a deliberately
strong low-data binned baseline.
