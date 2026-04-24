# SAGE-Reg Paper Outline

## Working Title

**SAGE-Reg: Self-Agreement Distributional Self-Training for Regression**

## Core Claim

Unlabeled regression samples should be trusted when their **full predictive laws**
remain stable across perturbations, not when a single scalar confidence proxy looks large.

## Method Summary

For each unlabeled input $x$:

1. generate $K$ stochastic predictive views,
2. map every view into a common predictive representation,
3. form a consensus predictive law $q_x$,
4. measure cross-view disagreement $A(x)$,
5. define a trust weight $w(x)=\exp(-A(x)/\tau)$,
6. optimize supervised loss plus a weighted agreement penalty.

The minimal prototype in `torchregress.semi_supervised` implements this with:

- `PredictiveBatch` as the common container,
- a shared 1D support grid per unlabeled batch item,
- generalized Jensen-Shannon-style disagreement to the consensus law,
- detached trust weights so the model cannot reduce loss by collapsing the weight itself.

## Objective

Let $\{p_{\theta}^{(k)}(y \mid \tilde{x}_k)\}_{k=1}^{K}$ be stochastic predictive views for the same unlabeled sample.
Define the consensus law as the arithmetic mean density:

$$
q_x(y) = \frac{1}{K}\sum_{k=1}^{K} p_{\theta}^{(k)}(y \mid x).
$$

Define self-disagreement as the average pairwise divergence:

$$
A(x) =
\frac{2}{K(K-1)}
\sum_{k < l}
D\!\left(p_{\theta}^{(k)}(\cdot \mid \tilde{x}_k), p_{\theta}^{(l)}(\cdot \mid \tilde{x}_l)\right).
$$

Trust weights are then

$$
w(x) = \exp\!\left(-\frac{A(x)}{\tau}\right).
$$

The training objective is

$$
\mathcal{L}_{\text{SAGE-Reg}}
=
\mathcal{L}_{\text{sup}}
+ \lambda_{\text{u}}
\frac{\sum_{x \in \mathcal{U}} \operatorname{sg}(w(x))
\;S\!\left(p_{\theta}(\cdot \mid x), q_x\right)}
{\sum_{x \in \mathcal{U}} \operatorname{sg}(w(x))}.
$$

Here `sg` denotes stop-gradient on the weight path in the prototype, and
$S(\cdot,\cdot)$ is the distributional pseudo-supervision score.

## Supported Predictive Families in v1

- **Gaussian heads**: `PredictiveBatch(mean=..., std=...)`
- **Quantile heads**: `PredictiveBatch(quantiles=..., quantile_levels=...)`
- **Bar / binned PDF heads**: `PredictiveBatch(bar_logits=..., bin_edges=...)`

The prototype is intentionally restricted to **1D targets** and uses a shared-grid density approximation.

## What SAGE-Reg Is Not

### Not RaC-centric

RaC-style scalar confidence ranking is not the method's core. A scalar ranking signal may be
used as an ablation or diagnostic, but SAGE-Reg's main object is the **predictive distribution**.

### Not reconstruction-based

There is no autoencoder, reconstruction head, or latent consistency loss. The only unlabeled
signal is agreement among stochastic predictive laws.

### Not iterative pseudo-label retraining

The v1 prototype does not implement curriculum pseudo-label loops, periodic relabeling, or
full retraining rounds.

## How This Differs From Generic Uncertainty-Guided Pseudo-Labeling

Generic uncertainty-guided pseudo-labeling usually does the following:

- compute one prediction per unlabeled point,
- convert that prediction into a scalar pseudo-target,
- accept or weight the sample using a scalar confidence/uncertainty score,
- train the student to regress toward that pseudo-target.

SAGE-Reg differs in three concrete ways:

1. **No single pseudo-target is central.**
   The method never needs to collapse the unlabeled prediction into one scalar target before scoring trust.
2. **Trust is distributional.**
   Stability is measured from the entire predictive law across perturbations, not just predictive variance or confidence of one pass.
3. **Agreement determines trust, while consensus supplies the target law.**
   Disagreement sets the weight; the student is trained toward a consensus predictive law rather than a fixed scalar pseudo target.

## How This Differs From CURE-Style Methods

The reference conversation was unavailable at implementation time, so this outline uses
**CURE-style** to mean semi-supervised regression methods where the central unlabeled mechanism is:

- a scalar confidence / uncertainty estimate,
- optional sample filtering or reweighting,
- and training toward a scalar pseudo-target or residual target.

Under that framing, SAGE-Reg differs because:

1. **The trusted object is a law, not a scalar.**
   CURE-style methods make scalar confidence central; SAGE-Reg makes cross-perturbation distributional stability central.
2. **Consensus is predictive-law consensus.**
   The prototype constructs $q_x$ from multiple stochastic predictive views instead of choosing one teacher prediction as the unlabeled target.
3. **Disagreement is internal to the predictive family.**
   Gaussian, quantile, and bar predictors are all compared after conversion to a common density representation.

If later paper-specific CURE details differ, this section should be revised against the actual source before publication.

## Minimal API

```python
import torch
from torchregress.prediction import PredictiveBatch
from torchregress.semi_supervised import SelfAgreementTrainer

trainer = SelfAgreementTrainer(
    optimizer=optimizer,
    supervised_loss_fn=supervised_loss_fn,
    predictive_batch_fn=predictive_batch_fn,
    n_views=4,
    tau=0.15,
    agreement_weight=0.5,
)
history = trainer.fit(model, labeled_loader, unlabeled_loader, epochs=10)
```

## Evaluation Targets for a Paper Pass

- labeled-data efficiency vs supervised-only baselines
- robustness to perturbation type and perturbation strength
- calibration and interval quality under limited labels
- failure modes under multimodal or mis-specified heads
- agreement-weight histograms versus downstream error
- representation sensitivity across Gaussian, quantile, and bar heads

For a dedicated note on how to interpret cross-backbone differences, see
[Representation Sensitivity](sage_reg_representation_sensitivity.md).

## Prototype Limitations

- 1D regression only
- shared-grid density approximation rather than exact family-specific divergences
- no explicit multimodal mixture-family support in v1
- only a minimal EMA teacher path
- no claim of theoretical consistency yet
