# Uncertain Ground Truth

These losses target settings where labels are noisy, partially trusted, weakly supervised, or augmented with pseudo labels.

→ API: [`NoisyTargetGaussianNLL`](../api/losses.md), [`PseudoLabelNLL`](../api/losses.md), [`ConsistencyRegLoss`](../api/losses.md), [`PseudoLabelConsistencyLoss`](../api/losses.md).

Available losses:

- `NoisyTargetGaussianNLL`
- `ConsistencyRegLoss`
- `PseudoLabelNLL`
- `PseudoLabelConsistencyLoss`

Related utilities:

- `torchregress.utils.generate_pseudo_labels`
- `torchregress.utils.update_ema_teacher_`

## When To Use Which

| Loss | Start here when | Main signal | API |
|:-----|:----------------|:------------|:----|
| `NoisyTargetGaussianNLL` | labels come with reported variance/error bars | explicit target-noise variance | [`NoisyTargetGaussianNLL`](../api/losses.md) |
| `PseudoLabelNLL` | Gaussian head + pseudo labels + confidence weights | probabilistic self-training | [`PseudoLabelNLL`](../api/losses.md) |
| `ConsistencyRegLoss` | you already have a trusted teacher prediction | student-teacher agreement | [`ConsistencyRegLoss`](../api/losses.md) |
| `PseudoLabelConsistencyLoss` | you want pseudo labels plus teacher consistency in one point-regression loss | supervised + pseudo-label + teacher consistency | [`PseudoLabelConsistencyLoss`](../api/losses.md) |

## Objectives

### Noise-aware Gaussian NLL

If the observed target has known variance $\sigma_y^2$ and the model predicts $(\mu, \sigma_{pred}^2)$:

$$
\mathcal{L}_{noise} =
\frac{1}{2}\log\!\left(2\pi(\sigma_{pred}^2 + \sigma_y^2)\right)
+ \frac{(y - \mu)^2}{2(\sigma_{pred}^2 + \sigma_y^2)}.
$$

### Consistency regularization

$$
\mathcal{L}_{consistency} = \lVert f_{student}(x) - \operatorname{sg}(f_{teacher}(x)) \rVert_2^2.
$$

### Semi-supervised composite loss

`PseudoLabelConsistencyLoss` is intentionally narrow: it is the loss for the common
student-teacher semi-supervised regression pattern where you have:

1. a labeled subset with real targets,
2. an unlabeled subset with pseudo targets,
3. an optional teacher prediction used for consistency regularization.

It combines:

$$
\mathcal{L} =
\mathcal{L}_{sup}
+ \lambda_{pseudo}\,\mathcal{L}_{pseudo}
+ \lambda_{cons}\,\mathcal{L}_{consistency},
$$

with `label_mask` controlling which samples contribute to the supervised term and `pseudo_confidence` controlling how strongly unlabeled pseudo labels contribute.

## API Examples

### Known target uncertainty

```python
import torch
from torchregress.losses import NoisyTargetGaussianNLL

loss_fn = NoisyTargetGaussianNLL()

mean = torch.randn(64, 1)
log_var = torch.randn(64, 1)
y_obs = torch.randn(64, 1)
y_obs_var = torch.rand(64, 1) * 0.05

loss = loss_fn((mean, log_var), y_obs, target_variance=y_obs_var)
```

### Pseudo-label Gaussian training

```python
from torchregress.losses import PseudoLabelNLL

loss_fn = PseudoLabelNLL(pseudo_weight=0.8)
loss = loss_fn(
    (mean, log_var),
    target,
    pseudo_target=pseudo_target,
    pseudo_confidence=pseudo_confidence,
    label_mask=label_mask,
)
```

### Point-regression semi-supervision

```python
from torchregress.losses import PseudoLabelConsistencyLoss

loss_fn = PseudoLabelConsistencyLoss(
    pseudo_weight=0.8,
    consistency_weight=0.25,
    confidence_threshold=0.35,
)
loss = loss_fn(
    student_pred,
    target_with_placeholders,
    pseudo_target=pseudo_target,
    pseudo_confidence=pseudo_confidence,
    teacher_pred=teacher_pred,
    label_mask=label_mask,
)
```

### Pseudo-label generation and EMA teacher update

```python
from torchregress.utils import generate_pseudo_labels, update_ema_teacher_

pseudo_target, pseudo_confidence, accepted = generate_pseudo_labels(
    teacher_mean,
    log_variance=teacher_log_var,
    confidence_threshold=0.35,
)
update_ema_teacher_(ema_teacher, student, momentum=0.95)
```

## Selection Bias / Propensity Support

`NoisyTargetGaussianNLL`, `PseudoLabelNLL`, and `PseudoLabelConsistencyLoss` support optional `propensity_scores` or `propensity_weights` so weak supervision can be combined with covariate-dependent label-observation correction.

## Comparison Examples

- [Semi-Supervised Regression Comparison](../examples/semi_supervised_regression_comparison.md)
- [Uncertain-GT + Density Conformal Comparison](../examples/uncertain_gt_density_conformal_comparison.md)
- [Uncertain-GT + Density Conformal Comparison (Real Data)](../examples/uncertain_gt_density_conformal_realdata_comparison.md)

## Practical Notes

!!! tip
    Use `NoisyTargetGaussianNLL` when you trust the reported measurement uncertainty more than any teacher model.

!!! tip
    Use `PseudoLabelConsistencyLoss` when you want a point-regression loss that explicitly matches the pseudo-label + teacher-consistency workflow. It is not meant to stand in for every semi-supervised regression method.

!!! warning "Confirmation Bias & Error Propagation"
    Pseudo-labeling is prone to **confirmation bias** — when a model repeatedly fits its own incorrect but highly confident predictions, amplifying errors over epochs.
    - **Mitigation**: Use an EMA teacher (via `update_ema_teacher_`) to smooth predictions, apply threshold gating on `pseudo_confidence` (e.g., via `generate_pseudo_labels`), and **never** include pseudo-labeled samples in the held-out validation or test splits. Always maintain a clean, fully-supervised evaluation set.

!!! warning "Hyperparameter sensitivity"
    - **Confidence threshold tuning**: `PseudoLabelNLL` and `PseudoLabelConsistencyLoss` use `confidence_threshold` to gate which pseudo-labels are trusted. Setting this too low admits noisy pseudo-labels; setting it too high rejects most unlabeled data, negating the semi-supervised benefit. Tune on a small held-out labeled set, not the unlabeled data.
    - **Target variance quality**: `NoisyTargetGaussianNLL` requires `target_variance` to be reasonably accurate. If the target variance is poorly estimated (e.g., guessed rather than measured), the loss can perform **worse** than standard `GaussianNLLLoss` which learns variance from data. Only use when you trust the variance estimates more than the model's ability to learn them.

## Next steps

- [Noisy labels](noisy_labels.md) — when noise variance is unknown
- [Semi-supervised comparison](../examples/semi_supervised_regression_comparison.md) — benchmark all approaches
- [Gaussian losses](gaussian.md) — standard heteroscedastic regression
- [Conformal + Uncertain GT](../examples/uncertain_gt_density_conformal_comparison.md) — density-based conformal on noisy labels

---

## References

| # | Reference |
|:-:|:----------|
| 1 | D.-H. Lee. ["Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks."](https://www.semanticscholar.org/paper/Pseudo-Label-:-The-Simple-and-Efficient-Method-for-Lee/8d1d828a2524c7fbd52a0a2df335029d29e7943d) *ICML Workshop*, **2013**. |
| 2 | S. Laine, T. Aila. ["Temporal Ensembling for Semi-Supervised Learning."](https://arxiv.org/abs/1610.02242) *ICLR*, **2017**. |
| 3 | A. Tarvainen, H. Valpola. ["Mean Teachers are Better Role Models."](https://arxiv.org/abs/1703.01780) *NeurIPS*, **2017**. |
