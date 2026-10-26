# Continuous Group DRO

> ← [LatentNN](latentnn.md) | [SIMEX](simex.md) → [Post-hoc calibration](../calibration.md)

**Group Distributionally Robust Optimization (Group DRO)** replaces the average training risk with the *worst-group* risk over a small set of physical regimes. For regression on a long-tailed continuous target, the target range is split into interpretable bins, and the model is trained to equalize its error across them instead of averaging a catastrophic tail away with a well-populated bulk.

`torchregress` already has a **CVaR** loss that focuses on the worst individual samples. CVaR is the wrong tool for noisy astronomical data: the highest-loss individuals are usually corrupted spectra or outliers in the *bulk*, not the metal-poor tail. Group DRO first **averages the individual noise inside each target regime**, then equalizes the regime means, so a noisy solar-metallicity sample cannot dominate the objective.

!!! abstract "The idea"
    Bin the target into $G$ regimes. Maintain adversarial weights $q_g$ on the simplex, update them with exponentiated-gradient ascent toward the worst-performing regime, and train on $\sum_g q_g \mathcal L_g$.

---

## Mathematical formulation

Let $\ell(f_\theta(x), y)$ be a per-sample loss and $P_g$ the conditional distribution of $(x,y)$ restricted to target bin $g$. Group DRO solves the minimax problem

$$\min_{\theta}\ \max_{g\in\{1,\dots,G\}}\ \mathbb{E}_{(x,y)\sim P_g}\bigl[\ell(f_\theta(x), y)\bigr].$$

The inner maximization is over the simplex of group weights $q$; the solution is found by **exponentiated-gradient (EG) ascent** on the group losses $\mathcal L_g^{(t)}$ of the current batch:

$$q_g^{(t)} \propto q_g^{(t-1)}\,\exp\!\bigl(\eta\,\mathcal L_g^{(t)}\bigr), \qquad \sum_{g=1}^{G} q_g = 1,$$

and the training objective is the $q$-weighted mixture

$$\mathcal L_{\text{DRO}}(\theta) = \sum_{g=1}^{G} q_g\,\mathcal L_g(\theta),$$

where $q$ is treated as a constant in the gradient step (the adversarial weights are updated in a separate, detached step — this is the standard two-player interpretation).

!!! tip "Numerical form"
    The implementation performs the update in log space, $\log q_g \leftarrow \log q_g + \eta\mathcal L_g$ followed by a log-sum-exp normalization, so a badly predicted bin with a huge loss cannot overflow the weights. Groups absent from a batch keep their previous weight before renormalization. An optional per-group floor `min_group_weight` projects the simplex onto $\{q_g \ge m,\ \sum_g q_g = 1\}$, trading exact worst-group optimality for lower variance when a group has very few samples.

---

## Usage

### End-to-end trainer

```python
import torch
import torch.nn as nn
from torchregress.algorithms import GroupDROTrainer

torch.manual_seed(0)
# Three regimes with very different population sizes and scatter:
# solar-like (many, small scatter), metal-poor (few), extreme (very few).
y = torch.cat(
    [
        0.1 + 0.1 * torch.randn(500),   # bulk
        -1.5 + 0.3 * torch.randn(200),  # sparse regime
        -3.0 + 0.4 * torch.randn(100),  # extreme regime
    ]
)
x = torch.stack([y, torch.sin(y)], dim=1) + 0.05 * torch.randn(y.shape[0], 2)

def make_model():
    return nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 1))

# Physical regimes: sparse (-4, -2.5), (-2.5, -1), bulk (-1, 0.5).
trainer = GroupDROTrainer(
    model_factory=make_model,
    group_edges=[-4.0, -2.5, -1.0, 0.5],
    base_loss="mse",
    eta=0.01,
    epochs=100,
    lr=1e-3,
    batch_size=64,
    seed=0,
)
trainer.fit(x, y)
y_hat = trainer.predict(x)

print("group weights:", trainer.group_weights_.tolist())
print("per-regime losses:", trainer.group_losses_.tolist())
```


The fitted attributes are:

| Attribute | Meaning |
|:----------|:--------|
| `trainer.model` | Fitted model (in `eval` mode). |
| `trainer.group_weights_` | Final simplex weights $q$ (sum to 1). |
| `trainer.group_losses_` | Per-regime final losses (data-weighted over batches; `NaN` for empty regimes). |
| `trainer.history_` | Per-epoch `{"loss": ..., "q": [...]}` records. |

### The loss in a custom training loop

`ContinuousGroupDRO` is a `torch.nn.Module`, so it composes with any optimizer:

```python
from torchregress.algorithms import ContinuousGroupDRO

dro = ContinuousGroupDRO([-4.0, -3.0, -2.0, -1.0, 0.5], eta=0.01, base_loss="mse")
dro.fit_group_edges(y_train)          # not needed when edges are isotropic/physical

model = make_model()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for x, y in loader:
    optimizer.zero_grad()
    loss = dro(model(x), y)           # updates q in training mode
    loss.backward()
    optimizer.step()

# Validation: freeze the adversarial weights.
dro.eval()
with torch.no_grad():
    val_loss = dro(model(x_val), y_val)
```

`group_losses(y_pred, target)` returns the per-regime losses without updating $q$ (useful for diagnostics), and `group_weights()` returns the current simplex weights.

---

## Comparison with related objectives

| Objective | What is equalized / penalized | Needs target bins? | Robust to corrupted bulk samples? |
|:----------|:------------------------------|:-------------------|:----------------------------------|
| Plain MSE | Mean error over the empirical mixture | No | No — bulk dominates |
| [LDS](../../losses/imbalanced.md) / [Balanced MSE](../../losses/imbalanced.md) | Re-weighted or distribution-corrected mean | Density estimate | No — reweights points, not regimes |
| CVaR (`CVaRLoss`) | Mean of the worst $\alpha$ fraction of **individual** samples | No | No — corrupted bulk samples are the highest losses |
| **Group DRO** | Worst **target-regime mean** | Yes (physical bins) | Yes — noise averages out inside each bin before equalization |
| [PRIME / ConR](../../losses/prime_conr.md) | Feature-space representation of sparse targets | No (continuous targets) | Partially |
| [IRLS](irls.md) | Robust point-wise residual weighting | No | Yes (per-point robustness, not per-regime) |

!!! tip "Choosing bins"
    Bins should be *physically meaningful* and each should contain enough objects for a stable mean (tens of samples at least). For metallicity, the natural bins are `[-4, -3]`, `[-3, -2]`, `[-2, -1]`, `[-1, +0.5]`; for photometric redshift, use redshift intervals of comparable comoving volume. If a bin is nearly empty, either merge it or set `min_group_weight` so it cannot be starved, and report the per-bin loss with its sample count.

---

## Caveats

!!! warning "Assumptions and failure modes"
    * **Group labels are needed.** The bins are derived from the target; if the target is the quantity you are predicting, bins are available at training time (where labels exist). Group DRO assumes the test distribution still contains all regimes.
    * **Worst-group average is not worst-case robustness.** A regime mean still includes corrupted samples; it just averages them. Combine with a robust base loss (`base_loss="mae"` or a Huber module with `reduction="none"`) when outliers are expected.
    * **Sample-starved groups are high variance.** With few samples per bin, EG can oscillate; lower `eta`, increase `update_every`, or set a floor on the group weights.
    * **Adversarial weights can concentrate.** With a large `eta`, $q$ can put almost all mass on one regime, sacrificing the others. Monitor `group_weights()` and cap `eta` for stability.
    * **Validation must not update $q$.** Call `.eval()` (or `GroupDROTrainer.predict`) when measuring validation metrics; otherwise the adversarial weights follow the validation losses.
    * **Not a calibration method.** It changes the fitted model, not the predictive intervals or coverage; re-run conformal/post-hoc calibration after switching to Group DRO.

---

## References

| # | Reference |
|:-:|:----------|
| 1 | Sagawa, S., Koh, P. W., Hashimoto, T. B., & Liang, P. (2020). Distributionally robust neural networks for group shifts: on the importance of regularization for worst-case generalization. *ICLR 2020*. [arXiv:1911.08731](https://arxiv.org/abs/1911.08731) |
| 2 | Oren, Y., Sagawa, S., & Hashimoto, T. B. (2019). Distributionally robust language modeling. *EMNLP 2019*. [arXiv:1909.02060](https://arxiv.org/abs/1909.02060) |
| 3 | Hashimoto, T. B., Srivastava, M., Namkoong, H., & Liang, P. (2018). Fairness without demographics in repeated loss minimization. *ICML 2018*. [arXiv:1806.08010](https://arxiv.org/abs/1806.08010) |
| 4 | Rockafellar, R. T., & Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of Risk*, 2, 21–42. (CVaR comparison.) |

## See also

* [Imbalanced targets](../../losses/imbalanced.md) — LDS, FDS, Balanced MSE, CVaR.
* [PRIME & ConR](../../losses/prime_conr.md) — representation-level imbalanced regression.
* [IRLS](irls.md) — robust reweighting with a different objective.
* [Imbalanced / rare-target regression](../../guide/method-selection.md) — task-level selection matrix.
