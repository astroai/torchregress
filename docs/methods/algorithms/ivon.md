# IVON Optimizer (Improved Variational Online Newton)

IVON is a natural-gradient-based optimization algorithm derived from the **Bayesian Learning Rule** framework (Khan & Rue, 2023). It fits a Gaussian variational posterior $q(\theta) = \mathcal{N}(\theta \mid \mu, \Sigma)$ over model parameters by tracking first and second moments, providing both optimization and lightweight uncertainty quantification in a single algorithm.

!!! abstract "What makes IVON special"
    Unlike Adam/SGD which only produce point estimates, IVON maintains a **full diagonal Gaussian posterior** over every parameter. This means you get:
    - **Parameter uncertainty** estimates "for free" during training
    - **Natural-gradient** updates that are invariant to parameter rescaling
    - **Built-in weight sampling** for Monte Carlo prediction

---

## Mathematical Background

### Variational Gaussian Posterior

IVON maintains a diagonal Gaussian variational posterior:

$$q_t(\theta) = \mathcal{N}(\theta \mid \mu_t, \operatorname{diag}(\sigma_t^2))$$

At each step, it samples $\theta \sim q_t$, computes the gradient, and updates both $\mu$ and the precision $\Sigma^{-1}$ using natural-gradient-style updates.

### Hessian Approximation

IVON uses one of two Hessian approximations to update the precision:

| Method | Formula | Characteristic |
|:-------|:--------|:--------------|
| `"price"` | $H \approx \mathbb{E}[\epsilon \cdot g]$ | Uses Price's theorem / Stein's lemma; noise-grad correlation |
| `"gradsq"` | $H \approx \mathbb{E}[g^2]$ | Squared gradients (similar to RMSprop/Adam second moment) |

where $\epsilon \sim \mathcal{N}(0, \Sigma)$ is the parameter noise and $g = \nabla_\theta \mathcal{L}$ is the gradient at the perturbed parameters.

### Update Rules

For parameter group with learning rate $\eta$, effective sample size $N$, and decay rates $\beta_1, \beta_2$:

$$\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) \bar{g}_t \\
h_t &= \beta_2 h_{t-1} + (1 - \beta_2) H_t + \frac{(1 - \beta_2)^2}{2} \frac{(h_{t-1} - H_t)^2}{h_{t-1} + \lambda} \\
\mu_{t+1} &= \mu_t - \eta \cdot \operatorname{clip}\!\left(\frac{m_t / d_t + \lambda \mu_t}{h_t + \lambda},\; \pm r\right)
\end{aligned}$$

where $d_t = 1 - \beta_1^t$ is a debiasing factor and $r$ is the clipping radius.

---

## Usage

### Basic Training Loop

The key API pattern: wrap the forward/backward pass in `optimizer.sampled_params(train=True)` to activate weight sampling:

```python
import torch
import torch.nn as nn
from torchregress.algorithms import IVON

model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
optimizer = IVON(
    model.parameters(),
    lr=0.1,
    ess=100.0,          # Effective sample size (typically dataset size)
    mc_samples=1,        # MC samples per step
    hess_approx="price", # Hessian approximation method
    weight_decay=1e-4,
)

for x, y in train_loader:
    with optimizer.sampled_params(train=True):
        optimizer.zero_grad()
        pred = model(x)
        loss = nn.functional.mse_loss(pred, y)
        loss.backward()
    optimizer.step()
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `lr` | `float` | — | Learning rate |
| `ess` | `float` | — | Effective sample size (set to training set size) |
| `hess_init` | `float` | `1.0` | Initial Hessian diagonal value |
| `beta1` | `float` | `0.9` | First moment decay |
| `beta2` | `float` | `0.99999` | Second moment (Hessian) decay |
| `weight_decay` | `float` | `1e-4` | L2 regularization coefficient |
| `mc_samples` | `int` | `1` | Number of MC samples per `step()` |
| `hess_approx` | `str` | `"price"` | `"price"` (Stein's lemma) or `"gradsq"` (squared gradients) |
| `clip_radius` | `float` | `inf` | Gradient clipping radius |
| `debias` | `bool` | `True` | Apply bias correction to momentum |
| `rescale_lr` | `bool` | `True` | Rescale lr by `(hess_init + wd)` |

### Multi-Sample Training

For better gradient estimates, increase `mc_samples` — the optimizer averages gradients over multiple weight samples:

```python
optimizer = IVON(model.parameters(), lr=0.1, ess=100.0, mc_samples=4)

def closure():
    optimizer.zero_grad()
    with optimizer.sampled_params(train=True):
        pred = model(x)
        loss = nn.functional.mse_loss(pred, y)
        loss.backward()
    return loss

loss = optimizer.step(closure)
```

### Uncertainty Estimation

The learned posterior enables MC prediction at test time:

```python
model.eval()
predictions = []
for _ in range(30):
    with optimizer.sampled_params(train=False):
        pred = model(x_test)
        predictions.append(pred.detach())

preds = torch.stack(predictions)
mean = preds.mean(0)
epistemic_std = preds.std(0)
```

---

## When to Use IVON

| Scenario | Recommended? | Rationale |
|:---------|:-----------:|:----------|
| **You want parameter uncertainty** from a standard optimizer | ✅ **Yes** | IVON gives you posterior samples without changing your model architecture |
| **Large-scale training** (millions of params) | ✅ | Diagonal Gaussian scales linearly with parameter count |
| **You need fast iteration** with familiar optimizer API | ✅ | Drop-in replacement for Adam with similar hyperparameter semantics |
| **You need exact Bayesian inference** | ❌ | Diagonal Gaussian is a mean-field approximation; use HMC or full Laplace for higher fidelity |
| **Your loss is highly non-convex with many modes** | ⚠️ | A single Gaussian posterior captures only one mode |
| **Extremely small datasets** ($N < 100$) | ⚠️ | The `ess` parameter interacts with learning rate; tune carefully |

---

## Comparison with Other Optimizers

| Feature | Adam/AdamW | IVON | SGD + SWAG |
|:--------|:----------:|:----:|:----------:|
| Parameter uncertainty | ❌ | ✅ (diagonal posterior) | ✅ (post-hoc covariance) |
| Natural gradient | ❌ | ✅ | ❌ |
| Memory overhead | 2× params | 2× params | 3× params (SWAG) |
| Training speed | Fast | Slightly slower (sampling) | Fast |
| Drop-in replacement | — | ✅ (similar API) | ❌ (separate collection phase) |

---

## Limitations

!!! warning "Practical constraints"
    - **Single-device only**: All parameters must be on the same device and dtype. Multi-GPU training requires manual sharding.
    - **Mean-field approximation**: The diagonal Gaussian posterior ignores weight correlations, underestimating epistemic uncertainty compared to full-covariance methods.
    - **Learning rate sensitivity**: The effective learning rate depends on `ess`, `hess_init`, and `rescale_lr`. When `rescale_lr=True`, the actual step size is $\eta \cdot (\text{hess\_init} + \lambda)$, which differs from the Adam convention.
    - **`mc_samples > 1` overhead**: Each additional MC sample requires a separate forward/backward pass. For large models, `mc_samples=1` is recommended and the variance is managed through the optimizer's internal tracking.
    - **Not a posterior over predictions**: IVON's posterior is over weights, not directly over predictions. For well-calibrated predictive uncertainty, combine with a heteroscedastic head or ensemble.
    - **Cold posterior risk**: If `ess` is set too large or training runs for many steps, the posterior variance ($1/h$) can become extremely small, effectively freezing parameters — analogous to the "cold posterior" effect in Bayesian deep learning.
    - **`torch.compile` incompatibility**: The `sampled_params()` context manager modifies parameters in-place and restores them, which is generally incompatible with `torch.compile`. Use standard eager mode with IVON.
    - **Gradient accumulation**: `sampled_params(train=True)` resets internal state on each call. Standard gradient accumulation patterns (multiple backward passes before one optimizer step) are not supported.

---

## Next steps

- [Ensemble methods](../ensemble/index.md) — combine multiple IVON-trained models for richer uncertainty
- [Effective Bayesian Laplace](heteroscedastic_laplace.md) — post-hoc uncertainty decomposition via natural-parameter Laplace
- [Adaptive-Prior VI](adaptive_prior_vi.md) — context-aware variational inference for covariate-shifted test data
- [Gaussian losses](../../losses/gaussian.md) — pair IVON's weight uncertainty with a heteroscedastic output head

---

## References

| # | Reference |
|:-:|:----------|
| 1 | M.E. Khan, H. Rue. ["The Bayesian Learning Rule."](https://arxiv.org/abs/2107.04562) *JMLR*, 24(214):1–45, **2023**. |
| 2 | Y. Shen et al. ["Variational Learning is Effective for Large Deep Networks."](https://arxiv.org/abs/2402.17641) *ICML*, **2024**. |
