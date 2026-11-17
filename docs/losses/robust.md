# Robust Loss Functions

> ← [Wasserstein Bound](gaussian_wasserstein.md) | [Quantile & Expectile](quantile_expectile.md) →

Robust loss functions are designed to mitigate the influence of **outliers** — observations that deviate significantly from the primary data-generating process. Unlike standard Squared Error (MSE), which grows quadratically with the residual $r = y - \hat{y}$, robust losses grow more slowly (linearly or sub-linearly) at the tails.

---

## The Influence Function

To understand robustness, we examine the **influence function** $\psi(r) = \frac{\partial \rho}{\partial r}$. For MSE, $\psi(r) = 2r$, meaning a single large residual has an unbounded, linear influence on the gradients. Robust losses **bound** or **suppress** this influence.

| Loss | Function $\rho(r)$ | Influence $\psi(r)$ | API Reference |
|:-----|:-------------------|:-------------------|:--------------|
| **MSE** | $r^2$ | $2r$ (unbounded) | [WeightedMSELoss](../api/losses.md) |
| **MAE** | $|r|$ | $\text{sgn}(r)$ (bounded) | [WeightedL1Loss](../api/losses.md) |
| **Huber** | Piecewise $L_2/L_1$ | $\text{clamp}(r, -\delta, \delta)$ | [WeightedHuberLoss](../api/losses.md) |
| **Cauchy** | $\log(1 + r^2/c^2)$ | $\frac{2r}{c^2 + r^2} \rightarrow 0$ | [CauchyLoss](../api/losses.md) |
| **Tukey** | Redescending | $0$ for $|r| > c$ | [TukeyBiweightLoss](../api/losses.md) |
| **Barron** | General robust family | Varies with $\alpha$ | [BarronLoss](../api/losses.md) |

---

## Core Robust Losses

### Huber & Pseudo-Huber

The **Huber Loss** \[1\] is the "gold standard" for robust regression. It provides the stability of $L_2$ near zero (ensuring a unique minimum) and the robustness of $L_1$ for large residuals.

$$\mathcal{L}_{\text{Huber}}(r; \delta) = \begin{cases} \tfrac{1}{2}r^2 & |r| \leq \delta \\ \delta\,|r| - \tfrac{1}{2}\delta^2 & |r| > \delta \end{cases}$$

The **Pseudo-Huber Loss** ([PseudoHuberLoss](../api/losses.md)) is a smooth, $C^\infty$ approximation that is easier to optimise with second-order methods:

$$\mathcal{L}_{\text{PH}}(r; \delta) = \delta^2 \left( \sqrt{1 + (r/\delta)^2} - 1 \right)$$

```python
from torchregress.losses import WeightedHuberLoss, PseudoHuberLoss

# Traditional Huber (weighted wrapper)
loss_fn = WeightedHuberLoss(delta=1.345) # 95% efficiency for Gaussian

# Smooth Pseudo-Huber
loss_fn = PseudoHuberLoss(delta=1.0)
```

---

### Redescending Losses (Cauchy & Tukey)

"Redescending" losses are the most aggressive form of robustness. Their influence function $\psi(r)$ actually **returns to zero** for extreme residuals, meaning the model effectively "ignores" outliers once they are confirmed as such.

**Cauchy Loss** ([CauchyLoss](../api/losses.md)):

$$\mathcal{L}_{\text{Cauchy}}(r; c) = \log\left(1 + \left(\frac{r}{c}\right)^2\right)$$

**Tukey Biweight** ([TukeyBiweightLoss](../api/losses.md)):

The ultimate outlier rejection tool \[2\]. Samples with $|r| > c$ contribute **zero gradient** to the model.

$$\mathcal{L}_{\text{Tukey}}(r; c) = \begin{cases} \frac{c^2}{6} \left[ 1 - (1 - (r/c)^2)^3 \right] & |r| \leq c \\ \frac{c^2}{6} & |r| > c \end{cases}$$

!!! warning "Optimization Limitations of Redescending Losses"
    - **Non-Convexity & Local Minima**: Cauchy and Tukey losses are highly non-convex. If initialized randomly, optimization can get stuck in poor local minima or fail to converge.
    - **Zero Gradients (Tukey)**: Tukey biweight completely zeroes out the gradients of samples with residuals $|r| > c$. If $c$ is set too small initially, the model may stop learning.
    - **Warm-Up Strategy**: **Always** pre-train (warm-up) your model for a few epochs using a convex loss function like `WeightedMSELoss` or `WeightedHuberLoss` before switching to Cauchy or Tukey losses.

---

### General Robust Family (Barron & AdaptiveRobust)

[BarronLoss](../api/losses.md) provides a single smooth family that spans several useful robustness regimes through the shape parameter $\alpha$ \[4\]:

$$
\mathcal{L}_{\text{Barron}}(r; \alpha, c) =
\begin{cases}
\frac{1}{2}(r/c)^2 & \alpha = 2 \\
\log\left(1 + \frac{1}{2}(r/c)^2\right) & \alpha = 0 \\
\frac{|\alpha - 2|}{\alpha}\left(\left(\frac{(r/c)^2}{|\alpha - 2|} + 1\right)^{\alpha/2} - 1\right) & \text{otherwise}
\end{cases}
$$

Practical reading:

- $\alpha \approx 2$: near-quadratic, fastest on clean data
- $\alpha \approx 1$: Huber-like compromise
- $\alpha \approx 0$: Cauchy-like heavy-tail robustness
- $\alpha < 0$: increasingly redescending behavior

[AdaptiveRobustLoss](../api/losses.md) keeps the same loss family but makes $\alpha$ and $c$ trainable. This is useful when the tail behavior is unknown up front and you want the optimizer to learn a robust regime jointly with the model.

```python
import torch
from torchregress.losses import AdaptiveRobustLoss, BarronLoss

# Fixed-shape Barron loss
loss_fn = BarronLoss(alpha=0.5, scale=1.0)

# Jointly learn the robust shape and scale with the model
adaptive_loss = AdaptiveRobustLoss(alpha_init=1.0, scale_init=1.0)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(adaptive_loss.parameters()),
    lr=1e-3,
)
```

!!! tip
    Start with `BarronLoss(alpha=1.0, scale=1.0)` when you want one default that sits between Huber and Cauchy. Use `AdaptiveRobustLoss` when the residual tail shape is unclear and you are willing to optimize a few extra parameters.

---

### Student-t likelihood with known precisions

When every observation carries a known inverse variance $\lambda_i = 1/\sigma_i^2$ (survey pixels, photometry, instrument noise models), the Student-$t$ likelihood is a robust *likelihood*, not just a robust loss: its normalising constants make deviances comparable across models and data sets \[6\].

[`student_t_nll`](../api/losses.md#robust-losses) is the exact negative log-likelihood of $t_\nu(0, \sigma_i^2)$, masked and summed per row:

$$
-\log p(\mathbf r \mid \nu, \boldsymbol\lambda)
= \sum_{i \in \mathcal O} \left[\log\Gamma\!\left(\tfrac{\nu}{2}\right) - \log\Gamma\!\left(\tfrac{\nu+1}{2}\right) + \tfrac12\log(\nu\pi) - \tfrac12\log\lambda_i + \tfrac{\nu+1}{2}\log\!\left(1 + \frac{\lambda_i r_i^2}{\nu}\right)\right],
$$

where $\mathcal O$ is the set of observed (unmasked) entries. `nu=math.inf` is the Gaussian NLL $\sum_i \tfrac12\lambda_i r_i^2 - \tfrac12\log\lambda_i + \tfrac12\log 2\pi$. The scale is $\sigma_i$ itself, so any $\nu > 0$ is valid and the IRLS weight is

$$
w_i = \lambda_i\,\frac{\nu + 1}{\nu + \lambda_i r_i^2},
$$

which is `robust_weights(u, "student_t", {"nu": nu})` at the standardized residual $u_i = r_i\sqrt{\lambda_i}$, times $\lambda_i$.

**Variance model.** [`effective_precision`](../api/losses.md#robust-losses) adds an error proportional to the model $m_i$ (a fractional template or calibration error $\eta$) to the known noise:

$$
\lambda_{\mathrm{eff},i} = \frac{1}{1/\lambda_i + \eta^2 m_i^2} = \frac{\lambda_i}{1 + \eta^2 m_i^2 \lambda_i}.
$$

`eta` is a float or a tensor broadcastable to the precisions (e.g. one value per row, `[B, 1]`). A tensor `eta` is validated (finite, $\ge 0$) with one value read back to the host, so on a GPU each call synchronises the device; a float `eta` costs no synchronisation.

**Efficiency.** [`student_t_efficiency`](../api/losses.md#robust-losses) returns $E[\psi']^2/E[\psi^2]$ for the score $\psi(z) = (\nu+1)z/(\nu+z^2)$: the information the $t_\nu$ estimator extracts per unit Gaussian information. Under $t_\nu$ noise (`noise="student_t"`) it is the Fisher factor $(\nu+1)/(\nu+3)$; under Gaussian noise (`noise="gaussian"`, the default) it is a closed form in $\operatorname{erfc}$ for $\nu < 3$ and 200-node Gauss–Hermite quadrature above, both within a few $10^{-15}$ (relative) of high-precision quadrature for every $\nu > 0$ ($0.942$ at $\nu = 5$; it falls like $\sqrt\nu$ as $\nu \to 0$). Use it to turn a Gaussian Fisher information into the asymptotic information of a $t_\nu$ fit.

```python
import math

import torch
from torchregress.losses import effective_precision, student_t_efficiency, student_t_nll

torch.manual_seed(0)
model = torch.ones(4, 256, dtype=torch.float64)
precision = torch.full_like(model, 100.0)
target = model + 0.1 * torch.randn_like(model)
target[:, ::50] += 5.0  # cosmic rays
mask = torch.ones_like(model, dtype=torch.bool)
mask[:, :8] = False  # bad pixels

prec_eff = effective_precision(precision, model, eta=0.02)
nll_t = student_t_nll(target - model, prec_eff, nu=5.0, mask=mask)  # [4]
nll_gauss = student_t_nll(target - model, prec_eff, nu=math.inf, mask=mask)
print(nll_t, nll_gauss, student_t_efficiency(5.0), student_t_efficiency(5.0, "student_t"))
```

!!! tip "Many small fits"
    To *fit* thousands of independent rows under this likelihood (one spectrum, light curve, or source per row), use [`batched_iteratively_reweighted_fit`](../methods/algorithms/irls.md#batched-irls-for-many-independent-problems): it reweights with $w_i$, applies $\eta$, and reports this NLL per row.

---

## Comparison & Selection

### Tradeoff Matrix

| Metric | MSE | Huber | Barron | Student-$t$ | Cauchy | Tukey |
|:-------|:---:|:-----:|:------:|:-----------:|:------:|:-----:|
| **Convergence Speed** | 🚀 | 🚄 | 🚄 | 🚗 | 🚗 | 🐢 |
| **Outlier Rejection** | ❌ | 🆗 | ✅ | ✅ | ✅ | 🏆 |
| **Convexity** | ✅ | ✅ | depends on $\alpha$ | ❌ | ❌ | ❌ |
| **Smoothness ($C^2$)** | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ |
| **Normalised likelihood** | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |

### Decision Flowchart

```mermaid
graph TD
    Start["New Regression Problem"] --> Outliers{"Contains Outliers?"}
    Outliers -->|No| MSE["WeightedMSELoss (Standard)"]
    Outliers -->|Maybe/Mild| Huber["WeightedHuberLoss (delta=1.0)"]
    Outliers -->|Unknown tail shape| Barron["BarronLoss (alpha≈1.0)"]
    Outliers -->|Confirmed/Severe| Redescending{"Severe/Adversarial?"}
    Redescending -->|Aggressive| Cauchy["CauchyLoss (c=1.0)"]
    Redescending -->|Extreme| Tukey["TukeyBiweight (c=4.685)"]
    Barron --> Adaptive{"Need alpha/scale learned?"}
    Adaptive -->|Yes| ABarron["AdaptiveRobustLoss"]
    Adaptive -->|No| Barron

    Huber --> Smooth{"Need Smoothness?"}
    Smooth -->|Yes| PHuber["PseudoHuberLoss"]
    Smooth -->|No| Huber
```

---

## Advanced: Tail Sensitivity with CVaR

While robust losses *ignore* outliers, **CVaR (Conditional Value at Risk)** \[3\] does the opposite — it focuses exclusively on the **hardest** samples. This is useful for:

- Ensuring fairness across sub-populations.
- Minimising worst-case error.
- Training models that must be reliable on the tails.

```python
from torchregress.losses import CVaRLoss

# Train on the worst 5% of samples using Huber as the base loss
loss_fn = CVaRLoss(alpha=0.05, base_loss="huber", delta=1.5)
```

---

## References

| # | Reference |
|:-:|:----------|
| 1 | Huber, P. J. ["Robust Estimation of a Location Parameter."](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-35/issue-1/Robust-Estimation-of-a-Location-Parameter/10.1214/aoms/1177703732.full) *Annals of Math. Stat.*, 1964. |
| 2 | Tukey, J. W. *Exploratory Data Analysis*. Addison-Wesley, 1977. |
| 3 | Rockafellar, R. T., & Uryasev, S. ["Conditional Value-at-Risk for General Loss Distributions."](https://www.sciencedirect.com/science/article/pii/S037842660200271X) *J. Banking & Finance*, 2002. |
| 4 | Barron, J. T. ["A General and Adaptive Robust Loss Function."](https://arxiv.org/abs/1701.03077) *CVPR*, 2019. |
| 5 | Belagiannis et al. ["Robust Optimization for Deep Regression."](https://arxiv.org/abs/1505.06641) *ICCV*, 2015. |
| 6 | Lange, K. L., Little, R. J. A., & Taylor, J. M. G. ["Robust Statistical Modeling Using the t Distribution."](https://doi.org/10.1080/01621459.1989.10478852) *J. Amer. Statist. Assoc.*, 84(408):881–896, 1989. |

---

## Next steps

- [Gaussian losses](gaussian.md) — when you don't need robustness (standard regression)
- [CVaRLoss](../api/losses.md) — focus on the worst-α fraction instead of ignoring outliers
- [Evaluation with proper scoring rules](../metrics/distribution.md) — CRPS, NLL, energy score
- [Mathematical Foundations](../guide/math/index.md) — CVaR derivation and scoring rule proofs
