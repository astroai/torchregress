# Gaussian Loss Functions

> ← [Base Classes](base.md) | [Beta-NLL](beta_nll.md) →

Gaussian (or Normal) loss functions are the workhorses of regression. They range from the standard Mean Squared Error (MSE) to sophisticated multivariate models that capture complex correlations between targets.

---

## Mathematical Background

In the probabilistic framework, regression is viewed as maximizing the likelihood of the data under a Gaussian noise model $p(y \mid x) = \mathcal{N}(\mu(x), \sigma^2(x))$. Minimising the **Negative Log-Likelihood (NLL)** is equivalent to:

$$\boxed{\;\mathcal{L}_{\text{NLL}}(y, \mu, \sigma^2) = \frac{1}{2}\log(2\pi\sigma^2) + \frac{(y - \mu)^2}{2\sigma^2}\;}$$

- **The Residual Term**: $(y - \mu)^2 / 2\sigma^2$ penalises predictions far from the truth, scaled by the uncertainty.
- **The Penalty Term**: $\frac{1}{2}\log(2\pi\sigma^2)$ prevents the model from simply predicting infinite variance to "cheat" the residual term.

---

## Comparison of Gaussian Losses

| Loss | Covariance Type | Outputs | API Reference | Best For |
|:-----|:----------------|:--------|:--------------|:---------|
| **`WeightedMSELoss`** | Fixed ($\sigma^2=1$) | $\mu$ | [WeightedMSELoss](../api/losses.md) | Homoscedastic, clean data |
| **`GaussianNLLLoss`** | Diagonal | $(\mu, \log\sigma^2)$ | [GaussianNLLLoss](../api/losses.md) | Heteroscedastic, independent targets |
| **`BetaNLLLoss`** | Diagonal | $(\mu, \log\sigma^2)$ | [BetaNLLLoss](../api/losses.md) | Same head as NLL; detached variance rescaling (β-NLL) |
| **`FaithfulGaussianLoss`** | Diagonal | $(\mu, \log\sigma^2)$ | [FaithfulGaussianLoss](../api/losses.md) | MSE on $\mu$ + NLL on variance with **detach($\mu$)** in residual |
| **`GaussianWassersteinBoundLoss`** | Configurable | $\mu$, cov / Cholesky / root | [GaussianWassersteinBoundLoss](../api/losses.md) | Supervise mean + covariance vs labels or pseudo-labels |
| **`MultivariateGaussianLoss`** | Full | $(\mu, \Sigma)$ | [MultivariateGaussianLoss](../api/losses.md) | Correlated multi-output (small $k$) |
| **`LowRankGaussianLoss`** | Low-Rank + Diag | $(\mu, W, d)$ | [LowRankGaussianLoss](../api/losses.md) | Correlated multi-output (large $k$) |

---

## 1. Univariate: [GaussianNLLLoss](../api/losses.md)

Used for standard regression where you want to estimate per-sample uncertainty.

```python
from torchregress.losses import GaussianNLLLoss

# Model must output [batch, 2] -> [mean, log_var]
loss_fn = GaussianNLLLoss()
loss = loss_fn(y_pred, y_true)
```

!!! tip "Numerical Stability"
    Always predict **log-variance** ($s$) rather than raw variance ($\sigma^2$). This ensures positivity ($e^s > 0$) and provides a more stable loss landscape for optimization.

---

## 1b. Faithful heteroscedastic: [FaithfulGaussianLoss](../api/losses.md)

Joint Gaussian NLL couples gradients from the variance head into the mean through the residual $(y-\mu)^2/\sigma^2$. **FaithfulGaussianLoss** adds a direct MSE term on the mean and uses **stop-gradient** on $\mu$ inside the NLL residual so variance calibration does not distort point prediction.

$$\mathcal{L}_{\text{Faithful}}(y, \mu, \sigma^2) = w_{\text{mean}} (y - \mu)^2 + w_{\text{var}} \left[ \frac{1}{2}\log(2\pi\sigma^2) + \frac{(y - \operatorname{sg}(\mu))^2}{2\sigma^2} \right]$$

```python
from torchregress.losses import FaithfulGaussianLoss

loss_fn = FaithfulGaussianLoss(mean_weight=1.0, variance_weight=1.0)
loss = loss_fn((mean, log_var), y_true)
```

Compare with [`BetaNLLLoss`](beta_nll.md): β-NLL keeps a single joint NLL and rescales it with a detached variance; faithful loss **explicitly splits** mean vs variance terms.

---

## 2. Multivariate: Full Covariance ([MultivariateGaussianLoss](../api/losses.md))

When targets are correlated (e.g., predicting $x, y, z$ coordinates), use `MultivariateGaussianLoss`.

$$\mathcal{L}_{\text{MV}}(y, \mu, \Sigma) = \frac{1}{2} \left[ \log|\Sigma| + (y - \mu)^\top \Sigma^{-1} (y - \mu) + k\log(2\pi) \right]$$

!!! warning "Limitations"
    - **Computational Scaling**: Solvers and determinants for a $K \times K$ covariance matrix scale as $\mathcal{O}(K^3)$ with target dimension, which makes full covariance impractical for high-dimensional outputs ($K > 10$).
    - **PSD Violations**: Cholesky factors must be regularized/clamped on the diagonal to prevent non-positive semi-definite covariance matrices during optimization.
    - **FaithfulGaussianLoss tradeoff**: The `stop_gradient` on $\mu$ inside the NLL residual decouples the variance head's gradients from the mean head — preventing variance miscalibration from distorting point predictions, but also meaning the mean head does **not** receive curvature information about heteroscedasticity. This can slow convergence on datasets where mean and variance are strongly coupled.
    - **LowRankGaussianLoss rank selection**: The rank $r$ is a fixed hyperparameter. Start with $r = \lfloor K/3 \rfloor$ and tune on validation NLL. If $r$ is too small, the diagonal component must absorb residual correlation, inflating marginal variances. If too large, parameter count approaches the full-covariance case and computational benefits vanish.

```python
from torchregress.losses import MultivariateGaussianLoss

# Requires full covariance matrix Σ (batch, k, k)
loss_fn = MultivariateGaussianLoss()
loss = loss_fn(y_mu, y_true, covariance_matrix)
```


---

## 3. Multivariate: Low-Rank Covariance ([LowRankGaussianLoss](../api/losses.md))

For high-dimensional targets, a full covariance matrix has $O(k^2)$ parameters. **Low-Rank** approximation reduces this to $O(k \cdot r)$, where $r$ is the rank.

$$\Sigma = W W^\top + \text{diag}(d)$$

```python
from torchregress.losses import LowRankGaussianLoss

# W is factor matrix (batch, k, r), d is diagonal (batch, k)
loss_fn = LowRankGaussianLoss()
loss = loss_fn(mu, y_true, W, d)
```

---

## Limitations

1. **Computational scaling of full covariance**: `MultivariateGaussianLoss` requires $\mathcal{O}(K^3)$ operations per sample for determinants and solves, making it impractical beyond $K \approx 10$. Use `LowRankGaussianLoss` with rank $r = \lfloor K/3 \rfloor$ for higher-dimensional targets.
2. **PSD violations**: Cholesky factors must be regularised on the diagonal to prevent non-positive-semidefinite covariance matrices. Always use a minimum variance floor (`min_variance` or jitter $\ge 10^{-6}$).
3. **LowRank rank selection**: If $r$ is too small, the diagonal component absorbs residual correlation, inflating marginal variances. If too large, parameter count approaches the full-covariance case. Tune $r$ on validation NLL.
4. **Gaussian assumption**: The NLL is only a proper scoring rule when $Y \mid X$ is truly Gaussian. For heavy-tailed, multimodal, or bounded targets, consider robust alternatives ([Robust losses](robust.md)), [MDN](mdn.md), or [Normalizing flows](nflows.md).
5. **Variance collapse**: Without a minimum variance floor, the model can learn $\sigma^2 \to 0$ to drive the NLL to $-\infty$, especially on small datasets. Always set `min_variance` and monitor predicted variance during training.

## Best Practices

!!! success "Stable Training"

    1. **Initialise log-var to zero**: This starts the model with $\sigma^2 \approx 1$.
    2. **Use a small `min_variance`**: Prevents the loss from blowing up if the model becomes too confident (default $10^{-6}$).
    3. **Weight Decay**: Apply higher weight decay to the uncertainty head to prevent "explaining away" all error as aleatoric noise.

---

## References

| # | Reference |
|:-:|:----------|
| 1 | Nix & Weigend. ["Estimating the Mean and Variance of the Target Probability Distribution."](https://ieeexplore.ieee.org/document/341257) *ICNN*, 1994. |
| 2 | Kendall & Gal. ["What Uncertainties Do We Need in Bayesian Deep Learning?"](https://arxiv.org/abs/1703.04977) *NeurIPS*, 2017. |
| 3 | Skafte et al. ["Reliable Training and Estimation of Variance Networks."](https://arxiv.org/abs/1906.01511) *NeurIPS*, 2019. |

---

## Next steps

Continue with the Gaussian family:

- [Beta-NLL](beta_nll.md) — stabilised heteroscedastic training via detached variance rescaling
- [Faithful Gaussian](faithful_gaussian.md) — decouple mean and variance gradients for cleaner calibration
- [Gaussian Wasserstein bound](gaussian_wasserstein.md) — supervise covariance directly against labels or pseudo-labels

Or branch out:

- [Robust losses](robust.md) — when Gaussian assumptions fail due to outliers
- [Ensemble methods](../methods/ensemble/index.md) — decompose into aleatoric + epistemic
- [Multivariate example](../examples/normalizing_flows_multitarget.md) — correlated multi-output in practice

## Recommendations

- **Start simple**: Begin with `WeightedMSELoss` (homoscedastic) or `GaussianNLLLoss` (heteroscedastic). Move to `FaithfulGaussianLoss` only if you observe variance inflation distorting mean predictions.
- **Low-rank for high-dimensional targets**: For $K > 5$ correlated targets, use `LowRankGaussianLoss`. For $K \le 5$, `MultivariateGaussianLoss` is simpler and more interpretable.
- **Stable training recipe**: Initialise `log_var` $\approx 0$ (variance $\approx 1$), set `min_variance = 1e-6`, and apply higher weight decay to the variance head (2–5× the mean head).
- **Monitor with CRPS**: Track `crps_gaussian` alongside NLL during training. CRPS is in target units and less sensitive to tail events than NLL. See [Distribution metrics](../metrics/distribution.md).
- **Variance calibration**: After training, validate with a PIT histogram or calibration error (ECE). If variance is systematically over- or under-estimated, apply post-hoc [Variance Temperature Scaling](../methods/calibration.md). See [FaithfulGaussianLoss demo](../losses/faithful_gaussian.md).
- **For coverage guarantees**: Gaussian likelihood losses provide no coverage guarantees. For finite-sample prediction intervals, wrap with [SplitConformal or CQR](../methods/conformal/predictors.md).
