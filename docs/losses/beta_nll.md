# Beta-NLL (heteroscedastic Gaussian)

**β-NLL** rescales the per-element diagonal Gaussian negative log-likelihood with a **detached** function of the predicted variance so that gradients do not collapse variance for the wrong reasons. It is a drop-in sibling of [`GaussianNLLLoss`](gaussian.md) for the same model outputs $(\mu, \log\sigma^2)$.

---

## When to use which

| Situation | Prefer |
|:----------|:-------|
| Stable heteroscedastic training; variance collapse or trivial σ² | [`BetaNLLLoss`](../api/losses.md) with $\beta \in (0, 1)$ (often $0.5$) |
| Standard NLL baseline; well-tuned optimisation | [`GaussianNLLLoss`](../api/losses.md) |
| Finite-sample **coverage** guarantees | [Conformal prediction](../methods/conformal/index.md) on top of a calibrated probabilistic model — β-NLL is **not** conformal |

---

## Mathematical definition

Let $\sigma^2_i$ be the predicted variance for dimension $i$ (from $\log\sigma^2_i$), and let $\mathcal{L}_{\text{NLL},i}$ be the usual diagonal Gaussian NLL contribution including the $\log(2\pi)$ term (matching [`GaussianNLLLoss`](../api/losses.md)).

The **β-NLL** uses a prefactor with **stopped variance**:

$$
\boxed{\;
\mathcal{L}_{\beta\text{-NLL},i}
=
(\sigma_i^2 + \varepsilon)^{-\beta}
\cdot
\mathcal{L}_{\text{NLL},i}
\;}
$$

where $(\sigma_i^2 + \varepsilon)^{-\beta}$ is computed from **detached** $\sigma_i^2$ (no gradient through the prefactor). For $\beta = 0$ this is exactly [`GaussianNLLLoss`](../api/losses.md).

---

## API

```python
from torchregress.losses import BetaNLLLoss, beta_nll_loss, GaussianNLLLoss

# Class (same y_pred layouts as GaussianNLLLoss: tuple or concatenated)
loss_fn = BetaNLLLoss(beta=0.5, reduction="mean")
loss = loss_fn((mean, log_var), target, mask=mask, weights=w)

# Functional
loss = beta_nll_loss((mean, log_var), target, beta=0.5, reduction="mean")
```

[create_loss_from_config](../api/losses.md) accepts `"type": "beta_nll"` with optional `"beta"`.

---

## Runnable example

See the [Beta-NLL heteroscedastic demo](../examples/heteroscedastic_beta_nll.md) and the script `examples/heteroscedastic_beta_nll_demo.py` in the repository.

---

## References

| # | Reference |
|:-:|:----------|
| 1 | Seitzer et al. ["On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks"](https://arxiv.org/abs/2205.11310) *NeurIPS*, 2022. |

---

## Next steps

- [Gaussian losses](gaussian.md) — multivariate and low-rank variants
- [Conformal prediction](../methods/conformal/index.md) — intervals with coverage guarantees
