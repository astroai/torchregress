# Beta-NLL (heteroscedastic Gaussian)

> ← [Gaussian](gaussian.md) | [Faithful Gaussian](faithful_gaussian.md) →

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

## Limitations

1. **Beta selection is heuristic**: $\beta$ controls the variance-rescaling strength but has no closed-form optimum. Typical values are $\beta = 0.5$ (recommended default from Seitzer et al., 2022). Values near 0 approach standard NLL; values near 1 aggressively downweight variance gradients. Tune on validation NLL.
2. **Not a conformal method**: β-NLL stabilises variance training but provides no coverage guarantees. For finite-sample prediction intervals, wrap with [SplitConformal or CQR](../methods/conformal/predictors.md).
3. **Same head as GaussianNLL**: β-NLL uses the same $(\mu, \log\sigma^2)$ output as standard NLL. It cannot fix a poorly designed model head — only how variance gradients flow during training.

## Recommendations

- **Default**: Start with `BetaNLLLoss(beta=0.5)`. This is the recommended value from the Seitzer et al. paper and works well across a broad range of regression problems.
- **When NLL is sufficient**: If standard `GaussianNLLLoss` trains stably without variance collapse, β-NLL offers no benefit. Don't add complexity you don't need.
- **Monitor variance**: Track predicted $\sigma^2$ during training. If it drifts below $10^{-3}$ or above $10^2$, the β parameter may need adjustment.
- **Alternative for decoupling**: If you specifically want to preserve point prediction accuracy while learning variance, consider [FaithfulGaussianLoss](faithful_gaussian.md) instead — it explicitly splits mean and variance objectives.

## Next steps

- [Gaussian losses](gaussian.md) — standard NLL, CRPS, multivariate and low-rank variants
- [Faithful Gaussian](faithful_gaussian.md) — explicit mean/variance decoupling (alternative to β-rescaling)
- [Conformal prediction](../methods/conformal/index.md) — intervals with finite-sample coverage guarantees
- [Calibration metrics](../metrics/calibration.md) — validate that β-NLL variance estimates are well-calibrated
