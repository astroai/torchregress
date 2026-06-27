# Gaussian Wasserstein bound surrogate

> ← [Faithful Gaussian](faithful_gaussian.md) | [Robust Losses](robust.md) →

This loss supervises a **predicted Gaussian** $ \mathcal{N}(\hat\mu, \hat\Sigma) $ against a **target mean** $ \mu $ and **target covariance** $ \Sigma $ (or a pseudo-covariance) using a standard **Frobenius surrogate** built from principal matrix square roots. It is useful for joint mean–covariance learning; it is **not** a drop-in replacement for [`GaussianNLLLoss`](gaussian.md) on raw targets.

---

## When to use which

| Situation | Prefer |
|:----------|:-------|
| Supervised / pseudo-supervised covariance targets | [GaussianWassersteinBoundLoss](../api/losses.md) with matching `covariance_parameterization` |
| Likelihood training on $(y \mid x)$ with heteroscedastic noise | [`GaussianNLLLoss`](gaussian.md) or [`BetaNLLLoss`](beta_nll.md) |
| Finite-sample **coverage** guarantees | [Conformal prediction](../methods/conformal/index.md) on calibrated predictive models |

---

## Mathematical definition

Let $ \hat S $ and $ S $ denote symmetric **principal square roots** of $ \hat\Sigma $ and $ \Sigma $ (implemented via eigen-decomposition with a small eigenvalue floor). The per-sample objective is

$$
\boxed{\;
\mathcal{L}
=
\lambda_{\mu}\,\|\hat\mu - \mu\|_2^2
+
\lambda_{\Sigma}\,\|\hat S - S\|_F^2
\;}
$$

In **diagonal** mode, the covariance term is
$ \sum_i \bigl(\sqrt{\hat v_i} - \sqrt{v_i}\bigr)^2 $ for positive variances $ \hat v_i, v_i $.

!!! warning "Surrogate, not exact 2-W"
    This objective is a common **upper-bound style surrogate** related to Gaussian 2-Wasserstein ideas; it should be documented and used as a training signal, not interpreted as the exact Wasserstein-2 distance in all non-commutative cases.

---

## Parameterisations

| `covariance_parameterization` | `pred_covariance` / `target_covariance` |
|:------------------------------|:----------------------------------------|
| `"diagonal"` | Positive variances, same shape as `pred_mean` |
| `"covariance"` | SPD matrices `[B, D, D]` or shared `[D, D]` |
| `"cholesky"` | Lower Cholesky factors $L$ with $\Sigma = L L^\top$ |
| `"sqrt"` | Symmetric roots $S$ (Frobenius term compares them directly) |

The helper `symmetric_spd_matrix_sqrt` (exported from `torchregress.losses`) applies the same root used internally for full matrices; see [`GaussianWassersteinBoundLoss`](../api/losses.md).

---

## API

```python
from torchregress.losses import GaussianWassersteinBoundLoss, gaussian_wasserstein_bound_loss

loss_fn = GaussianWassersteinBoundLoss(
    covariance_parameterization="covariance",
    mean_weight=1.0,
    covariance_weight=1.0,
    jitter=1e-6,
    reduction="mean",
)
loss = loss_fn(pred_mean, target_mean, pred_cov, target_cov, mask=mask, weights=w)
```

[create_loss_from_config](../api/losses.md) accepts ``"type": "gaussian_wasserstein_bound"``.

---

## Runnable example

See [Gaussian Wasserstein bound demo](../examples/gaussian_wasserstein_bound.md) and ``examples/gaussian_wasserstein_bound_demo.py``.

For a **two-phase** pseudo-covariance pretrain then Gaussian NLL fine-tune, see [Wasserstein-bound hybrid pretrain](../examples/wasserstein_bound_hybrid_pretrain.md).

---

## Covariance pseudo-labels (experimental)

When you do not have oracle covariance targets, the research plan suggests **neighbour-weighted local covariance** of ``y`` in input space as a heuristic supervision signal. v1 lives under algorithms (not losses):

- ``torchregress.algorithms.NeighborhoodCovariancePseudoLabeler`` with ``fit_predict(x, y) -> [n, d, d]``
- ``torchregress.algorithms.mahalanobis_covariance_pseudo_labels`` functional wrapper

Metrics supported: pooled **Mahalanobis** in ``x`` (default) or **Euclidean** distances; weights are a **softmax** over neighbour distances. Outputs are symmetrised and eigenvalue-clamped for numerical SPD-ish matrices. Treat as **experimental** and validate on your modality before relying on gradients.

---

## References

| # | Reference |
|:-:|:----------|
| 1 | D.C. Dowson, B.V. Landau. ["The Fréchet Distance between Multivariate Normal Distributions."](https://doi.org/10.1016/0047-259X(82)90077-X) *J. Multivariate Analysis*, 12(3):450–455, **1982**. |
| 2 | J. Delon, A. Desolneux. ["A Wasserstein-type Distance in the Space of Gaussian Mixture Models."](https://doi.org/10.1137/19M1301047) *SIAM J. Imaging Sciences*, 13(2):936–970, **2020**. |

---

## Limitations

1. **Surrogate, not exact Wasserstein**: This objective is a Frobenius-norm upper-bound surrogate related to Gaussian 2-Wasserstein ideas. It is not the exact Wasserstein-2 distance in all non-commutative cases. Treat it as a training signal, not an interpretable distance metric.
2. **Requires covariance targets**: Unlike NLL which learns from raw targets $(x, y)$, this loss requires a target covariance $\Sigma$ per sample (or a shared $\Sigma$). Covariance targets are rarely available — they must come from problem structure, pseudo-labeling, or external estimation.
3. **Pseudo-label quality**: The `NeighborhoodCovariancePseudoLabeler` in `torchregress.algorithms` provides heuristic covariance targets from neighbour-weighted statistics. These are experimental — validate on your modality before relying on gradients.
4. **Eigenvalue floor**: The matrix square root requires eigenvalue decomposition with a floor (jitter) to guarantee positive-semidefiniteness. If jitter is too small, gradients become unstable; if too large, covariance information is washed out.

## Recommendations

- **Use only with covariance targets**: This loss is for supervised or pseudo-supervised covariance learning. For standard likelihood training on $(x, y)$ pairs, use `GaussianNLLLoss`.
- **Start with diagonal parameterisation**: `covariance_parameterization="diagonal"` is simpler, faster, and numerically more stable than full covariance or Cholesky modes.
- **Two-phase strategy**: Consider a pseudo-covariance pretrain with `GaussianWassersteinBoundLoss` followed by `GaussianNLLLoss` fine-tuning. See the [hybrid pretrain demo](../../examples/wasserstein_bound_hybrid_pretrain_demo.py).
- **Validate pseudo-labels**: When using `NeighborhoodCovariancePseudoLabeler`, check that the produced covariance matrices are plausible (positive eigenvalues, reasonable scale) before training.

## Next steps

- [Gaussian losses](gaussian.md) — NLL, CRPS, and multivariate likelihoods for standard Gaussian training
- [Beta-NLL](beta_nll.md) — stabilized heteroscedastic likelihood training without covariance targets
- [TIC-TAC](../methods/algorithms/tictac.md) — Taylor-Induced Covariance for improved covariance estimation
- [Wasserstein demo](../examples/gaussian_wasserstein_bound.md) — runnable example with pseudo-covariance pretraining
