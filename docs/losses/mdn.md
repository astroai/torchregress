# Mixture Density Networks (MDN)

> ← [Evidential Regression](advanced.md) | [Normalizing Flows](nflows.md) →

Mixture Density Networks model the conditional distribution $p(y \mid x)$ as a **mixture of Gaussian components**, capturing multimodality, heteroscedastic noise, and complex distributional shapes.

!!! abstract "When to use"
    When the output distribution has **multiple modes** — e.g., inverse problems with multiple solutions, financial regimes, or multi-path trajectory prediction.

!!! tip "Noisy Inputs"
    If your input features ($X$) are noisy, standard MDNs may produce biased results. Use [**InputNoiseMDNLoss**](eiv.md#multimodal-eiv-variants) to marginalize over the input noise for more robust predictions.

---

## Mathematical Background

$$\boxed{\;p(y \mid x) = \sum_{k=1}^K \pi_k(x)\;\mathcal{N}\!\bigl(y \mid \mu_k(x),\, \Sigma_k(x)\bigr)\;}$$

| Symbol | Predicted by model | Constraint |
|:------:|:------------------|:-----------|
| $\pi_k$ | Mixture weights | $\pi_k > 0$, $\sum_k \pi_k = 1$ (softmax) |
| $\mu_k$ | Component means | Unconstrained |
| $\Sigma_k$ | Component covariance | PD (softplus + min_std) |

The NLL is:

$$\mathcal{L}_{\text{MDN}} = -\log \sum_{k=1}^K \pi_k\;\mathcal{N}(y \mid \mu_k, \Sigma_k)$$

computed via log-sum-exp for numerical stability.

See the [MixtureDensityLoss API](../api/losses.md) for constructor parameters and the [`MixtureDensityLoss`](../api/losses.md) alias.

## Usage

```python
from torchregress.losses import MixtureDensityLoss

loss_fn = MixtureDensityLoss(
    n_components=3,    # K = number of Gaussian components
    n_features=2,      # d = dimensionality of targets
    covariance_type="diagonal",  # or "full"
    min_std=1e-3,
)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `n_components` | `int` | — | Number of mixture components $K$ |
| `n_features` | `int` | — | Target dimensionality $d$ |
| `covariance_type` | `str` | `"diagonal"` | `"diagonal"` or `"full"` |
| `full_parameterization` | `str` | `"precision"` | For `"full"`: the triangular factor parameterises the precision or the covariance (see [below](#full-covariance-precision-or-covariance-factor)) |
| `min_std` | `float` | `1e-3` | Minimum std for stability |

### Model Output Size

=== "Diagonal covariance"

    $$\text{output\_dim} = \underbrace{K}_{\text{weights}} + \underbrace{K \cdot d}_{\text{means}} + \underbrace{K \cdot d}_{\text{log-stds}} = K(1 + 2d)$$

    ```python
    # K=3, d=2 → output_dim = 3(1 + 4) = 15
    model = nn.Sequential(
        nn.Linear(in_dim, 128), nn.ReLU(),
        nn.Linear(128, 15),
    )
    ```

=== "Full covariance"

    $$\text{output\_dim} = K + K \cdot d + K \cdot \frac{d(d+1)}{2}$$

    The last block parameterises lower-triangular factors $T_k$ (positive diagonal). By default they are Cholesky factors of the **precision**, $\Sigma_k^{-1} = T_k T_k^\top$; with `full_parameterization="covariance"`, $\Sigma_k = T_k T_k^\top$. The layout is the same for both.

### Full covariance: precision or covariance factor

With `covariance_type="full"` the network outputs, per component, a lower-triangular $T_k$. `full_parameterization` sets what it means:

| `full_parameterization` | Meaning of $T_k$ | Component log-density |
|:------------------------|:-----------------|:----------------------|
| `"precision"` (default since 0.3.0, as in sbi's MDN) | $P_k = \Sigma_k^{-1} = T_k T_k^\top$ | $-\tfrac12 \lVert T_k^\top (y-\mu_k)\rVert^2 + \sum_i \log T_{k,ii} - \tfrac d2 \log 2\pi$ |
| `"covariance"` (pre-0.3.0 behaviour) | $\Sigma_k = T_k T_k^\top$ | $-\tfrac12 \lVert T_k^{-1} (y-\mu_k)\rVert^2 - \sum_i \log T_{k,ii} - \tfrac d2 \log 2\pi$ |

The precision form is linear in $T_k$ and needs no triangular solve, which makes it much easier to optimise when there are many targets. It is not better everywhere. Harness run #6 (`density_parity`, test NLL, mean of 3 seeds, lower is better) against run #5 (covariance form), with sbi's MDN on the same splits:

| Dataset | Targets $d$ | `"covariance"` (run #5) | `"precision"` (run #6) | sbi MDN |
|:--------|:-----------:|------------------------:|-----------------------:|--------:|
| mt_scm20d | 16 | 3.11 | **0.33** | -0.01 |
| mt_jura | 3 | 2.53 | **1.95** | 1.92 |
| energy_efficiency | 2 | **-1.32** | -0.76 | -1.33 |
| mt_enb | 2 | **-1.18** | -0.85 | -1.24 |

!!! tip "Which parameterisation to use"
    Keep the default `"precision"`: it wins clearly on many targets (scm20d: 3.11 → 0.33). For low-dimensional targets ($d \le 2$ here) the covariance form did better, so also train with `full_parameterization="covariance"` and keep whichever has the lower **validation** NLL.

!!! warning "Sampling from a precision factor"
    In precision mode `_extract_distribution_parameters` returns the precision factor $T$ ($P = T T^\top$), not a covariance Cholesky factor. A covariance square root is $T^{-\top}$: a draw is $\mu + T^{-\top}\varepsilon$, **not** $\mu + T\varepsilon$. Use `loss_fn.sample(...)` (and `predict_interval`), which handle both parameterisations, rather than sampling from the extracted factors yourself. A harness suite that used $T$ as if it were $L$ got 0.00 interval coverage.

### Training

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    params = model(x_train)           # (batch, output_dim)
    loss = loss_fn(params, y_train)   # MDN NLL
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## Inference Methods

### `predict_mean_std`

Gaussian-approximation summary (mean and variance of the mixture):

```python
mean, std = loss_fn.predict_mean_std(model(x_test))
```

!!! warning "Multimodal caveat"
    For genuinely multimodal distributions, the mean may fall **between** modes where no data exists.  Use `sample` or `predict_interval` instead. Also note that `predict_interval` using MC sampling requires `n_samples` large enough to resolve all modes — for $K$ well-separated components, 10,000+ samples may be needed for stable interval boundaries.

### `predict_interval`

MC-based prediction intervals that respect multimodality:

```python
lower, upper = loss_fn.predict_interval(
    model(x_test), confidence=0.95, n_samples=10_000,
)
```

### `sample`

Draw samples from the learned mixture:

```python
samples = loss_fn.sample(model(x_test), n_samples=1000)
# samples shape: [n_samples, batch, n_features]
```

---

## Practical Tips

!!! tip "Avoiding component collapse"
    Monitor component usage during training:
    ```python
    log_weights, means, scales = loss_fn._extract_distribution_parameters(params)
    active = (log_weights.exp() > 0.05).float().mean()  # should stay > 0.5
    ```

!!! tip "Initialisation"
    Initialise the final layer with small weights to prevent early commitment to one component.

!!! tip "Start simple"
    Begin with $K = 3$ diagonal components. Increase $K$ only if validation NLL improves.

---

## Limitations & Optimization Challenges

While MDNs are highly expressive, they present several training and optimization challenges:

1. **Numerical Instability & Log-Sum-Exp**:
   Computing the NLL directly via $p(y \mid x) = \sum_k \pi_k \mathcal{N}(y \mid \mu_k, \sigma_k^2)$ involves exponentials that can easily underflow or overflow. **torchregress** internally utilizes the `log-sum-exp` trick:
   $$\log p(y \mid x) = \text{log-sum-exp}\left(\log \pi_k + \log \mathcal{N}(y \mid \mu_k, \sigma_k^2)\right)$$
   Always use a non-zero `min_std` (e.g., $10^{-3}$) to prevent variance collapse ($\sigma_k^2 \rightarrow 0$), which causes division by zero.

2. **Computational Scaling**:
   The forward pass cost is $\mathcal{O}(K \cdot d)$ for diagonal covariance and $\mathcal{O}(K \cdot d^2)$ for full covariance per sample. NLL evaluation adds determinant and solve operations that scale as $\mathcal{O}(d^3)$ for full covariance. For large $K$ or high-dimensional targets, prefer diagonal covariance or consider flow-based alternatives.

3. **Mode Collapse**:
   The network may collapse into using only a single mixture component, effectively ignoring the others ($\pi_k \approx 0$ for $k > 1$). To monitor this, check active components during validation:
   $$\text{active\_fraction} = \frac{1}{B} \sum_{i=1}^B \sum_{k=1}^K \mathbf{1}_{\pi_k^{(i)} > 0.05}$$
   Mode collapse is especially common with large $K$ (> 10) or strong regularization.

4. **Initialization Sensitivity**:
   If initialized poorly, components can "claim" the same regions of target space. Initialize the final output layer weight matrices with small random values to diversify components early in training.

5. **Dtype Sensitivity**:
   The log-sum-exp trick is numerically sensitive in `float16` or `bfloat16`. Use `float32` for the NLL computation and only cast down after the loss is computed. Mixed-precision training with MDNs requires careful gradient scaling.

6. **Covariance Type Constraints**:
   With `"full"` covariance the head has $K \cdot d(d+1)/2$ factor outputs and memory grows as $\mathcal{O}(K \cdot d^2)$ per sample. The default precision parameterisation trains on 16 targets (scm20d above); for very large $d$ prefer diagonal components or a flow.

---

## Next steps

- [Normalizing Flows](nflows.md) — more flexible distributions for complex targets
- [EIV + MDN](eiv.md#multimodal-eiv-variants) — marginalize over input noise with MDN heads
- [MDN Ensembles](../methods/ensemble/methods.md#mdnensemblemodel) — mixture-of-mixtures for decomposed uncertainty
- [Multimodal comparison](../examples/multimodal_method_comparison.md) — benchmark MDN vs flows vs SLS

---

## References

| # | Reference |
|:-:|:----------|
| 1 | C.M. Bishop. ["Mixture Density Networks."](https://publications.aston.ac.uk/id/eprint/373/) Neural Computing Research Group Report NCRG/94/004, **1994**. |
| 2 | C.M. Bishop. *Pattern Recognition and Machine Learning*, §5.6. Springer, **2006**. |
