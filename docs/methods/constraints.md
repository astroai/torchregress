# Output Constraints & Regularisation

Output constraints are architectural layers that **enforce** structural properties on your model's predictions — such as non-negativity, boundedness, or monotonicity — **by construction**. Unlike penalty-based methods (which only "encourage" constraints), these layers guarantee zero violations for every single forward pass.

---

## Why Use Architectural Constraints?

| Benefit | Description |
|:--------|:------------|
| **Exact Enforcement** | Zero constraint violations at any point in the input space. |
| **Differentiability** | All layers are fully differentiable (even sorting), enabling end-to-end training. |
| **Physical Validity** | Ensure predictions respect known physical bounds (e.g., mass > 0, probability ∈ [0, 1]). |
| **Numeric Stability** | Prevents $NaN$s and exploding gradients by keeping outputs within valid ranges. |

---

## Available Constraints

**torchregress** provides several constraint-enforcing layers that can be added as the final stage of any PyTorch model.

### 1. Positivity ([`NonNegativeHead`](../api/constraints.md#torchregress.constraints.NonNegativeHead))

Ensures all outputs are non-negative ($\hat{y} \geq 0$) via a **Softplus** transformation:

$$\hat{y} = \log(1 + e^{x})$$

```python
from torchregress.constraints import NonNegativeHead

# Guarantees all outputs are positive
model = nn.Sequential(nn.Linear(64, 1), NonNegativeHead())
```

### 2. Boundedness ([`BoundedHead`](../api/constraints.md#torchregress.constraints.BoundedHead))

Enforces strict lower and upper bounds ($a \leq \hat{y} \leq b$) via a scaled **Sigmoid**:

$$\hat{y} = a + (b - a) \cdot \sigma(x)$$

```python
from torchregress.constraints import BoundedHead

# Enforce output between 0 and 100
model = nn.Sequential(nn.Linear(64, 1), BoundedHead(lower=0, upper=100))
```

### 3. Non-Crossing ([`NonCrossingSort`](../api/constraints.md#torchregress.constraints.NonCrossingSort))

Ensures that multiple outputs are **monotonically ordered**. This is critical for **Quantile Regression** to prevent "quantile crossing" (where the 90th quantile is predicted to be less than the 50th).

```python
from torchregress.constraints import NonCrossingSort

# Predict 5 ordered quantiles
model = nn.Sequential(nn.Linear(64, 5), NonCrossingSort())
```

### 4. Simplex ([`SimplexHead`](../api/constraints.md#torchregress.constraints.SimplexHead))

Ensures that outputs sum to 1 and are non-negative ($\sum \hat{y}_i = 1, \hat{y}_i \geq 0$) via **Softmax**. Ideal for mixture weights or compositional data.

---

## Method Selection Matrix

| Layer | Constraint | API Reference | Best For |
|:------|:-----------|:--------------|:---------|
| **`NonNegativeHead`** | $\hat{y} \geq 0$ | [`NonNegativeHead`](../api/constraints.md#torchregress.constraints.NonNegativeHead) | Price, Variance, Count |
| **`BoundedHead`** | $[a, b]$ | [`BoundedHead`](../api/constraints.md#torchregress.constraints.BoundedHead) | Probabilities, Percentages |
| **`NonCrossingSort`** | $y_1 \leq y_2 \dots$ | [`NonCrossingSort`](../api/constraints.md#torchregress.constraints.NonCrossingSort) | Quantile Regression |
| **`SimplexHead`** | $\sum y_i = 1$ | [`SimplexHead`](../api/constraints.md#torchregress.constraints.SimplexHead) | Mixture Models, Weights |

---

## Advanced: Spectral Normalisation ([`SpectralNormWrapper`](../api/constraints.md#torchregress.constraints.SpectralNormWrapper))

For stability and **Out-of-Distribution (OOD)** detection, **torchregress** provides a `SpectralNormWrapper`. This bounds the **Lipschitz constant** of the model, ensuring it remains smooth and well-behaved even for inputs far from the training data [1].

```python
from torchregress.constraints import SpectralNormWrapper

# Bound the model's Lipschitz constant to 1.0
model = SpectralNormWrapper(base_model, coeff=1.0)
```

!!! tip "OOD Sensitivity"

    Lipschitz-bounded models produce more reliable uncertainty estimates for OOD data, as they cannot produce arbitrarily large changes in output for small changes in input.

---

## Best Practices

!!! tip "Softplus vs Exponential"

    Prefer `Softplus` over `Exponential` for enforcing positivity. `Exp` is numerically unstable for large inputs and can lead to $NaN$ gradients.

!!! warning "Sorting vs Monotonicity"

    `NonCrossingSort` enforces ordering across different output units. If you need a model to be monotonic with respect to an **input** feature, see our [Monotonic Regression Guide](../guides/concepts.md).

---

## References

| # | Reference |
|:-:|:----------|
| 1 | Miyato et al. ["Spectral Normalization for Generative Adversarial Networks."](https://arxiv.org/abs/1802.05957) *ICLR*, 2018. |
| 2 | Gasthaus et al. ["How to Avoid Being Mean: Non-Crossing Quantile Regression."](https://arxiv.org/abs/1912.01166) *NeurIPS*, 2019. |
| 3 | Koenker, R. ["Quantile Regression."](https://www.cambridge.org/core/books/quantile-regression/D316277B1A409C3B14B8881E60B2163A) *Cambridge University Press*, 2005. |

---

## Next Steps
- Learn about [Quantile & Expectile Losses](../losses/quantile_expectile.md)
- Explore [Post-Hoc Calibration](calibration.md)
- View the [Constraints & Calibration Example](../examples/constraints_calibration_comparison.md)
