# Constraints API

Complete reference for `torchregress.constraints`. This package provides
**output-head wrappers** that enforce structural properties (non-negativity,
bounded ranges, simplex, non-crossing order, Lipschitz bound via spectral
normalisation). They are all differentiable `nn.Module` subclasses and can be
used as drop-in heads.

For background, see [Output constraints](../methods/constraints.md).

---

## Head wrappers

| Symbol | Wraps | Output | Use case |
|:-------|:------|:-------|:---------|
| `NonNegativeHead(module, beta=1.0)` | any `nn.Module` | `softplus(module(x), beta=beta)` | Mass, count, variance, magnitude |
| `BoundedHead(module, low=0.0, high=1.0)` | any `nn.Module` | `low + (high − low) * sigmoid(module(x))` | Probabilities, rates, fractions |
| `SimplexHead(module, dim=-1)` | any `nn.Module` | `softmax(module(x), dim=dim)` | Mixture weights, K-class probability vectors |
| `NonCrossingSort(dim=-1)` | tensor | `sort(x, dim=dim).values` | Non-crossing quantile / probability curves |
| `SpectralNormWrapper(module, name="weight")` | `nn.Module` with a weight param | Lipschitz-bounded output | OOD detection, stable training |

All wrappers preserve gradients end-to-end (even `NonCrossingSort` — the
sort is differentiable through straight-through estimators or by sorting in
the forward pass only).

```python
import torch.nn as nn
from torchregress.constraints import (
    NonNegativeHead, BoundedHead, SimplexHead, NonCrossingSort, SpectralNormWrapper,
)

# Non-negative regression (e.g. variance, count)
head = NonNegativeHead(nn.Linear(64, 1), beta=1.0)

# Probability in [0, 1] (e.g. rate)
head = BoundedHead(nn.Linear(64, 1), low=0.0, high=1.0)

# Mixture weights over K components
head = SimplexHead(nn.Linear(64, 5), dim=-1)

# Non-crossing quantile / CDF outputs
head = NonCrossingSort(dim=-1)

# Lipschitz-bounded model
head = SpectralNormWrapper(nn.Linear(64, 32))
```

---

## Quick example

```python
import torch
import torch.nn as nn
from torchregress.constraints import NonNegativeHead, SimplexHead

# 1. Non-negative heteroscedastic variance head
mu_layer = nn.Linear(64, 1)
var_layer = NonNegativeHead(nn.Linear(64, 1), beta=1.0)

x = torch.randn(8, 64)
mu = mu_layer(x)
var = var_layer(x)            # always > 0
assert (var >= 0).all()

# 2. Mixture-of-experts gating head
gate = SimplexHead(nn.Linear(64, 4), dim=-1)  # output sums to 1, non-negative
weights = gate(x)            # \[8, 4\]
assert torch.allclose(weights.sum(dim=-1), torch.ones(8))
```

---

## Why these exist vs `torch.nn` constraints

| torchregress | PyTorch nearest | Why this exists |
|:-------------|:----------------|:----------------|
| `BoundedHead` | `nn.Hardtanh` (not in standard `nn`; exists in `torch.nn.functional`) | Differentiably smooth via sigmoid; no gradient at the boundary |
| `NonNegativeHead` | `nn.Softplus` | Wraps a head; preserves learnable shape through the underlying module |
| `SimplexHead` | `nn.Softmax` | Wraps a head rather than acting on raw logits; maintains module identity |
| `NonCrossingSort` | — | No native PyTorch equivalent; needed for monotone quantile / CDF heads |
| `SpectralNormWrapper` | `torch.nn.utils.parametrizations.spectral_norm` | Module-wrapper API; one-liner for full-module normalisation |

## Next steps

- [Output constraints](../methods/constraints.md)
- [Probabilistic losses](../losses/gaussian.md) — when to combine a constraint head with a probabilistic loss


## Constraints Reference

### BoundedHead

Applies upper and lower bounds to output predictions using scaled sigmoid functions.

### NonCrossingSort

Enforces non-crossing constraints on quantile predictions by sorting along the quantile dimension.

### NonNegativeHead

Enforces non-negativity using softplus or exponential transformations.

### SimplexHead

Projects outputs onto the probability simplex using softmax.

### SpectralNormWrapper

Applies spectral norm to module weights.
