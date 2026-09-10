# Grid & basis densities: Wasserstein-1 and Rank-N-Contrast

> ← [Normalizing Flows](nflows.md) | [Evidential Regression](advanced.md) →

Tools for heads that predict a **whole 1D distribution** over a fixed support (binned
classification, spline bases) rather than a parametric family, plus a representation
loss that orders the feature space by the continuous target.

---

## `BSplineDensityBasis` (`torchregress.utils`)

A density on $[lo, hi]$ is written as

$$
p(x) = \sum_{m=1}^{M} c_m\, M_m(x), \qquad c \in \Delta^{M-1},
\qquad M_m(x) = \frac{k\, B_{m,k}(x)}{t_{m+k} - t_m},
$$

where $B_{m,k}$ are order-$k$ (degree $k-1$) B-splines on a clamped knot vector and
$M_m$ are their **unit-integral** (M-spline) rescalings. Standard B-splines satisfy
$\sum_m B_m = 1$, *not* $\int B_m = 1$; the rescaling is what makes a softmax over
coefficients produce a normalised, non-negative density with no extra pass.

| Method | Returns | Notes |
|:-------|:--------|:------|
| `from_uniform(lo, hi, n_intervals, degree=3)` | basis | Equally spaced breakpoints |
| `from_quantiles(samples, n_interior, lo, hi)` | basis | Interior knots at empirical quantiles (balanced gradient flow) |
| `evaluate(x)` | `(..., M)` | Cox–de Boor recursion; zero outside `[lo, hi]` |
| `density(coeffs, grid)` | `(..., G)` | $\sum_m c_m M_m(z_g)$ |
| `bin_integrals(edges)` | `(K, M)` | $T_{km} = \int_{e_k}^{e_{k+1}} M_m$; exact via piecewise Gauss–Legendre |
| `cdf(x)` / `cumulative_moment(x, r)` | `(len(x), M)` | $\int_{lo}^{x} u^r M_m(u)\,du$ |
| `absolute_deviation(x)` | `(len(x), M)` | $\mathbb{E}_{M_m}\lvert U - x\rvert$ — exact $W_1$ to a point mass |

Bin masses are a fixed linear map of the coefficients, `masses = coeffs @ T.T`, so
tomographic/binned summaries of a predicted density need no sampling.

```python
import torch
from torchregress.utils import BSplineDensityBasis

basis = BSplineDensityBasis.from_quantiles(train_targets, n_interior=12, lo=0.0, hi=3.0)
coeffs = torch.softmax(head(features), dim=-1)          # [N, basis.n_basis]
nll = -(coeffs * basis.evaluate(y)).sum(-1).log()       # exact density at the labels
w1 = (coeffs * basis.absolute_deviation(y)).sum(-1)     # exact W1(p, delta_y) = E|U - y|
```

---

## `DiscreteWasserstein1Loss`

For mass vectors $p, q$ located at bin centres $c_1<\dots<c_K$,

$$
W_1(p, q) = \sum_{k=1}^{K-1} \lvert F_k - G_k \rvert\,(c_{k+1} - c_k),
$$

with $F, G$ the cumulative sums. Unlike KL or cross-entropy, the penalty grows with
*how far* misplaced mass is from the target, which complements a local divergence when
training binned or basis heads. Targets may be `[N, K]` masses or `[N]` scalar values
(quantised to the containing bin; for continuous parameterisations prefer
`BSplineDensityBasis.absolute_deviation`).

```python
from torchregress.losses import DiscreteWasserstein1Loss

loss = DiscreteWasserstein1Loss(bin_edges, from_logits=True)
value = loss(logits, target_masses)          # or loss(logits, scalar_targets)
```

---

## `RankNContrastLoss`

Rank-N-Contrast (Zha et al., NeurIPS 2023) trains features $h_i$ so that similarity
is ordered by label distance. For anchor $i$ and any $j \ne i$,

$$
\mathcal{L}_i = \frac{1}{N-1}\sum_{j\ne i} -\log
\frac{\exp(s_{ij}/\tau)}{\sum_{k\in S_{ij}} \exp(s_{ik}/\tau)},
\qquad
S_{ij} = \{k \ne i : \lvert y_i - y_k\rvert \ge \lvert y_i - y_j\rvert\},
$$

with $s_{ij} = -\lVert h_i - h_j\rVert_2$ (default) or cosine similarity. Ties in label
distance are included in $S_{ij}$. The implementation sorts by label distance per anchor
and uses a cumulative log-sum-exp, so it is $O(N^2 \log N)$ rather than $O(N^3)$.

```python
from torchregress.losses import RankNContrastLoss

rnc = RankNContrastLoss(temperature=2.0)
value = rnc(features, labels)                # features [N, D], labels [N] or [N, L]
```

`mask` removes samples from both the anchor set and every denominator; `weights` are
per anchor.

!!! note "Scope"
    These are building blocks. The library does not claim any particular downstream
    accuracy from them; validate on your own held-out data.
