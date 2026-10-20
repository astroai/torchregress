# Conditional Flow Matching

> ← [Normalizing Flows](../losses/nflows.md) | [Forward Generative Inversion](algorithms/forward_inversion.md) →

Conditional flow matching (CFM) learns a time-dependent velocity field
$v_\theta(x,t,c)$ that transports Gaussian noise to the conditional target
density.  It is simulation-free: no ODE solve is needed during training.

For an optimal-transport interpolation,

$$
x_t=(1-t)x_0+t x_1,\qquad x_0\sim\mathcal N(0,I),\quad x_1\sim p(y\mid c),
$$

the objective regresses the target velocity:

$$
\mathcal L_{\rm CFM}
=\mathbb E_{t\sim U(0,1),\,x_0,\,y}\left[
\left\|v_\theta(x_t,t,c)-(x_1-x_0)\right\|^2
\right].
$$

Samples are obtained by Euler integration from $t=0$ to $1$:

$$
x_{k+1}=x_k+\frac1K\,v_\theta(x_k,k/K,c).
$$

## When to use it

| Property | CFM | Exact flow (NSF/MAF/RealNVP/BPF) |
|:--|:--|:--|
| Exact likelihood | ❌ | ✅ |
| Sampling | short ODE | one forward pass |
| Multimodal / multi-target | ✅ | ✅ |
| Calibration/PIT from samples | ✅ | ✅ |
| NLL-based model selection | need ODE likelihood | direct |

CFM is a good fit for high-dimensional or multi-target **generation**, and as a
flexible sampler when exact NLL is not required.  For tabular regression with
calibrated NLL, an exact NSF/MDN is still the simpler default.  The companion
harness compares CFM against NSFs, cINN, BPF and MDN on shared encoders and
multi-target datasets (`density_parity`), and against noise-aware densities on
known-noise targets (`noise_aware_density`).

```python
import torch
from torchregress.models import FlowMatchingMLP, cfm_loss, sample_cfm

velocity = FlowMatchingMLP(n_features=2, context_dim=64, hidden=(128, 128))
context = torch.randn(32, 64)
y = torch.randn(32, 2)

loss = cfm_loss(velocity, context, y)
loss.backward()

samples = sample_cfm(velocity, context, n_samples=128, steps=32)
# [32, 128, 2]
```

## Comparison with related methods

- **Forward generative inversion** models $x_{\rm obs}=g(y)+\epsilon$ and
  inverts it at prediction time; it is aimed at errors-in-variables, not
  general conditional density estimation.
- **Contrastive normalizing flows** train a likelihood-ratio classifier and are
  used for hypothesis ranking/uncertainty-aware parameter inference; they do
  not replace a calibrated conditional density.
- **Noise-aware density losses** add the measurement model to an exact
  likelihood; CFM currently does not have an equivalent exact marginal
  likelihood.

## References

| # | Reference |
|:-:|:----------|
| 1 | Y. Lipman et al. ["Flow Matching for Generative Modeling."](https://arxiv.org/abs/2210.02747) 2022. |
| 2 | A. Tong et al. ["Improving and generalizing flow-based generative models with minibatch optimal transport."](https://arxiv.org/abs/2302.00482) TMLR 2023. |
| 3 | N. M. Boffi et al. ["Flow map matching with stochastic interpolants."](https://arxiv.org/abs/2406.07507) 2024. |
