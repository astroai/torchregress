# Normalizing Flow Maps

> ← [Normalizing Flows](../losses/nflows.md) | [Conditional Flow Matching](flow_matching.md) →

Normalizing Flow Maps (NFM; Ruscio et al., 2026) learn a **two-time invertible
pushforward** $f_{s,t}$ along a prescribed probability path.  The endpoint map
$f_{0,1}$ is a normalizing flow with an exact change-of-variables likelihood,
while intermediate time pairs provide few-step transport.

The torchregress tabular implementation uses time-conditioned affine coupling
layers.  For an admissible pair $s<t$,

$$
x_t=f_{s,t}(x_s\mid c),\qquad
\log p(y\mid c)=\log p_{\rm base}(f_{0,1}(y\mid c))
+\log\left|\det\frac{\partial f_{0,1}}{\partial y}\right|.
$$

Training combines pointwise flow-map matching,

$$
\mathcal L_{\rm map}
=\mathbb E_{s<t,x_0,y}\left[
\left\|f_{s,t}\big((1-s)x_0+s y\mid c\big)-((1-t)x_0+t y)\right\|^2
\right],
$$

with the endpoint NLL.  This is a lean tabular instantiation, not the
image-scale autoregressive NFM of the paper.

```python
from torchregress.models import NormalizingFlowMap

nfm = NormalizingFlowMap(n_features=2, context_dim=64, n_layers=4)
loss = nfm.training_loss(context, target)
logp = nfm.log_prob(context, target)
samples = nfm.sample(context, n_samples=128)   # [B, S, 2]
```

## When to use it

| Property | NFM | standard flow | CFM |
|:--|:--|:--|:--|
| Exact conditional NLL | ✅ | ✅ | ❌ |
| Invertible at intermediate times | ✅ | ❌ | ❌ |
| Few-step transport | ✅ | composition only | ODE |
| Mature tabular implementation | newer | mature | newer |
| Noise-aware measurement model | ❌ | via `NoiseAwareDensityLoss` | ❌ |

NFM is a research direction for combining exact likelihoods with flow-map
few-step transport.  For routine tabular regression with calibrated NLL, NSF
remains the safer default; test NFM when few-step/consistency behavior matters.

## References

| # | Reference |
|:-:|:----------|
| 1 | F. M. Ruscio et al. ["Normalizing Flow Maps."](https://arxiv.org/abs/2610.22711) 2026. |
| 2 | N. M. Boffi et al. ["Flow map matching with stochastic interpolants."](https://arxiv.org/abs/2406.07507) 2024. |
| 3 | Y. Lipman et al. ["Flow Matching for Generative Modeling."](https://arxiv.org/abs/2210.02747) 2022. |
