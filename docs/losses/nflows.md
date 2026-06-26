# Normalizing Flow Losses

Normalizing flows transform a simple base distribution (Gaussian) into a complex target distribution through learnable invertible transformations — enabling **arbitrary** conditional density estimation.

!!! warning "Dependency"
    Requires the [`zuko`](https://github.com/probabilists/zuko) package: `pip install zuko`

---

## Mathematical Background

A flow $f = f_1 \circ \cdots \circ f_K$ maps a simple base distribution $p_Z(z)$ to a complex distribution $p_X(x)$:

$$\boxed{\;p_X(x) = p_Z\bigl(f(x)\bigr)\;\left|\det\frac{\partial f(x)}{\partial x}\right|\;}$$

The NLL loss is:

$$\mathcal{L}_{\text{NF}}(x) = -\log p_Z(f(x)) - \sum_{k=1}^K \log\!\left|\det\frac{\partial f_k}{\partial f_{k-1}}\right|$$

where $f_0 = x$ is the input and $f_K = f(x)$ is the base-space representation.

See the [NormalizingFlowLoss API](../api/losses.md) for the training contract and `create_flow_loss` helper.

## Usage

### Step 1: Create a zuko flow

Use the `create_flow_model` helper or build your own zuko flow:

```python
from torchregress.losses import create_flow_model

flow = create_flow_model(
    n_features=2,        # target dimensionality
    context_dim=64,      # model's output dim (conditioning)
    flow_type="nsf",     # Neural Spline Flow
    n_transforms=5,      # number of invertible blocks
)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `n_features` | `int` | — | Target dimensionality |
| `context_dim` | `int` | `0` | Size of conditioning vector (model output) |
| `flow_type` | `str` | `"nsf"` | `"realnvp"`, `"maf"`, or `"nsf"` |
| `n_transforms` | `int` | `5` | Number of invertible blocks |
| `hidden_features` | `list[int]` or `None` | `None` | Hidden layer sizes (default: `[64, 64]`) |

### Step 2: Wrap in NormalizingFlowLoss

```python
from torchregress.losses import NormalizingFlowLoss

loss_fn = NormalizingFlowLoss(flow=flow)
```

Or use the shortcut:

```python
from torchregress.losses import create_flow_loss

loss_fn = create_flow_loss(
    n_features=2, context_dim=64, flow_type="nsf", n_transforms=5,
)
```

### Contrastive Flow Variant

When the downstream task is **parameter ranking** rather than generic density estimation,
use `ContrastiveFlowLoss`. It compares the observed target under the correct context against
one or more alternate contexts:

```python
from torchregress.losses import ContrastiveFlowLoss

loss_fn = ContrastiveFlowLoss(flow=flow, temperature=0.7, margin=0.2)
loss = loss_fn(
    positive_context,
    target,
    negative_context=negative_context,  # [batch, n_negatives, context_dim]
)
```

This is useful for nuisance-aware parameter estimation, simulator calibration, and domain-shift
settings where you care about the **likelihood ratio between hypotheses**, not only `p(y|x)`.

In practice, expect `ContrastiveFlowLoss` to help only when:

- the evaluation task is parameter ranking, scanning, or retrieval over hypotheses
- you can generate informative alternate contexts during training
- the parameter space is low-dimensional enough that discrimination between nearby hypotheses matters
- full-density calibration and sampling quality are secondary to getting the ordering right

If you mainly want a well-calibrated conditional density model, start with `NormalizingFlowLoss`.
`ContrastiveFlowLoss` is a task-specific objective, not a stronger default flow loss.

!!! info "Negative-context shapes"
    Use `[batch, n_negatives, context_dim]` for per-sample negative hypotheses or
    `[1, n_negatives, context_dim]` for a shared bank broadcast across the batch.
    A 2-D tensor `[N, context_dim]` is ambiguous when `N == batch_size`; the implementation now
    raises in that case unless you disambiguate explicitly.

### Step 3: Train

```python
class ContextModel(nn.Module):
    """Backbone outputs a context vector that conditions the flow."""
    def __init__(self, in_dim, context_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, context_dim),
        )
    def forward(self, x):
        return self.net(x)

model = ContextModel(in_dim=10, context_dim=64)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(loss_fn.parameters()), lr=1e-4,
)

for x, y in train_loader:
    context = model(x)                  # [batch, context_dim]
    loss = loss_fn(context, y)          # NLL
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

!!! tip "Include flow parameters in optimiser"
    The flow itself has learnable parameters — make sure to pass `loss_fn.parameters()` to the optimiser.

---

## Inference

### Sampling

```python
with torch.no_grad():
    context = model(x_test)
    samples = loss_fn.sample(context, n_samples=1000)
    # samples shape: [batch, 1000, n_features]
```

For a challenge-style parameter scan built on top of `ContrastiveFlowLoss`, see
[Contrastive Flow Parameter Estimation](../examples/contrastive_flow_parameter_estimation.md).
For shared-budget comparisons against Gaussian-summary and plain-flow baselines, see
[Contrastive Flow Parameter Estimation Comparison](../examples/contrastive_flow_parameter_estimation_comparison.md)

!!! warning "Mask semantics"
    Flow objectives model a **joint density** over all target dimensions. `NormalizingFlowLoss`
    and `ContrastiveFlowLoss` therefore only support **sample-level** masking, not partial
    feature-wise masking. If one target dimension is missing, drop the whole sample or switch to
    a model with an explicit missing-data strategy.

---

## Flow Architectures

| Flow | Expressivity | Sampling | Density Eval | Best For |
|:-----|:----------:|:--------:|:------------:|:---------|
| **RealNVP** | ⭐⭐ | Fast | Fast | Lower dimensions, quick baseline |
| **MAF** | ⭐⭐⭐ | Slow | Fast | Conditional density estimation |
| **NSF** | ⭐⭐⭐⭐ | Medium | Medium | Complex multimodal distributions |

## When To Use Which Flow Objective

| Objective | Start Here When | Tradeoff |
|:--|:--|:--|
| `NormalizingFlowLoss` | You need calibrated conditional densities, sampling, or a strong general-purpose flow baseline | Does not directly optimize parameter discrimination |
| `ContrastiveFlowLoss` | You need to rank the true hypothesis above alternate parameter settings with meaningful training negatives | Can underperform plain NLL when density calibration is the real objective |

---

## Next steps

- [MDN losses](mdn.md) — lighter-weight mixture models for simpler multimodality
- [SLS regression](sls.md) — volume-optimal prediction regions (flow-based frontier)
- [Contrastive flow estimation](../examples/contrastive_flow_parameter_estimation.md) — parameter ranking example
- [Flow comparison](../examples/contrastive_flow_parameter_estimation_comparison.md) — benchmark flows vs MDN vs Gaussian

---

## References

| # | Reference |
|:-:|:----------|
| 1 | D. Rezende, S. Mohamed. ["Variational Inference with Normalizing Flows."](https://arxiv.org/abs/1505.05770) *ICML*, **2015**. |
| 2 | G. Papamakarios et al. ["Normalizing Flows for Probabilistic Modeling and Inference."](https://arxiv.org/abs/1912.02762) *JMLR*, 22(57):1–64, **2021**. |
| 3 | F. Rozet et al. ["Zuko: Normalizing Flows in PyTorch."](https://github.com/probabilists/zuko) **2022**. |
| 4 | I. Elsharkawy, Y. Kahn. ["Contrastive Normalizing Flows for Uncertainty-Aware Parameter Estimation."](https://arxiv.org/abs/2505.08709) *arXiv:2505.08709*, **2025**. |
