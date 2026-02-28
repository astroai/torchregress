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

---

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

---

## Flow Architectures

| Flow | Expressivity | Sampling | Density Eval | Best For |
|:-----|:----------:|:--------:|:------------:|:---------|
| **RealNVP** | ⭐⭐ | Fast | Fast | Lower dimensions, quick baseline |
| **MAF** | ⭐⭐⭐ | Slow | Fast | Conditional density estimation |
| **NSF** | ⭐⭐⭐⭐ | Medium | Medium | Complex multimodal distributions |

---

## References

| # | Reference |
|:-:|:----------|
| 1 | D. Rezende, S. Mohamed. "Variational Inference with Normalizing Flows." *ICML*, **2015**. |
| 2 | G. Papamakarios et al. "Normalizing Flows for Probabilistic Modeling and Inference." *JMLR*, 22(57):1–64, **2021**. |
| 3 | F. Rozet et al. "Zuko: Normalizing Flows in PyTorch." **2022**. |
