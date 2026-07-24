# Normalizing Flow Losses

> ← [Mixture Density Networks](mdn.md) | [Error-in-Variables](eiv.md) →

Normalizing flows transform a simple base distribution (Gaussian) into a complex target distribution through learnable invertible transformations — enabling **arbitrary** conditional density estimation.

!!! warning "Dependency"
    Requires the [`zuko`](https://github.com/probabilists/zuko) package: `pip install zuko`

---

## Mathematical Background & Formulation for Supervised Regression

Conditional Normalizing Flows model complex, non-Gaussian, multi-dimensional target distributions $p(\mathbf{y} \mid \mathbf{x})$ for regression tasks.

### 1. Conditional Density Estimation in Regression

Given observed input features $\mathbf{x} \in \mathbb{R}^D$ (e.g. 1D stellar spectra, images, tabular covariates) and continuous target variables $\mathbf{y} \in \mathbb{R}^M$ (e.g. stellar parameters $[T_{\text{eff}}, \log g, [\text{Fe/H}]]$):

1. **Backbone Context Extractor**: A neural network (e.g. 1D CNN, MLP, or ResNet) $c_\theta(\mathbf{x})$ maps the input $\mathbf{x}$ to a conditioning context vector $\mathbf{c} \in \mathbb{R}^{d_{\text{ctx}}}$.
2. **Invertible Flow Head**: An invertible transformation $\mathbf{f}_\phi(\mathbf{y}; \mathbf{c})$ conditioned on context $\mathbf{c}$ maps the target $\mathbf{y} \in \mathbb{R}^M$ into a simple base Gaussian variable $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_M)$.

By change of variables, the exact conditional likelihood of target $\mathbf{y}$ given input $\mathbf{x}$ is:

$$\boxed{\;p(\mathbf{y} \mid \mathbf{x}) = p_{\mathbf{Z}}\bigl(\mathbf{f}_\phi(\mathbf{y}; c_\theta(\mathbf{x}))\bigr) \left| \det \frac{\partial \mathbf{f}_\phi(\mathbf{y}; c_\theta(\mathbf{x}))}{\partial \mathbf{y}} \right|\;}$$

Training minimizes the exact Negative Log-Likelihood (NLL) via standard gradient descent:

$$\mathcal{L}_{\text{NLL}}(\theta, \phi) = -\frac{1}{N} \sum_{i=1}^N \left[ \log p_{\mathbf{Z}}\bigl(\mathbf{f}_\phi(\mathbf{y}_i; c_\theta(\mathbf{x}_i))\bigr) + \log \left| \det \frac{\partial \mathbf{f}_\phi(\mathbf{y}_i; c_\theta(\mathbf{x}_i))}{\partial \mathbf{y}_i} \right| \right]$$

---

### 2. Standard Regression vs. Simulation-Based Inference (SBI)

| Setting | Data Source | Flow Role | Primary Objective |
|:--------|:------------|:----------|:------------------|
| **Standard Supervised Regression** | Observed paired dataset $\mathcal{D} = \{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^N$ | Direct parametric likelihood head $p(\mathbf{y} \mid \mathbf{x})$ | Maximum Likelihood Estimation (MLE) over target labels |
| **Simulation-Based Inference (SBI)** | Forward simulator $x \sim p(x \mid \theta)$ + prior $p(\theta)$ | Density ratio estimator or posterior surrogate $q(\theta \mid x)$ | Amortized Bayesian parameter retrieval from simulated $x$ |

In **standard regression**, conditional normalizing flows act as a drop-in replacement for `GaussianNLLLoss` or `MDNLoss`:
- **Beyond Gaussianity**: Unlike `GaussianNLLLoss` (which restricts $p(\mathbf{y} \mid \mathbf{x})$ to symmetric ellipses), flows capture non-linear parameter degeneracies (such as banana-shaped $T_{\text{eff}}$ vs. $\log g$ contours in astrophysics), heavy tails, and skewness.
- **Beyond Fixed Components**: Unlike `MDNLoss` (which requires choosing a fixed number $K$ of Gaussian mixture modes and can suffer from mode collapse during training), flows transform a continuous base space smoothly, providing stable likelihood gradients.
- **Fast Sampling at Inference**: Given a new input spectrum $\mathbf{x}_{\text{test}}$, evaluating context $\mathbf{c} = c_\theta(\mathbf{x}_{\text{test}})$ enables drawing $N = 5,000$ exact joint posterior samples $\mathbf{y}^{(s)} \sim p(\mathbf{y} \mid \mathbf{x}_{\text{test}})$ in milliseconds to produce full 1D and 2D corner plots.

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

## Limitations

1. **Computational cost**: Each training step evaluates $K$ invertible transforms and their Jacobian determinants. For `NSF` with 5+ transforms and large context dimensions, training can be 5–20× slower than an equivalent MDN.
2. **Mask semantics**: Flow objectives model a **joint density** over all target dimensions. Only **sample-level** masking is supported — if one target dimension is missing, drop the whole sample.
3. **Architecture choice matters**: `RealNVP` is fast but less expressive; `NSF` is most expressive but slower. Match the flow type to your target: use `NSF` for complex multimodality, `MAF` for conditional density estimation, `RealNVP` for a quick baseline.
4. **ContrastiveFlowLoss is task-specific**: It optimises hypothesis ranking, not density calibration. If you need well-calibrated conditional densities for sampling or downstream UQ, start with `NormalizingFlowLoss`. `ContrastiveFlowLoss` is for parameter ranking / retrieval tasks with informative negative contexts.
5. **Dependency**: Requires [`zuko`](https://github.com/probabilists/zuko). Not installed by default; use `pip install torchregress[flows]`.

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
