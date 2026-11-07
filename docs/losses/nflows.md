# Normalizing Flow Losses

> ← [Mixture Density Networks](mdn.md) | [Error-in-Variables](eiv.md) →

Normalizing flows transform a simple base distribution (Gaussian) into a complex target distribution through learnable invertible transformations — enabling **arbitrary** conditional density estimation.

!!! warning "Dependency"
    Requires the [`zuko`](https://github.com/probabilists/zuko) package: `pip install zuko`

---

## Mathematical Background & Formulation for Supervised Regression

Conditional Normalizing Flows model complex, non-Gaussian, multi-dimensional target distributions $p(\mathbf{y} \mid \mathbf{x})$ for regression tasks.

### 1. Conditional Density Estimation in Regression

Given observed input features $\mathbf{x} \in \mathbb{R}^D$ (e.g. tabular covariates or high-dimensional embeddings) and continuous target variables $\mathbf{y} \in \mathbb{R}^M$:

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

In **standard regression**, conditional normalizing flows act as a drop-in replacement for `GaussianNLLLoss` or `MixtureDensityLoss`:
- **Beyond Gaussianity**: Unlike `GaussianNLLLoss` (which restricts $p(\mathbf{y} \mid \mathbf{x})$ to symmetric ellipses), flows capture non-linear parameter degeneracies (such as banana-shaped $T_{\text{eff}}$ vs. $\log g$ contours in astrophysics), heavy tails, and skewness.
- **Beyond Fixed Components**: Unlike `MixtureDensityLoss` (which requires choosing a fixed number $K$ of Gaussian mixture modes and can suffer from mode collapse during training), flows transform a continuous base space smoothly, providing stable likelihood gradients.
- **Fast Sampling at Inference**: Given a new input spectrum $\mathbf{x}_{\text{test}}$, evaluating context $\mathbf{c} = c_\theta(\mathbf{x}_{\text{test}})$ enables drawing $N = 5,000$ exact joint posterior samples $\mathbf{y}^{(s)} \sim p(\mathbf{y} \mid \mathbf{x}_{\text{test}})$ in milliseconds to produce full 1D and 2D corner plots.

See the [NormalizingFlowLoss API](../api/losses.md) for the training contract and `create_flow_loss` helper.

### Choosing a flow family

| Flow | Best for | Caveat |
|:--|:--|:--|
| `nsf` / `ncsf` | scalar/multi-target continuous regression; best calibration per parameter | bounded spline domain; widen `bound` for heavy tails |
| `realnvp` / `cinn` | fast affine-coupling training, high-dimensional targets | less expressive per layer than NSF; see [stable affine layers](#stable-affine-layers-maf-realnvp-cinn) |
| `maf` | unbounded, autoregressive densities | sampling needs one pass per target coordinate; see [stable affine layers](#stable-affine-layers-maf-realnvp-cinn) |
| `bpf` | smooth Bernstein-polynomial scalar/joint densities | newer, less battle-tested than NSF |
| `tarflow` | 8 or more smooth, skewed targets that all depend on each other | loses to NSF/MAF on multimodal targets and for 2–4 targets; samples 30 to 75× slower than MAF; see [TarFlow](#transformer-autoregressive-flows-tarflow) |

For **noisy labels**, wrap the flow with
[`NoiseAwareDensityLoss`](noise_aware_density.md) instead of fitting the
observed label directly.  For **imbalanced targets**, remember that
target-density reweighting changes the population density (see
[imbalanced regression](imbalanced.md)).

## Usage

### Step 1: Create a zuko flow

For `bpf`, `n_transforms` and `hidden_features` control the autoregressive
architecture, while `degree` (at least 1) controls its polynomial coefficients.
The factory uses a local bounded Bernstein transform with
$f(x)=x$ for $|x|\geq 5$, continuous endpoint values and unit endpoint derivatives.
Its inverse solver tolerance controls numerical inversion, not a tail shift.
The corrected transform retains the parameter and buffer names of the upstream
BPF, so existing weights load when the actual layer count and widths match.
Older factory versions ignored the requested BPF layer count and widths; use the
checkpoint's actual architecture rather than that ignored request when loading.
Previously recorded predictions and benchmarks keep their original version and
must not be relabeled as results of this corrected map.

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
| `flow_type` | `str` | `"nsf"` | `"nsf"`/`"ncsf"`, `"maf"`, `"realnvp"`/`"cinn"`, `"bpf"`, or `"tarflow"` |
| `n_transforms` | `int` | `5` | Number of invertible blocks |
| `hidden_features` | `list[int]` or `None` | `None` | Hidden layer sizes (default: `[64, 64]`). For `tarflow`: Transformer width (largest entry) and, unless `n_layers=` is given, layers per block (number of entries). |
| `bound` | `float` or `None` | `None` | NSF: half-width `B` of the spline domain `[-B, B]` (zuko's fixed `5` when `None`); see [Training recipe](#training-recipe-for-nll-and-calibration). Spline `tarflow`: the same spline half-width. MAF / RealNVP / cINN and affine `tarflow`: conditioner soft-clip half-width (`10` when `None`); see [stable affine layers](#stable-affine-layers-maf-realnvp-cinn). Not accepted for BPF. |
| `slope` | `float` | NSF `1e-3`, affine `e^{-3}` | Minimum slope of each layer (forwarded to zuko's spline / affine transform). For the affine flows it bounds the log-scale: $|\log s| < -\log(\text{slope})$. |
| `bins` | `int` | `8` | NSF and spline `tarflow`: spline knots per transform. |
| `univariate`, `n_layers`, `n_heads`, `dropout` | | `"spline"`, from `hidden_features`, `4`, `0.0` | `tarflow` only: element-wise map (`"spline"` or `"affine"`), Transformer layers per block (an int, or one per block such as `(4, 1, 1)`), attention heads, dropout. |

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

### Contrastive Likelihood-Ratio Flow Variant (InfoNCE / NRE)

When the downstream task is **hypothesis ranking or likelihood-ratio estimation** rather than
generic density estimation, use `ContrastiveFlowLoss`. It is a multiclass InfoNCE objective over
the flow's conditional log-densities: the observed target is scored under the generating context
and one or more alternate contexts, and the flow is trained so that the true hypothesis scores
highest:

```python
from torchregress.losses import ContrastiveFlowLoss

# Likelihood-ratio mode: T=1, margin=0.
loss_fn = ContrastiveFlowLoss(flow=flow)
loss = loss_fn(
    positive_context,
    target,
    negative_context=negative_context,  # [batch, n_negatives, context_dim]
)
```

This is a **contrastive likelihood-ratio / neural-ratio-estimation (NRE) objective**, not a
density-calibration objective. Two regimes matter:

| Setting | What the scores mean |
|:--|:--|
| `temperature=1, margin=0` (`likelihood_ratio_mode is True`) | Score difference `log p(x|c+) - log p(x|c-)` estimates the log likelihood ratio between hypotheses (up to an offset constant in `x`). |
| `temperature != 1` or `margin != 0` | Tempered contrastive ranking score. `margin` acts like a log prior-odds offset only at `T=1`; otherwise the scores are **not** a calibrated likelihood ratio. |

!!! warning "What the ratio is relative to"
    The learned ratio is relative to the distribution the negative contexts are drawn from.
    If candidates are drawn from the prior (or the scan grid), then argmax over candidates is
    the maximum-likelihood estimate and is independent of the training label frequency; for
    posterior weights use [`posterior_log_weights`](../api/inference.md) with the correct
    `log_prior` / `log_reference` corrections. Arbitrary negative banks give rankings, not
    posterior probabilities.

For a dedicated classifier-based NRE loss (cheaper: no flow Jacobians, and the standard
ratio-estimation formulation), see [`NeuralRatioLoss`](nre.md) and the
[NRE guide](nre.md). The Elsharkawy & Kahn (2025) *Contrastive Normalizing Flow* is related work,
but its objective (contrastive density shaping between signal and background, followed by a
classifier and Neyman intervals) is different from the InfoNCE loss implemented here.

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

## Training recipe for NLL and calibration

Three things decide whether an NSF head is calibrated, in this order of importance
(measured with the flow settings above: 3 transforms, `[64, 64]` hidden, Adam at `1e-3`,
batch 64, 5 seeds, 1-D targets; NLL is test negative log-likelihood in native units,
`IS` the 90% interval score from 512 samples):

1. **Stop on validation NLL.** A fixed 60-epoch budget overfits small and medium tabular
   sets (the flow sharpens on the training rows, intervals under-cover). Hold out about
   15% of the training rows, evaluate `loss_fn` on them each epoch, stop after 10 epochs
   without improvement and restore the best weights.
2. **Standardise the target and derive the spline bound from it.** The spline is the
   identity outside `[-B, B]`, so a standardised skewed or heavy-tailed target with
   `|y| > 5` carries no learning signal there. Pass
   `bound=recommended_tail_bound(y_standardised)` (data range with 25% head-room, clipped
   to `[5, 25]`; it is `5` for light tails, so nothing changes there).
3. **Use `bins=16` with a widened bound** so the bulk of the data keeps its resolution.

```python
import copy
import torch
from torchregress.losses import NormalizingFlowLoss, create_flow_model, recommended_tail_bound

y_std = (y - y.mean()) / y.std()
flow = create_flow_model(
    n_features=1, context_dim=64, flow_type="nsf", n_transforms=3,
    hidden_features=[64, 64], bins=16, bound=recommended_tail_bound(y_std),
)
loss_fn = NormalizingFlowLoss(flow=flow)
n_val = int(0.15 * len(x))
x_val, y_val, x_tr, y_tr = x[:n_val], y_std[:n_val], x[n_val:], y_std[n_val:]
params = list(model.parameters()) + list(loss_fn.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3)

best, best_state, bad = float("inf"), None, 0
for epoch in range(200):
    model.train()
    for i in torch.randperm(len(x_tr)).split(64):
        optimizer.zero_grad()
        loss_fn(model(x_tr[i]), y_tr[i]).backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        val = float(loss_fn(model(x_val), y_val))
    if val < best - 1e-4:
        best, bad = val, 0
        best_state = copy.deepcopy((model.state_dict(), loss_fn.state_dict()))
    else:
        bad += 1
        if bad >= 10:
            break
model.load_state_dict(best_state[0]); loss_fn.load_state_dict(best_state[1])
```

| Target (test split, 5 seeds) | Recipe | NLL | Coverage (90%) | Interval score |
|:--|:--|--:|--:|--:|
| Skewed (log-normal noise, n = 3000) | fixed 60 epochs, `B = 5`, 8 bins | 1.272 | 0.832 | 5.259 |
| | early stopping, `B = 13.6`, 16 bins | **1.201** | 0.887 | **5.143** |
| Heavy-tailed (Student-t, 3 df) | fixed 60 epochs, `B = 5`, 8 bins | 1.935 | 0.857 | 9.367 |
| | early stopping, `B = 16.9`, 16 bins | **1.852** | 0.879 | **9.172** |
| `load_diabetes` (n = 309 train) | fixed 60 epochs, `B = 5`, 8 bins | 6.144 | 0.647 | 327.9 |
| | early stopping, `B = 5`, 16 bins | **5.417** | 0.880 | **222.0** |

On the `house_sales` row of the harness `pzflow_parity` suite the torchregress NSF
trains for a fixed 60 epochs on all training rows with no validation split, which is
the `fixed 60 epochs` recipe above; its 0.756 coverage at a nominal 0.90 (pzflow: 0.915)
is the over-sharpening signature that early stopping removes on the datasets above.

!!! tip "Why not change the library defaults"
    `create_flow_model` keeps zuko's `bins=8` and bound `5` and `NormalizingFlowLoss` does
    not train the flow, so the budget and stopping rule belong to the training loop. The
    data-dependent bound cannot be chosen at construction time; derive it from the
    standardised training target with `recommended_tail_bound` and pass it as `bound`.

!!! warning "Amortized posteriors over wide priors (SBI)"
    The same limit applies when the flow models `p(theta | x)`: a parameter whose
    training range exceeds the bound is outside the spline domain. On the sbibm
    `gaussian_mixture` task (`theta ~ U(-10, 10)^2`) an NSF trained on raw `theta` with the
    default bound put every posterior on the bound edge (posterior mean `-4.92` for a
    reference at `-9.26`; C2ST 0.997 against 0.51 for sbi NPE, which z-scores `theta`).
    Either z-score `theta` before training (and invert the samples, subtracting
    `sum(log scale)` from log-densities), or pass `bound=recommended_tail_bound(theta_train)`;
    in the harness `sbi_parity` suite both bring C2ST to about 0.6 at the same simulation
    budget. `NormalizingFlowLoss` warns once when a target exceeds the bound.

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
[Comparison Against Baselines](../examples/contrastive_flow_parameter_estimation.md#comparison-against-baselines) (`--mode compare`).

!!! warning "Mask semantics"
    Flow objectives model a **joint density** over all target dimensions. `NormalizingFlowLoss`
    and `ContrastiveFlowLoss` therefore only support **sample-level** masking, not partial
    feature-wise masking. If one target dimension is missing, drop the whole sample or switch to
    a model with an explicit missing-data strategy.

---

## Stable affine layers (MAF, RealNVP, cINN)

An affine layer maps coordinate $i$ as $z_i = s_i\,x_i + t_i$ with
$(\log s_i, t_i) = h(x_{<i}, \mathbf{c})$ for MAF ($x_{<i}$ are the preceding
coordinates in the layer's order) or $h(x_A, \mathbf{c})$ for a coupling layer
($x_A$ is the unchanged half). Sampling runs the inverse
$x_i = (z_i - t_i)/s_i$. The conditioner $h$ is a ReLU network, which
extrapolates linearly outside the training range. One far-tail base draw gives a
large $x_j$; $h$ answers with a large shift and a small scale, so $x_{j+1}$ is
larger still and the error grows geometrically over coordinates and layers. The
density stays correct (the test NLL looks fine), but a few in a million samples
are astronomically large, finite numbers, and sample-based scores (CRPS, energy
score, intervals) are ruined. torchregress observed samples of $10^{12}$ in
standardized units from a 6-target MAF trained on one million rows, and the same
tail from zuko's MAF used through sbi.

`create_flow_model` therefore builds the affine flows (zuko layout, masks and
state-dict keys) with two safeguards:

$$
\log s_i = \frac{a_i}{1 + |a_i| / L}, \qquad L = -\log(\text{slope}) = 3,
$$

so that each layer stretches or shrinks a coordinate by at most $e^{3}$ (zuko's
default $L = 6.9$ allows a factor $10^3$), and the conditioner sees
softly clipped inputs

$$
(\log s_i, t_i) = h\big(\operatorname{clip}_B(x_{<i}), \mathbf{c}\big), \qquad
\operatorname{clip}_B(x) = \frac{x}{\sqrt{1 + (x/B)^2}}, \qquad B = 10 .
$$

$\operatorname{clip}_B$ is the identity to second order near zero and saturates at
$\pm B$, so the conditioner's outputs stay in the range it learned and the
inverse cannot cascade. The same conditioner is used in both directions: the
layers remain exact bijections with an exact triangular Jacobian, and
`log_prob` and `sample` describe the same density. $B$ is in the target's
units, so standardize the target (as for NSF) or pass a wider `bound=`, e.g.
`recommended_tail_bound(y_standardised)` for heavy tails.

!!! warning "Changed in 0.3.0"
    Flows built by earlier versions used zuko's unbounded conditioners and
    $L = 6.9$. Their checkpoints load unchanged (same parameter names) but are
    evaluated with the safeguards; pass `slope=1e-3` and a very large `bound` to
    rebuild the old map.

```python
import torch
from torchregress.losses import NormalizingFlowLoss, create_flow_model

torch.manual_seed(0)
y = torch.distributions.StudentT(3.0).sample((4096, 3))  # heavy tails
x = torch.randn(4096, 8)
y_std = (y - y.mean(0)) / y.std(0)
encoder = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 32))
loss_fn = NormalizingFlowLoss(create_flow_model(3, 32, flow_type="maf", n_transforms=3))
params = list(encoder.parameters()) + list(loss_fn.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3)
for _ in range(200):
    optimizer.zero_grad()
    loss = loss_fn(encoder(x), y_std)
    loss.backward()
    optimizer.step()
with torch.no_grad():
    samples = loss_fn.sample(encoder(x[:100]), n_samples=1000)  # [100, 1000, 3]
print(float(samples.abs().max()))
```

NSF and BPF need no such safeguard: their monotone maps are the identity
outside $[-B, B]$ and map $[-B, B]$ onto itself, so a layer's inverse satisfies
$|x| \le \max(|z|, B)$.

## Transformer autoregressive flows (TarFlow)

`flow_type="tarflow"` builds a tabular, conditional version of TarFlow
(Zhai et al. 2025, ref. 14) and its deep-shallow scaling, STARFlow
(Gu et al. 2025, ref. 15). Block $k$ is an exact autoregressive bijection in a
coordinate order $\pi_k$:

$$
z_{\pi_k(j)} = f\!\left(y_{\pi_k(j)};\ \boldsymbol\theta_{k,j}\right),
\qquad
\boldsymbol\theta_{k,j} = \operatorname{CausalTransformer}_k\!\left(\mathbf{c},\, y_{\pi_k(0)}, \ldots, y_{\pi_k(j-1)}\right)_j ,
$$

where the token sequence is a context token followed by one token per
preceding target, and $f$ is a monotone rational-quadratic spline (as in NSF, ref. 16)
or, with `univariate="affine"`, TarFlow's affine map. Consecutive blocks reverse
the order. The Jacobian is triangular, so

$$
\log p(\mathbf{y}\mid\mathbf{c}) = \log \mathcal{N}(\mathbf{z}; \mathbf{0}, \mathbf{I})
+ \sum_k \sum_j \log \left|\frac{\partial f}{\partial y_{\pi_k(j)}}\right| .
$$

The difference from NSF and MAF is the conditioner: a masked MLP (MADE) gives
every target the same fixed-width view of its predecessors, while attention
lets each target weigh any of them. Density evaluation is one Transformer pass
per block; sampling needs one pass per target coordinate per block.

The image-specific parts of TarFlow (patch tokens, noise augmentation with a
score-based denoising step, guidance) and STARFlow's pretrained autoencoder
latent space are left out, because tabular targets are continuous,
low-dimensional and modelled directly.

```python
import torch

from torchregress.losses import NormalizingFlowLoss, create_flow_model, recommended_tail_bound

torch.manual_seed(0)
x = torch.randn(2000, 2)
z = torch.randn(2000, 8).cumsum(dim=1) / torch.arange(1, 9).sqrt()  # correlated
y = z + 0.3 * torch.tanh(x[:, :1]) * z**2  # skewed, context-dependent, 8 targets
y = (y - y.mean(0)) / y.std(0)

flow = create_flow_model(
    n_features=8,
    context_dim=2,
    flow_type="tarflow",
    n_transforms=3,
    hidden_features=[64],
    n_layers=(4, 1, 1),  # STARFlow-style: one deep block, then shallow ones
    bound=recommended_tail_bound(y),
)
loss_fn = NormalizingFlowLoss(flow)
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
for epoch in range(3):
    for batch in torch.randperm(2000).split(256):
        optimizer.zero_grad()
        loss_fn(x[batch], y[batch]).backward()
        optimizer.step()

samples = loss_fn.sample(x[:5], n_samples=1000)  # [5, 1000, 8], for corner plots
```

### TarFlow against NSF and MAF

Each flow was trained with Adam ($10^{-3}$), batch size 256 and early stopping
on validation NLL (patience 15). The data were 8000 training, 2000 validation and
4000 test pairs, with standardized targets and the spline bound from
`recommended_tail_bound`. Results are means over 3 seeds. The
tasks have known conditional densities. *KL gap* is
$\mathbb{E}[\log p - \log \hat p]$ on the test set in nats (0 is perfect).
*Calibration* is the mean absolute error of the sample-based joint HPD coverage
([`sample_hpd_coverage`](../metrics/distribution.md)) at 0.5, 0.68, 0.9 and 0.95.
*Sampling* is the wall time for 256 draws at each of 1000 contexts (8 CPU threads
shared by 4 jobs). NSF and MAF use 5 blocks of MLP widths `[128, 128]`.
TarFlow uses 3 blocks of width 64, with 2 layers per block, or 4, 1 and 1
layers in the deep-shallow variant.

| Task (targets) | Flow | KL gap (nats) | Calibration error | Sampling (s) |
|:--|:--|--:|--:|--:|
| Banana (2) | NSF | 0.068 ± 0.017 | 0.013 | 3.6 |
| | MAF | **0.038 ± 0.009** | 0.010 | 1.8 |
| | TarFlow, spline | 0.085 ± 0.036 | 0.021 | 11.7 |
| | TarFlow, deep-shallow | 0.076 ± 0.026 | 0.015 | 9.3 |
| 3-component Gaussian mixture (4) | NSF | **0.357 ± 0.026** | 0.029 | 14.2 |
| | MAF | 0.361 ± 0.032 | 0.033 | 3.7 |
| | TarFlow, spline | 0.438 ± 0.013 | 0.033 | 45.4 |
| | TarFlow, deep-shallow | 0.467 ± 0.019 | 0.024 | 44.3 |
| Student-t, 3 dof (3) | NSF | 0.088 ± 0.024 | 0.013 | 10.1 |
| | MAF | **0.058 ± 0.014** | 0.019 | 2.7 |
| | TarFlow, spline | 0.109 ± 0.009 | 0.028 | 27.4 |
| | TarFlow, deep-shallow | 0.088 ± 0.015 | 0.014 | 27.0 |
| Skewed Gaussian copula (8) | NSF | 0.297 ± 0.015 | 0.033 | 52.7 |
| | MAF | 0.242 ± 0.009 | 0.025 | 4.7 |
| | TarFlow, spline | 0.175 ± 0.021 | 0.021 | 140.8 |
| | TarFlow, deep-shallow | **0.147 ± 0.011** | **0.013** | 141.5 |
| 4-component Gaussian mixture (8) | NSF | 1.020 ± 0.032 | 0.054 | 26.1 |
| | MAF | **0.871 ± 0.089** | **0.037** | 2.7 |
| | TarFlow, spline | 1.207 ± 0.059 | 0.053 | 105.1 |
| | TarFlow, deep-shallow | 1.221 ± 0.026 | 0.057 | 103.0 |
| Skewed Gaussian copula (16)† | NSF | 0.735 ± 0.034 | 0.048 | 70.9 |
| | MAF | 0.771 ± 0.034 | 0.043 | 5.2 |
| | TarFlow, spline | **0.197 ± 0.012** | **0.027** | 383.0 |
| | TarFlow, deep-shallow | 0.203 ± 0.016 | 0.033 | 399.4 |

† One job per 8-core session (2 threads per job), so the sampling times are not
directly comparable with the other rows.

The affine variant (`univariate="affine"`, five blocks) was only run on the
first four tasks. It beat the spline variant on the banana (0.053 nats) but was
worse on the other three (0.547, 0.117 and 0.263 nats), and worst of all flows on
the mixture: one affine map per block cannot split a mode.

!!! info "When TarFlow is worth it"
    TarFlow is a specialist option, not a replacement for NSF.
    It won clearly on the **skewed copula**, where 8 or more smooth, skewed
    targets all depend on each other. There its KL gap was about half of NSF's at
    8 targets and about a quarter at 16 (0.20 against 0.74 nats), with the best
    joint calibration in both cases.
    It lost on **multimodal** targets (both mixtures), and for **2 to 4 targets**
    MAF and NSF were as accurate or better.
    It also costs 2 to 3× NSF's training time per epoch and samples 30 to 75×
    slower than MAF (one Transformer pass per target per block).
    These are synthetic tasks at one training budget, so compare held-out NLL
    and [`sample_hpd_coverage`](../metrics/distribution.md) on your own data
    before switching.

!!! warning "Sample through the loss, not the raw zuko flow"
    Inverting an autoregressive flow (NSF, MAF, BPF, TarFlow) rebuilds every
    layer's univariate parameters once per target per layer, and torch's cached
    inverse transforms keep them in reference cycles until Python's garbage
    collector runs. `NormalizingFlowLoss.sample` collects them after each
    block of at most 8192 draws (contexts × samples). Large requests are also
    faster that way: 51,200 NSF draws at 16 targets took 33 s instead of 84 s.
    Calling `flow(context).sample(...)` directly skips both: in the benchmark
    above, repeated raw calls at 16 targets grew one process to about 18 GB on Linux.

## Flow Architectures

| Flow | Expressivity | Sampling | Density Eval | Best For |
|:-----|:----------:|:--------:|:------------:|:---------|
| **RealNVP** | ⭐⭐ | Fast | Fast | Lower dimensions, quick baseline |
| **MAF** | ⭐⭐⭐ | Slow | Fast | Conditional density estimation |
| **NSF** | ⭐⭐⭐⭐ | Medium | Medium | Complex multimodal distributions |
| **TarFlow** | ⭐⭐⭐ | Slow | Medium | Many smooth, mutually dependent targets (8 or more) |

## Limitations

1. **Computational cost**: Each training step evaluates $K$ invertible transforms and their Jacobian determinants. For `NSF` with 5+ transforms and large context dimensions, training can be 5–20× slower than an equivalent MDN.
2. **Mask semantics**: Flow objectives model a **joint density** over all target dimensions. Only **sample-level** masking is supported — if one target dimension is missing, drop the whole sample.
3. **Architecture choice matters**: `RealNVP` is fast but less expressive; `NSF` is most expressive but slower. Match the flow type to your target: use `NSF` for complex multimodality, `MAF` for conditional density estimation, `RealNVP` for a quick baseline.
4. **ContrastiveFlowLoss is task-specific**: It optimises a contrastive likelihood-ratio/ranking objective, not density calibration. `temperature != 1` or `margin != 0` leave the calibrated-likelihood-ratio regime; negative contexts must follow a meaningful reference distribution for the ratios to be interpretable. If you need well-calibrated conditional densities for sampling or downstream UQ, start with `NormalizingFlowLoss`. For classifier-based NRE see [`NeuralRatioLoss`](nre.md).
5. **Dependency**: Requires [`zuko`](https://github.com/probabilists/zuko). Not installed by default; use `pip install torchregress[flows]`.

## When To Use Which Flow Objective

| Objective | Start Here When | Tradeoff |
|:--|:--|:--|
| `NormalizingFlowLoss` | You need calibrated conditional densities, sampling, or a strong general-purpose flow baseline | Does not directly optimize parameter discrimination |
| `ContrastiveFlowLoss` | You need to rank the true hypothesis above alternate parameter settings with meaningful training negatives | InfoNCE/ranking objective, not calibrated densities; temperature/margin and the negative distribution define the estimand |
| [`NeuralRatioLoss`](nre.md) | You want the standard binary/multiclass NRE ratio estimator with a classifier (cheaper, no flow Jacobians) | Needs a classifier and a good reference distribution; ratios are relative to that reference |

---

## Next steps

- [MDN losses](mdn.md) — lighter-weight mixture models for simpler multimodality
- [SLS regression](sls.md) — volume-optimal prediction regions (flow-based frontier)
- [Contrastive flow estimation](../examples/contrastive_flow_parameter_estimation.md) — parameter ranking example
- [Flow comparison](../examples/contrastive_flow_parameter_estimation.md#comparison-against-baselines) — Gaussian summary, flow NLL, contrastive flow, NRE and NPE on one budget

---

## Ensembles of flows and joint calibration

A single flow captures aleatoric spread, not the uncertainty in its own fit. Train $M$ flows
independently (different seeds or bootstraps) and combine them with
[`DensityEnsemble`](../api/ensemble.md), the equal-weight mixture

$$
p(y \mid x) = \frac{1}{M}\sum_{m=1}^{M} p_m(y \mid x),
$$

whose entropy splits into a mean member entropy (aleatoric) and the members' mutual information
with $y$ (epistemic, $\geq 0$):

$$
H\bigl[p(\cdot \mid x)\bigr] = \frac{1}{M}\sum_m H\bigl[p_m(\cdot \mid x)\bigr]
+ \frac{1}{M}\sum_m \mathrm{KL}\bigl(p_m(\cdot \mid x)\,\|\,p(\cdot \mid x)\bigr).
$$

Unlike a variance decomposition, this holds for multimodal and multi-target densities.

Per-dimension PIT and interval coverage cannot detect a wrong *joint* shape (for example, a
wrong correlation between targets). [`sample_hpd_level`](../api/metrics.md) works on joint
highest-density regions instead. For each target it counts how many predictive draws have a
higher density than the observed $y$:

$$
\ell(x, y) = \frac{1}{S}\sum_{s=1}^{S} \mathbf{1}\bigl[\log p(y_s \mid x) > \log p(y \mid x)\bigr],
\qquad y_s \sim p(\cdot \mid x).
$$

For a calibrated model $\ell$ is uniform on $[0, 1]$ in any dimension, and
[`sample_hpd_coverage`](../api/metrics.md) reports the coverage of the nominal 50/68/90/95%
regions. Values pile up near 1 when the model is over-confident and near 0 when it is
under-confident.

```python
import torch
from torch import nn
from torchregress.ensemble import DensityEnsemble
from torchregress.losses import create_flow_loss, recommended_tail_bound
from torchregress.metrics import model_hpd_level

torch.manual_seed(0)
x = torch.randn(2048, 4)
y = torch.stack([x[:, 0] + 0.3 * torch.randn(2048), x[:, 0] * x[:, 1] + 0.3 * torch.randn(2048)], 1)

bound = recommended_tail_bound(y)  # keep the spline range wide enough for x0 * x1
heads = []
for seed in range(3):
    torch.manual_seed(seed)
    head = create_flow_loss(n_features=2, context_dim=4, flow_type="nsf", bound=bound)
    optimiser = torch.optim.Adam(head.parameters(), lr=3e-3)
    for _ in range(200):
        loss = head(x, y)            # conditional NLL with the features as context
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    heads.append(head)

ensemble = DensityEnsemble(heads)
x_test = torch.randn(256, 4)
y_test = torch.stack(
    [x_test[:, 0] + 0.3 * torch.randn(256), x_test[:, 0] * x_test[:, 1] + 0.3 * torch.randn(256)], 1
)
with torch.no_grad():
    nll = -ensemble.log_prob(x_test, y_test).mean()
parts = ensemble.entropy_decomposition(x_test, n_samples=64)       # total / aleatoric / epistemic
levels = model_hpd_level(ensemble, x_test, y_test, n_samples=128)  # ~ Uniform(0, 1) if calibrated
coverage_90 = (levels <= 0.9).float().mean()  # well below 0.9 means over-confident
```

Short training budgets typically leave flow ensembles over-confident (90% joint coverage well
below 0.9 in this example). Validation-NLL early stopping and more members both help;
conformalising the ensemble density with [`ConformalizedDensity`](conformal.md) restores
marginal coverage.

**Fitted weights with error bars.** Equal weights are a choice, not an estimate.
`ensemble.fit_weights(x_val, y_val)` fits $\pi$ by maximum likelihood on held-out data (concave, solved
by EM) and stores a sandwich covariance; `ensemble.log_prob_std(x, y)` then returns the standard
error of $\log p(y \mid x)$ due to estimating $\pi$. This applies the *wifi ensemble* idea
(Benevedes & Thaler 2025, see [NRE](nre.md#frequentist-uncertainties-wifi-ensembles)) to densities.
It quantifies weight uncertainty only, not errors inside the frozen members, and assumes the
weights are away from zero.

```python
# Continuing the example above: fresh data that the heads were not trained on.
x_val = torch.randn(1024, 4)
y_val = torch.stack(
    [x_val[:, 0] + 0.3 * torch.randn(1024), x_val[:, 0] * x_val[:, 1] + 0.3 * torch.randn(1024)], 1
)
ensemble.fit_weights(x_val, y_val)
log_p = ensemble.log_prob(x_test, y_test)
log_p_std = ensemble.log_prob_std(x_test, y_test)  # [N]
```

!!! tip "Cost"
    `entropy_decomposition` scores every member's draws under every member: $M^2 S$ flow
    evaluations per row. Use small $S$ (32–128) for monitoring and larger $S$ only for reports.

## References

| # | Reference |
|:-:|:----------|
| 1 | D. Rezende, S. Mohamed. ["Variational Inference with Normalizing Flows."](https://arxiv.org/abs/1505.05770) *ICML*, **2015**. |
| 2 | G. Papamakarios et al. ["Normalizing Flows for Probabilistic Modeling and Inference."](https://arxiv.org/abs/1912.02762) *JMLR*, 22(57):1–64, **2021**. |
| 3 | F. Rozet et al. ["Zuko: Normalizing Flows in PyTorch."](https://github.com/probabilists/zuko) **2022**. |
| 4 | I. Elsharkawy, Y. Kahn. ["Contrastive Normalizing Flows for Uncertainty-Aware Parameter Estimation."](https://arxiv.org/abs/2505.08709) *arXiv:2505.08709*, **2025**. (Related work: density-shaping CNF + classifier + Neyman intervals; not the InfoNCE objective in `ContrastiveFlowLoss`.) |
| 5 | A. Oord, Y. Li, O. Vinyals. ["Representation Learning with Contrastive Predictive Coding."](https://arxiv.org/abs/1807.03748) *arXiv:1807.03748*, **2018**. (InfoNCE.) |
| 6 | K. Cranmer, J. Pavez, G. Louppe. ["Approximating Likelihood Ratios with Calibrated Discriminative Classifiers."](https://arxiv.org/abs/1506.02169) *arXiv:1506.02169*, **2015**. (Classifier-based ratio estimation.) |
| 7 | C. Durkan, I. Murray, G. Papamakarios. ["On Contrastive Learning for Likelihood-free Inference."](https://arxiv.org/abs/2002.03712) *ICML*, **2020**. (Balanced contrastive NRE.) |
| 8 | B. K. Miller, C. Weniger, P. Forré. ["Contrastive Neural Ratio Estimation."](https://arxiv.org/abs/2210.06170) *NeurIPS*, **2022**. (Bias-free multiclass NRE; `NeuralRatioLoss(criterion="cnre")`.) |
| 9 | G. Papamakarios, T. Pavlakou, I. Murray. ["Masked Autoregressive Flow for Density Estimation."](https://arxiv.org/abs/1705.07057) *NeurIPS*, **2017**. (MAF.) |
| 10 | L. Dinh, J. Sohl-Dickstein, S. Bengio. ["Density Estimation Using Real NVP."](https://arxiv.org/abs/1605.08803) *ICLR*, **2017**. (Affine coupling.) |
| 11 | L. Ardizzone et al. ["Guided Image Generation with Conditional Invertible Neural Networks."](https://arxiv.org/abs/1907.02392) *arXiv:1907.02392*, **2019**. (cINN; soft clamping of affine log-scales.) |
| 12 | B. Lakshminarayanan, A. Pritzel, C. Blundell. ["Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles."](https://arxiv.org/abs/1612.01474) *NeurIPS*, **2017**. (Deep ensembles.) |
| 13 | S. Depeweg et al. ["Decomposition of Uncertainty in Bayesian Deep Learning for Efficient and Risk-sensitive Learning."](https://arxiv.org/abs/1710.07283) *ICML*, **2018**. (Entropy decomposition.) |
| 14 | S. Zhai et al. ["Normalizing Flows are Capable Generative Models."](https://arxiv.org/abs/2412.06329) *ICML*, **2025**. (TarFlow; `flow_type="tarflow"`.) |
| 15 | J. Gu et al. ["STARFlow: Scaling Latent Normalizing Flows for High-resolution Image Synthesis."](https://arxiv.org/abs/2506.06276) *arXiv:2506.06276*, **2025**. (Deep-shallow design; `n_layers=(4, 1, 1)`.) |
| 16 | C. Durkan, A. Bekasov, I. Murray, G. Papamakarios. ["Neural Spline Flows."](https://arxiv.org/abs/1906.04032) *NeurIPS*, **2019**. (NSF; rational-quadratic splines.) |
