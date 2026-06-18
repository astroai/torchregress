# Error-Aware Feature Encoding

Error-aware encoders provide a **lightweight alternative** to explicit latent-input optimization (like [LatentNN](latentnn.md)) for handling noisy tabular inputs. Instead of learning per-sample latent inputs, they engineer **quality signals** from the known measurement uncertainties and feed them to the network as additional feature channels.

!!! abstract "Key idea"
    Give the network access to both the raw features **and** their estimated quality (log-uncertainty, signal-to-noise ratio, quality-gated features) so it can learn to **downweight unreliable inputs** through standard gradient-based training.

---

## Architecture Components

### `ErrorAwareFeatureEncoder`

A feature preprocessor that concatenates five channels per input feature:

| Channel | Formula | Purpose |
|:--------|:--------|:--------|
| **Quality-gated features** | $x \cdot g(\sigma_x)$ | Raw features gated by learned quality sigmoid |
| **Raw features** | $x$ | Preserves original signal |
| **Log uncertainty** | $\log \sigma_x$ | Explicit noise magnitude |
| **Signed SNR** | $x / \sigma_x$ | Normalized signal strength |
| **Gated precision** | $g(\sigma_x) / \sigma_x$ | Quality-weighted precision |

The **quality gate** $g(\sigma_x) = \sigma(\frac{\theta_{\text{ref}} - \log \sigma_x}{T})$ is a learned sigmoid that becomes active when the log-uncertainty falls below a reference level $\theta_{\text{ref}}$, with learnable temperature $T$. The gate is **per-feature**, allowing the network to learn which input dimensions are trustworthy.

### `NoiseAwareRegressor`

A convenience wrapper that pairs an `ErrorAwareFeatureEncoder` with a configurable prediction backbone (MLP with LayerNorm and GELU).

```
Input (x, σ_x) → ErrorAwareFeatureEncoder → Backbone MLP → Prediction
```

---

## Usage

### Basic Setup

```python
import torch
import torch.nn as nn
from torchregress.algorithms import NoiseAwareRegressor, ErrorAwareFeatureEncoder

# Option 1: Use the high-level wrapper
model = NoiseAwareRegressor(
    input_dim=10,           # Number of raw features
    output_dim=1,           # Prediction dimension
    encoder_hidden_dim=128,
    backbone_hidden_dims=(128, 64),
    dropout=0.1,
)

# Forward pass: model takes both features AND noise
x = torch.randn(64, 10)
sigma_x = 0.1 + 0.2 * torch.rand(64, 10)  # Per-sample noise
pred = model(x, sigma_x)
```

```python
# Option 2: Use encoder standalone
encoder = ErrorAwareFeatureEncoder(
    input_dim=10,
    hidden_dim=128,
    output_dim=64,
    dropout=0.0,
)

# Encode features + noise into quality-aware representation
features = encoder(x, sigma_x)  # (batch, 64)

# Then pass to any downstream model
downstream = nn.Linear(64, 1)
pred = downstream(features)
```

### Quality Gate Inspection

```python
# Inspect which features the encoder considers reliable
gate_values = encoder.quality_gate(x, sigma_x)
# gate_values[:, j] ≈ 1 → feature j is trusted
# gate_values[:, j] ≈ 0 → feature j is downweighted
print(f"Mean gate per feature: {gate_values.mean(dim=0)}")
```

### Training Loop

```python
from torchregress.losses import GaussianNLLLoss

model = NoiseAwareRegressor(input_dim=10, output_dim=2)  # mean + logvar
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = GaussianNLLLoss()

for x, y, sigma_x in train_loader:
    pred = model(x, sigma_x)       # Model uses noise info internally
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## When to Use

| Scenario | Recommended? | Rationale |
|:---------|:-----------:|:----------|
| **Known per-sample input noise, large dataset** | ✅ **Yes** | No per-sample parameters to store |
| **Unknown noise structure** | ❌ | Encoder needs $\sigma_x$ as input |
| **Noise varies strongly per feature** | ✅ | Per-feature quality gates learn to gate individually |
| **Need interpretable feature quality** | ✅ | `quality_gate()` exposes per-feature trust scores |
| **Very small models** | ⚠️ | The 5-channel expansion multiplies input dimension by 5 |

---

## Comparison: Error-Aware vs LatentNN vs Standard

| Method | Handles Noise | Per-Sample Storage | Interpretability | Training Cost |
|:-------|:------------:|:-----------------:|:----------------:|:------------:|
| **ErrorAwareFeatureEncoder** | ✅ (known $\sigma_x$) | None | ✅ (quality gates) | Standard |
| **LatentNN** | ✅ (known $\sigma_x$) | $N \times D$ | ✅ (latent inputs) | Higher (dual optimization) |
| **Standard MLP** | ❌ | None | ❌ | Standard |

---

## Limitations

!!! warning "Practical constraints"
    - **Requires known $\sigma_x$**: The encoder cannot estimate measurement noise internally — it must be provided. If noise is unknown, consider standard robust losses instead.
    - **Input dimension expansion**: The 5-channel encoding multiplies the input dimension by 5, which increases the first layer's parameter count proportionally.
    - **Gaussian noise assumption**: The SNR and precision channels are most informative under additive Gaussian noise. For non-Gaussian noise (quantization, Poisson), the derived channels may be less useful.
    - **No latent recovery**: Unlike `LatentNN`, the error-aware encoder does not produce denoised latent inputs — it only produces a quality-aware representation. If you need the cleaned inputs, use `LatentNN`.
    - **Feature-wise gating only**: The quality gate operates independently per feature. It cannot model interactions — e.g., "feature A is trustworthy only when feature B is also above some threshold." For cross-feature quality interactions, consider learned attention or a dedicated meta-model.

---

## Complete Example

```python
import torch
import torch.nn as nn
from torchregress.algorithms import NoiseAwareRegressor

torch.manual_seed(42)

# Generate data with input-dependent noise
n = 500
X_true = torch.randn(n, 5)
sigma_x = 0.1 + 0.4 * torch.rand(n, 5)       # Per-sample per-feature noise
X_obs = X_true + sigma_x * torch.randn(n, 5)
y = X_true @ torch.tensor([2.0, -1.0, 0.5, 0.3, -0.2]) + 0.1 * torch.randn(n)

# Model with error-aware encoding
model = NoiseAwareRegressor(input_dim=5, output_dim=1, dropout=0.1)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(200):
    pred = model(X_obs, sigma_x).squeeze(-1)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Compare with standard MLP (no noise info)
standard_model = nn.Sequential(
    nn.Linear(5, 128), nn.GELU(),
    nn.Linear(128, 64), nn.GELU(),
    nn.Linear(64, 1),
)
opt = torch.optim.Adam(standard_model.parameters(), lr=1e-3)
for epoch in range(200):
    pred = standard_model(X_obs).squeeze(-1)
    loss = loss_fn(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.no_grad():
    error_aware_mse = nn.functional.mse_loss(
        model(X_obs, sigma_x).squeeze(-1), y
    )
    standard_mse = nn.functional.mse_loss(
        standard_model(X_obs).squeeze(-1), y
    )
print(f"Error-Aware MSE: {error_aware_mse:.4f}")
print(f"Standard MLP MSE: {standard_mse:.4f}")
```

---

## References

| # | Reference |
|:-:|:----------|
| 1 | S. Fabbro et al. "Error-aware feature encoding for noisy tabular regression." *torchregress documentation*, **2025**. |
