# Advanced Losses

This page covers advanced loss functions for sophisticated uncertainty estimation and probabilistic regression, including methods that model full predictive distributions.

## Overview

Advanced losses go beyond simple point predictions or variance estimation to model complete probability distributions. These methods are essential when:

- The target distribution is **multi-modal** (multiple peaks)
- The distribution shape **varies with inputs**
- You need **flexible, non-parametric** density estimation
- Standard parametric assumptions (Gaussian, etc.) are too restrictive

## Quick Selection Guide

| Method | Complexity | Expressiveness | Training Difficulty | Best For |
|--------|-----------|----------------|---------------------|----------|
| **Gaussian NLL** | Low | Low | Easy | Simple unimodal distributions |
| **MDN** | Medium | Medium-High | Medium | Multi-modal with known component count |
| **DeepAR** | Medium | Medium | Medium | Probabilistic time series forecasting |
| **Evidential Regression** | Medium | Medium | Medium | Decomposed uncertainty (aleatoric/epistemic) |
| **Normalizing Flows** | High | Very High | Hard | Complex, arbitrary distributions |
| **Conformal Prediction** | Low | N/A | Easy | Distribution-free coverage guarantees |

## Mixture Density Networks (MDN)

Model the output distribution as a weighted mixture of simpler distributions (typically Gaussians).

### Mathematical Foundation

$$p(y|x) = \sum_{k=1}^{K} \pi_k(x) \mathcal{N}(y|\mu_k(x), \Sigma_k(x))$$

Where:
- $\pi_k(x)$: Mixture weights (sum to 1)
- $\mu_k(x)$: Component means
- $\Sigma_k(x)$: Component covariances

### When to Use MDNs

**Ideal scenarios:**
- Predicting arrival time with multiple possible routes
- Medical diagnosis with distinct disease subtypes
- Financial forecasting with market regime changes
- Inverse kinematics with multiple solutions

**Example: Inverse Problem**

```python
import torch
import torch.nn as nn
import torchregress as tr

# Inverse kinematics: multiple joint configurations can produce same endpoint
class InverseKinematicsModel(nn.Module):
    def __init__(self, input_dim=3, output_dim=6, n_components=4):
        super().__init__()

        # Calculate MDN output size: weights + means + log_stds
        mdn_size = n_components + 2 * n_components * output_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, mdn_size)
        )

    def forward(self, x):
        return self.network(x)

# Create model and loss
model = InverseKinematicsModel()
loss_fn = tr.losses.MixtureDensityLoss(
    n_components=4,
    n_features=6,
    covariance_type='diagonal',
    min_std=1e-3
)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    # Forward pass
    mdn_params = model(endpoint_positions)
    loss = loss_fn(mdn_params, joint_configurations)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Practical Tips for MDNs

**1. Number of Components:**
```python
# Start small and increase if needed
n_components = 3  # Good starting point

# Too many components can overfit
# If validation loss increases with more components, reduce
```

**2. Avoiding Component Collapse:**
```python
# Components may collapse during training
# Monitor component weights during training:
weights, means, stds = loss_fn._extract_distribution_parameters(pred)
component_usage = (weights > 0.1).float().mean()  # Should be > 0.8
```

**3. Initialization:**
```python
# Initialize final layer with small weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=0.01)
        m.bias.data.fill_(0.01)

model.network[-1].apply(init_weights)
```

**4. Sampling from MDN:**
```python
def sample_mdn(mdn_params, loss_fn, n_samples=100):
    """Generate samples from trained MDN"""
    weights, means, stds = loss_fn._extract_distribution_parameters(mdn_params)

    batch_size = means.shape[0]
    n_features = means.shape[-1]
    samples = []

    for b in range(batch_size):
        # Sample component indices
        probs = weights[b].cpu().numpy()
        components = torch.multinomial(
            torch.from_numpy(probs),
            n_samples,
            replacement=True
        )

        # Sample from selected components
        batch_samples = []
        for c in components:
            epsilon = torch.randn(n_features)
            sample = means[b, c] + stds[b, c] * epsilon
            batch_samples.append(sample)

        samples.append(torch.stack(batch_samples))

    return torch.stack(samples)
```

[Detailed MDN documentation →](mdn.md)

## Normalizing Flows

Transform a simple base distribution into a complex target distribution through learnable invertible transformations.

### Mathematical Foundation

$$p_X(x) = p_Z(f(x)) \left| \det\left(\frac{\partial f(x)}{\partial x}\right) \right|$$

Where $f$ is a composition of invertible transformations and $p_Z$ is a simple base distribution (e.g., Gaussian).

### Available Flow Types

**RealNVP:** Coupling layers, best for lower dimensions
**MAF:** Autoregressive, expressive but slow sampling
**NSF:** Neural splines, balanced expressivity and speed
**IAF:** Fast sampling, slow density evaluation

### When to Use Normalizing Flows

**Ideal scenarios:**
- Extremely complex distributions that can't be captured by mixtures
- High-dimensional outputs with intricate dependencies
- When you need both sampling and density evaluation
- Scientific applications requiring precise probability estimates

**Example: Multi-Output Regression with Complex Dependencies**

```python
import torch
import torch.nn as nn
import torchregress as tr

# Predict correlated variables (e.g., weather: temp, humidity, pressure)
class WeatherModel(nn.Module):
    def __init__(self, input_dim=50, output_dim=3):
        super().__init__()

        # Number of flow parameters depends on architecture
        # NSF with 5 blocks typically needs ~500-1000 params
        flow_params_dim = 800

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, flow_params_dim)
        )

        self.flow_loss = tr.losses.NormalizingFlowLoss(
            n_features=output_dim,
            flow_type='nsf',  # Neural Spline Flow
            n_blocks=5,
            hidden_features=128,
            n_hidden_layers=2
        )

    def forward(self, x):
        return self.encoder(x)

    def sample(self, x, n_samples=100):
        flow_params = self.forward(x)
        return self.flow_loss.sample(flow_params, n_samples)

# Training
model = WeatherModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(200):
    flow_params = model(input_features)
    loss = model.flow_loss(flow_params, weather_targets)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Important!
    optimizer.step()

# Inference: Sample multiple plausible weather scenarios
with torch.no_grad():
    samples = model.sample(test_input, n_samples=1000)
    # samples shape: [batch_size, 1000, 3] (temp, humidity, pressure)

    # Get mean prediction and uncertainty
    mean_pred = samples.mean(dim=1)
    std_pred = samples.std(dim=1)
```

### Practical Tips for Normalizing Flows

**1. Training Stability:**
```python
# Gradient clipping is essential
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Start with smaller learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Use learning rate scheduling
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)
```

**2. Architecture Selection:**
```python
# Start simple, increase complexity if needed
simple_flow = tr.losses.NormalizingFlowLoss(
    n_features=output_dim,
    flow_type='realnvp',  # Simpler, more stable
    n_blocks=3,           # Fewer blocks
    hidden_features=64    # Smaller networks
)

# Move to complex if simple doesn't fit
complex_flow = tr.losses.NormalizingFlowLoss(
    n_features=output_dim,
    flow_type='nsf',      # More expressive
    n_blocks=8,           # More transformation layers
    hidden_features=256   # Larger networks
)
```

**3. Memory Management:**
```python
# For large batches, use gradient accumulation
accumulation_steps = 4
optimizer.zero_grad()

for i, (x_batch, y_batch) in enumerate(dataloader):
    flow_params = model(x_batch)
    loss = model.flow_loss(flow_params, y_batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

[Detailed Normalizing Flows documentation →](nflows.md)

## Conformal Prediction

Provides prediction intervals with **rigorous coverage guarantees** without distributional assumptions.

### Key Concept

Unlike probabilistic methods that model distributions, conformal prediction provides **distribution-free** prediction sets with guaranteed coverage:

$$P(Y \in C(X)) \geq 1 - \alpha$$

This holds for **any** data distribution and **any** base model.

### When to Use Conformal Prediction

**Ideal scenarios:**
- Safety-critical applications requiring guaranteed coverage
- When you distrust distributional assumptions
- Finite-sample validity needed (not just asymptotic)
- Model-agnostic uncertainty quantification

**Important:** Conformal prediction does NOT provide:
- Uncertainty decomposition (epistemic vs aleatoric)
- Out-of-distribution detection
- Measure of model confidence

**Example: Safety-Critical Autonomous System**

```python
import torch
import torchregress as tr

# Base model (any architecture)
class SafetyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(10, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2)  # Lower and upper quantiles
        )

    def forward(self, x):
        return self.network(x)

# Train base model on quantiles
model = SafetyModel()
quantile_loss = tr.losses.MultiQuantileLoss(quantiles=[0.05, 0.95])
optimizer = torch.optim.Adam(model.parameters())

# Train on training set
for epoch in range(100):
    preds = model(X_train)
    loss = quantile_loss(preds, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Conformalize on calibration set
from torchregress.losses.conformal import ConformalLoss

conformal_loss = ConformalLoss(method='cqr', alpha=0.1)

with torch.no_grad():
    cal_preds = model(X_cal)

conformal_loss.calibrate(cal_preds, y_cal)

# Get guaranteed 90% coverage intervals on test set
with torch.no_grad():
    test_preds = model(X_test)
    lower, upper = conformal_loss.predict_interval(test_preds)

# Verify coverage (should be >= 90%)
coverage = ((y_test >= lower) & (y_test <= upper)).float().mean()
print(f"Coverage: {coverage:.2%}")  # Will be >= 90% with high probability
```

### Conformal Methods

**Split Conformal:** Simple, works with any model
```python
from torchregress.losses.conformal import ConformalLoss

conformal = ConformalLoss(method='split', alpha=0.1)
```

**CQR (Conformalized Quantile Regression):** More efficient intervals
```python
from torchregress.losses.conformal import ConformalLoss

conformal = ConformalLoss(method='cqr', alpha=0.1)
```

**ACI (Adaptive Conformal Inference):** Adaptive to input difficulty
```python
from torchregress.losses.conformal import ConformalLoss

conformal = ConformalLoss(method='aci', alpha=0.1, model=base_model)
```

[Detailed Conformal Prediction documentation →](conformal.md)

## DeepAR

DeepAR is a probabilistic forecasting method for time series data. It uses an autoregressive recurrent neural network to predict the parameters of a probability distribution for each time step.

### Mathematical Foundation

The DeepAR loss is typically the negative log-likelihood of a Gaussian distribution, where the mean and variance are predicted by the model at each time step.

$$p(y_t|y_{t-1}, ..., y_0, x_t) = \mathcal{N}(y_t|\mu(h_t), \sigma^2(h_t))$$

Where $h_t$ is the hidden state of the RNN at time $t$.

### When to Use DeepAR

**Ideal scenarios:**
- Time series forecasting
- When you need probabilistic forecasts (prediction intervals)
- When you have covariates that can be used to improve forecasts

**Example: Time Series Forecasting**

```python
import torch
import torch.nn as nn
import torchregress as tr

class DeepARModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear_mu = nn.Linear(hidden_size, 1)
        self.linear_sigma = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        mu = self.linear_mu(out)
        sigma = torch.nn.functional.softplus(self.linear_sigma(out)) + 1e-6
        return mu, sigma

# Create model and loss
model = DeepARModel(input_size=10, hidden_size=32, num_layers=2)
# DeepAR-style objectives are not currently implemented in torchregress.
# Consider GaussianNLLLoss or quantile/conformal methods for forecasting intervals.

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    # Forward pass
    mu, sigma = model(time_series_data)
    loss = loss_fn((mu, torch.log(sigma**2)), target_series)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Evidential Regression

Evidential Regression is a method for quantifying uncertainty from a single forward pass. It does this by placing a higher-order prior distribution over the parameters of the likelihood function.

### Mathematical Foundation

Evidential Regression uses a Normal-Inverse-Gamma (NIG) distribution as a prior over the mean and variance of a Gaussian likelihood. The model outputs four parameters $(\gamma, \nu, \alpha, \beta)$ that define the NIG distribution.

- **Aleatoric uncertainty**: $\mathbb{E}[\sigma^2] = \frac{\beta}{\alpha - 1}$
- **Epistemic uncertainty**: $\text{Var}[\mu] = \frac{\beta}{\nu(\alpha - 1)}$

### When to Use Evidential Regression

**Ideal scenarios:**
- When you need to distinguish between aleatoric and epistemic uncertainty.
- When you want to get uncertainty estimates from a single forward pass, without ensembles or sampling.
- For out-of-distribution detection.

**Example: Regression with Uncertainty Decomposition**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchregress as tr

class EvidentialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # Output: gamma, nu, alpha, beta
        )

    def forward(self, x):
        out = self.net(x)
        gamma = out[:, 0:1]  # mean
        nu = F.softplus(out[:, 1:2]) + 0.01  # > 0
        alpha = F.softplus(out[:, 2:3]) + 1.01  # > 1
        beta = F.softplus(out[:, 3:4]) + 0.01  # > 0
        return torch.cat([gamma, nu, alpha, beta], dim=1)

# Create model and loss
model = EvidentialModel()
loss_fn = tr.losses.EvidentialRegressionLoss(coeff_nig=0.01)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

x = torch.randn(32, 10)
y = torch.randn(32, 1)

for epoch in range(100):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Inference with uncertainty
model.eval()
with torch.no_grad():
    params = model(x)
    mean, ale_unc, epi_unc = loss_fn.predict_with_uncertainty(params)
```

## Choosing the Right Advanced Method

### Decision Tree

```
Do you need guaranteed coverage?
├─ YES → Use Conformal Prediction
└─ NO
   └─ Are you doing time series forecasting?
      ├─ YES → Use DeepAR
      └─ NO
         └─ Do you need to decompose uncertainty?
            ├─ YES → Use Evidential Regression
            └─ NO
               └─ Is the distribution multimodal?
                  ├─ YES, with known # of modes → Use MDN
                  ├─ YES, complex/unknown → Use Normalizing Flows
                  └─ NO → Use simpler methods (Gaussian NLL, Quantile)
```

### Computational Comparison

| Method | Training Time | Inference Time | Memory | Stability |
|--------|--------------|----------------|---------|-----------|
| Gaussian NLL | Fast | Very Fast | Low | High |
| MDN | Medium | Fast | Medium | Medium |
| DeepAR | Medium | Fast | Medium | Medium |
| Evidential Regression | Medium | Fast | Medium | Medium |
| Normalizing Flows | Slow | Medium | High | Low |
| Conformal | Fast (any base) | Fast | Low | High |

### Expressiveness vs Complexity Trade-off

```
Expressiveness
    ↑
    │                        ┌─ Normalizing Flows
    │                   ┌────┘
    │              ┌────┘ MDN
    │         ┌────┘
    │    ┌────┘ Gaussian NLL
    └────┘ MSE
         └────┴────┴────┴────→ Complexity
```

## Combining Methods

Advanced methods can be combined for even better results:

### 1. Ensemble of MDNs
```python
# Better uncertainty via model disagreement
mdns = [create_mdn_model() for _ in range(5)]

# Train each independently
for mdn in mdns:
    train_model(mdn, train_data)

# Average predictions
ensemble_samples = [mdn.sample(x) for mdn in mdns]
combined_samples = torch.cat(ensemble_samples, dim=1)
```

### 2. Conformal + Probabilistic
```python
# Get calibrated intervals on top of distributional predictions
base_model = create_mdn_model()
train_model(base_model, train_data)

# Use mean prediction for conformal
from torchregress.losses.conformal import ConformalLoss

conformal = ConformalLoss(method='split', alpha=0.1)
conformal.calibrate(base_model.mean(X_cal), y_cal)

# Now you have both:
# - Distributional predictions from MDN
# - Guaranteed coverage from conformal
```

### 3. Normalizing Flows with Ensemble
```python
# Epistemic + aleatoric uncertainty
flow_ensemble = [create_flow_model() for _ in range(5)]

# Each captures aleatoric via distribution
# Ensemble captures epistemic via disagreement
predictions = [flow.sample(x, 100) for flow in flow_ensemble]

# Total uncertainty = within-flow + between-flow
aleatoric = torch.cat(predictions).var(dim=1).mean(dim=0)
epistemic = torch.stack([p.mean(dim=1) for p in predictions]).var(dim=0)
```

## Best Practices

1. **Start Simple:** Begin with Gaussian NLL, move to advanced only if needed
2. **Validate Carefully:** Use proper scoring rules (NLL, CRPS) and calibration metrics
3. **Monitor Training:** Advanced methods can be unstable - watch for divergence
4. **Use Ensembles:** Combining multiple models often beats single complex model
5. **Test Coverage:** For safety-critical apps, empirically verify coverage guarantees

## Further Reading

- [Mathematical Foundations](../math/index.md) - Detailed mathematical background
- [Uncertainty Estimation](../math/index.md) - Understanding uncertainty types
- [Calibration Metrics](../metrics/calibration.md) - Evaluating uncertainty quality
