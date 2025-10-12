# Poisson & Tweedie Loss Functions

This page covers loss functions for specialized data types: count data (Poisson) and data with complex mean-variance relationships (Tweedie).

## Quick Selection Guide

| Data Type | Characteristics | Recommended Loss | Power (p) |
|-----------|----------------|------------------|-----------|
| Count data | Non-negative integers | `WeightedPoissonNLLLoss` | p=1 |
| Count data with excess zeros | Many zeros beyond Poisson expectation | `ZeroInflatedPoissonNLLLoss` | - |
| Overdispersed counts | Variance > mean | `NegativeBinomialNLLLoss` | - |
| Positive continuous | Always positive, no zeros | `GammaLoss` | p=2 |
| Continuous with zeros | Zero values mixed with positive | `CompoundPoissonLoss` | 1<p<2 |
| Highly skewed positive | Strong right skew | `InverseGaussianLoss` | p=3 |

## Poisson Loss Functions

Poisson losses are designed for modeling count data and events that occur with a known average rate.

### Mathematical Background

The Poisson distribution models the probability of observing $k$ events in a fixed interval:

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

The negative log-likelihood becomes:

$$\mathcal{L}_{\text{Poisson}}(y, \lambda) = \lambda - y \log(\lambda) + \log(y!)$$

### Use Cases

- **Event Counts**: Number of website visits, customer arrivals, equipment failures
- **Rare Events**: Accidents, disease occurrences, natural disasters
- **Histogram Fitting**: Scientific data binning (e.g., particle physics)
- **Rate Estimation**: Events per unit time/space/volume

### PoissonDevianceLoss

Goodness-of-fit measure for Poisson models, useful for histogram fitting and scientific applications.

```python
import torch
import torchregress as tr

# For histogram fitting in scientific analysis
loss_fn = tr.losses.PoissonDevianceLoss(log_input=True)

# Model predicts log(λ) for each bin
y_pred = torch.log(torch.tensor([10.0, 20.0, 30.0, 40.0]))  # Expected counts
target = torch.tensor([12.0, 18.0, 32.0, 38.0])  # Observed counts

loss = loss_fn(y_pred, target)
```

**When to use:**
- Evaluating model fit to count data
- Comparing nested Poisson models (via likelihood ratio tests)
- High-energy physics histogram fitting

### ZeroInflatedPoissonNLLLoss

For count data with more zeros than a standard Poisson distribution would predict.

```python
# Customer purchase counts (many customers purchase nothing)
model_output = torch.tensor([1.0, 2.0, 3.0])  # Lambda values
pi_logits = torch.tensor([-1.0, 0.0, 1.0])  # Zero-inflation logits
purchases = torch.tensor([0.0, 0.0, 3.0])  # Many zeros

loss_fn = tr.losses.ZeroInflatedPoissonNLLLoss()
loss = loss_fn(model_output, purchases, pi_logits)
```

**When to use:**
- Survey responses (many non-respondents)
- Insurance claims (many zero-claim policies)
- Customer behavior (inactive users)

### NegativeBinomialNLLLoss

For overdispersed count data where variance exceeds the mean.

```python
# Gene expression counts (often overdispersed)
loss_fn = tr.losses.NegativeBinomialNLLLoss(learn_theta=True)

y_pred = torch.tensor([10.0, 50.0, 100.0])  # Mean expression
target = torch.tensor([5.0, 45.0, 150.0])  # Observed counts (high variance)

loss = loss_fn(y_pred, target)
```

**When to use:**
- RNA-seq data analysis
- Social media engagement counts
- Traffic accident counts
- Any count data with variance > mean

## Tweedie Loss Functions

Tweedie distributions generalize several common distributions through a power parameter $p$, with variance:

$$\text{Var}(Y) = \phi \cdot \mu^p$$

### TweedieLoss

General-purpose implementation with configurable power parameter.

```python
import torch
import torchregress as tr

# Insurance claims: continuous amounts with exact zeros
loss_fn = tr.losses.TweedieLoss(p=1.5, link='log')

# Model predicts log(μ)
y_pred = torch.tensor([0.0, 1.0, 2.0, 3.0])  # log(μ)
target = torch.tensor([0.0, 0.0, 5.0, 15.0])  # Claim amounts

loss = loss_fn(y_pred, target)
```

**Power parameter guide:**
- p=0: Normal distribution (use for standard regression)
- p=1: Poisson (use for count data)
- p=1.5: Common for insurance (many zeros, continuous claims)
- p=2: Gamma (strictly positive continuous)
- p=3: Inverse Gaussian (highly skewed positive)

### GammaLoss

For strictly positive continuous data with constant coefficient of variation.

```python
# Product prices or service durations (always positive)
loss_fn = tr.losses.GammaLoss(link='log')

y_pred = torch.log(torch.tensor([10.0, 50.0, 100.0]))  # log(price)
target = torch.tensor([12.0, 45.0, 110.0])  # Observed prices

loss = loss_fn(y_pred, target)
```

**When to use:**
- Product pricing
- Service durations
- Survival times
- Positive-valued financial data

### CompoundPoissonLoss

For continuous data with exact zeros (1 < p < 2).

```python
# Rainfall amounts: zero on dry days, continuous on rainy days
loss_fn = tr.losses.CompoundPoissonLoss(p=1.6)

y_pred = torch.tensor([0.0, 0.5, 1.0, 2.0])  # log(expected rainfall)
target = torch.tensor([0.0, 0.0, 2.5, 8.3])  # Daily rainfall (mm)

loss = loss_fn(y_pred, target)
```

**When to use:**
- Insurance claim amounts
- Rainfall measurements
- Customer expenditures (non-participants = 0)
- Environmental pollutants with detection limits

### InverseGaussianLoss

For highly right-skewed positive data.

```python
# Time until failure (often highly skewed)
loss_fn = tr.losses.InverseGaussianLoss(link='log')

y_pred = torch.log(torch.tensor([100.0, 500.0, 1000.0]))
target = torch.tensor([120.0, 480.0, 1200.0])  # Time to failure (hours)

loss = loss_fn(y_pred, target)
```

**When to use:**
- Reliability analysis (time to failure)
- Response time data
- Financial time series with long tails

## Practical Examples

### Example 1: E-commerce Click Prediction

```python
import torch
import torch.nn as nn
import torchregress as tr

# Predict daily click counts per user
class ClickModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x)  # log(lambda)

model = ClickModel()
loss_fn = tr.losses.PoissonDevianceLoss(log_input=True)

# Training
X = torch.randn(100, 10)  # User features
y = torch.randint(0, 20, (100, 1)).float()  # Click counts

y_pred = model(X)
loss = loss_fn(y_pred, y)
```

### Example 2: Insurance Claim Prediction

```python
# Predict claim amounts (many zeros, continuous positive values)
class ClaimModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(20, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.features(x)  # log(mu)

model = ClaimModel()
loss_fn = tr.losses.CompoundPoissonLoss(p=1.5)

# Policy features
X = torch.randn(1000, 20)
# Claim amounts (many zeros)
y = torch.cat([
    torch.zeros(700, 1),  # 70% no claims
    torch.rand(300, 1) * 5000  # 30% with claims
])

y_pred = model(X)
loss = loss_fn(y_pred, y)
```

### Example 3: Gene Expression Analysis

```python
# RNA-seq data (overdispersed counts)
class ExpressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(50, 256),
            nn.ReLU(),
            nn.Linear(256, 100),
        )

    def forward(self, x):
        return self.encoder(x)  # Mean expression

model = ExpressionModel()
loss_fn = tr.losses.NegativeBinomialNLLLoss(learn_theta=True)

# Cell features
X = torch.randn(200, 50)
# Gene expression counts (high variance)
y = torch.randint(0, 1000, (200, 100)).float()

y_pred = model(X)
loss = loss_fn(y_pred, y)
```

## Comparison with Gaussian Losses

**When Gaussian (MSE) works:**
- Large count values (>30)
- Approximately symmetric distributions
- Constant variance across predictions

**When Poisson/Tweedie works better:**
- Small counts (<10)
- Skewed distributions
- Variance increases with mean
- Exact zeros are meaningful

## Mathematical Insights

### Poisson as Limiting Case

For large $\lambda$, Poisson $\approx$ Normal($\mu=\lambda$, $\sigma^2=\lambda$):

```python
# Compare Poisson and Gaussian for large counts
large_counts = torch.tensor([100.0, 200.0, 300.0])
pred = torch.log(large_counts)

poisson_loss = tr.losses.PoissonDevianceLoss(log_input=True)(pred, large_counts)
gaussian_loss = tr.losses.MSELoss()(large_counts, large_counts)
# These should be similar for large counts
```

### Tweedie Power Parameter Selection

If you don't know the appropriate $p$:

1. **Visual inspection**: Plot mean vs. variance of residuals
2. **Profile likelihood**: Try multiple $p$ values, select minimum loss
3. **Domain knowledge**: Use established values for your field

```python
# Profile likelihood approach
powers = [1.1, 1.3, 1.5, 1.7, 1.9]
losses = []

for p in powers:
    loss_fn = tr.losses.TweedieLoss(p=p)
    val_loss = evaluate_on_validation(model, loss_fn, val_data)
    losses.append(val_loss)

best_p = powers[np.argmin(losses)]
```

## Related Documentation

- [Gaussian Losses](gaussian.md) - For standard continuous data
- [Poisson-Gaussian Mixture](poisson_gaussian.md) - For imaging with mixed noise
- [Mathematical Formulations](../math/formulations.md) - Detailed mathematical background
