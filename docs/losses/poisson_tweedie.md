# Poisson & Tweedie Loss Functions

This page covers loss functions for modeling count data, and data with complex mean-variance relationships.

## Poisson Loss Functions

Poisson loss functions are designed for modeling count data and events that occur with a known average rate in a fixed interval of time or space. These losses are appropriate for scenarios where the target variable represents:

- Count data (non-negative integers)
- Event frequencies
- Rate data (events per unit time/space)
- Rare event occurrences

### Mathematical Background

The Poisson distribution models the probability of observing $k$ events in a fixed interval when events occur independently at a constant rate $\lambda$:

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

Where:
- $\lambda$ is the rate parameter (expected number of occurrences)
- $k$ is the number of occurrences (target value)

The negative log-likelihood of the Poisson distribution is:

$$\mathcal{L}_{\text{Poisson}}(y, \lambda) = \lambda - y \log(\lambda) + \log(y!)$$

In practice, the $\log(y!)$ term is often omitted during optimization as it doesn't depend on the model parameters.

### Available Poisson Losses

#### PoissonNLLLoss

For standard Poisson Negative Log-Likelihood, use `torch.nn.PoissonNLLLoss`.

#### PoissonDevianceLoss

Poisson deviance loss function, also known as G-statistic, which measures the goodness-of-fit for Poisson models. This loss is useful for assessing how well a model fits count data.

**Mathematical Formulation:**

The deviance is defined mathematically as:

$$D(y, \lambda) = 2 \sum_i \left[ y_i \log\left(\frac{y_i}{\lambda_i}\right) - (y_i - \lambda_i) \right]$$

For implementation, we use the equivalent form (without the factor of 2):

$$\mathcal{L}_{\text{Deviance}}(y, \lambda) = \lambda - y + y \log\left(\frac{y}{\lambda}\right)$$

Where $y=0$, the term $y \log(y/\lambda) = 0$.

**Example:**

```python
import torch
import torchregress as tr

# Create deviance loss
loss_fn = tr.losses.PoissonDevianceLoss(log_input=True)

# For a model that predicts log(λ)
y_pred = torch.tensor([[0.0, 1.0, 2.0], [1.0, 1.5, 0.5]])  # log(λ) values
target = torch.tensor([[1.0, 2.0, 7.0], [2.0, 5.0, 1.0]])  # count data

# Calculate deviance
loss = loss_fn(y_pred, target)
```

## Tweedie Loss Functions

Tweedie loss functions are derived from the Tweedie family of distributions, which include many common distributions as special cases. These losses are particularly valuable for modeling data with complex variance-mean relationships.

### Mathematical Background

The Tweedie distribution is defined by its variance function:

$$\text{Var}(Y) = \phi \cdot \mu^p$$

Where:
- $\mu$ is the mean
- $\phi$ is the dispersion parameter
- $p$ is the power parameter

This power parameter $p$ determines the specific distribution:
- $p = 0$: Normal distribution
- $p = 1$: Poisson distribution
- $p = 2$: Gamma distribution
- $p = 3$: Inverse Gaussian distribution
- $1 < p < 2$: Compound Poisson-Gamma (handles continuous data with exact zeros)

### Available Tweedie Losses

#### TweedieLoss

General-purpose Tweedie loss function for regression with configurable power parameter.

**Mathematical Formulation:**

The Tweedie deviance (loss) depends on the power parameter $p$:

For $p = 0$ (Normal):
$$\mathcal{L}(y, \mu) = \frac{1}{2}(y - \mu)^2$$

For $p = 1$ (Poisson):
$$\mathcal{L}(y, \mu) = \mu - y\log(\mu) + \log(y!)$$

For $p = 2$ (Gamma):
$$\mathcal{L}(y, \mu) = \log\left(\frac{\mu}{y}\right) + \frac{y}{\mu} - 1$$

For $1 < p < 2$ (Compound Poisson-Gamma):
$$\mathcal{L}(y, \mu) = \frac{2}{(2-p)(1-p)}\left[y^{2-p} - (2-p)y\mu^{1-p} + (1-p)\mu^{2-p}\right]$$

**Example:**

```python
import torch
import torchregress as tr

# For regression with compound Poisson-Gamma distribution (p=1.5)
loss_fn = tr.losses.TweedieLoss(p=1.5, link='log')

# Model predicts log of mean parameter
y_pred = torch.tensor([[0.0, 1.0, 2.0], [0.5, 1.5, 2.5]])  # log(μ) values
target = torch.tensor([[0.0, 2.0, 8.0], [1.0, 4.0, 12.0]])  # response values

# Calculate loss
loss = loss_fn(y_pred, target)
```

#### GammaLoss

Specialized loss for gamma regression (p=2), suitable for positive continuous response variables with constant coefficient of variation.

**Example:**

```python
import torch
import torchregress as tr

# For modeling positive continuous data like prices or durations
loss_fn = tr.losses.GammaLoss(link='log')

# Model predicts log(μ)
y_pred = torch.tensor([[1.0, 2.0, 3.0]])  # log(μ) values
target = torch.tensor([[3.0, 7.5, 20.0]])  # positive response values

# Calculate loss
loss = loss_fn(y_pred, target)
```