# Poisson Loss Functions

Poisson loss functions are designed for modeling count data and events that occur with a known average rate in a fixed interval of time or space. These losses are appropriate for scenarios where the target variable represents:

- Count data (non-negative integers)
- Event frequencies
- Rate data (events per unit time/space)
- Rare event occurrences

## Mathematical Background

The Poisson distribution models the probability of observing $k$ events in a fixed interval when events occur independently at a constant rate $\lambda$:

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

Where:
- $\lambda$ is the rate parameter (expected number of occurrences)
- $k$ is the number of occurrences (target value)

The negative log-likelihood of the Poisson distribution is:

$$\mathcal{L}_{\text{Poisson}}(y, \lambda) = \lambda - y \log(\lambda) + \log(y!)$$

In practice, the $\log(y!)$ term is often omitted during optimization as it doesn't depend on the model parameters.

## Available Poisson Losses

### PoissonNLLLoss

```python
class PoissonNLLLoss(RegressionLoss)
```

For standard Poisson Negative Log-Likelihood, use WeightedPoissonNLLLoss from the base module.

**Example:**

```python
import torch
import torchregress as tr

# Create Poisson NLL loss
loss_fn = tr.losses.WeightedPoissonNLLLoss(log_input=True, full=False)

# For a model that predicts log(λ)
y_pred = torch.tensor([[0.0, 1.0, 2.0], [1.0, 1.5, 0.5]])  # log(λ) values
target = torch.tensor([[1.0, 2.0, 7.0], [2.0, 5.0, 1.0]])  # count data

# Calculate loss
loss = loss_fn(y_pred, target)
```

### PoissonDevianceLoss

```python
class PoissonDevianceLoss(RegressionLoss)
```

Poisson deviance loss function, also known as G-statistic, which measures the goodness-of-fit for Poisson models. This loss is useful for assessing how well a model fits count data.

**Parameters:**

- `log_input` (bool, optional): If True, input is expected to be log(λ) rather than λ. Default: `True`
- `eps` (float, optional): Small constant for numerical stability. Default: `1e-8`
- `learn_variance` (bool, optional): Whether to use a learnable variance parameter. Default: `False`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `forward(y_pred, target, mask=None, weights=None)`: Computes the Poisson deviance loss

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

# With variance learning (useful for meta-learning or uncertainty quantification)
var_loss_fn = tr.losses.PoissonDevianceLoss(log_input=True, learn_variance=True)
var_loss = var_loss_fn(y_pred, target)
```

### PoissonLikelihoodRatioLoss

```python
class PoissonLikelihoodRatioLoss(RegressionLoss)
```

Poisson likelihood ratio test statistic for binned data, also known as Baker-Cousins loss. This is particularly useful for histogram fitting in high-energy physics and other scientific applications.

**Parameters:**

- `log_input` (bool, optional): If True, input is expected to be log(λ) rather than λ. Default: `True`
- `eps` (float, optional): Small constant for numerical stability. Default: `1e-8`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `forward(y_pred, target, mask=None, weights=None)`: Computes the likelihood ratio statistic

The likelihood ratio test statistic is defined as:

$$-2\ln(\lambda) = 2 \sum_i \left[ \lambda_i - n_i + n_i \ln\left(\frac{n_i}{\lambda_i}\right) \right]$$

Where:
- $n_i$ are the observed counts (target)
- $\lambda_i$ are the expected counts (predictions)
- For $n_i = 0$, the term $n_i \ln(n_i/\lambda_i) = 0$

This statistic follows a $\chi^2$ distribution asymptotically, making it useful for hypothesis testing and goodness-of-fit evaluation.

**Example:**

```python
import torch
import torchregress as tr

# Create loss for histogram fitting
loss_fn = tr.losses.PoissonLikelihoodRatioLoss(log_input=False)

# Expected bin counts from model
y_pred = torch.tensor([[10.0, 20.0, 30.0], [15.0, 25.0, 35.0]])
# Observed bin counts from data
target = torch.tensor([[12.0, 18.0, 32.0], [16.0, 23.0, 38.0]])

# Calculate test statistic
loss = loss_fn(y_pred, target)
```

### ZeroInflatedPoissonNLLLoss

```python
class ZeroInflatedPoissonNLLLoss(RegressionLoss)
```

Zero-Inflated Poisson Negative Log-Likelihood loss for count data with excess zeros.

**Parameters:**

- `log_input` (bool, optional): If True, input is expected to be log(λ) rather than λ. Default: `True`
- `eps` (float, optional): Small constant for numerical stability. Default: `1e-8`
- `learn_variance` (bool, optional): Whether to learn a global variance parameter. Default: `False`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Example:**

```python
import torch
import torchregress as tr

# Create loss
loss_fn = tr.losses.ZeroInflatedPoissonNLLLoss()

# Model predicts lambda values and zero-inflation logits
y_pred = torch.tensor([1.0, 2.0, 3.0])  # lambda values
pi_logits = torch.tensor([-1.0, 0.0, 1.0])  # zero-inflation logits
target = torch.tensor([0.0, 0.0, 3.0])  # counts

# Calculate loss
loss = loss_fn(y_pred, target, pi_logits)
```

### NegativeBinomialNLLLoss

```python
class NegativeBinomialNLLLoss(RegressionLoss)
```

Negative Binomial Negative Log-Likelihood loss for overdispersed count data (where variance > mean).

**Parameters:**

- `learn_theta` (bool, optional): Whether to learn the dispersion parameter θ. Default: `False`
- `eps` (float, optional): Small constant for numerical stability. Default: `1e-8`
- `min_theta` (float, optional): Minimum value for θ parameter. Default: `1e-6`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Example:**

```python
import torch
import torchregress as tr

# Create loss
loss_fn = tr.losses.NegativeBinomialNLLLoss(learn_theta=True)
y_pred = torch.tensor([1.0, 2.0, 3.0])  # mean values
target = torch.tensor([0.0, 3.0, 5.0])  # counts

# Calculate loss
loss = loss_fn(y_pred, target)
```

## When to Use Poisson Losses

1. **Count Data Regression**: When your target variable represents counts (number of events, occurrences, etc.)

2. **Low-Count Scenarios**: Especially important when counts are small (<10), where Gaussian approximations are inappropriate

3. **Histogram Fitting**: For fitting models to binned data, particularly in scientific applications

4. **Rate Estimation**: When modeling events per unit time, area, or volume

## Mathematical Insights

1. **Connection to MSE**: For large counts, the Poisson distribution can be approximated by a Gaussian with mean=variance=λ, making MSE a reasonable approximation in high-count scenarios.

2. **Variance Structure**: Unlike Gaussian models with constant variance, Poisson models have variance equal to the mean (λ), naturally capturing heteroscedasticity for count data.

3. **Zero Inflation**: Standard Poisson may underperform when data has excessive zeros beyond what's expected; consider Zero-Inflated Poisson models in these cases.

4. **Overdispersion**: When variance > mean in your data, consider NegativeBinomialNLLLoss as an alternative.
