# Statistical Interpretation

This page provides statistical interpretations of the loss functions and metrics in TorchRegression, helping users understand the theoretical foundations and make informed choices.

## Loss Functions as Negative Log-Likelihood

Many loss functions can be interpreted as negative log-likelihoods of different probability distributions. This interpretation helps in understanding the implicit assumptions being made when using a particular loss function.

### Mean Squared Error

The MSE loss corresponds to the negative log-likelihood of a Gaussian distribution with fixed variance:

$$\mathcal{L}_{\text{MSE}}(y, \hat{y}) \propto -\log p(y|\hat{y}) = -\log \mathcal{N}(y|\hat{y}, \sigma^2)$$

where $\sigma^2$ is constant. This means that minimizing MSE is equivalent to maximizing the likelihood under a homoscedastic Gaussian noise model.

### Mean Absolute Error

The MAE loss corresponds to the negative log-likelihood of a Laplace distribution:

$$\mathcal{L}_{\text{MAE}}(y, \hat{y}) \propto -\log p(y|\hat{y}) = -\log \text{Laplace}(y|\hat{y}, b)$$

where $b$ is the scale parameter of the Laplace distribution. This implies that MAE is more robust to outliers than MSE because the Laplace distribution has heavier tails than the Gaussian.

### Huber Loss

The Huber loss can be interpreted as a mixture of Gaussian and Laplace likelihoods, combining the benefits of both:

- For small residuals, it behaves like MSE (Gaussian)
- For large residuals, it behaves like MAE (Laplace)

This makes it robust to outliers while still being strongly convex near the minimum.

### Quantile Loss

The quantile loss at level $\tau$ corresponds to the negative log-likelihood of an asymmetric Laplace distribution:

$$\mathcal{L}_{\text{Quantile}}(y, \hat{q}_\tau, \tau) \propto -\log \text{ALD}(y|\hat{q}_\tau, \tau)$$

where ALD is the asymmetric Laplace distribution.

## Maximum Likelihood Estimation

When we minimize a loss function derived from the negative log-likelihood of a probability distribution, we are performing maximum likelihood estimation (MLE). 

For example, when using the GaussianNLLLoss, we are estimating the parameters of a conditional Gaussian distribution:

$$\hat{\theta} = \arg\max_\theta \sum_{i=1}^n \log p(y_i|x_i, \theta)$$

where $p(y_i|x_i, \theta) = \mathcal{N}(y_i|\mu_\theta(x_i), \sigma^2_\theta(x_i))$.

## Bayesian Interpretation

### Regularization as Prior

Regularization techniques can be interpreted as imposing prior distributions on the model parameters in a Bayesian framework:

- L2 regularization (weight decay) corresponds to a Gaussian prior on the parameters
- L1 regularization corresponds to a Laplace prior, promoting sparsity

### Prediction Intervals

In a Bayesian setting, prediction intervals account for both aleatoric and epistemic uncertainty:

$$p(y^*|x^*, \mathcal{D}) = \int p(y^*|x^*, \theta) p(\theta|\mathcal{D}) d\theta$$

where $p(y^*|x^*, \theta)$ captures aleatoric uncertainty and $p(\theta|\mathcal{D})$ captures epistemic uncertainty.

Methods like Monte Carlo dropout and deep ensembles approximate this integral through sampling.

## Distributional Regression

Traditional regression focuses on estimating the conditional mean $\mathbb{E}[Y|X=x]$, but distributional regression aims to estimate the entire conditional distribution $p(y|x)$.

### Parametric Approaches

- **Gaussian models**: Estimate mean and variance
- **Mixture Density Networks**: Estimate parameters of a Gaussian mixture
- **Normalizing Flows**: Estimate parameters of a flexible distribution

### Non-parametric Approaches

- **Quantile Regression**: Estimate specific quantiles of the distribution
- **Expectile Regression**: Estimate specific expectiles
- **Histogram Regression**: Estimate the entire discretized distribution

## Causal Interpretation

While most regression methods aim to model correlations, some techniques in TorchRegression can be used in causal inference scenarios:

### Error-in-Variables Models

EIV models account for measurement errors in predictors, which is important in causal inference where we need to estimate the true effect of a variable, not just its measured proxy.

$$Y = \beta X^* + \epsilon$$
$$X = X^* + \delta$$

where $X^*$ is the true value, $X$ is the measured value with error $\delta$, and $\epsilon$ is the model error.

### Instrumental Variables

Some robust regression techniques can be adapted for instrumental variable estimation, which is a common approach in causal inference with endogeneity issues.

## Metrics Interpretation

### R²

The coefficient of determination ($R^2$) represents the proportion of variance in the dependent variable that is explained by the model:

$$R^2 = 1 - \frac{\text{Unexplained Variance}}{\text{Total Variance}}$$

An $R^2$ close to 1 indicates that the model explains most of the variance in the target variable.

### Interval Coverage

PICP (Prediction Interval Coverage Probability) measures the calibration of uncertainty estimates:

- PICP ≈ target coverage rate (e.g., 0.95): Well-calibrated intervals
- PICP < target: Overconfident predictions
- PICP > target: Underconfident predictions

### NLL and CRPS

NLL (Negative Log-Likelihood) and CRPS (Continuous Ranked Probability Score) are proper scoring rules, meaning they are minimized in expectation when the predicted distribution matches the true data-generating process.

## When to Use Different Loss Functions

Based on the statistical interpretations above, here are guidelines for choosing loss functions:

1. **When data follows a Gaussian distribution**: MSE or GaussianNLL
2. **When data contains outliers**: Huber, LogCosh, or Cauchy
3. **When errors are asymmetric**: Quantile or Expectile loss
4. **When dealing with count data**: Poisson loss
5. **When the distribution is multi-modal**: MDN loss
6. **When modeling extreme values**: Tweedie loss or heavy-tailed distributions

Remember that the choice of loss function implicitly defines the conditional distribution your model is trying to approximate, so it should match the statistical properties of your data.
