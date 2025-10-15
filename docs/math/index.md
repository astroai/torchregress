# Mathematical Foundations

This page provides the mathematical foundations for the loss functions, metrics, and uncertainty estimation techniques implemented in TorchRegression.

## Notation

Throughout this documentation, we use the following notation:

- $y$: True target value (ground truth)
- $\hat{y}$: Predicted value (point prediction)
- $\hat{\sigma}$: Predicted standard deviation (uncertainty)
- $\hat{\mu}$: Predicted mean (for distributional predictions)
- $\hat{q}_\alpha$: Predicted quantile at level $\alpha$
- $\hat{e}_\tau$: Predicted expectile at level $\tau$
- $\mathcal{L}$: Loss function
- $n$: Number of samples
- $\mathbf{1}$: Indicator function (1 if condition is true, 0 otherwise)

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

### Quantile Loss

The quantile loss at level $\tau$ corresponds to the negative log-likelihood of an asymmetric Laplace distribution:

$$\mathcal{L}_{\text{Quantile}}(y, \hat{q}_\tau, \tau) \propto -\log \text{ALD}(y|\hat{q}_\tau, \tau)$$

where ALD is the asymmetric Laplace distribution.

### Poisson Loss

The Poisson loss corresponds to the negative log-likelihood of a Poisson distribution:

$$\mathcal{L}_{\text{Poisson}}(y, \hat{\lambda}) \propto -\log \text{Poisson}(y|\hat{\lambda})$$

This is appropriate for count data where the variance equals the mean.

### Tweedie Loss

The Tweedie loss corresponds to the negative log-likelihood of the Tweedie distribution, which includes many common distributions as special cases:

$$\mathcal{L}_{\text{Tweedie}}(y, \hat{\mu}, p) \propto -\log \text{Tweedie}(y|\hat{\mu}, p, \phi)$$

where $p$ is the power parameter and $\phi$ is the dispersion parameter.

## Uncertainty Estimation

### Types of Uncertainty

In regression modeling, we distinguish between two main types of uncertainty:

#### Aleatoric Uncertainty

Aleatoric uncertainty represents the inherent noise or randomness in the data. This type of uncertainty cannot be reduced by collecting more data and is sometimes called "data uncertainty" or "irreducible uncertainty."

- **Homoscedastic**: When the noise level is constant across all inputs
- **Heteroscedastic**: When the noise level varies with the input (more common in real-world data)

#### Epistemic Uncertainty

Epistemic uncertainty represents our model's ignorance or lack of knowledge. This type of uncertainty can be reduced by collecting more data or improving the model and is sometimes called "model uncertainty."

- **Parameter uncertainty**: Uncertainty about the true parameters of the model
- **Model structure uncertainty**: Uncertainty about the correct model form
- **Out-of-distribution uncertainty**: Uncertainty for inputs far from the training data

### Methods for Uncertainty Estimation

#### Direct Variance Estimation

The most common approach for estimating aleatoric uncertainty is to directly predict the variance along with the mean:

$$p(y|x) = \mathcal{N}(y|\mu_\theta(x), \sigma^2_\theta(x))$$

where $\mu_\theta(x)$ and $\sigma^2_\theta(x)$ are the predicted mean and variance from a model with parameters $\theta$.

The corresponding loss function is the negative log-likelihood:

$$\mathcal{L}_{\text{NLL}}(y, \mu_\theta, \sigma^2_\theta) = \frac{1}{2}\log(2\pi\sigma^2_\theta) + \frac{(y - \mu_\theta)^2}{2\sigma^2_\theta}$$

#### Quantile Regression

An alternative approach to uncertainty estimation is to predict specific quantiles of the conditional distribution:

$$P(Y \leq q_\tau(x)|X=x) = \tau$$

where $q_\tau(x)$ is the $\tau$-th quantile of the conditional distribution of $Y$ given $X=x$.

To get prediction intervals, we can predict two quantiles, such as the 0.05 and 0.95 quantiles for a 90% prediction interval.

#### Ensemble Methods

Ensemble methods combine multiple models to estimate epistemic uncertainty:

$$\mu_{\text{ensemble}}(x) = \frac{1}{M}\sum_{m=1}^M \mu_{\theta_m}(x)$$

$$\sigma^2_{\text{epistemic}}(x) = \frac{1}{M}\sum_{m=1}^M (\mu_{\theta_m}(x) - \mu_{\text{ensemble}}(x))^2$$

where $\mu_{\theta_m}(x)$ is the prediction from the $m$-th model in the ensemble.

If each model also predicts its own aleatoric uncertainty $\sigma^2_{\theta_m}(x)$, the total uncertainty is:

$$\sigma^2_{\text{total}}(x) = \underbrace{\frac{1}{M}\sum_{m=1}^M \sigma^2_{\theta_m}(x)}_{\text{aleatoric}} + \underbrace{\frac{1}{M}\sum_{m=1}^M (\mu_{\theta_m}(x) - \mu_{\text{ensemble}}(x))^2}_{\text{epistemic}}$$

#### Mixture Density Networks

Mixture Density Networks (MDNs) model the predictive distribution as a mixture of Gaussians:

$$p(y|x) = \sum_{k=1}^K \pi_k(x) \mathcal{N}(y|\mu_k(x), \sigma^2_k(x))$$

where $\pi_k(x)$ are the mixture weights, $\mu_k(x)$ are the means, and $\sigma^2_k(x)$ are the variances of the $K$ components.

This allows for modeling multi-modal and skewed distributions, capturing complex forms of aleatoric uncertainty.

#### Evidential Regression

Evidential Regression models the parameters of a Gaussian distribution as coming from a prior distribution, allowing for the separation of aleatoric and epistemic uncertainty:

$$p(y|x) = \int p(y|\mu, \sigma^2) p(\mu, \sigma^2|x) d\mu d\sigma^2$$

The model predicts the parameters of a Normal-Gamma prior over the mean and precision (inverse variance).
