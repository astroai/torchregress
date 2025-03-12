# Mathematical Formulations

This page provides the mathematical foundations for the loss functions and metrics implemented in TorchRegression.

## Notation

Throughout this documentation, we use the following notation:

- $y$: True target value (ground truth)
- $\hat{y}$: Predicted value (point prediction)
- $\hat{\sigma}$: Predicted standard deviation (uncertainty)
- $\hat{\mu}$: Predicted mean (for distributional predictions)
- $\hat{q}_\alpha$: Predicted quantile at level $\alpha$
- $\mathcal{L}$: Loss function
- $n$: Number of samples
- $\mathbf{1}$: Indicator function (1 if condition is true, 0 otherwise)

## Basic Loss Functions

### Mean Squared Error (MSE)

The Mean Squared Error is defined as:

$$\mathcal{L}_{\text{MSE}}(y, \hat{y}) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

### Mean Absolute Error (MAE)

The Mean Absolute Error is defined as:

$$\mathcal{L}_{\text{MAE}}(y, \hat{y}) = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

## Robust Loss Functions

### Huber Loss

The Huber Loss combines MSE and MAE to be robust to outliers:

$$\mathcal{L}_{\text{Huber}}(y, \hat{y}, \delta) = \frac{1}{n}\sum_{i=1}^{n}
\begin{cases}
\frac{1}{2}(y_i - \hat{y}_i)^2, & \text{if } |y_i - \hat{y}_i| \leq \delta \\
\delta (|y_i - \hat{y}_i| - \frac{\delta}{2}), & \text{otherwise}
\end{cases}$$

where $\delta$ is a threshold parameter that determines the transition point.

### Cauchy Loss

The Cauchy Loss is highly robust to extreme outliers:

$$\mathcal{L}_{\text{Cauchy}}(y, \hat{y}, \gamma) = \frac{1}{n}\sum_{i=1}^{n} \gamma^2 \log\left(1 + \frac{(y_i - \hat{y}_i)^2}{\gamma^2}\right)$$

where $\gamma$ controls the scale of the loss.

### Log-Cosh Loss

The Log-Cosh Loss is a smooth approximation of the Huber loss:

$$\mathcal{L}_{\text{LogCosh}}(y, \hat{y}) = \frac{1}{n}\sum_{i=1}^{n} \log(\cosh(y_i - \hat{y}_i))$$

## Quantile and Expectile Losses

### Quantile Loss

For a quantile level $\tau \in (0, 1)$, the Quantile Loss is defined as:

$$\mathcal{L}_{\text{Quantile}}(y, \hat{q}_\tau, \tau) = \frac{1}{n}\sum_{i=1}^{n} \rho_\tau(y_i - \hat{q}_\tau(i))$$

where $\rho_\tau(u) = u \cdot (\tau - \mathbf{1}_{u < 0})$ is the check function and $\hat{q}_\tau(i)$ is the predicted $\tau$-quantile for the $i$-th sample.

### Expectile Loss

For an expectile level $\tau \in (0, 1)$, the Expectile Loss is defined as:

$$\mathcal{L}_{\text{Expectile}}(y, \hat{e}_\tau, \tau) = \frac{1}{n}\sum_{i=1}^{n} |\tau - \mathbf{1}_{y_i < \hat{e}_\tau(i)}| \cdot (y_i - \hat{e}_\tau(i))^2$$

where $\hat{e}_\tau(i)$ is the predicted $\tau$-expectile for the $i$-th sample.

## Distribution-Based Losses

### Gaussian Negative Log-Likelihood

For a Gaussian model with predicted mean $\hat{\mu}$ and variance $\hat{\sigma}^2$:

$$\mathcal{L}_{\text{GaussNLL}}(y, \hat{\mu}, \hat{\sigma}^2) = \frac{1}{n}\sum_{i=1}^{n} \left( \frac{1}{2}\log(2\pi\hat{\sigma}_i^2) + \frac{(y_i - \hat{\mu}_i)^2}{2\hat{\sigma}_i^2} \right)$$

### Mixture Density Network Loss

For a mixture of $K$ Gaussians with means $\hat{\mu}_{i,k}$, variances $\hat{\sigma}_{i,k}^2$, and mixture weights $\hat{\pi}_{i,k}$:

$$\mathcal{L}_{\text{MDN}}(y, \hat{\mu}, \hat{\sigma}^2, \hat{\pi}) = -\frac{1}{n}\sum_{i=1}^{n} \log\left( \sum_{k=1}^{K} \hat{\pi}_{i,k} \mathcal{N}(y_i | \hat{\mu}_{i,k}, \hat{\sigma}_{i,k}^2) \right)$$

where $\mathcal{N}(y | \mu, \sigma^2)$ is the probability density function of the normal distribution.

### Poisson Loss

For count data with predicted rates $\hat{\lambda}$:

$$\mathcal{L}_{\text{Poisson}}(y, \hat{\lambda}) = \frac{1}{n}\sum_{i=1}^{n} (\hat{\lambda}_i - y_i \log(\hat{\lambda}_i) + \log(y_i!))$$

where we typically drop the constant term $\log(y_i!)$ during optimization.

## Error-in-Variables Losses

### Deming Regression Loss

For errors in both $X$ and $y$ with known error ratio $\delta$:

$$\mathcal{L}_{\text{Deming}}(y, X, \hat{\beta}, \delta) = \frac{1}{n}\sum_{i=1}^{n} \frac{(y_i - X_i\hat{\beta})^2}{1 + \delta\hat{\beta}^2}$$

where $\hat{\beta}$ is the regression coefficient.

## Metrics

### Root Mean Squared Error (RMSE)

$$\text{RMSE}(y, \hat{y}) = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

### Coefficient of Determination (R²)

$$\text{R}^2(y, \hat{y}) = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

where $\bar{y} = \frac{1}{n}\sum_{i=1}^{n} y_i$ is the mean of the true values.

### Prediction Interval Coverage Probability (PICP)

For a prediction interval $[\hat{y}_{\text{lower}}, \hat{y}_{\text{upper}}]$:

$$\text{PICP}(y, \hat{y}_{\text{lower}}, \hat{y}_{\text{upper}}) = \frac{1}{n}\sum_{i=1}^{n} \mathbf{1}_{y_i \in [\hat{y}_{\text{lower},i}, \hat{y}_{\text{upper},i}]}$$

### Continuous Ranked Probability Score (CRPS)

For a predicted cumulative distribution function $F$:

$$\text{CRPS}(F, y) = \int_{-\infty}^{\infty} (F(z) - \mathbf{1}_{z \geq y})^2 dz$$

For a Gaussian distribution with mean $\hat{\mu}$ and standard deviation $\hat{\sigma}$, this has the analytical form:

$$\text{CRPS}_{\text{Gaussian}}(y, \hat{\mu}, \hat{\sigma}) = \hat{\sigma} \left[ \frac{1}{\sqrt{\pi}} - 2\phi\left(\frac{y-\hat{\mu}}{\hat{\sigma}}\right) - \frac{y-\hat{\mu}}{\hat{\sigma}}\left(2\Phi\left(\frac{y-\hat{\mu}}{\hat{\sigma}}\right) - 1\right) \right]$$

where $\phi$ and $\Phi$ are the PDF and CDF of the standard normal distribution, respectively.
