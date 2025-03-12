# Mathematical Formulations

This page provides the mathematical foundations for the loss functions and metrics implemented in TorchRegression.

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

### Pseudo-Huber Loss

A smooth approximation of the Huber loss:

$$\mathcal{L}_{\text{PseudoHuber}}(y, \hat{y}, \delta) = \frac{1}{n}\sum_{i=1}^{n} \delta^2 \left( \sqrt{1 + \left(\frac{y_i - \hat{y}_i}{\delta}\right)^2} - 1 \right)$$

### Log-Cosh Loss

The Log-Cosh Loss approximates Huber loss but is twice differentiable:

$$\mathcal{L}_{\text{LogCosh}}(y, \hat{y}) = \frac{1}{n}\sum_{i=1}^{n} \log(\cosh(y_i - \hat{y}_i))$$

### Cauchy Loss

The Cauchy Loss is highly robust to extreme outliers:

$$\mathcal{L}_{\text{Cauchy}}(y, \hat{y}, \gamma) = \frac{1}{n}\sum_{i=1}^{n} \gamma^2 \log\left(1 + \frac{(y_i - \hat{y}_i)^2}{\gamma^2}\right)$$

where $\gamma$ controls the scale of the loss.

### Tukey Biweight Loss

Tukey's biweight loss completely ignores errors beyond a certain threshold:

$$\mathcal{L}_{\text{Tukey}}(y, \hat{y}, c) = \begin{cases}
\frac{c^2}{6} \left[1 - \left(1 - \left(\frac{y - \hat{y}}{c}\right)^2\right)^3\right], & \text{if } |y - \hat{y}| \leq c \\
\frac{c^2}{6}, & \text{otherwise}
\end{cases}$$

### Charbonnier Loss

A smooth alternative to L1 loss:

$$\mathcal{L}_{\text{Charbonnier}}(y, \hat{y}, \epsilon) = \frac{1}{n}\sum_{i=1}^{n} \sqrt{(y_i - \hat{y}_i)^2 + \epsilon^2}$$

### Lq Loss

A generalization of L1 and L2 losses:

$$\mathcal{L}_{\text{Lq}}(y, \hat{y}, q) = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|^q$$

### Fair Loss

Grows less than linearly with errors:

$$\mathcal{L}_{\text{Fair}}(y, \hat{y}, c) = \frac{1}{n}\sum_{i=1}^{n} c^2 \left(\frac{|y_i - \hat{y}_i|}{c} - \log\left(1 + \frac{|y_i - \hat{y}_i|}{c}\right)\right)$$

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

### Normalizing Flow Loss

For a normalizing flow model with transformation $f$ and base distribution $p_Z$:

$$\mathcal{L}_{\text{NF}}(y, \theta) = -\frac{1}{n}\sum_{i=1}^{n} \left[ \log p_Z(f_\theta(y_i)) + \log \left| \det\left(\frac{\partial f_\theta(y_i)}{\partial y_i}\right) \right| \right]$$

where $\theta$ represents the parameters of the flow.

### Poisson Loss

For count data with predicted rates $\hat{\lambda}$:

$$\mathcal{L}_{\text{Poisson}}(y, \hat{\lambda}) = \frac{1}{n}\sum_{i=1}^{n} (\hat{\lambda}_i - y_i \log(\hat{\lambda}_i) + \log(y_i!))$$

where the term $\log(y_i!)$ is typically dropped during optimization.

### Poisson Likelihood Ratio Loss

For binned count data:

$$\mathcal{L}_{\text{PoissonLR}}(y, \hat{\lambda}) = \frac{1}{n}\sum_{i=1}^{n} \left[ \hat{\lambda}_i - y_i + y_i \log\left(\frac{y_i}{\hat{\lambda}_i}\right) \right]$$

when $y_i > 0, and $\hat{\lambda}_i$ otherwise.

### Tweedie Loss

For the Tweedie family with power parameter $p$:

$$\mathcal{L}_{\text{Tweedie}}(y, \hat{\mu}, p, \phi) = \frac{1}{n\phi}\sum_{i=1}^{n} \begin{cases}
\frac{1}{2}(y_i - \hat{\mu}_i)^2, & \text{if } p = 0 \\
y_i\log\left(\frac{y_i}{\hat{\mu}_i}\right) - y_i + \hat{\mu}_i, & \text{if } p = 1 \\
-\log\left(\frac{y_i}{\hat{\mu}_i}\right) + \frac{y_i}{\hat{\mu}_i} - 1, & \text{if } p = 2 \\
\frac{1}{p-1}\left(\frac{y_i^{2-p}}{2-p} - \frac{y_i\hat{\mu}_i^{1-p}}{1-p}\right) + \frac{\hat{\mu}_i^{2-p}}{2-p}, & \text{if } p \neq 0,1,2
\end{cases}$$

## Error-in-Variables Losses

### Orthogonal Distance Regression Loss

For errors in both $X$ and $y$ with covariances $\Sigma_X$ and $\Sigma_y$:

$$\mathcal{L}_{\text{ODR}}(X, y, \hat{X}, f, \Sigma_X, \Sigma_y) = \frac{1}{n}\sum_{i=1}^{n} \left[ (X_i - \hat{X}_i)^\top \Sigma_X^{-1} (X_i - \hat{X}_i) + (y_i - f(\hat{X}_i))^\top \Sigma_y^{-1} (y_i - f(\hat{X}_i)) \right]$$

where $\hat{X}$ are the latent true input values (optimized during loss computation) and $f$ is the regression model.

## Regression-as-Classification Losses

### Standard Classification Regression Loss

Cross-entropy loss between binned target distribution and predicted distribution:

$$\mathcal{L}_{\text{RAG-CE}}(y, \hat{p}) = -\frac{1}{n}\sum_{i=1}^{n} \sum_{b=1}^{B} p(b|y_i) \log(\hat{p}_{i,b})$$

where $p(b|y_i)$ is the target distribution over bins and $\hat{p}_{i,b}$ is the predicted probability for bin $b$.

### Wasserstein RAG Loss

Wasserstein distance between binned target distribution and predicted distribution:

$$\mathcal{L}_{\text{RAG-W}}(y, \hat{p}, D) = \frac{1}{n}\sum_{i=1}^{n} \sum_{b=1}^{B} \sum_{c=1}^{B} D_{bc} |CDF_p(b) - CDF_{\hat{p}}(c)|$$

where $D_{bc}$ is the distance matrix between bins and $CDF$ represents cumulative distribution functions.

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

### Interval Score (Winkler Score)

For a prediction interval $[\hat{y}_{\text{lower}}, \hat{y}_{\text{upper}}]$ with confidence level $\alpha$:

$$\text{IS}_{\alpha}(y, \hat{y}_{\text{lower}}, \hat{y}_{\text{upper}}) = (\hat{y}_{\text{upper}} - \hat{y}_{\text{lower}}) + \frac{2}{\alpha}(\hat{y}_{\text{lower}} - y) \mathbf{1}_{y < \hat{y}_{\text{lower}}} + \frac{2}{\alpha}(y - \hat{y}_{\text{upper}}) \mathbf{1}_{y > \hat{y}_{\text{upper}}}$$

This score rewards narrow intervals and penalizes when observations fall outside the interval.

### Energy Score

For multivariate predictions, the Energy Score is a generalization of CRPS:

$$\text{ES}(y, S) = \frac{1}{|S|}\sum_{i=1}^{|S|} \|y - s_i\| - \frac{1}{2|S|^2}\sum_{i=1}^{|S|}\sum_{j=1}^{|S|} \|s_i - s_j\|$$

where $S$ is a set of samples from the predictive distribution and $\|\cdot\|$ represents the Euclidean norm.

### Expected Calibration Error (ECE)

For predicted quantiles $\hat{q}_\alpha$ at level $\alpha$:

$$\text{ECE} = \frac{1}{K}\sum_{k=1}^{K} \left| \hat{\alpha}_k - \alpha_k \right|$$

where $\hat{\alpha}_k$ is the observed frequency of target values below the predicted $\alpha_k$-quantile.

### Probability Integral Transform (PIT)

For a predicted cumulative distribution function $F$ and observation $y$:

$$\text{PIT} = F(y)$$

For well-calibrated forecasts, PIT values should follow a uniform distribution on $[0, 1]$.
