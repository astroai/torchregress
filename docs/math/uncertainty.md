# Uncertainty Estimation

This page explains the theoretical foundations of uncertainty estimation in regression and how TorchRegression implements different uncertainty quantification methods.

## Types of Uncertainty

In regression modeling, we distinguish between two main types of uncertainty:

### Aleatoric Uncertainty

Aleatoric uncertainty represents the inherent noise or randomness in the data. This type of uncertainty cannot be reduced by collecting more data and is sometimes called "data uncertainty" or "irreducible uncertainty."

- **Homoscedastic**: When the noise level is constant across all inputs
- **Heteroscedastic**: When the noise level varies with the input (more common in real-world data)

### Epistemic Uncertainty

Epistemic uncertainty represents our model's ignorance or lack of knowledge. This type of uncertainty can be reduced by collecting more data or improving the model and is sometimes called "model uncertainty."

- **Parameter uncertainty**: Uncertainty about the true parameters of the model
- **Model structure uncertainty**: Uncertainty about the correct model form
- **Out-of-distribution uncertainty**: Uncertainty for inputs far from the training data

## Mathematical Formulation

### Direct Variance Estimation

The most common approach for estimating aleatoric uncertainty is to directly predict the variance along with the mean:

$$p(y|x) = \mathcal{N}(y|\mu_\theta(x), \sigma^2_\theta(x))$$

where $\mu_\theta(x)$ and $\sigma^2_\theta(x)$ are the predicted mean and variance from a model with parameters $\theta$.

The corresponding loss function is the negative log-likelihood:

$$\mathcal{L}_{\text{NLL}}(y, \mu_\theta, \sigma^2_\theta) = \frac{1}{2}\log(2\pi\sigma^2_\theta) + \frac{(y - \mu_\theta)^2}{2\sigma^2_\theta}$$

### Quantile Regression

An alternative approach to uncertainty estimation is to predict specific quantiles of the conditional distribution:

$$P(Y \leq q_\tau(x)|X=x) = \tau$$

where $q_\tau(x)$ is the $\tau$-th quantile of the conditional distribution of $Y$ given $X=x$.

To get prediction intervals, we can predict two quantiles, such as the 0.05 and 0.95 quantiles for a 90% prediction interval.

### Ensemble Methods

Ensemble methods combine multiple models to estimate epistemic uncertainty:

$$\mu_{\text{ensemble}}(x) = \frac{1}{M}\sum_{m=1}^M \mu_{\theta_m}(x)$$

$$\sigma^2_{\text{epistemic}}(x) = \frac{1}{M}\sum_{m=1}^M (\mu_{\theta_m}(x) - \mu_{\text{ensemble}}(x))^2$$

where $\mu_{\theta_m}(x)$ is the prediction from the $m$-th model in the ensemble.

If each model also predicts its own aleatoric uncertainty $\sigma^2_{\theta_m}(x)$, the total uncertainty is:

$$\sigma^2_{\text{total}}(x) = \underbrace{\frac{1}{M}\sum_{m=1}^M \sigma^2_{\theta_m}(x)}_{\text{aleatoric}} + \underbrace{\frac{1}{M}\sum_{m=1}^M (\mu_{\theta_m}(x) - \mu_{\text{ensemble}}(x))^2}_{\text{epistemic}}$$

### Monte Carlo Dropout

Monte Carlo Dropout uses dropout at inference time to approximate Bayesian inference:

$$p(y|x, \mathcal{D}) \approx \frac{1}{T} \sum_{t=1}^T p(y|x, \omega_t)$$

where $\omega_t \sim q_{\theta}(\omega)$ are dropout-sampled weights. The predictive mean and variance are:

$$\mathbb{E}[y|x, \mathcal{D}] \approx \frac{1}{T} \sum_{t=1}^T f_{\omega_t}(x)$$

$$\text{Var}[y|x, \mathcal{D}] \approx \underbrace{\frac{1}{T} \sum_{t=1}^T \sigma^2_{\omega_t}(x)}_{\text{aleatoric}} + \underbrace{\frac{1}{T} \sum_{t=1}^T (f_{\omega_t}(x) - \mathbb{E}[y|x, \mathcal{D}])^2}_{\text{epistemic}}$$

### Mixture Density Networks

Mixture Density Networks (MDNs) model the predictive distribution as a mixture of Gaussians:

$$p(y|x) = \sum_{k=1}^K \pi_k(x) \mathcal{N}(y|\mu_k(x), \sigma^2_k(x))$$

where $\pi_k(x)$ are the mixture weights, $\mu_k(x)$ are the means, and $\sigma^2_k(x)$ are the variances of the $K$ components.

This allows for modeling multi-modal and skewed distributions, capturing complex forms of aleatoric uncertainty.

### Normalizing Flows

Normalizing Flows provide non-parametric density estimation by transforming a simple base distribution into a complex target distribution through a series of invertible transformations:

$$p_X(x) = p_Z(f(x)) \left| \det\left(\frac{\partial f(x)}{\partial x}\right) \right|$$

where $p_Z$ is a base distribution (usually Gaussian), and $f$ is an invertible transformation.

### Evidential Regression

Evidential Regression models the parameters of a Gaussian distribution as coming from a prior distribution, allowing for the separation of aleatoric and epistemic uncertainty:

$$p(y|x) = \int p(y|\mu, \sigma^2) p(\mu, \sigma^2|x) d\mu d\sigma^2$$

The model predicts the parameters of a Normal-Gamma prior over the mean and precision (inverse variance).

## Calibration Methods

Uncertainty estimates often need to be calibrated to ensure that they match empirical frequencies:

### Temperature Scaling

Temperature scaling adjusts the confidence of predictions by dividing logits by a temperature parameter $T$:

$$q_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$

where higher values of $T$ produce softer (more uncertain) predictions.

### Isotonic Regression

Isotonic regression learns a non-parametric monotonic mapping from predicted probabilities to calibrated probabilities:

$$\hat{p}_{\text{calibrated}} = f_{\text{isotonic}}(\hat{p}_{\text{uncalibrated}})$$

where $f_{\text{isotonic}}$ is a piecewise constant function that preserves the rank order of predictions.

### Conformal Prediction

Conformal prediction provides prediction intervals with guaranteed coverage properties:

$$C_n(X_{n+1}) = \{\hat{y} \in \mathbb{R} : s(\hat{y}, X_{n+1}) \leq q_{\alpha}\}$$

where $s$ is a nonconformity score and $q_{\alpha}$ is a quantile of the calibration scores chosen to achieve $(1-\alpha)$ coverage.

## Evaluation Metrics

Several metrics can be used to evaluate the quality of uncertainty estimates:

### Negative Log-Likelihood (NLL)

$$\text{NLL} = -\frac{1}{n}\sum_{i=1}^{n} \log p(y_i|x_i)$$

### Continuous Ranked Probability Score (CRPS)

$$\text{CRPS}(F, y) = \int_{-\infty}^{\infty} (F(z) - \mathbbm{1}\{z \geq y\})^2 dz$$

where $F$ is the predicted cumulative distribution function.

### Prediction Interval Coverage Probability (PICP)

$$\text{PICP} = \frac{1}{n}\sum_{i=1}^{n} \mathbbm{1}\{y_i \in [\hat{y}_{\text{lower},i}, \hat{y}_{\text{upper},i}]\}$$

### Mean Prediction Interval Width (MPIW)

$$\text{MPIW} = \frac{1}{n}\sum_{i=1}^{n} (\hat{y}_{\text{upper},i} - \hat{y}_{\text{lower},i})$$

### Calibration Error

$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{n} |p_m - \hat{p}_m|$$

where $p_m$ is the observed frequency and $\hat{p}_m$ is the predicted probability in bin $m$.

## Implementation in TorchRegression

TorchRegression provides several methods for uncertainty estimation:

### Gaussian NLL Loss

```python
loss_fn = tr.losses.DiagonalGaussianNLL(n_features=1)
```

### Quantile Regression

```python
# For predicting upper and lower bounds of a 90% prediction interval
loss_fn = tr.losses.MultiQuantileLoss(quantiles=[0.05, 0.95])
```

### Mixture Density Networks

```python
loss_fn = tr.losses.MixtureDensityLoss(n_components=5, n_features=1)
```

### Normalizing Flows

```python
loss_fn = tr.losses.NormalizingFlowLoss(n_features=1, flow_type='nsf')
```

### Deep Ensembles

```python
ensemble = tr.ensemble.DeepEnsemble(base_model=MyModel, ensemble_size=5)
```

### Monte Carlo Dropout

```python
# Define a model with dropout
model = create_model_with_dropout(dropout_prob=0.1)

# Generate MC dropout samples at inference time
results = tr.ensemble.utils.generate_prediction_samples(model, x, n_samples=30)
```

## Uncertainty Decomposition: Which Methods Support It?

**IMPORTANT:** Not all uncertainty quantification methods support the decomposition of total uncertainty into epistemic and aleatoric components. Understanding which methods provide this capability is crucial for choosing the right approach for your application.

### Total Predictive Uncertainty

Total uncertainty = Epistemic + Aleatoric

This decomposition is only possible with certain methods. Below is a comprehensive guide:

### Methods Supporting Decomposition

| Method | Epistemic | Aleatoric | Notes |
|--------|-----------|-----------|-------|
| **Heteroscedastic Ensemble** | ✅ Variance of means | ✅ Mean of variances | Requires ensemble + variance prediction. Most reliable decomposition. |
| **MDN (Mixture Density Network)** | ✅ Mixture entropy | ✅ Component variances | From mixture weights and component variances. Good for multimodal distributions. |
| **Normalizing Flows (ensemble)** | ✅ Via ensemble | ✅ From distribution | Requires multiple flows or Bayesian approach. |
| **SWAG/MultiSWAG** | ✅ Weight posterior | ⚠️ Requires additional modeling | Epistemic via weight sampling. Aleatoric needs variance prediction. |
| **Monte Carlo Dropout** | ✅ Dropout variance | ⚠️ Requires variance prediction | Can approximate epistemic. Aleatoric needs explicit modeling. |
| **Deep Ensembles** | ✅ Variance of means | ❌ Not available | Unless combined with heteroscedastic outputs (variance prediction). |
| **Quantile Regression** | ❌ | ❌ | Provides intervals, not decomposition. Distribution-free. |
| **Conformal Prediction** | ❌ | ❌ | Distribution-free coverage guarantees, NOT uncertainty decomposition. |
| **Simple Point Losses** | ❌ | ❌ | No uncertainty quantification at all. |

### When to Use What

**Need epistemic/aleatoric decomposition?**
- ✅ Use: Heteroscedastic Ensemble or MDN
- ⚠️ Alternative: SWAG with variance prediction
- ❌ Don't use: Conformal prediction or quantile regression

**Distribution-free prediction intervals?**
- ✅ Use: Conformal Prediction or Quantile Regression
- Note: These provide coverage guarantees but cannot decompose uncertainty

**Calibrated coverage guarantees?**
- ✅ Use: Conformal Prediction
- Note: Provides valid coverage without distributional assumptions

**Multimodal uncertainty?**
- ✅ Use: MDN or Normalizing Flows
- Good for: Complex, multi-peaked distributions

**Simple uncertainty estimate without decomposition?**
- ✅ Use: Deep Ensemble or Quantile Regression
- Good for: When you only need total uncertainty

**Out-of-distribution detection?**
- ✅ Use: Deep Ensemble or SWAG (high epistemic uncertainty indicates OOD)
- ❌ Don't rely on: Conformal prediction (provides coverage, not OOD detection)

### Important Distinction: Conformal Prediction

**Conformal prediction is NOT an uncertainty decomposition method.** It provides:
- ✅ Distribution-free prediction intervals with guaranteed marginal coverage
- ✅ Finite-sample validity (coverage holds for finite samples, not just asymptotically)
- ❌ NO separation of epistemic vs aleatoric uncertainty
- ❌ NO measure of model confidence or knowledge

Use conformal prediction when you need:
1. Rigorous coverage guarantees without distributional assumptions
2. Calibrated prediction intervals
3. Robust performance across different data distributions

Do NOT use conformal prediction when you need:
1. To understand sources of uncertainty (epistemic vs aleatoric)
2. To detect out-of-distribution samples (use ensemble disagreement instead)
3. To know if the model is confident or uncertain about its predictions

### Practical Examples

#### Example 1: Medical Diagnosis (Need Decomposition)

```python
# Goal: Understand if uncertainty comes from inherent patient variability
# (aleatoric) or lack of data (epistemic)

# Use heteroscedastic ensemble
from torchregress.ensemble import DeepEnsemble
from torchregress.losses import HeteroscedasticGaussianLoss

model = DeepEnsemble(
    base_model=MyModel(output_dim=2),  # Predicts (mean, log_var)
    ensemble_size=5
)
loss_fn = HeteroscedasticGaussianLoss()

# At inference, get decomposed uncertainties
mean, epistemic_var, aleatoric_var = model.predict_with_uncertainty(x_test)

# High epistemic variance → need more data
# High aleatoric variance → inherent noise in the problem
```

#### Example 2: Safety-Critical Application (Need Coverage Guarantees)

```python
# Goal: Ensure prediction intervals have valid coverage for safety

# Use conformal prediction
from torchregress.losses import TorchCPConformalLoss

loss_fn = TorchCPConformalLoss(method='cqr', alpha=0.1)  # 90% coverage

# Train model
train_model(model, loss_fn, train_loader)

# Calibrate on hold-out set
loss_fn.calibrate(cal_predictions, cal_targets)

# Get intervals with guaranteed 90% coverage
lower, upper = loss_fn.predict(test_predictions)

# Note: These intervals have coverage guarantees but don't tell you
# whether uncertainty comes from model or data
```

#### Example 3: Active Learning (Need Epistemic Uncertainty)

```python
# Goal: Select samples where model is most uncertain (needs more data)

# Use deep ensemble for epistemic uncertainty
from torchregress.ensemble import DeepEnsemble

ensemble = DeepEnsemble(base_model=MyModel(), ensemble_size=10)

# Get epistemic uncertainty (disagreement between ensemble members)
predictions = ensemble.predict(unlabeled_data)
epistemic_uncertainty = predictions.var(dim=0)

# Select samples with highest epistemic uncertainty for labeling
samples_to_label = torch.topk(epistemic_uncertainty, k=100)
```

## Best Practices

1. **Begin with simpler methods**: Start with direct variance estimation before moving to more complex approaches.

2. **Consider the data distribution**: Choose an appropriate uncertainty estimation method based on the properties of your data.
   - For approximately Gaussian data: Direct variance estimation
   - For skewed or multi-modal data: MDNs or normalizing flows
   - For count data: Appropriate likelihood (e.g., Poisson, Negative Binomial)

3. **Choose based on your needs**:
   - Need uncertainty decomposition? → Heteroscedastic Ensemble or MDN
   - Need coverage guarantees? → Conformal Prediction
   - Need OOD detection? → Deep Ensemble or SWAG
   - Need multimodal distributions? → MDN or Normalizing Flows

4. **Evaluate calibration** of your uncertainty estimates using proper scoring rules and calibration metrics.

5. **Combine multiple methods** for more robust uncertainty estimation (e.g., ensembles of heteroscedastic models, or conformal prediction on top of probabilistic models).
