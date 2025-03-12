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

## Implemented Methods in TorchRegression

### Parametric Methods

TorchRegression implements several parametric uncertainty estimation methods:

#### Gaussian Models

```python
# Direct variance estimation
model = UncertaintyModel()  # Model that outputs mean and log_variance
loss_fn = tr.losses.GaussianNLLLoss()
```

#### Mixture Density Networks

For multi-modal distributions:

```python
# MDN with 5 components
model = MDNModel(components=5)  # Model that outputs mixture parameters
loss_fn = tr.losses.MDNLoss(components=5)
```

### Non-parametric Methods

#### Quantile Regression

```python
# For 90% prediction intervals
model = QuantileModel()  # Model that outputs multiple quantiles
loss_fn = tr.losses.MultiQuantileLoss(quantiles=[0.05, 0.95])
```

#### Conformalized Quantile Regression

```python
# Train with quantile loss
model = QuantileModel()
loss_fn = tr.losses.QuantileLoss(quantile=0.5)

# Then apply conformal calibration
conformal_predictor = tr.calibration.ConformalPredictor(model, alpha=0.1)
conformal_predictor.calibrate(X_calib, y_calib)
lower, upper = conformal_predictor.predict(X_test)
```

### Ensemble Methods

#### Deep Ensembles

```python
# Create an ensemble of 5 models
models = [create_model() for _ in range(5)]
ensemble = tr.ensemble.DeepEnsemble(models)

# Get predictions with uncertainty
mean, variance = ensemble.predict(X_test)
```

#### Monte Carlo Dropout

```python
# Enable dropout at test time
model.train()  # Keep dropout active during inference

# Perform multiple forward passes
predictions = []
for _ in range(50):
    pred = model(X_test)
    predictions.append(pred)

# Calculate mean and variance
stacked_preds = torch.stack(predictions)
mean = torch.mean(stacked_preds, dim=0)
variance = torch.var(stacked_preds, dim=0)
```

## Evaluation of Uncertainty Estimates

To evaluate the quality of uncertainty estimates, TorchRegression provides several metrics:

### Calibration Metrics

- **Prediction Interval Coverage Probability (PICP)**: Measures what fraction of true values fall within the prediction intervals.
- **Expected Calibration Error (ECE)**: Measures the difference between confidence and accuracy.

### Sharpness Metrics

- **Mean Prediction Interval Width (MPIW)**: Measures how wide the prediction intervals are.
- **Negative Log-Likelihood (NLL)**: Evaluates both calibration and sharpness.

### Proper Scoring Rules

- **Continuous Ranked Probability Score (CRPS)**: A proper scoring rule for probabilistic forecasts.

```python
# Evaluate uncertainty estimates
picp = tr.metrics.picp(y_test, lower, upper)
mpiw = tr.metrics.mpiw(lower, upper)
nll = tr.metrics.gaussian_nll(mean, y_test, variance)
crps = tr.metrics.crps_gaussian(mean, y_test, torch.sqrt(variance))
```

## Visualization Tools

TorchRegression provides tools for visualizing uncertainty estimates:

```python
# Plot predictions with uncertainty
tr.viz.plot_predictions(X_test, y_test, mean, lower, upper)

# Plot calibration curve
tr.viz.plot_calibration_curve(mean, torch.sqrt(variance), y_test)

# Plot reliability diagram for prediction intervals
tr.viz.plot_reliability_diagram(lower, upper, y_test)
```
