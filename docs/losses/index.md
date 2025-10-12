# Loss Functions

TorchRegression provides a comprehensive collection of loss functions for regression tasks, ranging from basic losses to advanced uncertainty-aware losses.

## Loss Categories

The library organizes loss functions into several categories:

### Standard Losses

The building blocks of regression, including basic losses and wrapped PyTorch losses:

- **Mean Squared Error (MSE)**: Standard quadratic loss function
- **Mean Absolute Error (MAE)**: L1 loss function
- **Binary Cross Entropy (BCE)**: For regression with binary targets
- **Wrapped PyTorch Losses**: Adding mask and weight support to PyTorch's losses

See [base losses documentation](base.md) for the foundation classes and wrapped PyTorch losses.

### Gaussian Losses

These are the most common regression losses, based on Gaussian distribution assumptions:

- **Mean Squared Error (MSE)**: Standard L2 loss function
- **Mean Absolute Error (MAE)**: L1 loss function
- **Gaussian Negative Log-Likelihood (NLL)**: Jointly optimizes prediction mean and variance

[Learn more about Gaussian losses →](gaussian.md)

### Robust Losses

Loss functions that are less sensitive to outliers:

- **Huber Loss**: Combines MSE and MAE benefits
- **Pseudo-Huber Loss**: Smooth approximation of Huber loss
- **Log-cosh Loss**: Smooth approximation that's twice differentiable
- **Cauchy Loss**: Highly robust to extreme outliers

[Learn more about robust losses →](robust.md)

### Quantile & Expectile Losses

Distribution-free alternatives that target specific parts of the conditional distribution:

- **Quantile Loss**: For predicting conditional quantiles
- **Expectile Loss**: For predicting conditional expectiles
- **Multiple Quantile Loss**: For simultaneous prediction of multiple quantiles

[Learn more about quantile & expectile losses →](quantile_expectile.md)

### Poisson Losses

Specialized losses for count data:

- **Poisson Loss**: For count data regression
- **Poisson Likelihood Ratio**: For histogram fitting and goodness-of-fit tests

[Learn more about Poisson losses →](poisson.md)

### Poisson-Gaussian Mixture Losses

Losses that combine Poisson and Gaussian components for modeling mixed noise processes:

- **PoissonGaussianMixtureLoss**: Basic mixture implementation
- **EnhancedPoissonGaussianMixtureLoss**: With gain, offset and noise components

[Learn more about Poisson-Gaussian mixture losses →](poisson_gaussian.md)

### Tweedie Losses

Flexible family of losses for data with various mean-variance relationships:

- **TweedieLoss**: General implementation with configurable power parameter
- **GammaLoss**: For positive continuous data (p=2)
- **CompoundPoissonLoss**: For continuous data with exact zeros (1<p<2)

[Learn more about Tweedie losses →](tweedie.md)

### Conformal Prediction Losses

Methods that provide rigorous prediction intervals with statistical guarantees:

- **ConformalLoss**: Unified wrapper for various `torchcp` methods (Split, CQR, ACI)

[Learn more about conformal prediction losses →](conformal.md)

### Advanced Uncertainty-Aware Losses

Sophisticated losses for full predictive distributions:

- **Mixture Density Network (MDN) Loss**: For multi-modal distribution prediction
- **Normalizing Flows (NF) Loss**: For flexible distribution prediction
- **Residual Adaptive Gaussian (RAG) Loss**: For heteroscedastic uncertainty estimation

[Learn more about advanced losses →](advanced.md)

### Error-in-Variables Losses

Losses that account for uncertainty in both inputs and outputs:

- **Deming Regression Loss**: Classical error-in-variables loss
- **Orthogonal Distance Regression Loss**: Minimizes perpendicular distances

[Learn more about error-in-variables losses →](eiv.md)

### Ensemble Methods

Approaches to combine multiple models:

- **Bootstrapped Ensemble Loss**: Training with resampled data
- **Adversarial Training Loss**: Ensembles with adversarial perturbations
- **Snapshot Ensemble Loss**: Time-based ensemble creation

[Learn more about ensemble methods →](ensemble.md)

## Loss Selection Guide

| If you need... | Consider using... |
|----------------|-------------------|
| Simple regression | `MSELoss` or `L1Loss` |
| Robustness to outliers | `HuberLoss` or `CauchyLoss` |
| Prediction intervals | `QuantileLoss` (with multiple quantiles) |
| Guaranteed coverage intervals | `ConformalLoss` |
| Full uncertainty | `GaussianNLLLoss` or `MDNLoss` |
| Count data | `PoissonNLLLoss` |
| Continuous data with zeros | `TweedieLoss(p=1.5)` |
| Positive continuous data | `GammaLoss` |
| Multi-modal predictions | `MDNLoss` or `NormalizingFlowLoss` |
| Accounting for input errors | `DemingLoss` or `ODRLoss` |

For detailed guidance on loss selection, see the [practical usage guide](../usage/practical_usage.md).
