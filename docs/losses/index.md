# Loss Functions

torchregress provides a comprehensive collection of loss functions for regression tasks, ranging from basic losses to advanced uncertainty-aware losses.

Task-first entry point (recommended): use the
[Task-First Method Selection Matrix](../guides/method_selection_matrix.md) when you are
choosing between loss families for outliers, multimodal targets, calibration, OOD
robustness, imbalance, or noisy features.

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
- **Barron Loss**: A generalization of Cauchy and L2 losses.

[Learn more about robust losses →](robust.md)

### Quantile & Expectile Losses

Distribution-free alternatives that target specific parts of the conditional distribution:

- **Quantile Loss**: For predicting conditional quantiles
- **Expectile Loss**: For predicting conditional expectiles
- **Multiple Quantile Loss**: For simultaneous prediction of multiple quantiles

[Learn more about quantile & expectile losses →](quantile_expectile.md)

### Ordinal Losses

Losses for ordered discrete targets (for example ratings/bins where class distance matters):

- **OrdinalCrossEntropyLoss**: class-logit baseline for ordinal labels
- **CumulativeLinkLoss**: cumulative-threshold objective with `K-1` logits
- **CORALLoss**: CORAL-style cumulative objective

[Learn more about ordinal losses →](ordinal.md)

### Poisson & Tweedie Losses

Specialized losses for count data and data with non-normal distributions:

- **Poisson Loss**: For count data regression
- **Tweedie Loss**: For data with various mean-variance relationships (e.g., insurance claims)
- **Poisson-Gaussian Mixture Loss**: For modeling mixed noise processes

[Learn more about Poisson & Tweedie losses →](poisson_tweedie.md)

### Imbalanced & Noisy Data Losses

Losses designed to handle specific data challenges:

- **Imbalanced Loss**: For imbalanced regression tasks where some target values are more frequent than others.
- **Noisy Loss**: For training with noisy labels.

[Learn more about imbalanced & noisy data losses →](imbalanced.md)

### Advanced Uncertainty-Aware Losses

Sophisticated losses for full predictive distributions:

- **Mixture Density Network (MDN) Loss**: For multi-modal distribution prediction
- **Normalizing Flows (NF) Loss**: For flexible distribution prediction
- **Evidential Loss**: For evidential regression and uncertainty quantification.
- **DeepAR Loss**: For probabilistic forecasting with autoregressive models.
- **Squared Quantile Score (SQS) Loss**: For quantile regression with a scoring rule.

[Learn more about advanced losses →](advanced.md)

### Error-in-Variables Losses

Losses that account for uncertainty in both inputs and outputs:

- **Deming Regression Loss**: Classical error-in-variables loss
- **Orthogonal Distance Regression Loss**: Minimizes perpendicular distances

[Learn more about error-in-variables losses →](eiv.md)

### Censored Regression Losses

Losses for right/left and interval-censored outcomes:

- **CensoredGaussianNLLLoss**
- **CensoredQuantileLoss**
- **AFTLoss**

[Learn more about censored regression losses →](censored.md)

### Conformal Prediction Losses

Methods that provide rigorous prediction intervals with statistical guarantees:

- **ConformalLoss**: Unified wrapper for various `torchcp` methods (Split, CQR, ACI)

[Learn more about conformal prediction losses →](conformal.md)

### Transform Losses

Apply transformations to the target variable before computing the loss:

- **TransformedLoss**: A wrapper that applies a transformation to the target variable.

[Learn more about transform losses →](transforms.md)

## Loss Selection Guide

| If you need... | Consider using... |
|----------------|-------------------|
| Simple regression | `MSELoss` or `L1Loss` |
| Robustness to outliers | `HuberLoss` or `BarronLoss` |
| Prediction intervals | `QuantileLoss` (with multiple quantiles) |
| Guaranteed coverage intervals | `ConformalLoss` |
| Full uncertainty | `GaussianNLLLoss`, `MDNLoss`, or `EvidentialLoss` |
| Count data | `PoissonNLLLoss` |
| Continuous data with zeros | `TweedieLoss(p=1.5)` |
| Imbalanced data | `ImbalancedLoss` |
| Noisy labels | `NoisyLoss` |
| Time series forecasting | `DeepARLoss` |
| Accounting for input errors | `DemingLoss` or `ODRLoss` |

For detailed guidance on loss selection, see the [practical usage guide](../usage/practical_usage.md).

For end-to-end comparisons and runtime tradeoffs, see the [examples index](../examples/index.md).
