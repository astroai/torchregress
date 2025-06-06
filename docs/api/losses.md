# Losses API Reference

This page provides a complete API reference for all loss functions in TorchRegression.

## Base Classes

Abstract base classes that define the interface for all loss functions.

### BaseLoss

```python
class BaseLoss(torch.nn.Module)
```

Base class for all loss functions in TorchRegression.

**Parameters:**

- `reduction` (str): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `forward(y_pred, target, **kwargs)`: Abstract method for computing the loss
- `_reduce(loss, mask=None, weights=None)`: Applies reduction to the loss tensor

### RegressionLoss

```python
class RegressionLoss(BaseLoss)
```

Base class for standard regression loss functions that operate on point predictions.

**Parameters:**

- `reduction` (str): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `forward(y_pred, target, mask=None, weights=None)`: Abstract method for computing regression loss

### DistributionLoss

```python
class DistributionLoss(BaseLoss)
```

Base class for losses that model full probability distributions.

**Parameters:**

- `reduction` (str): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `_extract_distribution_parameters(y_pred)`: Extract distribution parameters from model outputs
- `_calculate_nll(y_pred, target, mask)`: Calculate negative log-likelihood

## Standard Losses

### MSELoss

```python
class MSELoss(RegressionLoss)
```

Mean Squared Error loss with support for masking and per-sample weights.

### L1Loss

```python
class L1Loss(RegressionLoss)
```

Mean Absolute Error (L1) loss with support for masking and per-sample weights.

### BCELoss

```python
class BCELoss(BaseLoss)
```

Binary Cross Entropy loss with support for masking, weighting, and positive class weighting.

## PyTorch Loss Wrappers

### TorchLossWrapper

```python
class TorchLossWrapper(BaseLoss)
```

Wrapper for PyTorch loss functions to add masking and weighting support.

The library provides wrapped versions of standard PyTorch losses:

- `MaskedMSELoss`
- `MaskedL1Loss`
- `MaskedHuberLoss`
- `MaskedCrossEntropyLoss`
- `MaskedBCEWithLogitsLoss`
- `MaskedKLDivLoss`
- `MaskedNLLLoss`
- `MaskedSmoothL1Loss`
- `MaskedPoissonNLLLoss`

## Gaussian Losses

### WeightedMSELoss

```python
class WeightedMSELoss(RegressionLoss)
```

An extension of the standard Mean Squared Error loss with support for sample/feature weighting and masking.

### DiagonalGaussianNLL

```python
class DiagonalGaussianNLL(DistributionLoss)
```

Negative Log-Likelihood loss for diagonal Gaussian distributions, which models each output dimension with an independent Gaussian distribution.

### GaussianNLLWithCovariance

```python
class GaussianNLLWithCovariance(DistributionLoss)
```

Negative Log-Likelihood loss for multivariate Gaussian with full covariance matrices.

## Robust Losses

### HuberLoss

```python
class HuberLoss(RegressionLoss)
```

The Huber loss combines the best properties of MSE and MAE: it behaves like MSE for small errors and like MAE for large errors.

### PseudoHuberLoss

```python
class PseudoHuberLoss(RegressionLoss)
```

The Pseudo-Huber loss is a smooth approximation of the Huber loss that ensures continuous derivatives everywhere.

### LogCoshLoss

```python
class LogCoshLoss(RegressionLoss)
```

The Log-Cosh loss computes the logarithm of the hyperbolic cosine of the prediction error.

### CauchyLoss

```python
class CauchyLoss(RegressionLoss)
```

The Cauchy loss is based on the Cauchy distribution and is highly robust to extreme outliers.

### TukeyLoss

```python
class TukeyLoss(RegressionLoss)
```

Tukey's biweight loss completely ignores errors beyond a certain threshold.

## Quantile & Expectile Losses

### QuantileLoss

```python
class QuantileLoss(RegressionLoss)
```

Basic quantile regression loss function for estimating a single quantile level.

### MultiQuantileLoss

```python
class MultiQuantileLoss(RegressionLoss)
```

Loss for simultaneously estimating multiple quantile levels, useful for generating prediction intervals.

### QuantileCrossover

```python
class QuantileCrossover(RegressionLoss)
```

Loss that encourages proper ordering of quantile predictions.

### ExpectileLoss

```python
class ExpectileLoss(RegressionLoss)
```

Basic expectile regression loss function for estimating a single expectile level.

### MultiExpectileLoss

```python
class MultiExpectileLoss(RegressionLoss)
```

Loss for simultaneously estimating multiple expectile levels.

### ExpectileCrossover

```python
class ExpectileCrossover(RegressionLoss)
```

Loss that encourages proper ordering of expectile predictions.

### AsymmetricLeastSquaresLoss

```python
class AsymmetricLeastSquaresLoss(ExpectileLoss)
```

Alias for ExpectileLoss, provided for compatibility with different naming conventions.

## Poisson & Tweedie Losses

### PoissonNLLLoss

```python
class PoissonNLLLoss(RegressionLoss)
```

Negative log-likelihood loss for Poisson regression, suitable for modeling count data.

### PoissonLikelihoodRatioLoss

```python
class PoissonLikelihoodRatioLoss(RegressionLoss)
```

Poisson likelihood ratio test statistic for binned data, also known as Baker-Cousins loss.

### TweedieLoss

```python
class TweedieLoss(RegressionLoss)
```

Loss function for Tweedie regression, which generalizes several common distributions.

### GammaLoss

```python
class GammaLoss(TweedieLoss)
```

Specialized loss for gamma regression (p=2), suitable for positive continuous response variables.

### InverseGaussianLoss

```python
class InverseGaussianLoss(TweedieLoss)
```

Loss function for inverse Gaussian regression (p=3), suitable for positive data with variance that increases with the cube of the mean.

### CompoundPoissonLoss

```python
class CompoundPoissonLoss(TweedieLoss)
```

Specialized loss for compound Poisson-Gamma regression (1<p<2).

## Poisson-Gaussian Mixture Losses

### PoissonGaussianMixtureLoss

```python
class PoissonGaussianMixtureLoss(RegressionLoss)
```

Negative log-likelihood loss for a mixture of Gaussian (readout noise) and Poisson (count) noise.

### EnhancedPoissonGaussianMixtureLoss

```python
class EnhancedPoissonGaussianMixtureLoss(RegressionLoss)
```

Advanced Poisson-Gaussian mixture loss with additional features for scientific applications.

## Advanced Uncertainty-Aware Losses

### MixtureDensityLoss

```python
class MixtureDensityLoss(DistributionLoss)
```

Negative Log-Likelihood loss for Mixture Density Networks, supporting diagonal or full covariance matrices.

### NormalizingFlowLoss

```python
class NormalizingFlowLoss(DistributionLoss)
```

Negative Log-Likelihood loss for normalizing flow models, supporting various flow architectures.

### RAGLoss

```python
class RAGLoss(DistributionLoss)
```

Residual Adaptive Gaussian (RAG) loss implements a heteroscedastic regression approach.

### DeepEvidentialLoss

```python
class DeepEvidentialLoss(DistributionLoss)
```

Implements evidential regression loss, which uses the theory of subjective logic to quantify uncertainty.

## Error-in-Variables Losses

### DemingLoss

```python
class DemingLoss(RegressionLoss)
```

Deming regression is a generalization of orthogonal least squares that accounts for errors in both the input features and the target variable.

### ODRLoss

```python
class ODRLoss(RegressionLoss)
```

Orthogonal Distance Regression (ODR) loss generalizes Deming regression to non-linear functions.

### WeightedEIVLoss

```python
class WeightedEIVLoss(RegressionLoss)
```

Weighted Error-in-Variables loss extends the EIV concept to allow for different weights.

## Ensemble Method Losses

### BootstrappedEnsembleLoss

```python
class BootstrappedEnsembleLoss(RegressionLoss)
```

Implements a bootstrapped ensemble approach that trains on different resampled subsets.

### DiversityRegularizedEnsembleLoss

```python
class DiversityRegularizedEnsembleLoss(RegressionLoss)
```

Trains an ensemble by combining a standard loss with a diversity-promoting regularization term.

### SnapshotEnsembleLoss

```python
class SnapshotEnsembleLoss(RegressionLoss)
```

Implements the Snapshot Ensemble approach with a cyclic learning rate schedule.

### AdversarialEnsembleLoss

```python
class AdversarialEnsembleLoss(RegressionLoss)
```

Enhances ensemble robustness by adding adversarial perturbations during training.

## Factory Functions

TorchRegression provides factory functions to create losses with specific configurations:

- `create_gaussian_nll`: Create a Gaussian NLL loss
- `create_robust_loss`: Create a robust loss
- `create_mdn_loss`: Create a Mixture Density Network loss
- `create_flow_loss`: Create a Normalizing Flow loss
- `create_eiv_loss`: Create an error-in-variables loss
- `create_ensemble_loss`: Create an ensemble loss
- `poisson_gaussian_mixture_loss`: Create a Poisson-Gaussian mixture loss
- `enhanced_poisson_gaussian_loss`: Create an enhanced Poisson-Gaussian mixture loss
