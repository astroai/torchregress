# Torch Regression Losses

A comprehensive collection of PyTorch loss functions for regression tasks with uncertainty modeling, designed for numerical stability and efficiency.

## Installation

### Using pip

```bash
pip install torchregression
```

### Using uv (recommended)

```bash
uv pip install torchregression
```

### Development installation

```bash
# Clone the repository
git clone https://github.com/sfabbro/torchregression.git
cd torchregression

# Using uv (recommended)
uv pip install -e ".[dev,test]"

# Using pip
pip install -e ".[dev,test]"
```

## Features

TorchRegression provides a comprehensive collection of loss functions for regression tasks:

- **Basic loss functions**: MSE, MAE, Gaussian NLL
- **Robust loss functions**: Huber, Pseudo-Huber, Log-cosh, Cauchy
- **Distributional losses**: Quantile, Expectile, Poisson, Tweedie
- **Uncertainty estimation**: RAG, MDN, Normalizing Flows
- **Error-in-variables models**: Total Least Squares, Deming Regression
- **Ensemble methods**: Various ensembling techniques

## Loss Functions

### Gaussian Losses
- **WeightedMSELoss**: Weighted mean squared error with fixed diagonal covariance
- **DiagonalGaussianNLL**: Gaussian NLL with learned diagonal covariance
- **GaussianNLLWithCovariance**: Gaussian NLL with fixed full covariance matrices
- **AdjustedGaussianNLL**: Gaussian NLL with learned diagonal adjustments to covariance
- **HeteroscedasticGaussianNLL**: Combined loss with covariance and diagonal components
- **LearnedGaussianNLL**: Gaussian NLL with fully learned covariance matrix
- **LowRankGaussianNLL**: Gaussian NLL with low-rank plus diagonal covariance
- **GaussianPoissonMixtureNLL**: Mixed Gaussian-Poisson noise model

### Poisson Losses
- **PoissonNLL**: Standard Poisson negative log-likelihood
- **ModifiedPoissonNLL**: Baker-Cousins modification of Poisson NLL
- **ZeroInflatedPoissonNLL**: For data with excess zeros
- **NegativeBinomialNLL**: For overdispersed count data

### Robust Losses
- **L1Loss**: Mean absolute error (L1 norm)
- **HuberLoss**: Combination of L1 and L2 losses
- **PseudoHuberLoss**: Smooth approximation to Huber loss
- **LogCoshLoss**: Log of hyperbolic cosine of error
- **CharbonnierLoss**: Square root of squared error plus epsilon
- **LqLoss**: Generalization of L1/L2 losses with arbitrary power

### Quantile Losses
- **QuantileLoss/PinballLoss**: For quantile regression
- **MultiQuantileLoss**: For predicting multiple quantiles simultaneously
- **LogLinQuantileLoss**: Log-linearized quantile loss

### Other Losses
- **ExpectileLoss**: Asymmetric generalization of squared error
- **HistogramLoss**: For direct distribution learning with histograms
- **TweedieLoss**: For modeling data with Tweedie distribution

### Utilities
- **iteratively_reweighted_least_squares**: IRLS algorithm for robust regression

## Mathematical Formulations

### `WeightedMSELoss`

$$\mathcal{L} = \frac{1}{\sum_{i=1}^{B} \sum_{j=1}^{N} m_{ij}} \sum_{i=1}^{B} \sum_{j=1}^{N} w_{ij} (x_{ij} - x'_{ij})^2$$

where $B$ is batch size, $N$ is feature count, $w_{ij}$ are weights (inverse variances), and $m_{ij}$ is the mask.

### `DiagonalGaussianNLL`

$$\mathcal{L} = \frac{1}{\sum_{i=1}^{B} \sum_{j=1}^{N} m_{ij}} \sum_{i=1}^{B} \left[ \sum_{j=1}^{N} \frac{(x_{ij} - x'_{ij})^2}{v_{j}} + \sum_{j=1}^{N} \log(v_{j}) + N \log(2\pi) \right]$$

where $v_{j}$ is the learned variance for feature $j$.

### `GaussianNLLWithCovariance`

$$\mathcal{L} = \frac{1}{\sum_{i=1}^{B} \sum_{j=1}^{N} m_{ij}} \sum_{i=1}^{B} \left[ \frac{1}{2} (\mathbf{x}_i - \mathbf{x}'_i)^T \mathbf{\Sigma}_i^{-1} (\mathbf{x}_i - \mathbf{x}'_i) + \frac{1}{2} \log |\mathbf{\Sigma}_i| + \frac{N}{2} \log(2\pi) \right]$$

where $\mathbf{\Sigma}_i$ is the fixed covariance matrix for sample $i$.

### `AdjustedGaussianNLL`

$$\mathcal{L} = \frac{1}{\sum_{i=1}^{B} \sum_{j=1}^{N} m_{ij}} \sum_{i=1}^{B} \left[ \frac{1}{2} (\mathbf{x}_i - \mathbf{x}'_i)^T \mathbf{M}_i (\mathbf{x}_i - \mathbf{x}'_i) + \frac{1}{2} \log |\mathbf{\Sigma}'_i| + \frac{N}{2} \log(2\pi) \right]$$

where $\mathbf{\Sigma}'_i = \mathbf{\Sigma}_i + \text{diag}(\mathbf{v}) + \epsilon \mathbf{I}$. Here, $\mathbf{v}$ is the vector of learned additional variances.

### `HeteroscedasticGaussianNLL`

$$\mathcal{L}_{total} = \mathcal{L}_{cov} + \mathcal{L}_{diag}$$

where $\mathcal{L}_{cov}$ is from `AdjustedGaussianNLL` and $\mathcal{L}_{diag}$ is from either `WeightedMSELoss` or `DiagonalGaussianNLL`.

## Usage Example

```python
import torch
from torch_regression_losses import (
    DiagonalGaussianNLL,
    HuberLoss,
    QuantileLoss
)

# Gaussian NLL with learned variance
loss_fn = DiagonalGaussianNLL(n_features=10)
y_true = torch.randn(32, 10)
y_pred = torch.randn(32, 10)
loss = loss_fn(y_true, y_pred)

# Robust regression with Huber loss
huber = HuberLoss(delta=1.0)
# Create an optional mask to handle missing data
mask = torch.ones_like(y_true).bool()
mask[:, 5:8] = False  # Mask out some features
loss = huber(y_true, y_pred, mask=mask)

# Quantile regression (predicting 90th percentile)
q_loss = QuantileLoss(tau=0.9)
loss = q_loss(y_true, y_pred)
```

## License

MIT
