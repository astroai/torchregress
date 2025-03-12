# Advanced Uncertainty-Aware Losses

This page documents advanced loss functions in TorchRegression designed for modeling complex predictive distributions and sophisticated uncertainty estimation.

## Mixture Density Networks (MDN)

Mixture Density Networks model the conditional output distribution as a mixture of parametric distributions (typically Gaussians).

[Learn more about MDN losses →](mdn.md)

## Normalizing Flows

Normalizing Flows transform a simple base distribution into a complex target distribution through a series of invertible transformations.

[Learn more about Normalizing Flow losses →](nflows.md)

## Residual Adaptive Gaussian Losses

### RAGLoss

```python
class RAGLoss(DistributionLoss)
```

Residual Adaptive Gaussian (RAG) loss implements a heteroscedastic regression approach that adapts to different noise characteristics in the data.

**Parameters:**

- `alpha` (float, optional): Heavy-tail parameter controlling the adaptive distribution. Default: `0.1`
- `beta` (float, optional): Regularization parameter for variance prediction. Default: `1.0`
- `min_sigma` (float, optional): Minimum allowed standard deviation. Default: `1e-5`
- `eps` (float, optional): Small constant for numerical stability. Default: `1e-8`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Mathematical Formulation:**

RAG loss uses a robust error formulation by having the model predict both the mean and the variance, then adapting the loss based on the residuals:

$$\mathcal{L}_{\text{RAG}}(y, \hat{y}, \hat{\sigma}) = \frac{1}{2} \left( \frac{(y - \hat{y})^2}{\hat{\sigma}^2 + r^2} + \log(\hat{\sigma}^2 + r^2) \right)$$

where:
- $\hat{y}$ is the predicted mean
- $y$ is the true value
- $\hat{\sigma}$ is the predicted standard deviation
- $r$ is a function of the residual that adapts based on outliers

**Example:**

```python
import torch
import torchregression as tr

# Create RAG loss
loss_fn = tr.losses.RAGLoss(alpha=0.1, beta=1.0)

# Model predicts both mean and log(sigma)
mean = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
log_sigma = torch.tensor([[-1.0, -0.5], [-0.5, 0.0]])
target = torch.tensor([[1.2, 2.5], [2.8, 3.9]])

# Calculate loss
loss = loss_fn((mean, log_sigma), target)
```

### EnsembleDistributionLoss

```python
class EnsembleDistributionLoss(DistributionLoss)
```

Loss function for training ensemble-based uncertainty models, which combines individual model losses with consistency regularization.

**Parameters:**

- `base_loss` (DistributionLoss): Base loss function to use for each individual model
- `consistency_weight` (float, optional): Weight for consistency regularization. Default: `0.1`
- `diversity_weight` (float, optional): Weight for diversity promotion. Default: `0.05`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Example:**

```python
import torch
import torchregression as tr

# Create base loss (e.g. Gaussian NLL)
base_loss = tr.losses.DiagonalGaussianNLL(n_features=2)

# Create ensemble loss
loss_fn = tr.losses.EnsembleDistributionLoss(
    base_loss=base_loss,
    consistency_weight=0.1,
    diversity_weight=0.05
)

# Model predictions from multiple ensemble members
# List of (mean, log_var) tuples from different models
ensemble_preds = [
    (torch.randn(8, 2), torch.randn(8, 2)),
    (torch.randn(8, 2), torch.randn(8, 2)),
    (torch.randn(8, 2), torch.randn(8, 2))
]

# Target values
target = torch.randn(8, 2)

# Calculate ensemble loss
loss = loss_fn(ensemble_preds, target)
```

### DeepEvidentialLoss

```python
class DeepEvidentialLoss(DistributionLoss)
```

Implements evidential regression loss, which uses the theory of subjective logic to quantify both aleatoric and epistemic uncertainty.

**Parameters:**

- `kl_weight` (float, optional): Weight for KL divergence regularization term. Default: `0.01`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Mathematical Background:**

Evidential regression models the target as a Normal distribution with unknown mean and precision (inverse variance). These parameters are modeled using a Normal-Gamma prior, where the model predicts the parameters of this prior:

$$\mathcal{L} = \mathcal{L}_{\text{NLL}} + \mathcal{L}_{\text{KL}}$$

Where NLL is the negative log likelihood of the target and KL is a regularization term.

**Example:**

```python
import torch
import torchregression as tr

# Create evidential loss
loss_fn = tr.losses.DeepEvidentialLoss(kl_weight=0.01)

# Model predicts 4 parameters: (gamma, nu, alpha, beta)
y_pred = torch.tensor([
    [1.0, 2.0, 3.0, 0.5],
    [2.0, 1.0, 4.0, 0.6]
])
target = torch.tensor([[1.2], [1.8]])

# Calculate loss
loss = loss_fn(y_pred, target)
```

## Choosing the Right Advanced Loss

| Loss Function | Best For | Uncertainty Types | Computational Cost |
|---------------|----------|------------------|-------------------|
| DiagonalGaussianNLL | Simple heteroscedastic uncertainty | Aleatoric | Low |
| GaussianNLLWithCovariance | Correlated outputs | Aleatoric | Medium |
| MixtureDensityLoss | Multi-modal distributions | Aleatoric | Medium |
| NormalizingFlowLoss | Complex arbitrary distributions | Aleatoric | High |
| RAGLoss | Robust regression with outliers | Aleatoric | Low |
| EnsembleDistributionLoss | Combined model training | Both | High |
| DeepEvidentialLoss | Separate uncertainty types | Both | Medium |

## Practical Guidelines

1. **Start Simple**: Begin with DiagonalGaussianNLL before moving to more complex losses

2. **Complexity Progression**:
   - For heteroscedastic data → DiagonalGaussianNLL
   - For multi-modal data → MixtureDensityLoss
   - For highly complex distributions → NormalizingFlowLoss
   - For robust regression → RAGLoss
   - For uncertainty decomposition → DeepEvidentialLoss

3. **Model Architecture Considerations**:
   - Advanced losses generally require larger models
   - Output layer size depends on the loss function requirements
   - Consider separate heads for different distribution parameters

4. **Evaluation**:
   - Use proper scoring rules (NLL, CRPS)
   - Check calibration with reliability diagrams
   - Evaluate both accuracy and uncertainty quality

5. **Computational Resources**:
   - More advanced losses require more computation
   - NormalizingFlowLoss is the most computationally intensive
   - EnsembleDistributionLoss requires multiple forward passes
