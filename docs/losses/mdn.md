# Mixture Density Network Losses

Mixture Density Networks (MDNs) model complex output distributions as a mixture of simpler parametric distributions (typically Gaussians). This allows neural networks to represent multi-modal outputs, uncertainty, and complex conditional distributions.

## Mathematical Background

A Mixture Density Network outputs parameters for a mixture of probability distributions. For a mixture of $K$ Gaussians, the probability density function is:

$$p(y|x) = \sum_{k=1}^{K} \pi_k(x) \mathcal{N}(y|\mu_k(x), \Sigma_k(x))$$

Where:
- $\pi_k(x)$ are the mixture weights (probabilities) for each component
- $\mu_k(x)$ are the means for each component
- $\Sigma_k(x)$ are the covariance matrices for each component
- $\mathcal{N}(y|\mu, \Sigma)$ is the Gaussian probability density function

The negative log-likelihood loss for this mixture model is:

$$\mathcal{L}_{\text{MDN}}(y) = -\log p(y|x) = -\log \sum_{k=1}^{K} \pi_k(x) \mathcal{N}(y|\mu_k(x), \Sigma_k(x))$$

## Available MDN Losses

### MixtureDensityLoss

```python
class MixtureDensityLoss(DistributionLoss)
```

Negative Log-Likelihood loss for Mixture Density Networks, supporting diagonal or full covariance matrices.

**Parameters:**

- `n_components` (int): Number of mixture components (Gaussian distributions)
- `n_features` (int): Number of output features (dimensionality of the target)
- `covariance_type` (str, optional): Type of covariance matrices: 'diagonal' | 'full'. Default: 'diagonal'
- `min_std` (float, optional): Minimum standard deviation for numerical stability. Default: `1e-3`
- `eps` (float, optional): Small constant for numerical stability in calculations. Default: `1e-8`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `_extract_distribution_parameters(y_pred)`: Extracts mixture weights, means, and std/covariance factors
- `_log_prob_diagonal(target, means, stds)`: Calculates log probability for diagonal covariance
- `_log_prob_full(target, means, L_matrices)`: Calculates log probability for full covariance
- `_calculate_nll(target, params, mask)`: Calculates the negative log-likelihood
- `forward(y_pred, target, mask=None, weights=None)`: Computes the MDN loss

**Example:**

```python
import torch
import torchregress as tr

# Create an MDN loss with 3 components (mixture of 3 Gaussians)
loss_fn = tr.losses.MixtureDensityLoss(
    n_components=3,
    n_features=2,
    covariance_type='diagonal'
)

# Shape depends on output parameterization
# For diagonal: [batch_size, n_components + 2*n_components*n_features]
# Here: [batch_size, 3 + 2*3*2] = [batch_size, 15]
y_pred = torch.randn(10, 15)  # Parameters for 3 components, 2D output
target = torch.randn(10, 2)   # Target values

# Calculate loss
loss = loss_fn(y_pred, target)
```

## Factory Function

### create_mdn_loss

```python
create_mdn_loss(n_components, n_features, covariance_type='diagonal', min_std=1e-3, 
               eps=1e-8, reduction='mean')
```

Factory function to create a Mixture Density Network loss with the specified parameters.

**Example:**

```python
# Create an MDN loss with 5 components and full covariance matrices
mdn_loss = tr.losses.create_mdn_loss(
    n_components=5,
    n_features=3,
    covariance_type='full',
    min_std=1e-2  # Increase for better numerical stability
)
```

## Output Parameterization

The model's output for an MDN must provide all necessary distribution parameters. For a mixture with `K` components and `D` features:

### Diagonal Covariance

Model output size: `K + 2*K*D` (weights + means + stds)
- First `K` elements: Mixture weights (logits before softmax)
- Next `K*D` elements: Means for each component and feature
- Last `K*D` elements: Log standard deviations for each component and feature

```python
# Example model architecture for diagonal covariance
class MDNModel(torch.nn.Module):
    def __init__(self, input_dim, n_components=3, output_dim=2):
        super().__init__()
        self.n_components = n_components
        self.output_dim = output_dim
        
        # Size of MDN output (weights + means + stds)
        mdn_output_size = n_components + 2 * n_components * output_dim
        
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, mdn_output_size)
        )
        
    def forward(self, x):
        return self.network(x)
```

### Full Covariance

Model output size: `K + K*D + K*D*(D+1)/2` (weights + means + lower triangular elements)
- First `K` elements: Mixture weights (logits before softmax)
- Next `K*D` elements: Means for each component and feature
- Last `K*D*(D+1)/2` elements: Parameters for lower triangular matrices (Cholesky factors)

## Parameter Interpretation

The MDN loss function internally handles the transformation of raw model outputs:

1. **Mixture Weights**: Logits are transformed with softmax to ensure they sum to 1
2. **Standard Deviations**: Log-stds are transformed with softplus and a minimum value to ensure positivity
3. **Covariance Matrices**: For full covariance, lower triangular matrices are constructed with positive diagonal elements

## Sampling from MDNs

While the loss function itself doesn't provide a sampling method, you can implement one using the extracted parameters:

```python
def sample_from_mdn(params, n_samples=1):
    weights, means, stds_or_L = loss_fn._extract_distribution_parameters(params)
    
    # Convert weights to probabilities
    probs = weights.cpu().numpy()
    
    # Initialize storage for samples
    batch_size = means.shape[0]
    n_features = means.shape[-1]
    samples = torch.zeros(batch_size, n_samples, n_features, device=means.device)
    
    # For each item in the batch
    for b in range(batch_size):
        # Choose components according to mixture weights
        component_indices = np.random.choice(
            weights.shape[-1], size=n_samples, p=probs[b]/probs[b].sum()
        )
        
        # For each sample
        for i, c in enumerate(component_indices):
            if isinstance(stds_or_L, tuple) or stds_or_L.ndim == 4:  # Full covariance
                # Using Cholesky decomposition for multivariate sampling
                L = stds_or_L[b, c]
                epsilon = torch.randn(n_features, device=L.device)
                samples[b, i] = means[b, c] + torch.mv(L, epsilon)
            else:  # Diagonal covariance
                std = stds_or_L[b, c]
                epsilon = torch.randn_like(std)
                samples[b, i] = means[b, c] + std * epsilon
                
    return samples
```

## Advanced Usage: Model Evaluation

To evaluate an MDN model beyond just the loss value, you can compute various metrics:

```python
def evaluate_mdn(model, loss_fn, x, y, n_samples=1000):
    # Get model predictions (MDN parameters)
    with torch.no_grad():
        mdn_params = model(x)
    
    # Extract distribution parameters
    weights, means, stds = loss_fn._extract_distribution_parameters(mdn_params)
    
    # Compute negative log-likelihood
    nll = loss_fn(mdn_params, y)
    
    # Generate samples
    samples = sample_from_mdn(mdn_params, n_samples)
    
    # Calculate sample statistics
    sample_mean = samples.mean(dim=1)
    sample_std = samples.std(dim=1)
    
    # RMSE of the sample mean
    rmse = torch.sqrt(((sample_mean - y) ** 2).mean())
    
    # Compute calibration metrics (e.g., quantile coverage)
    lower_5 = torch.quantile(samples, 0.05, dim=1)
    upper_95 = torch.quantile(samples, 0.95, dim=1)
    coverage_90 = ((y >= lower_5) & (y <= upper_95)).float().mean()
    
    return {
        'nll': nll.item(),
        'rmse': rmse.item(),
        'coverage_90': coverage_90.item()
    }
```

## When to Use MDN Losses

Mixture Density Networks are particularly useful when:

1. **The conditional distribution is multi-modal**: When the target can have multiple distinct "modes" or peaks

2. **The output has heteroscedastic noise**: When uncertainty varies across the input space

3. **You need probabilistic predictions**: When point estimates are insufficient

4. **The data follows a complex but structured distribution**: When you can approximate the true distribution with a mixture

5. **You need interpretable uncertainty**: When you want to identify different modes in the prediction

## Practical Considerations

1. **Number of Components**:
   - Start with a small number (2-5) and increase if needed
   - Too many components can lead to overfitting
   - Too few components might not capture the true distribution

2. **Covariance Type**:
   - Diagonal: Faster, more stable, sufficient for many applications
   - Full: More expressive, captures correlations, but harder to train

3. **Numerical Stability**:
   - MDNs can be unstable during training
   - Use constraints on minimum standard deviations (controlled by `min_std`)
   - Consider gradient clipping
   - Initialize the final layer with small weights

4. **Mixture Collapse**:
   - MDNs can suffer from component collapse where one component dominates
   - Regularize the mixture weights or use temperature scaling
   - Ensure diverse initialization of component parameters

5. **Evaluation**:
   - NLL alone is not sufficient to evaluate MDNs
   - Use calibration metrics and sample-based evaluations
   - Visualize the predicted distributions when possible

## Comparison with Other Distribution Losses

| Feature | Gaussian NLL | MDN Loss | Normalizing Flow Loss | Quantile Loss |
|---------|-------------|----------|---------------------|---------------|
| Expressivity | Low | Medium-High | Very High | Medium |
| Multi-modality | No | Yes | Yes | Indirect |
| Speed | Fast | Moderate | Slow | Fast |
| Parameters | Few | Moderate | Many | Few |
| Training Stability | High | Medium | Low | High |
| Interpretability | High | Medium | Low | Medium |
