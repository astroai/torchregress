# Normalizing Flow Losses

Normalizing flows are powerful density estimation models that transform a simple base distribution (like a Gaussian) into a complex target distribution through a series of invertible transformations. In regression contexts, they enable modeling of complex, multimodal output distributions with high flexibility.

## Mathematical Background

A normalizing flow defines a sequence of invertible transformations $f = f_1 \circ f_2 \circ \cdots \circ f_K$ that map a simple base distribution $p_Z(z)$ to a more complex distribution $p_X(x)$:

$$p_X(x) = p_Z(f(x)) \left| \det\left(\frac{\partial f(x)}{\partial x}\right) \right|$$

The negative log-likelihood (NLL) loss for a normalizing flow model is:

$$\mathcal{L}_{\text{NF}}(x) = -\log p_X(x) = -\log p_Z(f(x)) - \log \left| \det\left(\frac{\partial f(x)}{\partial x}\right) \right|$$

This allows the model to capture complex, multimodal distributions by learning the parameters of these transformations.

## Available Normalizing Flow Losses

### NormalizingFlowLoss

```python
class NormalizingFlowLoss(DistributionLoss)
```

Negative Log-Likelihood loss for normalizing flow models, supporting various flow architectures for flexible distributional regression.

**Dependencies:**
- Requires the `zuko` package: `pip install zuko`

**Parameters:**

- `n_features` (int): Number of output features (dimensions)
- `flow_type` (str, optional): Type of flow architecture to use: 'realnvp' | 'maf' | 'nsf' | 'iaf'. Default: 'realnvp'
- `n_blocks` (int, optional): Number of transformation blocks. Default: `3`
- `hidden_features` (int, optional): Size of hidden layers in coupling/autoregressive networks. Default: `64`
- `n_hidden_layers` (int, optional): Number of hidden layers in networks. Default: `2`
- `base_distribution` (str, optional): Base distribution: 'normal' | 'uniform'. Default: 'normal'
- `activation` (str, optional): Activation function for hidden layers. Default: 'relu'
- `dropout` (float, optional): Dropout rate. Default: `0.0`
- `batch_norm` (bool, optional): Whether to use batch normalization. Default: `False`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `_validate_flow_type()`: Validates the chosen flow type
- `_create_flow(params_dict)`: Creates the flow model from parameters
- `_extract_distribution_parameters(y_pred)`: Gets flow parameters from predictions
- `_calculate_nll(y_true, params, mask)`: Calculates negative log-likelihood
- `forward(y_pred, y_true, mask=None, weights=None)`: Computes the NLL loss
- `sample(y_pred, n_samples=1)`: Generates samples from the flow distribution

**Example:**

```python
import torch
import torchregress as tr

# Create a normalizing flow loss
loss_fn = tr.losses.NormalizingFlowLoss(
    n_features=2,          # 2D outputs
    flow_type='nsf',       # Neural Spline Flow
    n_blocks=5,            # 5 transformation blocks
    hidden_features=128,   # Size of hidden layers
    n_hidden_layers=2,     # Number of hidden layers
    activation='relu'
)

# Model predicts parameters for the flow
batch_size = 32
# These parameters would come from your model
flow_params = torch.randn(batch_size, 500)  
y_true = torch.randn(batch_size, 2)

# Calculate loss
loss = loss_fn(flow_params, y_true)

# Generate samples from the distribution
samples = loss_fn.sample(flow_params, n_samples=10)  # [batch_size, 10, 2]
```

## Flow Types

TorchRegression supports four main types of normalizing flows, each with different characteristics:

### RealNVP (Real-valued Non-Volume Preserving)

- Uses **coupling layers** that split the input and transform part of it conditioned on the other part
- Efficient forward and inverse operations
- Good for lower-dimensional data
- Less expressive than autoregressive flows
- Default option and best starting point

### MAF (Masked Autoregressive Flow)

- **Autoregressive structure** where each variable depends on previous ones
- More expressive than RealNVP
- Slow sampling but fast density evaluation
- Good for conditional density estimation

### NSF (Neural Spline Flow)

- Uses **monotonic rational-quadratic splines** for transformations
- Highly expressive with compact representations
- Balances computational efficiency and modeling power
- Great for complex, multi-modal distributions

### IAF (Inverse Autoregressive Flow)

- Inverse of MAF
- Fast sampling but slow density evaluation
- Good when drawing samples is the priority

## Factory Function

### create_flow_loss

```python
create_flow_loss(n_features, flow_type='realnvp', n_blocks=3, hidden_features=64, 
                n_hidden_layers=2, base_distribution='normal', activation='relu',
                dropout=0.0, batch_norm=False, reduction='mean')
```

Factory function to create a normalizing flow loss with the specified parameters.

**Example:**

```python
# Create a Neural Spline Flow loss with custom parameters
flow_loss = tr.losses.create_flow_loss(
    n_features=3,
    flow_type='nsf',
    n_blocks=5,
    hidden_features=128,
    activation='swish'  # Using Swish activation
)
```

## Model Integration

To use normalizing flows in your PyTorch models, you need to design a network that outputs parameters for the flow. Here's an example:

```python
class FlowRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, flow_params_dim):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, flow_params_dim)
        )
        
        # Create the loss function which contains the flow
        self.flow_loss = tr.losses.create_flow_loss(
            n_features=output_dim,
            flow_type='nsf',
            n_blocks=5
        )
        
    def forward(self, x):
        # Output flow parameters
        flow_params = self.backbone(x)
        return flow_params
        
    def sample(self, x, n_samples=1):
        # Generate flow parameters
        flow_params = self.forward(x)
        # Use the loss function's sample method
        return self.flow_loss.sample(flow_params, n_samples)
        
    def loss(self, flow_params, y_true):
        return self.flow_loss(flow_params, y_true)
```

## Practical Considerations

1. **Parameter Dimensionality**: Different flow architectures require different numbers of parameters. You may need to adjust your model's output size accordingly.

2. **Training Stability**: Normalizing flows can be sensitive to initialization and learning rate. Start with simpler architectures and gradually increase complexity.

3. **Flow Depth**: More transformation blocks (higher `n_blocks`) create more expressive models but are harder to train. Start with 3-5 blocks.

4. **Memory Requirements**: Complex flows, especially with many blocks and large hidden layers, can require significant memory.

5. **External Dependencies**: 
   - Requires the `zuko` package: `pip install zuko`
   - For highly complex distributions, consider installing PyTorch with CUDA support

## When to Use Normalizing Flows

Normalizing flows are particularly useful for:

1. **Multi-modal distributions**: When your target variable can have multiple distinct "modes" or peaks

2. **Complex conditional distributions**: When the shape of the output distribution changes significantly based on the input

3. **Non-parametric density estimation**: When standard parametric distributions (Gaussian, etc.) are too restrictive

4. **Uncertainty quantification**: When you need detailed, flexible uncertainty estimates beyond just mean and variance

5. **Sampling tasks**: When you need to generate realistic samples from the learned distribution

## Comparison with Other Distribution Losses

| Loss Type | Expressivity | Computational Cost | Ease of Training | Best For |
|-----------|-------------|-------------------|-----------------|----------|
| Gaussian NLL | Low | Low | High | Simple unimodal distributions |
| Mixture Density | Medium | Medium | Medium | Simple multi-modal distributions |
| Normalizing Flows | High | High | Low | Complex, arbitrary distributions |
| Quantile Loss | Medium | Low | Medium | Distribution-free intervals |
