# Iteratively Reweighted Least Squares (IRLS)

## Overview

Iteratively Reweighted Least Squares (IRLS) is a powerful algorithm for robust regression that handles outliers by iteratively adjusting the weights of data points. This implementation provides:

- Multiple weight functions (Huber, Tukey, Power)
- Support for various loss functions (Gaussian, Huber, L1)
- Efficient batched processing for large datasets
- PyTorch integration with GPU acceleration

## Algorithm Description

The IRLS algorithm works by iteratively:
1. Computing predictions with the current model
2. Calculating residuals between predictions and targets
3. Determining weights based on those residuals
4. Updating the model using the weighted loss

This process automatically reduces the influence of outliers, producing robust regression results.

## Functions

### `iteratively_reweighted_least_squares`

Core implementation of the IRLS algorithm.

```python
def iteratively_reweighted_least_squares(
    model: nn.Module,
    x: torch.Tensor,
    y_true: torch.Tensor,
    initial_precision: Optional[torch.Tensor] = None,
    covariance_matrices: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    base_loss: str = "gaussian",
    max_iter: int = 10,
    tol: float = 1e-4,
    delta: float = 1.0,
    weight_fn: Union[str, Callable] = "huber",
    weight_params: Optional[Dict[str, Any]] = None,
    variance_type: str = "predicted",
    epsilon: float = EPS,
    return_all_predictions: bool = False,
    callbacks: Optional[List[CallbackFn]] = None,
    use_compile: bool = False,
    compile_kwargs: Optional[Dict[str, Any]] = None,
    use_tqdm: bool = False,
) -> Union[
    Tuple[torch.Tensor, List[float], torch.Tensor],
    Tuple[torch.Tensor, List[float], torch.Tensor, List[torch.Tensor]],
]
```

#### Parameters

- **model**: PyTorch model for making predictions
- **x**: Input data tensor of shape `(batch_size, n_features_x)`
- **y_true**: Target data tensor of shape `(batch_size, n_features_y)`
- **initial_precision**: Optional initial precision (inverse variance) weights
- **covariance_matrices**: Optional covariance matrices for multivariate Gaussian models
- **mask**: Optional mask for ignoring certain values in the computation
- **base_loss**: Base loss function: 'gaussian', 'huber', or 'l1'
- **max_iter**: Maximum number of IRLS iterations
- **tol**: Convergence tolerance for early stopping
- **delta**: Parameter for Huber loss/weight function
- **weight_fn**: Weighting function: 'huber', 'tukey', 'power', or a callable
- **weight_params**: Parameters for the chosen weighting function
- **variance_type**: Variance estimation method: 'predicted', 'fixed', or 'robust'
- **epsilon**: Small value for numerical stability
- **return_all_predictions**: Whether to return predictions from all iterations
- **callbacks**: List of callback functions for monitoring the algorithm
- **use_compile**: Whether to use torch.compile for acceleration (PyTorch 2.0+)
- **compile_kwargs**: Additional arguments for torch.compile
- **use_tqdm**: Whether to display a progress bar for iterations

#### Returns

- **y_pred**: Final model predictions
- **loss_history**: List of loss values over iterations
- **final_precision**: Final precision tensor (weights)
- **[optional] all_predictions**: List of predictions from all iterations if requested

### `IRLS`

User-friendly implementation for training models with IRLS.

```python
def IRLS(
    model: nn.Module,
    train_data: Union[DataLoader, Tuple[torch.Tensor, torch.Tensor], IterableDataset],
    loss_fn: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    num_epochs: int = 1,
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 32,
    irls_max_iter: int = 10,
    irls_tol: float = 1e-4,
    delta: float = 1.0,
    weight_fn: Union[str, Callable] = "huber",
    weight_params: Optional[Dict[str, Any]] = None,
    variance_type: str = "predicted",
    initial_precision: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    covariance_matrices: Optional[torch.Tensor] = None,
    verbose: bool = True,
    progress_bar: bool = True,
    update_weights: str = "epoch",
    val_data: Optional[Union[DataLoader, Tuple[torch.Tensor, torch.Tensor]]] = None,
    val_freq: int = 1,
    clip_grad_norm: Optional[float] = None,
    base_loss: Optional[str] = None,
    use_compile: bool = False,
    compile_kwargs: Optional[Dict[str, Any]] = None,
    callbacks: Optional[List[Callable]] = None,
) -> Dict[str, Any]
```

#### Parameters

- **model**: PyTorch model to train
- **train_data**: Training data as DataLoader or (x, y) tensor tuple
- **loss_fn**: Loss function (inferred from base_loss if not provided)
- **optimizer**: PyTorch optimizer (defaults to Adam if not provided)
- **num_epochs**: Number of training epochs
- **device**: Device to use ('cpu', 'cuda', etc.)
- **batch_size**: Batch size for training
- **irls_max_iter**: Maximum iterations for IRLS per reweighting
- **irls_tol**: Convergence tolerance for IRLS
- **delta**: Huber loss delta parameter
- **weight_fn**: Weight function ('huber', 'tukey', 'power', or callable)
- **weight_params**: Parameters for weight function
- **variance_type**: Variance estimation method
- **initial_precision**: Initial precision weights
- **mask**: Optional mask for ignoring values
- **covariance_matrices**: Optional covariance matrices
- **verbose**: Whether to print progress information
- **progress_bar**: Show progress bars using tqdm
- **update_weights**: When to update IRLS weights:
  - "epoch": Reweight once per epoch
  - "batch": Reweight after each batch
  - "iter:N": Reweight every N iterations
- **val_data**: Optional validation data
- **val_freq**: Validation frequency (epochs)
- **clip_grad_norm**: Optional gradient clipping value
- **base_loss**: Base loss type ('gaussian', 'huber', 'l1')
- **use_compile**: Whether to use torch.compile for speedup
- **compile_kwargs**: Additional kwargs for torch.compile
- **callbacks**: Optional list of callbacks

#### Returns

Dictionary containing:
- **model**: Trained model
- **train_loss_history**: Training loss history
- **final_precision**: Final precision weights
- **val_loss_history**: (if validation data provided)
- **all_iterations**: (if return_all_iterations=True)

## Weight Functions

The IRLS implementation supports several weight functions for robust regression:

### Huber Weights

```python
def huber_weights(residuals, delta=1.0, **kwargs):
    """
    Huber weights for robust regression.
    
    Returns 1.0 for |residuals| <= delta, and delta/|residuals| otherwise
    """
```

### Tukey Biweight (Bisquare) Weights

```python
def tukey_weights(residuals, c=4.685, **kwargs):
    """
    Tukey biweight (bisquare) weights for robust regression.
    
    Returns (1-(residuals/c)²)² for |residuals| <= c, and 0 otherwise
    """
```

### Power Weights

```python
def power_weights(residuals, a=1.0, b=2.0, **kwargs):
    """
    Power weights for robust regression.
    
    Returns 1/((a + |residuals|^b))
    """
```

## Usage Examples

### Basic IRLS with a Linear Model

```python
import torch
import torch.nn as nn
from torchregress.algorithms.irls import IRLS

# Define a simple linear model
class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

# Create synthetic data with outliers
X = torch.randn(1000, 5)
y_true = X @ torch.tensor([1.0, -0.5, 0.2, 0.7, -0.3]) + 0.1 * torch.randn(1000)
# Add some outliers
outlier_idx = torch.randperm(1000)[:50]
y_true[outlier_idx] += 5.0 * torch.randn(50)

# Create and train the model
model = LinearModel(5, 1)
result = IRLS(
    model=model,
    train_data=(X, y_true),
    weight_fn='tukey',  # Use Tukey biweight for robust regression
    num_epochs=5,
    verbose=True
)

# Access the trained model
trained_model = result['model']
```

### Advanced Usage with Custom Weight Function

```python
import torch
import torch.nn as nn
from torchregress.algorithms.irls import iteratively_reweighted_least_squares

# Define a custom weight function
def custom_weights(residuals, cutoff=2.5, **kwargs):
    weights = torch.ones_like(residuals)
    weights[torch.abs(residuals) > cutoff] = 0.0  # Hard rejection of outliers
    return weights

# Use with a pre-trained model for inference
with torch.no_grad():
    y_pred, loss_history, precision = iteratively_reweighted_least_squares(
        model=my_model,
        x=x_test,
        y_true=y_test,
        weight_fn=custom_weights,
        weight_params={'cutoff': 3.0},
        max_iter=20
    )
```

## References

1. Holland, P. W., & Welsch, R. E. (1977). Robust regression using iteratively reweighted least-squares. Communications in Statistics - Theory and Methods, 6(9), 813–827.
2. Huber, P. J. (1973). Robust regression: asymptotics, conjectures, and Monte Carlo. Annals of Statistics, 1(5), 799-821.
3. Beaton, A. E., & Tukey, J. W. (1974). The fitting of power series, meaning polynomials, illustrated on band-spectroscopic data. Technometrics, 16(2), 147-185.
