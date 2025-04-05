"""
Easy wrappers for commonly used tools and utilities in torchregress.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Callable, Dict, Any

# Import base losses
from .losses.base import MaskedLoss, RegressionLoss, WeightedLossWrapper, DistributionLoss, WeightedMSELoss, WeightedHuberLoss, WeightedL1Loss

# Import specific loss implementations
from .losses.gaussian import HeteroscedasticGaussianLoss as DiagonalGaussianNLL, MultivariateGaussianLoss as GaussianNLLWithCovariance
from .losses.robust import PseudoHuberLoss, LogCoshLoss, CharbonnierLoss
from .losses.quantile import QuantileLoss, MultiQuantileLoss
from .losses.mdn import create_mdn_loss
from .losses.eiv import FunctionalEIVLoss as ChamferEIVLoss, OrthogonalDistanceRegressionLoss as RobustEIVLoss

# Import ensemble components
from .ensemble import DeepEnsemble, HeteroscedasticEnsembleModel

# Import utilities
from .utils.augment import GaussianNoiseAugmentation, AdversarialAugmentation


def create_gaussian_regression(
    in_features: int,
    out_features: int,
    hidden_sizes: List[int] = [64, 64],
    activation: nn.Module = nn.ReLU(),
    heteroscedastic: bool = False,
    eps: float = 1e-6,
) -> Tuple[nn.Module, nn.Module]:
    """
    Creates a Gaussian regression model and corresponding loss function.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        hidden_sizes: List of hidden layer sizes
        activation: Activation function
        heteroscedastic: Whether to predict both mean and variance
        eps: Small constant for numerical stability

    Returns:
        Tuple of (model, loss_function)
    """
    layers = []
    layer_sizes = [in_features] + hidden_sizes

    # Create hidden layers
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        layers.append(activation)

    # Create output layer
    if heteroscedastic:
        # Output both mean and log_std
        output_layer = nn.Linear(layer_sizes[-1], 2 * out_features)
        model = nn.Sequential(*layers, output_layer)
        loss_fn = DiagonalGaussianNLL(out_features, eps=eps)
    else:
        # Output only mean
        output_layer = nn.Linear(layer_sizes[-1], out_features)
        model = nn.Sequential(*layers, output_layer)
        loss_fn = nn.MSELoss()

    return model, loss_fn


def create_robust_regression(
    in_features: int,
    out_features: int,
    loss_type: str = "huber",
    delta: float = 1.0,
    hidden_sizes: List[int] = [64, 64],
    activation: nn.Module = nn.ReLU(),
) -> Tuple[nn.Module, nn.Module]:
    """
    Creates a robust regression model with the specified loss.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        loss_type: Type of loss ('huber', 'l1', 'pseudo_huber', 'log_cosh')
        delta: Delta parameter for Huber and Pseudo-Huber losses
        hidden_sizes: List of hidden layer sizes
        activation: Activation function

    Returns:
        Tuple of (model, loss_function)
    """
    layers = []
    layer_sizes = [in_features] + hidden_sizes

    # Create hidden layers
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        layers.append(activation)

    # Create output layer
    output_layer = nn.Linear(layer_sizes[-1], out_features)
    model = nn.Sequential(*layers, output_layer)

    # Create loss function
    if loss_type == "huber":
        loss_fn = WeightedHuberLoss(delta=delta)
    elif loss_type == "l1":
        loss_fn = WeightedL1Loss()
    elif loss_type == "pseudo_huber":
        loss_fn = PseudoHuberLoss(delta=delta)
    elif loss_type == "log_cosh":
        loss_fn = LogCoshLoss()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    return model, loss_fn


def create_quantile_regression(
    in_features: int,
    out_features: int,
    quantiles: Union[float, List[float], torch.Tensor],
    hidden_sizes: List[int] = [64, 64],
    activation: nn.Module = nn.ReLU(),
) -> Tuple[nn.Module, nn.Module]:
    """
    Creates a quantile regression model.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        quantiles: Quantile level(s) to estimate
        hidden_sizes: List of hidden layer sizes
        activation: Activation function

    Returns:
        Tuple of (model, loss_function)
    """
    layers = []
    layer_sizes = [in_features] + hidden_sizes

    # Create hidden layers
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        layers.append(activation)

    # Handle multiple quantiles case
    if isinstance(quantiles, (list, torch.Tensor)) and len(quantiles) > 1:
        num_quantiles = len(quantiles) if isinstance(quantiles, list) else quantiles.numel()
        output_layer = nn.Linear(layer_sizes[-1], out_features * num_quantiles)
        model = nn.Sequential(*layers, output_layer)
        loss_fn = MultiQuantileLoss(quantiles)
    else:
        # Single quantile case
        output_layer = nn.Linear(layer_sizes[-1], out_features)
        model = nn.Sequential(*layers, output_layer)
        loss_fn = QuantileLoss(tau=quantiles)

    return model, loss_fn


def create_histogram_regression(
    in_features: int,
    out_features: int,
    num_bins: int,
    bin_min: float,
    bin_max: float,
    hidden_sizes: List[int] = [64, 64],
    activation: nn.Module = nn.ReLU(),
    target_distribution: str = "gaussian",
) -> Tuple[nn.Module, nn.Module]:
    """
    Creates a histogram regression model.

    Args:
        in_features: Input dimension
        out_features: Output dimension (number of regression targets)
        num_bins: Number of bins in the histogram
        bin_min: Minimum value of the support of the histogram
        bin_max: Maximum value of the support of the histogram
        hidden_sizes: List of hidden layer sizes
        activation: Activation function
        target_distribution: Type of the target distribution ('gaussian' or 'one-bin')

    Returns:
        Tuple of (model, loss_function)
    """
    layers = []
    layer_sizes = [in_features] + hidden_sizes

    # Create hidden layers
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        layers.append(activation)

    # Create output layer
    output_layer = nn.Linear(layer_sizes[-1], num_bins * out_features)

    class HistogramRegressionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_extractor = nn.Sequential(*layers)
            self.output_layer = output_layer
            self.out_features = out_features
            self.num_bins = num_bins

        def forward(self, x):
            x = self.feature_extractor(x)
            x = self.output_layer(x)
            # Reshape to [batch_size, out_features, num_bins]
            return x.reshape(-1, self.out_features, self.num_bins)

    model = HistogramRegressionModel()
    loss_fn = HistogramLoss(
        num_bins=num_bins, bin_min=bin_min, bin_max=bin_max, target_distribution=target_distribution
    )

    return model, loss_fn


def create_mdn_model(
    in_features: int,
    out_features: int,
    num_components: int = 5,
    hidden_sizes: List[int] = [64, 64],
    activation: nn.Module = nn.ReLU(),
    distribution: str = "gaussian",
) -> Tuple[nn.Module, nn.Module]:
    """
    Creates a Mixture Density Network (MDN) model.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        num_components: Number of mixture components
        hidden_sizes: List of hidden layer sizes
        activation: Activation function
        distribution: Type of distribution ('gaussian', 'student-t', etc.)

    Returns:
        Tuple of (model, loss_function)
    """
    layers = []
    layer_sizes = [in_features] + hidden_sizes

    # Create hidden layers
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        layers.append(activation)

    # Get distribution module to determine output size
    from .losses.mdn import get_distribution_module

    dist_module = get_distribution_module(distribution)
    dist_params_size = dist_module.get_params_size(out_features)
    total_params_size = dist_params_size * num_components + num_components

    # Create output layer
    output_layer = nn.Linear(layer_sizes[-1], total_params_size)
    model = nn.Sequential(*layers, output_layer)

    # Create loss function
    loss_fn = create_mdn_loss(num_components, out_features, distribution=distribution)

    return model, loss_fn


def create_deep_ensemble(
    model_fn: Callable[..., Tuple[nn.Module, nn.Module]],
    ensemble_size: int = 5,
    augmentation: str = "none",
    augmentation_params: Optional[Dict[str, Any]] = None,
    **model_kwargs,
) -> Tuple[nn.Module, nn.Module]:
    """
    Creates a deep ensemble model using the provided model factory function.

    Args:
        model_fn: Function that creates a single model and loss function
        ensemble_size: Number of models in the ensemble
        augmentation: Type of augmentation ('none', 'gaussian', 'adversarial')
        augmentation_params: Parameters for the augmentation
        **model_kwargs: Arguments to pass to the model factory function

    Returns:
        Tuple of (ensemble_model, loss_function)
    """
    # Create a single model to get its configuration
    model, loss_fn = model_fn(**model_kwargs)

    # Set up augmentation
    aug_params = augmentation_params or {}
    if augmentation == "gaussian":
        aug = GaussianNoiseAugmentation(**aug_params)
    elif augmentation == "adversarial":
        aug = AdversarialAugmentation(**aug_params)
    else:
        aug = None

    # Create an ensemble model factory
    def model_factory():
        return model_fn(**model_kwargs)[0]

    # Create the ensemble
    ensemble = DeepEnsemble.from_model_factory(
        model_factory=model_factory, ensemble_size=ensemble_size, augmentation=aug
    )

    # Set the loss function
    ensemble.set_loss_fn(loss_fn)

    return ensemble, loss_fn


def wrap_pytorch_loss(loss_class: type, **kwargs) -> MaskedLoss:
    """
    Wrap any PyTorch loss function with torchregress's masking and weighting capabilities.

    Args:
        loss_class: PyTorch loss class (e.g., nn.MSELoss, nn.L1Loss)
        **kwargs: Arguments to pass to the loss constructor

    Returns:
        A wrapped loss function with masking and weighting support
    """
    return WeightedLossWrapper(loss_class, **kwargs)


def create_regression_model(
    in_features: int,
    out_features: int,
    hidden_sizes: List[int] = None,
    activation: nn.Module = None,
    dropout: float = 0.0,
    batch_norm: bool = False,
    output_activation: Optional[nn.Module] = None,
) -> nn.Module:
    """
    Create a generic MLP regression model with configurable architecture.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        hidden_sizes: List of hidden layer sizes (default: [64, 64])
        activation: Activation function (default: ReLU)
        dropout: Dropout probability (default: 0.0)
        batch_norm: Whether to use batch normalization (default: False)
        output_activation: Optional activation for the output layer

    Returns:
        A PyTorch model
    """
    if hidden_sizes is None:
        hidden_sizes = [64, 64]
    if activation is None:
        activation = nn.ReLU()

    layers = []
    layer_sizes = [in_features] + hidden_sizes

    # Create hidden layers
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        if batch_norm:
            layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))

        layers.append(activation)

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

    # Create output layer
    layers.append(nn.Linear(layer_sizes[-1], out_features))

    # Add output activation if specified
    if output_activation is not None:
        layers.append(output_activation)

    return nn.Sequential(*layers)


def create_loss_from_config(config: Dict[str, Any]) -> MaskedLoss:
    """
    Create a loss function from a configuration dictionary.

    This provides a standardized way to instantiate any loss function
    based on a configuration dict, useful for experiments and hyperparameter tuning.

    Args:
        config: Configuration dictionary with at least a 'type' key
               and optional parameters for the specific loss

    Returns:
        Instantiated loss function

    Example:
        >>> loss_config = {
        >>>     'type': 'huber',
        >>>     'delta': 1.0,
        >>>     'reduction': 'mean'
        >>> }
        >>> loss_fn = create_loss_from_config(loss_config)
    """
    loss_type = config.pop("type", "").lower()

    if loss_type == "mse" or loss_type == "mseloss":
        return WeightedMSELoss(**config)

    elif loss_type == "l1" or loss_type == "mae":
        return WeightedL1Loss(**config)

    elif loss_type == "huber":
        return WeightedHuberLoss(**config)

    elif loss_type == "gaussian" or loss_type == "gaussiannll":
        return DiagonalGaussianNLL(**config)

    elif loss_type == "quantile" or loss_type == "pinball":
        return QuantileLoss(**config)

    elif loss_type == "multiquantile":
        return MultiQuantileLoss(**config)

    elif loss_type == "histogram":
        return HistogramLoss(**config)

    elif loss_type == "mdn":
        return create_mdn_loss(**config)

    elif loss_type == "pytorch":
        loss_class = config.pop("class")
        if isinstance(loss_class, str):
            # Try to get the loss class from torch.nn
            import torch.nn as nn

            if hasattr(nn, loss_class):
                loss_class = getattr(nn, loss_class)
            else:
                raise ValueError(f"Unknown PyTorch loss: {loss_class}")
        return wrap_pytorch_loss(loss_class, **config)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
