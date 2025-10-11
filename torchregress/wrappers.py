"""
High-level wrappers and convenience functions for torchregress.

This module provides simplified interfaces for common regression tasks,
including model creation, loss function setup, and ensemble methods.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch.nn as nn

# Import ensemble components
from .ensemble import DeepEnsemble

# Import base losses
from .losses.base import (
    MaskedLoss,
    WeightedHuberLoss,
    WeightedL1Loss,
    WeightedLossWrapper,
    WeightedMSELoss,
)

# Import specific loss implementations
from .losses.gaussian import (
    HeteroscedasticGaussianLoss as DiagonalGaussianNLL,
)
from .losses.mdn import create_mdn_loss
from .losses.quantile import MultiQuantileLoss, QuantileLoss
from .losses.robust import LogCoshLoss, PseudoHuberLoss

# Import utilities


def _get_defaults(
    hidden_sizes: Optional[List[int]], activation: Optional[nn.Module]
) -> Tuple[List[int], nn.Module]:
    """
    Get default values for hidden sizes and activation if not provided.

    Args:
        hidden_sizes: User-provided hidden sizes or None
        activation: User-provided activation or None

    Returns:
        Tuple of (hidden_sizes, activation) with defaults applied
    """
    if hidden_sizes is None:
        hidden_sizes = [64, 64]
    if activation is None:
        activation = nn.ReLU()
    return hidden_sizes, activation


def _build_mlp_backbone(
    in_features: int,
    hidden_sizes: List[int],
    activation: nn.Module,
    dropout: float = 0.0,
    batch_norm: bool = False,
) -> List[nn.Module]:
    """
    Build MLP backbone layers (shared code for all model builders).

    This function creates a standard MLP architecture with optional batch normalization
    and dropout. It's used internally by all model creation functions to avoid code duplication.

    Args:
        in_features: Input dimension (number of input features)
        hidden_sizes: List of hidden layer sizes (e.g., [64, 32] for two hidden layers)
        activation: Activation function instance (e.g., nn.ReLU())
        dropout: Dropout probability (0.0 = no dropout, default: 0.0)
        batch_norm: Whether to use batch normalization after each linear layer (default: False)

    Returns:
        List of nn.Module layers that can be unpacked into nn.Sequential()

    Example:
        >>> import torch.nn as nn
        >>> layers = _build_mlp_backbone(10, [64, 32], nn.ReLU(), dropout=0.1)
        >>> model = nn.Sequential(*layers, nn.Linear(32, 1))
    """
    layers: List[nn.Module] = []
    layer_sizes = [in_features] + hidden_sizes

    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        if batch_norm:
            layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))

        layers.append(activation)

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

    return layers


def create_gaussian_model(
    in_features: int,
    out_features: int,
    hidden_sizes: Optional[List[int]] = None,
    activation: Optional[nn.Module] = None,
    dropout: float = 0.0,
    batch_norm: bool = False,
    learn_variance: bool = False,
) -> Tuple[nn.Module, nn.Module]:
    """
    Creates a Gaussian regression model with optional variance learning.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        hidden_sizes: List of hidden layer sizes
        activation: Activation function
        dropout: Dropout probability
        batch_norm: Whether to use batch normalization
        learn_variance: Whether to learn variance as well as mean

    Returns:
        Tuple of (model, loss_function)
    """
    # Set default values
    hidden_sizes, activation = _get_defaults(hidden_sizes, activation)

    # Build MLP backbone
    layers = _build_mlp_backbone(in_features, hidden_sizes, activation, dropout, batch_norm)

    # Create output layer
    if learn_variance:
        # Output both mean and log variance
        output_layer = nn.Linear(hidden_sizes[-1], out_features * 2)
        model = nn.Sequential(*layers, output_layer)
        loss_fn = DiagonalGaussianNLL(n_features=out_features)
    else:
        # Output only mean
        output_layer = nn.Linear(hidden_sizes[-1], out_features)
        model = nn.Sequential(*layers, output_layer)
        loss_fn = WeightedMSELoss()

    return model, loss_fn


def create_quantile_model(
    in_features: int,
    out_features: int,
    quantiles: Union[float, List[float]] = [0.1, 0.5, 0.9],
    hidden_sizes: Optional[List[int]] = None,
    activation: Optional[nn.Module] = None,
    dropout: float = 0.0,
    batch_norm: bool = False,
) -> Tuple[nn.Module, nn.Module]:
    """
    Creates a quantile regression model.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        quantiles: Quantile(s) to predict
        hidden_sizes: List of hidden layer sizes
        activation: Activation function
        dropout: Dropout probability
        batch_norm: Whether to use batch normalization

    Returns:
        Tuple of (model, loss_function)
    """
    # Set default values
    hidden_sizes, activation = _get_defaults(hidden_sizes, activation)

    # Build MLP backbone
    layers = _build_mlp_backbone(in_features, hidden_sizes, activation, dropout, batch_norm)

    # Handle quantiles
    if isinstance(quantiles, (int, float)):
        quantiles = [quantiles]
        single_quantile = True
    else:
        single_quantile = False

    # Create output layer
    output_layer = nn.Linear(hidden_sizes[-1], out_features * len(quantiles))
    model = nn.Sequential(*layers, output_layer)

    # Create loss function
    if single_quantile:
        loss_fn: Union[QuantileLoss, MultiQuantileLoss] = QuantileLoss(quantile=quantiles[0])
    else:
        loss_fn = MultiQuantileLoss(quantiles=quantiles)

    return model, loss_fn


def create_robust_model(
    in_features: int,
    out_features: int,
    loss_type: str = "huber",
    hidden_sizes: Optional[List[int]] = None,
    activation: Optional[nn.Module] = None,
    dropout: float = 0.0,
    batch_norm: bool = False,
    **loss_kwargs: Any,
) -> Tuple[nn.Module, nn.Module]:
    """
    Creates a robust regression model.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        loss_type: Type of robust loss ('huber', 'pseudo-huber', 'log-cosh')
        hidden_sizes: List of hidden layer sizes
        activation: Activation function
        dropout: Dropout probability
        batch_norm: Whether to use batch normalization
        **loss_kwargs: Additional arguments for the loss function

    Returns:
        Tuple of (model, loss_function)
    """
    # Set default values
    hidden_sizes, activation = _get_defaults(hidden_sizes, activation)

    # Build MLP backbone
    layers = _build_mlp_backbone(in_features, hidden_sizes, activation, dropout, batch_norm)

    # Create output layer
    output_layer = nn.Linear(hidden_sizes[-1], out_features)
    model = nn.Sequential(*layers, output_layer)

    # Create loss function
    if loss_type == "huber":
        loss_fn = WeightedHuberLoss(**loss_kwargs)
    elif loss_type == "pseudo-huber":
        loss_fn = PseudoHuberLoss(**loss_kwargs)
    elif loss_type == "log-cosh":
        loss_fn = LogCoshLoss(**loss_kwargs)
    else:
        raise ValueError(f"Unknown robust loss type: {loss_type}")

    return model, loss_fn


def create_mdn_model(
    in_features: int,
    out_features: int,
    num_components: int = 5,
    hidden_sizes: Optional[List[int]] = None,
    activation: Optional[nn.Module] = None,
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
    # Set default values
    hidden_sizes, activation = _get_defaults(hidden_sizes, activation)

    # Build MLP backbone (no dropout/batch_norm for MDN by default)
    layers = _build_mlp_backbone(in_features, hidden_sizes, activation, dropout=0.0, batch_norm=False)

    # Create output layer for MDN
    # For diagonal covariance: n_components + 2*n_components*n_features
    if distribution == "gaussian":
        expected_output_size = num_components + 2 * num_components * out_features
    else:
        # Default to gaussian for other distributions
        expected_output_size = num_components + 2 * num_components * out_features

    output_layer = nn.Linear(hidden_sizes[-1], expected_output_size)
    model = nn.Sequential(*layers, output_layer)

    # Create loss function
    loss_fn = create_mdn_loss(
        n_components=num_components, n_features=out_features, covariance_type="diagonal"
    )

    return model, loss_fn


def create_deep_ensemble(
    model_fn: Callable[..., Tuple[nn.Module, nn.Module]],
    ensemble_size: int = 5,
    augmentation: str = "none",
    augmentation_params: Optional[Dict[str, Any]] = None,
    **model_kwargs: Any,
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

    # Create an ensemble model factory
    def model_factory() -> nn.Module:
        return model_fn(**model_kwargs)[0]

    # Create the ensemble
    ensemble = DeepEnsemble(base_model=model_factory, ensemble_size=ensemble_size, device="cpu")

    # Set the loss function
    # ensemble.set_loss_fn(loss_fn)  # This method doesn't exist

    return ensemble, loss_fn


def wrap_pytorch_loss(loss_class: type, **kwargs: Any) -> MaskedLoss:
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
    hidden_sizes: Optional[List[int]] = None,
    activation: Optional[nn.Module] = None,
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
    # Set default values
    hidden_sizes, activation = _get_defaults(hidden_sizes, activation)

    # Build MLP backbone
    layers = _build_mlp_backbone(in_features, hidden_sizes, activation, dropout, batch_norm)

    # Create output layer
    layers.append(nn.Linear(hidden_sizes[-1], out_features))

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

    loss_fn: MaskedLoss
    if loss_type == "mse" or loss_type == "mseloss":
        loss_fn = WeightedMSELoss(**config)

    elif loss_type == "l1" or loss_type == "mae":
        loss_fn = WeightedL1Loss(**config)

    elif loss_type == "huber":
        loss_fn = WeightedHuberLoss(**config)

    elif loss_type == "gaussian" or loss_type == "gaussiannll":
        loss_fn = DiagonalGaussianNLL(**config)

    elif loss_type == "quantile" or loss_type == "pinball":
        loss_fn = QuantileLoss(**config)

    elif loss_type == "multiquantile":
        loss_fn = MultiQuantileLoss(**config)

    elif loss_type == "mdn":
        loss_fn = create_mdn_loss(**config)

    elif loss_type == "pytorch":
        loss_class = config.pop("class")
        if isinstance(loss_class, str):
            # Try to get the loss class from torch.nn
            import torch.nn as nn

            if hasattr(nn, loss_class):
                loss_class = getattr(nn, loss_class)
            else:
                raise ValueError(f"Unknown PyTorch loss: {loss_class}")
        loss_fn = wrap_pytorch_loss(loss_class, **config)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return loss_fn
