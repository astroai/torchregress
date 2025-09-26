"""
Error-in-Variables (EIV) loss functions for regression with uncertain inputs.

This module provides implementations of various loss functions that account for
measurement error in both features and targets.
"""

from typing import Callable, List, Optional, Union

import torch

from .eiv.eiv_chamfer import ChamferEIVLoss, HybridEIVChamferLoss
from .eiv.eiv_mdn import MDNEIVLoss, MDNEIVModel
from .eiv.eiv_quantile import MultiQuantileEIVLoss, QuantileEIVLoss
from .eiv.eiv_rfit import (
    RobustEIVLoss,
    adversarial_variation,
    bootstrap_variation,
    gaussian_variation,
    structured_variation,
    uniform_variation,
)
from .eiv.eiv_standard import (
    EnsembleEIVLoss,
    FunctionalEIVLoss,
    OrthogonalDistanceRegressionLoss,
    StructuralEIVLoss,
)

# Factory functions with improved parameter handling


def create_eiv_loss(
    model: Callable,
    sigma_x: Union[float, torch.Tensor],
    sigma_y: Optional[Union[float, torch.Tensor]] = None,
    monte_carlo: bool = False,
    n_samples: int = 20,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> FunctionalEIVLoss:
    """
    Create a Functional Error-in-Variables loss function.

    Args:
        model: Model function f(x) that predicts y
        sigma_x: Standard deviation of feature noise
        sigma_y: Standard deviation of target noise (optional)
        monte_carlo: Whether to use Monte Carlo sampling for gradient estimation
        n_samples: Number of MC samples if monte_carlo=True
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        eps: Small value for numerical stability

    Returns:
        FunctionalEIVLoss instance
    """
    return FunctionalEIVLoss(
        model=model,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        monte_carlo=monte_carlo,
        n_samples=n_samples,
        reduction=reduction,
        eps=eps,
    )


def create_structural_eiv_loss(
    model: Callable,
    sigma_x: Union[float, torch.Tensor],
    sigma_y: Union[float, torch.Tensor],
    sigma_xy: torch.Tensor,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> StructuralEIVLoss:
    """
    Create a Structural EIV loss that handles correlated errors.

    Args:
        model: Model function f(x) that predicts y
        sigma_x: Covariance of feature noise
        sigma_y: Covariance of target noise
        sigma_xy: Cross-covariance between feature and target noise
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        eps: Small value for numerical stability

    Returns:
        StructuralEIVLoss instance
    """
    return StructuralEIVLoss(
        model=model,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        sigma_xy=sigma_xy,
        reduction=reduction,
        eps=eps,
    )


def create_odr_loss(
    model: Callable,
    sigma_x: Union[float, torch.Tensor],
    sigma_y: Union[float, torch.Tensor],
    learning_rate: float = 0.01,
    max_iterations: int = 10,
    tolerance: float = 1e-6,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> OrthogonalDistanceRegressionLoss:
    """
    Create an Orthogonal Distance Regression loss.

    Args:
        model: Model function f(x) that predicts y
        sigma_x: Standard deviation of feature noise
        sigma_y: Standard deviation of target noise
        learning_rate: Learning rate for latent x optimization
        max_iterations: Maximum iterations for optimization
        tolerance: Convergence criterion for optimization
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        eps: Small value for numerical stability

    Returns:
        OrthogonalDistanceRegressionLoss instance
    """
    return OrthogonalDistanceRegressionLoss(
        model=model,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        tolerance=tolerance,
        reduction=reduction,
        eps=eps,
    )


def create_chamfer_eiv_loss(
    model: Callable,
    sigma_x: Optional[Union[float, torch.Tensor]] = None,
    method: str = "monte_carlo",
    n_samples: int = 100,
    optim_steps: int = 50,
    optim_lr: float = 0.01,
    early_stopping_tol: float = 1e-5,
    reduction: str = "mean",
) -> ChamferEIVLoss:
    """
    Create a Chamfer distance-based EIV loss.

    Args:
        model: Model function f(x) that predicts y
        sigma_x: Standard deviation of feature noise
        method: Method for finding closest point ('monte_carlo', 'optimization')
        n_samples: Number of Monte Carlo samples
        optim_steps: Number of optimization steps
        optim_lr: Learning rate for optimization
        early_stopping_tol: Tolerance for early stopping
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'

    Returns:
        ChamferEIVLoss instance
    """
    return ChamferEIVLoss(
        model=model,
        sigma_x=sigma_x,
        method=method,
        n_samples=n_samples,
        optim_steps=optim_steps,
        optim_lr=optim_lr,
        early_stopping_tol=early_stopping_tol,
        reduction=reduction,
    )


def create_hybrid_eiv_loss(
    model: Callable,
    sigma_x: Union[float, torch.Tensor],
    alpha: float = 0.5,
    chamfer_method: str = "monte_carlo",
    n_samples: int = 100,
    reduction: str = "mean",
    **kwargs,
) -> HybridEIVChamferLoss:
    """
    Create a hybrid EIV-Chamfer loss.

    Args:
        model: Model function f(x) that predicts y
        sigma_x: Standard deviation of feature noise
        alpha: Weight for EIV component (0-1)
        chamfer_method: Method for Chamfer component ('monte_carlo', 'optimization')
        n_samples: Number of Monte Carlo samples
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        **kwargs: Additional arguments for component losses

    Returns:
        HybridEIVChamferLoss instance
    """
    # Extract kwargs for each component
    eiv_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in ["method", "optim_steps", "optim_lr", "early_stopping_tol"]
    }
    chamfer_kwargs = {k: v for k, v in kwargs.items() if k not in ["monte_carlo"]}

    # Create component losses with 'none' reduction
    eiv_loss = FunctionalEIVLoss(model=model, sigma_x=sigma_x, reduction="none", **eiv_kwargs)

    chamfer_loss = ChamferEIVLoss(
        model=model,
        sigma_x=sigma_x,
        method=chamfer_method,
        n_samples=n_samples,
        reduction="none",
        **chamfer_kwargs,
    )

    return HybridEIVChamferLoss(
        eiv_loss=eiv_loss, chamfer_loss=chamfer_loss, alpha=alpha, reduction=reduction
    )


def create_mdn_eiv_loss(
    num_components: int,
    n_features_y: int,
    sigma_x: Union[float, torch.Tensor],
    sigma_y: Optional[Union[float, torch.Tensor]] = None,
    min_sigma: float = 1e-4,
    eps: float = 1e-8,
    uncertainty_method: str = "fixed",
    mc_samples: int = 100,
    reduction: str = "mean",
) -> MDNEIVLoss:
    """
    Create an EIV loss for Mixture Density Networks.

    Args:
        num_components: Number of mixture components in the MDN
        n_features_y: Dimensionality of target variable
        sigma_x: Standard deviation of feature noise
        sigma_y: Standard deviation of target noise (optional)
        min_sigma: Minimum value for standard deviation
        eps: Small value for numerical stability
        uncertainty_method: Method for uncertainty propagation ('fixed', 'gradient', 'monte_carlo')
        mc_samples: Number of Monte Carlo samples
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'

    Returns:
        MDNEIVLoss instance
    """
    return MDNEIVLoss(
        num_components=num_components,
        n_features=n_features_y,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        min_sigma=min_sigma,
        eps=eps,
        uncertainty_method=uncertainty_method,
        mc_samples=mc_samples,
        reduction=reduction,
    )


def create_mdn_model(
    input_size: int,
    hidden_layers: List[int],
    output_size: int = 1,
    num_components: int = 5,
    activation: str = "relu",
    dropout_rate: float = 0.0,
) -> MDNEIVModel:
    """
    Create a Mixture Density Network model with Error-in-Variables capabilities.

    Args:
        input_size: Input feature dimension
        hidden_layers: List of hidden layer sizes
        output_size: Dimensionality of output variable
        num_components: Number of mixture components
        activation: Activation function ('relu', 'tanh', 'sigmoid', 'leaky_relu')
        dropout_rate: Dropout probability (0 to disable)

    Returns:
        MDNEIVModel instance
    """
    # Map activation string to torch module
    activation_map = {
        "relu": torch.nn.ReLU(),
        "tanh": torch.nn.Tanh(),
        "sigmoid": torch.nn.Sigmoid(),
        "leaky_relu": torch.nn.LeakyReLU(0.1),
    }

    if activation not in activation_map:
        raise ValueError(
            f"Unsupported activation: {activation}. " f"Choose from {list(activation_map.keys())}"
        )

    return MDNEIVModel(
        input_size=input_size,
        hidden_layers=hidden_layers,
        output_size=output_size,
        num_components=num_components,
        activation=activation_map[activation],
        dropout_rate=dropout_rate,
    )


def create_robust_eiv_loss(
    model: Callable,
    sigma_x: Union[float, torch.Tensor] = 1.0,
    base_loss: str = "huber",
    delta: float = 1.0,
    variation_method: str = "gaussian",
    n_samples: int = 10,
    batch_size: Optional[int] = None,
    aggregation: str = "median",
    quantile: float = 0.95,
    reduction: str = "mean",
    **variation_params,
) -> RobustEIVLoss:
    """
    Create a robust EIV loss that uses multiple forward passes.

    Args:
        model: Model function f(x) that predicts y
        sigma_x: Standard deviation of feature noise
        base_loss: Base loss function ('huber', 'l1', or 'mse')
        delta: Delta parameter for Huber loss
        variation_method: Method for generating input variations
            ('gaussian', 'uniform', 'bootstrap', 'structured', 'adversarial')
        n_samples: Number of samples to generate
        batch_size: Batch size for processing variations
        aggregation: How to aggregate losses ('mean', 'median', 'max', or 'quantile')
        quantile: Quantile level for 'quantile' aggregation
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        **variation_params: Additional parameters for variation function

    Returns:
        RobustEIVLoss instance
    """
    # Select variation function based on variation_method
    variation_fn_map = {
        "gaussian": gaussian_variation,
        "uniform": uniform_variation,
        "bootstrap": bootstrap_variation,
        "structured": structured_variation,
        "adversarial": adversarial_variation,
    }

    if variation_method not in variation_fn_map:
        raise ValueError(
            f"Unsupported variation method: {variation_method}. "
            f"Choose from {list(variation_fn_map.keys())}"
        )

    variation_fn = variation_fn_map[variation_method]

    return RobustEIVLoss(
        model=model,
        sigma_x=sigma_x,
        base_loss=base_loss,
        delta=delta,
        variation_fn=variation_fn,
        n_samples=n_samples,
        batch_size=batch_size,
        aggregation=aggregation,
        quantile=quantile,
        reduction=reduction,
        variation_params=variation_params,
    )


def create_ensemble_eiv_loss(
    model: Callable,
    sigma_x: Union[float, torch.Tensor],
    n_samples: int = 20,
    perturb_method: str = "gaussian",
    reduction: str = "mean",
    eps: float = 1e-8,
) -> EnsembleEIVLoss:
    """
    Create a simple ensemble-based EIV loss.

    Args:
        model: Model function f(x) that predicts y
        sigma_x: Standard deviation of feature noise
        n_samples: Number of perturbed samples
        perturb_method: Method for perturbing inputs ('gaussian', 'uniform')
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        eps: Small value for numerical stability

    Returns:
        EnsembleEIVLoss instance
    """
    return EnsembleEIVLoss(
        model=model,
        sigma_x=sigma_x,
        n_samples=n_samples,
        perturb_method=perturb_method,
        reduction=reduction,
        eps=eps,
    )


def create_quantile_eiv_loss(
    model: Callable,
    sigma_x: Union[float, torch.Tensor],
    quantile: float = 0.5,
    n_samples: int = 20,
    perturb_method: str = "gaussian",
    reduction: str = "mean",
    eps: float = 1e-8,
) -> QuantileEIVLoss:
    """
    Create a Quantile Error-in-Variables loss.

    Args:
        model: Model function f(x) that predicts y
        sigma_x: Standard deviation of feature noise
        quantile: Quantile level to estimate (0 to 1)
        n_samples: Number of perturbed samples
        perturb_method: Method for perturbing inputs ('gaussian', 'uniform')
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        eps: Small value for numerical stability

    Returns:
        QuantileEIVLoss instance
    """
    return QuantileEIVLoss(
        model=model,
        sigma_x=sigma_x,
        quantile=quantile,
        n_samples=n_samples,
        perturb_method=perturb_method,
        reduction=reduction,
        eps=eps,
    )


def create_multi_quantile_eiv_loss(
    model: Callable,
    sigma_x: Union[float, torch.Tensor],
    quantiles: list = [0.1, 0.5, 0.9],
    n_samples: int = 20,
    perturb_method: str = "gaussian",
    reduction: str = "mean",
    eps: float = 1e-8,
) -> MultiQuantileEIVLoss:
    """
    Create a Multi-Quantile Error-in-Variables loss.

    Args:
        model: Model function f(x) that predicts y
        sigma_x: Standard deviation of feature noise
        quantiles: List of quantile levels to estimate
        n_samples: Number of perturbed samples
        perturb_method: Method for perturbing inputs ('gaussian', 'uniform')
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        eps: Small value for numerical stability

    Returns:
        MultiQuantileEIVLoss instance
    """
    return MultiQuantileEIVLoss(
        model=model,
        sigma_x=sigma_x,
        quantiles=quantiles,
        n_samples=n_samples,
        perturb_method=perturb_method,
        reduction=reduction,
        eps=eps,
    )
