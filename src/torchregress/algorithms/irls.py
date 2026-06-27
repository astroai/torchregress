"""
Iteratively Reweighted Least Squares (IRLS) implementation.

This module provides implementations of IRLS for robust regression,
with support for various weighting schemes and loss functions.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn

from ..losses.base import (
    WeightedLossWrapper,
)
from ..losses.gaussian import (
    GaussianNLLLoss,
    MultivariateGaussianLoss,
)
from ..utils.validation import check_tensor


@dataclass(frozen=True)
class IRLSConfig:
    """
    Configuration options for Iteratively Reweighted Least Squares (IRLS) algorithm.

    Parameters
    ----------
    base_loss : str
        The base loss function to use ('gaussian', 'huber', or 'l1').
    max_iter : int
        Maximum number of IRLS iterations.
    tol : float
        Convergence tolerance for loss changes.
    delta : float
        Huber loss parameter delta.
    weight_fn : str | Callable
        Weighting function ('huber', 'tukey', 'power', or a callable).
    weight_params : dict[str, Any] | None
        Optional parameters for the weighting function.
    variance_type : str
        Variance estimation method ('predicted', 'fixed', or 'robust').
    epsilon : float
        Small constant for numerical stability.
    return_all_predictions : bool
        Whether to return intermediate predictions.
    batch_size : int
        Batch size for inference checks.
    """

    base_loss: str = "gaussian"
    max_iter: int = 10
    tol: float = 1e-4
    delta: float = 1.0
    weight_fn: str | Callable = "huber"
    weight_params: dict[str, Any] | None = None
    variance_type: str = "predicted"
    epsilon: float = 1e-8
    return_all_predictions: bool = False
    batch_size: int = 1024


# Get machine epsilon for numerical stability
EPS = torch.finfo(torch.float32).eps


# --- Weighting Functions ---
def huber_weights(scaled_residuals: torch.Tensor, delta: float) -> torch.Tensor:
    """
    Huber weighting function.

    Args:
        scaled_residuals: Residuals scaled by standard deviation
        delta: Threshold parameter

    Returns:
        Weight tensor with same shape as input
    """
    # Vectorized implementation for better GPU performance
    abs_res = torch.abs(scaled_residuals)
    return torch.where(abs_res <= delta, torch.ones_like(scaled_residuals), delta / (abs_res + EPS))


def tukey_weights(scaled_residuals: torch.Tensor, c: float) -> torch.Tensor:
    """
    Tukey's biweight weighting function.

    Args:
        scaled_residuals: Residuals scaled by standard deviation
        c: Tuning parameter (typically 4.685)

    Returns:
        Weight tensor with same shape as input
    """
    abs_res = torch.abs(scaled_residuals)
    return torch.where(
        abs_res <= c, (1 - (scaled_residuals / c) ** 2) ** 2, torch.zeros_like(scaled_residuals)
    )


def power_weights(scaled_residuals: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """
    Power-law weighting function (generalization of DAOPHOT-like weighting).

    Args:
        scaled_residuals: Residuals scaled by standard deviation
        a: Scale parameter
        b: Power parameter

    Returns:
        Weight tensor with same shape as input
    """
    return 1.0 / (1.0 + (torch.abs(scaled_residuals) / a) ** b)


def calculate_mad(residuals: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Calculates the Median Absolute Deviation (MAD) along the specified dimension.

    Args:
        residuals: Residual tensor
        dim: Dimension along which to calculate MAD

    Returns:
        MAD tensor
    """
    median = torch.median(residuals, dim=dim, keepdim=True)[0]
    return torch.median(torch.abs(residuals - median), dim=dim, keepdim=True)[0]


# --- Variance Estimation Functions ---
def estimate_variance(
    residuals: torch.Tensor,
    y_pred: torch.Tensor | tuple[torch.Tensor, ...],
    covariance_matrices: torch.Tensor | None = None,
    variance_type: str = "predicted",
    loss_fn: nn.Module | None = None,
) -> torch.Tensor:
    """
    Estimates the variance of the residuals based on the specified method.

    Args:
        residuals: The residuals tensor
        y_pred: Model predictions, which may include variance components
        covariance_matrices: Optional covariance matrices for multivariate Gaussian
        variance_type: One of 'predicted', 'fixed', or 'robust'
        loss_fn: Loss function instance that may contain variance information

    Returns:
        variance: Estimated variance tensor with same shape as residuals
    """
    # Optimized implementation for better GPU utilization
    if variance_type == "predicted":
        # Handle covariance matrices case (full multivariate Gaussian)
        if covariance_matrices is not None:
            return torch.diagonal(covariance_matrices, dim1=-2, dim2=-1)

        # Handle DiagonalGaussianNLL case with learnable variances
        elif loss_fn is not None and hasattr(loss_fn, "log_variances"):
            log_variances = cast(torch.Tensor, getattr(loss_fn, "log_variances"))
            variance = torch.exp(log_variances.data)  # Use .data to avoid gradient tracking
            if variance.device != residuals.device:
                variance = variance.to(residuals.device)
            return variance.unsqueeze(0).expand(residuals.shape[0], -1)

        # Handle heteroscedastic output case (mean and log_std outputs)
        elif isinstance(y_pred, tuple) and len(y_pred) == 2:
            _, log_std = y_pred
            return torch.exp(2 * log_std)

        # Handle heteroscedastic output case (concatenated outputs)
        elif isinstance(y_pred, torch.Tensor) and y_pred.shape[-1] == 2 * residuals.shape[-1]:
            n_features = residuals.shape[-1]
            log_sigma = y_pred[..., n_features:]
            return torch.exp(2 * log_sigma)

        else:
            raise ValueError(
                "Cannot determine predicted variance. Model output format not recognized."
            )

    elif variance_type == "fixed":
        if loss_fn is None or not hasattr(loss_fn, "fixed_variance"):
            raise ValueError(
                "Fixed variance requested, but loss_fn has no 'fixed_variance' attribute."
            )
        fixed_variance = cast(torch.Tensor, getattr(loss_fn, "fixed_variance"))
        variance = fixed_variance.to(residuals.device)
        return variance.expand_as(residuals) if variance.ndim < residuals.ndim else variance

    elif variance_type == "robust":
        mad = calculate_mad(residuals)
        return (1.4826 * mad) ** 2  # Consistent estimator for Gaussian distribution

    else:
        raise ValueError(
            f"Invalid variance_type: {variance_type}. Must be 'predicted', 'fixed', or 'robust'."
        )


def extract_mean_and_residuals(
    y_pred: torch.Tensor | tuple[torch.Tensor, ...], y_true: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts mean predictions and calculates residuals based on model output format.

    Args:
        y_pred: Model predictions (can be tensor or tuple)
        y_true: Ground truth values

    Returns:
        mean: Mean predictions
        residuals: Residuals (y_true - mean)
    """
    # Handle tuple output case (mean, log_std)
    if isinstance(y_pred, tuple) and len(y_pred) >= 2:
        mean, _ = y_pred
        return mean, y_true - mean

    # Handle heteroscedastic output case for bellshape-like loss
    elif isinstance(y_pred, torch.Tensor) and y_pred.shape[-1] == 2 * y_true.shape[-1]:
        n_features = y_true.shape[-1]
        mean = y_pred[..., :n_features]
        return mean, y_true - mean

    # Standard case: direct prediction
    else:
        y_pred_t = cast(torch.Tensor, y_pred)
        return y_pred_t, y_true - y_pred_t


def _setup_irls(  # noqa: PLR0913
    model: nn.Module,
    x: torch.Tensor,
    y_true: torch.Tensor,
    covariance_matrices: torch.Tensor | None,
    config: IRLSConfig,
) -> tuple[nn.Module, nn.Module, Callable, dict[str, Any]]:
    """Helper function to set up IRLS components."""
    # Loss Function Setup
    loss_fn: nn.Module
    if config.base_loss == "gaussian":
        loss_fn = (
            MultivariateGaussianLoss()
            if covariance_matrices is not None
            else GaussianNLLLoss(fixed_variance=1.0)
        )
    elif config.base_loss == "huber":
        loss_fn = WeightedLossWrapper(nn.HuberLoss, delta=config.delta)
    elif config.base_loss == "l1":
        loss_fn = WeightedLossWrapper(nn.L1Loss)
    else:
        raise ValueError(
            f"Invalid base_loss: {config.base_loss}. Must be 'gaussian', 'huber', or 'l1'."
        )

    # Weight Function Setup
    weight_params = config.weight_params or {}
    _weight_fn: Callable[..., torch.Tensor]
    if isinstance(config.weight_fn, str):
        if config.weight_fn == "huber":
            _weight_fn = huber_weights
            weight_params = {"delta": config.delta, **weight_params}
        elif config.weight_fn == "tukey":
            _weight_fn = tukey_weights
            weight_params = {"c": 4.685, **weight_params}
        elif config.weight_fn == "power":
            _weight_fn = power_weights
            weight_params = {"a": 1.0, "b": 2.0, **weight_params}
        else:
            raise ValueError(
                f"Invalid weight_fn: {config.weight_fn}. Must be 'huber', 'tukey', or 'power'."
            )
    elif callable(config.weight_fn):
        _weight_fn = config.weight_fn
    else:
        raise TypeError("weight_fn must be a string or a callable")

    return model, loss_fn, _weight_fn, weight_params


def _compute_irls_loss(  # noqa: PLR0913
    y_pred: torch.Tensor | tuple[torch.Tensor, ...],
    y_true: torch.Tensor,
    precision: torch.Tensor,
    loss_fn: nn.Module,
    covariance_matrices: torch.Tensor | None,
    mask: torch.Tensor | None,
    config: IRLSConfig,
) -> torch.Tensor:
    if config.base_loss == "gaussian":
        return loss_fn(
            y_pred=y_pred,
            target=y_true,
            covariance_matrices=covariance_matrices,
            mask=mask,
            weights=precision,
        )
    else:
        return loss_fn(y_pred=y_pred, target=y_true, mask=mask, weights=precision)


def _update_precision(  # noqa: PLR0913
    residuals: torch.Tensor,
    y_pred: torch.Tensor | tuple[torch.Tensor, ...],
    precision: torch.Tensor,
    loss_fn: nn.Module,
    _weight_fn: Callable,
    weight_params: dict[str, Any],
    covariance_matrices: torch.Tensor | None,
    config: IRLSConfig,
) -> torch.Tensor:
    variance = estimate_variance(
        residuals, y_pred, covariance_matrices, config.variance_type, loss_fn
    )
    scaled_residuals = residuals / (torch.sqrt(variance) + config.epsilon)
    iter_weights = _weight_fn(scaled_residuals, **weight_params)
    return precision * iter_weights


def _perform_irls_iteration(  # noqa: PLR0913
    y_pred: torch.Tensor | tuple[torch.Tensor, ...],
    residuals: torch.Tensor,
    y_true: torch.Tensor,
    precision: torch.Tensor,
    loss_fn: nn.Module,
    _weight_fn: Callable,
    weight_params: dict[str, Any],
    covariance_matrices: torch.Tensor | None,
    mask: torch.Tensor | None,
    iteration: int,
    all_predictions: list[torch.Tensor] | None,
    config: IRLSConfig,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor] | None]:
    """Helper function to perform a single IRLS iteration."""
    # Note: y_pred is now passed in, not computed from model(x)

    with torch.no_grad():
        if config.return_all_predictions and all_predictions is not None:
            all_predictions.append(y_pred[0] if isinstance(y_pred, tuple) else y_pred)

        loss_value = _compute_irls_loss(
            y_pred, y_true, precision, loss_fn, covariance_matrices, mask, config
        )

        precision = _update_precision(
            residuals,
            y_pred,
            precision,
            loss_fn,
            _weight_fn,
            weight_params,
            covariance_matrices,
            config,
        )

    return precision, loss_value, all_predictions


def _batched_predict(
    model: nn.Module,
    x: torch.Tensor,
    batch_size: int = 1024,
    device: str | torch.device | None = None,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """
    Predicts in batches to avoid OOM.

    Args:
        model: The model to use for prediction
        x: Input tensor (can be on CPU or GPU)
        batch_size: Batch size for inference
        device: Target device for output (defaults to x.device)

    Returns:
        Prediction tensor(s) on target device
    """
    model_device = next(model.parameters()).device
    target_device = device if device is not None else x.device

    # If x fits in memory or is already on model device, just run
    # Note: We rely on batch_size to decide if we should split,
    # but here we force batching if x is not on model device to be safe,
    # or just respect batch_size.

    num_samples = x.shape[0]

    # Simple case: if x is on correct device and small enough, or if we don't want to batch
    if x.device == model_device and num_samples <= batch_size:
        with torch.no_grad():
            pred = model(x)
            if isinstance(pred, tuple):
                return tuple(p.to(target_device) for p in pred)
            return cast(torch.Tensor, pred).to(target_device)

    # Batched inference
    batch_preds: list[torch.Tensor | tuple[torch.Tensor, ...]] = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)

            batch_x = x[start_idx:end_idx].to(model_device)
            batch_pred = model(batch_x)

            # Handle tuple output
            if isinstance(batch_pred, tuple):
                batch_preds.append(tuple(p.to(target_device) for p in batch_pred))
            else:
                batch_preds.append(batch_pred.to(target_device))

    # Concatenate results
    if not batch_preds:
        return cast(torch.Tensor, torch.tensor([]).to(target_device))

    if isinstance(batch_preds[0], tuple):
        # Using zip(*batch_preds) to transpose and then cat column-wise
        return tuple(torch.cat(column, dim=0) for column in zip(*batch_preds))
    else:
        # Direct cat of the list of tensors for single-output models
        return torch.cat(cast(list[torch.Tensor], batch_preds), dim=0)


def _run_irls_loop(  # noqa: PLR0913
    y_pred: torch.Tensor | tuple[torch.Tensor, ...],
    residuals: torch.Tensor,
    y_true: torch.Tensor,
    precision: torch.Tensor,
    loss_fn: nn.Module,
    _weight_fn: Callable,
    weight_params: dict[str, Any],
    covariance_matrices: torch.Tensor | None,
    mask: torch.Tensor | None,
    all_predictions: list[torch.Tensor] | None,
    config: IRLSConfig,
) -> tuple[torch.Tensor, list[float], list[torch.Tensor] | None]:
    """Executes the main IRLS iteration loop."""
    loss_history: list[float] = []

    for iteration in range(config.max_iter):
        precision, loss_tensor, all_predictions = _perform_irls_iteration(
            y_pred,
            residuals,
            y_true,
            precision,
            loss_fn,
            _weight_fn,
            weight_params,
            covariance_matrices,
            mask,
            iteration,
            all_predictions,
            config,
        )
        # Deferring .item() call to here allows GPU to execute subsequent operations
        # (variance estimation, weight calculation, precision update) which were
        # queued in _perform_irls_iteration, while CPU waits for the loss value.
        loss_value = loss_tensor.item()
        loss_history.append(loss_value)

        if iteration > 0 and abs(loss_history[-1] - loss_history[-2]) < config.tol:
            break

    return precision, loss_history, all_predictions


def iteratively_reweighted_least_squares(
    model: nn.Module,
    x: torch.Tensor,
    y_true: torch.Tensor,
    initial_precision: torch.Tensor | None = None,
    covariance_matrices: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    config: IRLSConfig | None = None,
    **kwargs: Any,
) -> (
    tuple[torch.Tensor, list[float], torch.Tensor]
    | tuple[torch.Tensor, list[float], torch.Tensor, list[torch.Tensor]]
):
    """
    Applies iteratively reweighted least squares (IRLS) for robust regression.

    This function is a performance-optimized implementation that supports PyTorch's
    latest features, including `torch.compile`. The core logic is broken down into
    helper functions for clarity and maintainability.

    Args:
        model: PyTorch model
        x: Input data (batch_size, n_features_x)
        y_true: Target data (batch_size, n_features_y)
        initial_precision: Initial precision (inverse variance) (batch_size, n_features_y)
        covariance_matrices: Covariance matrices for multivariate Gaussian
        mask: Optional mask for ignoring certain values
        base_loss: Base loss function: 'gaussian', 'huber', or 'l1'
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        delta: Delta parameter for Huber loss
        weight_fn: Weighting function: 'huber', 'tukey', 'power', or a callable
        weight_params: Parameters for the weighting function
        variance_type: Variance estimation method: 'predicted', 'fixed', or 'robust'
        epsilon: Small value for numerical stability
        return_all_predictions: Whether to return predictions from all iterations
        batch_size: Batch size for inference (default: 1024)

    Returns:
        y_pred: Final predicted values
        loss_history: List of loss values over iterations
        final_precision: Final precision tensor
        [optional] all_predictions: List of predictions from all iterations

    References
    ----------
    .. [1] Beaton, A. E., & Tukey, J. W. (1974). The Fitting of Power Series,
       Meaning Polynomials, Illustrated on Band-Spectroscopic Data.
       In *Technometrics*, 16(2), 147-185.
       https://doi.org/10.1080/00401706.1974.10489171
    """
    if config is None:
        config = IRLSConfig(**kwargs)

    check_tensor(x, "x")
    check_tensor(y_true, "y_true")
    if initial_precision is not None:
        check_tensor(initial_precision, "initial_precision")

    # x might be on CPU. We keep it there if so.
    x = x.detach()
    device = x.device

    model, loss_fn, _weight_fn, weight_params = _setup_irls(  # noqa: E501
        model, x, y_true, covariance_matrices, config
    )

    if initial_precision is None:
        precision = torch.ones_like(y_true)
    else:
        if initial_precision.shape != y_true.shape:
            raise ValueError(
                f"initial_precision shape {initial_precision.shape} must match "
                f"y_true shape {y_true.shape}"
            )
        precision = initial_precision.clone().detach().to(device)

    all_predictions: list[torch.Tensor] | None = [] if config.return_all_predictions else None

    # --- Precompute Predictions and Residuals ---
    # This avoids running the model repeatedly in the loop, which is redundant
    # since model weights are not updated within this function.
    # We use batched inference to avoid OOM if x is large.
    y_pred = _batched_predict(model, x, batch_size=config.batch_size, device=device)

    # Compute residuals once
    _, residuals = extract_mean_and_residuals(y_pred, y_true)

    # --- IRLS Iterations ---
    precision, loss_history, all_predictions = _run_irls_loop(
        y_pred,
        residuals,
        y_true,
        precision,
        loss_fn,
        _weight_fn,
        weight_params,
        covariance_matrices,
        mask,
        all_predictions,
        config,
    )

    final_y_pred = cast(torch.Tensor, y_pred)  # Public API contract returns Tensor

    if config.return_all_predictions:
        assert all_predictions is not None
        return final_y_pred, loss_history, precision, all_predictions
    else:
        return final_y_pred, loss_history, precision
