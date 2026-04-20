"""
Iteratively Reweighted Least Squares (IRLS) implementation.

This module provides implementations of IRLS for robust regression,
with support for various weighting schemes and loss functions.
"""

import warnings
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Tuple, Union, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

from ..losses.base import (
    WeightedLossWrapper,
)
from ..losses.gaussian import (
    GaussianNLLLoss,
    LowRankGaussianLoss,
    MultivariateGaussianLoss,
)
from ..losses.robust import TukeyBiweightLoss
from ..utils.validation import check_tensor


class CallbackFn(Protocol):
    def __call__(
        self,
        *,
        iteration: int,
        model: nn.Module,
        y_pred: torch.Tensor,
        mean: torch.Tensor,
        residuals: torch.Tensor,
        precision: torch.Tensor,
        loss: float,
        **kwargs: Any,
    ) -> None: ...


# Get machine epsilon for numerical stability
EPS = torch.finfo(torch.float32).eps
PredictionOutput = Union[torch.Tensor, Tuple[torch.Tensor, ...]]


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
    return cast(torch.Tensor, 1.0 / (1.0 + (torch.abs(scaled_residuals) / a) ** b))


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
    y_pred: PredictionOutput,
    covariance_matrices: Optional[torch.Tensor] = None,
    variance_type: str = "predicted",
    loss_fn: Optional[nn.Module] = None,
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
    y_pred: PredictionOutput, y_true: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        y_pred_tensor = cast(torch.Tensor, y_pred)
        return y_pred_tensor, y_true - y_pred_tensor


# --- Data Handling Utilities ---
def parse_update_frequency(update_weights: str) -> Tuple[str, int]:
    """
    Parses the update_weights string to determine update type and frequency.

    Args:
        update_weights: String specifying update method ("epoch", "batch", or "iter:N")

    Returns:
        tuple: (update_type, update_frequency)
    """
    if update_weights == "epoch":
        return "epoch", 1
    elif update_weights == "batch":
        return "batch", 1
    elif update_weights.startswith("iter:"):
        try:
            freq = int(update_weights.split(":")[1])
            if freq <= 0:
                raise ValueError("Iteration frequency must be positive")
            return "iter", freq
        except (ValueError, IndexError):
            raise ValueError("Invalid update_weights format. Use 'epoch', 'batch', or 'iter:N'")
    else:
        raise ValueError("Invalid update_weights. Use 'epoch', 'batch', or 'iter:N'")


def setup_data_loader(
    train_data: Union[DataLoader, Tuple[torch.Tensor, torch.Tensor], IterableDataset],
    device: Union[str, torch.device],
    batch_size: int,
    num_epochs: int,
) -> Tuple[DataLoader, bool]:
    """
    Set up data loader based on the provided training data.

    Args:
        train_data: The training data
        device: Device to use
        batch_size: Batch size for DataLoader
        num_epochs: Number of epochs

    Returns:
        tuple: (data_loader, is_minibatch)
    """
    if isinstance(train_data, DataLoader):
        return train_data, True

    elif isinstance(train_data, tuple) and len(train_data) == 2:
        x_train, y_train = train_data
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        # Use full batch for non-minibatch training
        return (
            DataLoader(TensorDataset(x_train, y_train), batch_size=len(x_train), shuffle=False),
            False,
        )

    elif isinstance(train_data, IterableDataset):
        return DataLoader(train_data, batch_size=batch_size), True

    else:
        raise TypeError("train_data must be DataLoader, tuple of Tensors, or IterableDataset")


def setup_validation_loader(
    val_data: Union[DataLoader, Tuple[torch.Tensor, torch.Tensor]], device: Union[str, torch.device]
) -> DataLoader:
    """
    Set up validation data loader.

    Args:
        val_data: Validation data
        device: Device to use

    Returns:
        DataLoader: Validation data loader
    """
    if isinstance(val_data, DataLoader):
        return val_data

    elif isinstance(val_data, tuple) and len(val_data) == 2:
        x_val, y_val = val_data
        x_val = x_val.to(device)
        y_val = y_val.to(device)
        return DataLoader(TensorDataset(x_val, y_val), batch_size=len(x_val), shuffle=False)

    else:
        raise TypeError("val_data must be DataLoader or tuple of Tensors")


def iterate_batches(data_loader: DataLoader) -> Iterator:
    """
    Safely iterate through batches with proper error handling.

    Args:
        data_loader: DataLoader to iterate

    Yields:
        Batch data
    """
    try:
        yield from data_loader
    except Exception as e:
        warnings.warn(f"Error during batch iteration: {str(e)}")
        yield from []


def unpack_batch_data(
    batch_data: Union[Tuple, List], device: Union[str, torch.device]
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Unpacks batch data and moves tensors to the specified device.

    Args:
        batch_data: Batch data from DataLoader
        device: Device to move tensors to

    Returns:
        tuple: (batch_x, batch_y, batch_cov, batch_mask)
    """
    if not batch_data:
        raise ValueError("Empty batch data received")

    if len(batch_data) >= 2:
        batch_x, batch_y = batch_data[0].to(device), batch_data[1].to(device)

        # Handle optional covariance matrices and masks
        batch_cov = batch_data[2].to(device) if len(batch_data) > 2 else None
        batch_mask = batch_data[3].to(device) if len(batch_data) > 3 else None

        return batch_x, batch_y, batch_cov, batch_mask
    else:
        raise ValueError("Batch data must contain at least input and target tensors")


def buffer_data(
    data_loader: DataLoader, device: Union[str, torch.device]
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Buffer a full epoch of data from a DataLoader.

    Args:
        data_loader: DataLoader to buffer
        device: Device to store tensors on

    Returns:
        tuple: (all_x, all_y_true, all_cov, all_masks)
    """
    all_x, all_y_true = [], []
    all_cov, all_masks = [], []

    try:
        for batch_data in data_loader:
            batch_x, batch_y, batch_cov, batch_mask = unpack_batch_data(batch_data, device)

            all_x.append(batch_x)
            all_y_true.append(batch_y)

            if batch_cov is not None:
                all_cov.append(batch_cov)
            if batch_mask is not None:
                all_masks.append(batch_mask)
    except Exception as e:
        warnings.warn(f"Error during data buffering: {str(e)}")

    # Handle empty case
    if not all_x:
        return torch.tensor([]), torch.tensor([]), None, None

    # Concatenate results
    x_tensor = torch.cat(all_x, dim=0)
    y_tensor = torch.cat(all_y_true, dim=0)

    cov_tensor = torch.cat(all_cov, dim=0) if all_cov else None
    mask_tensor = torch.cat(all_masks, dim=0) if all_masks else None

    return x_tensor, y_tensor, cov_tensor, mask_tensor


def get_batch_precision(
    previous_epoch_precision: Optional[torch.Tensor],
    batch_idx: int,
    batch_size: int,
    batch_size_current: int,
    all_x: Optional[torch.Tensor],
    batch_y: torch.Tensor,
    is_minibatch: bool,
) -> torch.Tensor:
    """
    Get precision tensor for the current batch.

    Args:
        previous_epoch_precision: Overall precision tensor
        batch_idx: Current batch index
        batch_size: Nominal batch size
        batch_size_current: Actual current batch size
        all_x: Buffered input data
        batch_y: Current batch targets
        is_minibatch: Whether using minibatch training

    Returns:
        torch.Tensor: Precision for current batch
    """
    if previous_epoch_precision is None:
        return torch.ones_like(batch_y)

    if is_minibatch and all_x is not None:
        start_idx = (batch_idx * batch_size) % len(all_x)
        end_idx = min(start_idx + batch_size_current, len(all_x))
        return previous_epoch_precision[start_idx:end_idx]
    else:
        return previous_epoch_precision


def validate_model(
    model: nn.Module, val_loader: DataLoader, loss_fn: nn.Module, device: Union[str, torch.device]
) -> float:
    """
    Evaluate model on validation set.

    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device for computation

    Returns:
        float: Average validation loss
    """
    model.eval()
    losses = []
    total_samples = 0

    with torch.no_grad():
        for batch_data in val_loader:
            batch_x, batch_y, batch_cov, batch_mask = unpack_batch_data(batch_data, device)
            batch_size = batch_x.shape[0]

            # Forward pass
            y_pred = model(batch_x)

            # Calculate loss (without precision weighting for validation)
            if hasattr(loss_fn, "forward"):
                if (
                    batch_cov is not None
                    and "covariance_matrices" in loss_fn.forward.__code__.co_varnames
                ):
                    loss = loss_fn(y_pred, batch_y, covariance_matrices=batch_cov, mask=batch_mask)
                else:
                    loss = loss_fn(y_pred, batch_y, mask=batch_mask)
            else:
                # Fallback for standard PyTorch losses
                loss = loss_fn(y_pred, batch_y)

            losses.append(loss * batch_size)
            total_samples += batch_size

    model.train()  # Set back to training mode

    if not losses:
        return 0.0

    return float(torch.sum(torch.stack(losses)).item() / max(1, total_samples))


def _setup_irls(
    model: nn.Module,
    x: torch.Tensor,
    y_true: torch.Tensor,
    base_loss: str,
    weight_fn: Union[str, Callable],
    delta: float,
    weight_params: Optional[Dict[str, Any]],
    covariance_matrices: Optional[torch.Tensor] = None,
) -> Tuple[nn.Module, nn.Module, Callable, Dict[str, Any]]:
    """Helper function to set up IRLS components."""
    # Loss Function Setup
    loss_fn: nn.Module
    if base_loss == "gaussian":
        loss_fn = (
            MultivariateGaussianLoss()
            if covariance_matrices is not None
            else GaussianNLLLoss(fixed_variance=1.0)
        )
    elif base_loss == "huber":
        loss_fn = WeightedLossWrapper(nn.HuberLoss, delta=delta)
    elif base_loss == "l1":
        loss_fn = WeightedLossWrapper(nn.L1Loss)
    else:
        raise ValueError(f"Invalid base_loss: {base_loss}. Must be 'gaussian', 'huber', or 'l1'.")

    # Weight Function Setup
    weight_params = weight_params or {}
    _weight_fn: Callable[..., torch.Tensor]
    if isinstance(weight_fn, str):
        if weight_fn == "huber":
            _weight_fn = huber_weights
            weight_params = {"delta": delta, **weight_params}
        elif weight_fn == "tukey":
            _weight_fn = tukey_weights
            weight_params = {"c": 4.685, **weight_params}
        elif weight_fn == "power":
            _weight_fn = power_weights
            weight_params = {"a": 1.0, "b": 2.0, **weight_params}
        else:
            raise ValueError(
                f"Invalid weight_fn: {weight_fn}. Must be 'huber', 'tukey', or 'power'."
            )
    elif callable(weight_fn):
        _weight_fn = weight_fn
    else:
        raise TypeError("weight_fn must be a string or a callable")

    return model, loss_fn, _weight_fn, weight_params


def _record_predictions(
    y_pred: PredictionOutput,
    return_all_predictions: bool,
    all_predictions: Optional[List[torch.Tensor]],
) -> Optional[List[torch.Tensor]]:
    if return_all_predictions and all_predictions is not None:
        if isinstance(y_pred, tuple):
            all_predictions.append(y_pred[0])
        else:
            all_predictions.append(y_pred)
    return all_predictions


def _compute_irls_loss(
    y_pred: PredictionOutput,
    y_true: torch.Tensor,
    precision: torch.Tensor,
    loss_fn: nn.Module,
    base_loss: str,
    covariance_matrices: Optional[torch.Tensor],
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if base_loss == "gaussian":
        return cast(
            torch.Tensor,
            loss_fn(
                y_pred=y_pred,
                target=y_true,
                covariance_matrices=covariance_matrices,
                mask=mask,
            ),
        )
    else:
        return cast(
            torch.Tensor,
            loss_fn(y_pred=y_pred, target=y_true, mask=mask, weights=precision),
        )


def _update_precision(
    residuals: torch.Tensor,
    y_pred: PredictionOutput,
    precision: torch.Tensor,
    loss_fn: nn.Module,
    _weight_fn: Callable,
    weight_params: Dict[str, Any],
    variance_type: str,
    covariance_matrices: Optional[torch.Tensor],
) -> torch.Tensor:
    variance = estimate_variance(residuals, y_pred, covariance_matrices, variance_type, loss_fn)
    scaled_residuals = residuals / (torch.sqrt(variance) + EPS)
    iter_weights = _weight_fn(scaled_residuals, **weight_params)
    return precision * iter_weights


def _perform_irls_iteration(
    y_pred: PredictionOutput,
    residuals: torch.Tensor,
    y_true: torch.Tensor,
    precision: torch.Tensor,
    loss_fn: nn.Module,
    _weight_fn: Callable,
    weight_params: Dict[str, Any],
    base_loss: str,
    variance_type: str,
    covariance_matrices: Optional[torch.Tensor],
    mask: Optional[torch.Tensor],
    iteration: int,
    return_all_predictions: bool,
    all_predictions: Optional[List[torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
    """Helper function to perform a single IRLS iteration."""
    # Note: y_pred is now passed in, not computed from model(x)

    with torch.no_grad():
        all_predictions = _record_predictions(y_pred, return_all_predictions, all_predictions)

        loss_value = _compute_irls_loss(
            y_pred, y_true, precision, loss_fn, base_loss, covariance_matrices, mask
        )

        precision = _update_precision(
            residuals,
            y_pred,
            precision,
            loss_fn,
            _weight_fn,
            weight_params,
            variance_type,
            covariance_matrices,
        )

    return precision, loss_value, all_predictions


def _batched_predict(
    model: nn.Module,
    x: torch.Tensor,
    batch_size: int = 1024,
    device: Optional[Union[str, torch.device]] = None,
) -> PredictionOutput:
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
    batch_preds: list[PredictionOutput] = []
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
        return torch.cat(cast(List[torch.Tensor], batch_preds), dim=0)


def iteratively_reweighted_least_squares(
    model: nn.Module,
    x: torch.Tensor,
    y_true: torch.Tensor,
    initial_precision: torch.Tensor | None = None,
    covariance_matrices: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    base_loss: str = "gaussian",
    max_iter: int = 10,
    tol: float = 1e-4,
    delta: float = 1.0,
    weight_fn: str | Callable = "huber",
    weight_params: Dict[str, Any] | None = None,
    variance_type: str = "predicted",
    epsilon: float = EPS,
    return_all_predictions: bool = False,
    batch_size: int = 1024,
) -> (
    Tuple[torch.Tensor, List[float], torch.Tensor]
    | Tuple[torch.Tensor, List[float], torch.Tensor, List[torch.Tensor]]
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
    """
    # Validate inputs
    check_tensor(x, "x")
    check_tensor(y_true, "y_true")
    if initial_precision is not None:
        check_tensor(initial_precision, "initial_precision")

    # x might be on CPU. We keep it there if so.
    x = x.detach()  # No clone needed if we don't modify in place, but safer?
    # Actually, we don't need clone if we are careful. But to be safe vs side effects:
    # x = x.clone() # Maybe skip clone to save memory if x is large?
    # The original code did clone. Let's trust user not to modify x in place.

    device = x.device

    model, loss_fn, _weight_fn, weight_params = _setup_irls(
        model,
        x,
        y_true,
        base_loss,
        weight_fn,
        delta,
        weight_params,
        covariance_matrices,
    )

    # --- Initial Precision ---
    if initial_precision is None:
        precision = torch.ones_like(y_true)  # Initialize equal precision weights
    else:
        if initial_precision.shape != y_true.shape:
            raise ValueError(
                f"initial_precision shape {initial_precision.shape} must match "
                f"y_true shape {y_true.shape}"
            )
        precision = initial_precision.clone().detach().to(device)

    loss_history: List[float] = []
    all_predictions: Optional[List[torch.Tensor]] = [] if return_all_predictions else None

    # --- Precompute Predictions and Residuals ---
    # This avoids running the model repeatedly in the loop, which is redundant
    # since model weights are not updated within this function.
    # We use batched inference to avoid OOM if x is large.
    y_pred = _batched_predict(model, x, batch_size=batch_size, device=device)

    # Compute residuals once
    _, residuals = extract_mean_and_residuals(y_pred, y_true)

    iter_range = range(max_iter)

    # --- IRLS Iterations ---
    for iteration in iter_range:
        precision, loss_tensor, all_predictions = _perform_irls_iteration(
            y_pred,
            residuals,
            y_true,
            precision,
            loss_fn,
            _weight_fn,
            weight_params,
            base_loss,
            variance_type,
            covariance_matrices,
            mask,
            iteration,
            return_all_predictions,
            all_predictions,
        )
        # Deferring .item() call to here allows GPU to execute subsequent operations
        # (variance estimation, weight calculation, precision update) which were
        # queued in _perform_irls_iteration, while CPU waits for the loss value.
        loss_value = loss_tensor.item()
        loss_history.append(loss_value)

        if iteration > 0 and abs(loss_history[-1] - loss_history[-2]) < tol:
            break

    final_y_pred = cast(torch.Tensor, y_pred)  # Public API contract returns Tensor

    if return_all_predictions:
        assert all_predictions is not None
        return final_y_pred, loss_history, precision, all_predictions
    else:
        return final_y_pred, loss_history, precision


def calculate_loss(
    loss_fn: nn.Module,
    y_pred: PredictionOutput,
    y_true: torch.Tensor,
    precision: Optional[torch.Tensor] = None,
    covariance_matrices: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculate loss based on the loss function type and model outputs.

    Args:
        loss_fn: Loss function module
        y_pred: Model predictions
        y_true: Ground truth values
        precision: Precision weights (for weighted loss)
        covariance_matrices: Covariance matrices (for Gaussian models)
        mask: Optional mask

    Returns:
        torch.Tensor: Computed loss
    """
    # Handle tuple output (mu, log_sigma)
    if isinstance(y_pred, tuple) and len(y_pred) == 2:
        mu, log_sigma = y_pred
        if isinstance(loss_fn, GaussianNLLLoss):
            return cast(torch.Tensor, loss_fn(y_pred=(mu, log_sigma), target=y_true, mask=mask))
        else:
            # For other losses, just use the mean prediction
            return cast(
                torch.Tensor,
                loss_fn(y_pred=mu, target=y_true, mask=mask, weights=precision),
            )

    # Handle MultivariateGaussianLoss
    elif isinstance(loss_fn, MultivariateGaussianLoss):
        return cast(
            torch.Tensor,
            loss_fn(
                y_pred=y_pred,
                target=y_true,
                covariance_matrices=covariance_matrices,
                mask=mask,
            ),
        )
    # Handle LowRankGaussianLoss
    elif isinstance(loss_fn, LowRankGaussianLoss):
        if (
            covariance_matrices is None
            or not isinstance(covariance_matrices, (tuple, list))
            or len(covariance_matrices) != 2
        ):
            raise ValueError(
                "LowRankGaussianLoss requires covariance_matrices=(cov_factor, cov_diag)."
            )
        cov_factor, cov_diag = covariance_matrices
        return cast(
            torch.Tensor,
            loss_fn(
                y_pred=y_pred,
                target=y_true,
                cov_factor=cov_factor,
                cov_diag=cov_diag,
                mask=mask,
            ),
        )

    # Handle robust losses with weights
    elif (
        isinstance(loss_fn, WeightedLossWrapper) or isinstance(loss_fn, TukeyBiweightLoss)
    ) and precision is not None:
        return cast(
            torch.Tensor,
            loss_fn(y_pred=y_pred, target=y_true, mask=mask, weights=precision),
        )

    # Standard case
    else:
        return cast(torch.Tensor, loss_fn(y_pred=y_pred, target=y_true, mask=mask))


def _setup_irls_loss(
    loss_fn: Optional[nn.Module],
    base_loss: Optional[str],
    covariance_matrices: Optional[torch.Tensor],
    delta: float,
) -> Tuple[str, nn.Module]:
    """Helper to determine base_loss type and set up loss_fn for IRLS if not provided."""
    if base_loss is None:
        if isinstance(loss_fn, (GaussianNLLLoss, MultivariateGaussianLoss)):
            base_loss = "gaussian"
        elif isinstance(loss_fn, WeightedLossWrapper) and isinstance(
            loss_fn.torch_loss, nn.HuberLoss
        ):
            base_loss = "huber"
        elif isinstance(loss_fn, WeightedLossWrapper) and isinstance(loss_fn.torch_loss, nn.L1Loss):
            base_loss = "l1"
        else:
            base_loss = "gaussian"
            cls_name = getattr(loss_fn, "__class__", type(None)).__name__
            warnings.warn(
                f"Could not determine base_loss type from {cls_name}. "
                f"Using default 'gaussian'. Specify base_loss explicitly if needed."
            )

    if loss_fn is None:
        if base_loss == "gaussian":
            loss_fn = (
                MultivariateGaussianLoss()
                if covariance_matrices is not None
                else GaussianNLLLoss(fixed_variance=1.0)
            )
        elif base_loss == "huber":
            loss_fn = WeightedLossWrapper(nn.HuberLoss, delta=delta)
        elif base_loss == "l1":
            loss_fn = WeightedLossWrapper(nn.L1Loss)
        else:
            raise ValueError(f"Unsupported base_loss for IRLS training API: {base_loss}")

    return base_loss, loss_fn


def _perform_epoch_reweighting(
    model: nn.Module,
    data_loader: DataLoader,
    is_minibatch: bool,
    device: Union[str, torch.device],
    all_x: Optional[torch.Tensor],
    all_y_true: Optional[torch.Tensor],
    all_cov: Optional[torch.Tensor],
    all_masks: Optional[torch.Tensor],
    covariance_matrices: Optional[torch.Tensor],
    mask: Optional[torch.Tensor],
    previous_epoch_precision: Optional[torch.Tensor],
    base_loss: str,
    irls_max_iter: int,
    irls_tol: float,
    delta: float,
    weight_fn: Union[str, Callable],
    weight_params: Optional[Dict[str, Any]],
    variance_type: str,
    epsilon: float,
    return_all_iterations: bool,
    batch_size: int,
    epoch: int,
    all_iterations_data: Optional[List[Dict[str, Any]]],
    train_loss_history: List[float],
) -> torch.Tensor:
    """Helper to perform epoch-level IRLS reweighting."""
    x_for_irls = all_x if is_minibatch else next(iter(data_loader))[0].to(device)
    y_for_irls = all_y_true if is_minibatch else next(iter(data_loader))[1].to(device)
    cov_for_irls = all_cov if is_minibatch else covariance_matrices
    mask_for_irls = all_masks if is_minibatch else mask
    assert x_for_irls is not None
    assert y_for_irls is not None

    with torch.no_grad():
        irls_result = iteratively_reweighted_least_squares(
            model=model,
            x=x_for_irls,
            y_true=y_for_irls,
            initial_precision=previous_epoch_precision,
            covariance_matrices=cov_for_irls,
            mask=mask_for_irls,
            base_loss=base_loss,
            max_iter=irls_max_iter,
            tol=irls_tol,
            delta=delta,
            weight_fn=weight_fn,
            weight_params=weight_params,
            variance_type=variance_type,
            epsilon=epsilon,
            return_all_predictions=return_all_iterations,
            batch_size=batch_size,
        )
        if return_all_iterations:
            (
                _,
                epoch_loss_history,
                new_precision,
                epoch_iterations,
            ) = cast(
                Tuple[torch.Tensor, List[float], torch.Tensor, List[torch.Tensor]],
                irls_result,
            )
            if all_iterations_data is not None:
                all_iterations_data.append({"epoch": epoch, "iterations": epoch_iterations})
        else:
            _, epoch_loss_history, new_precision = cast(
                Tuple[torch.Tensor, List[float], torch.Tensor],
                irls_result,
            )

    if not is_minibatch:
        train_loss_history.extend(epoch_loss_history)

    return new_precision


def _perform_batch_reweighting(
    model: nn.Module,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    batch_cov: Optional[torch.Tensor],
    batch_mask: Optional[torch.Tensor],
    previous_epoch_precision: Optional[torch.Tensor],
    base_loss: str,
    irls_max_iter: int,
    irls_tol: float,
    delta: float,
    weight_fn: Union[str, Callable],
    weight_params: Optional[Dict[str, Any]],
    variance_type: str,
    epsilon: float,
    return_all_iterations: bool,
    epoch: int,
    batch_idx: int,
    all_iterations_data: Optional[List[Dict[str, Any]]],
    is_minibatch: bool,
    all_x: Optional[torch.Tensor],
    batch_size: int,
    batch_size_current: int,
    device: Union[str, torch.device],
) -> torch.Tensor:
    """Helper to perform batch-level IRLS reweighting and update global precision."""
    with torch.no_grad():
        irls_result = iteratively_reweighted_least_squares(
            model=model,
            x=batch_x,
            y_true=batch_y,
            initial_precision=previous_epoch_precision,
            covariance_matrices=batch_cov,
            mask=batch_mask,
            base_loss=base_loss,
            max_iter=irls_max_iter,
            tol=irls_tol,
            delta=delta,
            weight_fn=weight_fn,
            weight_params=weight_params,
            variance_type=variance_type,
            epsilon=epsilon,
            return_all_predictions=return_all_iterations,
        )
        if return_all_iterations:
            _, _, batch_precision, batch_iterations = cast(
                Tuple[torch.Tensor, List[float], torch.Tensor, List[torch.Tensor]],
                irls_result,
            )
            if all_iterations_data is not None:
                all_iterations_data.append(
                    {"epoch": epoch, "batch": batch_idx, "iterations": batch_iterations}
                )
        else:
            _, _, batch_precision = cast(
                Tuple[torch.Tensor, List[float], torch.Tensor],
                irls_result,
            )

        if previous_epoch_precision is not None and is_minibatch and all_x is not None:
            start_idx = (batch_idx * batch_size) % all_x.shape[0]
            end_idx = min(start_idx + batch_size_current, all_x.shape[0])

            if previous_epoch_precision is None:
                previous_epoch_precision = torch.ones_like(
                    batch_y
                    if all_x is None
                    else torch.zeros((all_x.shape[0], batch_y.shape[1]), device=device)
                )

            if start_idx < end_idx:
                previous_epoch_precision[start_idx:end_idx] = batch_precision
            return previous_epoch_precision
        else:
            return batch_precision


def _train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    batch_precision: Optional[torch.Tensor],
    batch_cov: Optional[torch.Tensor],
    batch_mask: Optional[torch.Tensor],
    clip_grad_norm: Optional[float],
) -> float:
    """Helper for a single training step."""
    optimizer.zero_grad()
    y_pred = model(batch_x)
    loss = calculate_loss(loss_fn, y_pred, batch_y, batch_precision, batch_cov, batch_mask)
    loss.backward()

    if clip_grad_norm is not None and clip_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

    optimizer.step()
    return float(loss.item())


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
    update_weights: str = "epoch",  # "epoch", "batch", or "iter:N"
    val_data: Optional[Union[DataLoader, Tuple[torch.Tensor, torch.Tensor]]] = None,
    val_freq: int = 1,
    clip_grad_norm: Optional[float] = None,
    base_loss: Optional[str] = None,
    use_compile: bool = False,
    compile_kwargs: Optional[Dict[str, Any]] = None,
    callbacks: Optional[List[Callable]] = None,
    return_all_iterations: bool = False,
    verbose_epoch_freq: int = 1,
    verbose_batch_freq: int = 5,
    epsilon: float = EPS,
) -> Dict[str, Any]:
    warnings.warn(
        "The IRLS function is deprecated and will be removed in a future version. "
        "Please use the iteratively_reweighted_least_squares function in your own training loop.",
        DeprecationWarning,
        stacklevel=2,
    )
    """
    Trains a PyTorch model using Iteratively Reweighted Least Squares (IRLS).

    User-friendly implementation supporting common PyTorch training workflows.

    Args:
        model: The PyTorch model to train
        train_data: Training data as DataLoader or (x, y) tensor tuple
        loss_fn: Optional loss function (inferred from base_loss if not provided)
        optimizer: Optional optimizer (Adam used by default if not provided)
        num_epochs: Number of training epochs
        device: Device to use ('cpu', 'cuda', etc.)
        batch_size: Batch size for training
        irls_max_iter: Maximum iterations for IRLS per reweighting
        irls_tol: Convergence tolerance for IRLS
        delta: Huber loss delta parameter
        weight_fn: Weight function ('huber', 'tukey', 'power', or callable)
        weight_params: Parameters for the weighting function
        variance_type: Variance estimation method ('predicted', 'fixed', 'robust')
        initial_precision: Initial precision weights
        mask: Optional mask for ignoring values
        covariance_matrices: Optional covariance matrices
        verbose: Whether to print progress information
        progress_bar: Show progress bars using tqdm (if installed)
        update_weights: When to update IRLS weights ('epoch', 'batch', or 'iter:N')
        val_data: Optional validation data
        val_freq: Validation frequency (epochs)
        clip_grad_norm: Optional gradient clipping
        base_loss: Base loss type ('gaussian', 'huber', 'l1')
        use_compile: Whether to use torch.compile for speedup (PyTorch 2.0+)
        compile_kwargs: Additional kwargs for torch.compile
        callbacks: Optional list of callbacks for monitoring training

    Returns:
        Dictionary containing trained model and training history
    """
    # Create optimizer if not provided
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())
        if verbose:
            print("No optimizer provided. Using default Adam optimizer with learning_rate=0.001")

    model.to(device)
    model.train()
    previous_epoch_precision = initial_precision
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    all_iterations_data: Optional[list[dict[str, Any]]] = [] if return_all_iterations else None

    # --- Determine base loss type and setup loss function ---
    base_loss, loss_fn = _setup_irls_loss(loss_fn, base_loss, covariance_matrices, delta)
    assert optimizer is not None

    # --- Parse update_weights ---
    update_type, update_freq = parse_update_frequency(update_weights)

    # --- Data Handling ---
    data_loader, is_minibatch = setup_data_loader(train_data, device, batch_size, num_epochs)

    # --- Validation Data Setup ---
    val_loader = None
    if val_data is not None:
        val_loader = setup_validation_loader(val_data, device)

    # Initialize global step counter for batch/iter updates
    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0

        # --- Print epoch start if verbose ---
        should_print_epoch = verbose and (
            epoch % verbose_epoch_freq == 0 or epoch == num_epochs - 1
        )
        if should_print_epoch:
            print(f"Epoch {epoch + 1}/{num_epochs} started")

        # --- Buffer data for epoch-level IRLS if needed ---
        all_x = all_y_true = all_cov = all_masks = None
        if is_minibatch and update_type == "epoch":
            # OPTIMIZATION: Buffer data to CPU to avoid GPU OOM for large datasets
            buffer_device = "cpu"
            all_x, all_y_true, all_cov, all_masks = buffer_data(data_loader, buffer_device)
            # Skip epoch if no data (possible with IterableDataset)
            if len(all_x) == 0:
                if verbose:
                    print(f"Epoch {epoch + 1}: No data available, skipping.")
                continue

        # --- Perform IRLS reweighting at epoch level if required ---
        do_epoch_reweight = update_type == "epoch" and epoch % update_freq == 0
        if do_epoch_reweight:
            if should_print_epoch:
                print(f"Epoch {epoch + 1}: Performing IRLS reweighting with {base_loss} base loss")

            previous_epoch_precision = _perform_epoch_reweighting(
                model=model,
                data_loader=data_loader,
                is_minibatch=is_minibatch,
                device=device,
                all_x=all_x,
                all_y_true=all_y_true,
                all_cov=all_cov,
                all_masks=all_masks,
                covariance_matrices=covariance_matrices,
                mask=mask,
                previous_epoch_precision=previous_epoch_precision,
                base_loss=base_loss,
                irls_max_iter=irls_max_iter,
                irls_tol=irls_tol,
                delta=delta,
                weight_fn=weight_fn,
                weight_params=weight_params,
                variance_type=variance_type,
                epsilon=epsilon,
                return_all_iterations=return_all_iterations,
                batch_size=batch_size,
                epoch=epoch,
                all_iterations_data=all_iterations_data,
                train_loss_history=train_loss_history,
            )

        # --- Training Loop ---
        for i, batch_data in enumerate(iterate_batches(data_loader)):
            batch_x, batch_y, batch_cov, batch_mask = unpack_batch_data(batch_data, device)
            batch_size_current = batch_x.shape[0]

            # --- Batch-level IRLS reweighting if requested ---
            if update_type == "batch" and global_step % update_freq == 0:
                should_print_batch = verbose and (
                    i % (len(data_loader) // verbose_batch_freq + 1) == 0
                )
                if should_print_batch:
                    print(
                        f"Epoch {epoch + 1}, Batch {i + 1}/{len(data_loader)}: "
                        f"Performing IRLS reweighting with {base_loss} base loss"
                    )

                previous_epoch_precision = _perform_batch_reweighting(
                    model=model,
                    batch_x=batch_x,
                    batch_y=batch_y,
                    batch_cov=batch_cov,
                    batch_mask=batch_mask,
                    previous_epoch_precision=previous_epoch_precision,
                    base_loss=base_loss,
                    irls_max_iter=irls_max_iter,
                    irls_tol=irls_tol,
                    delta=delta,
                    weight_fn=weight_fn,
                    weight_params=weight_params,
                    variance_type=variance_type,
                    epsilon=epsilon,
                    return_all_iterations=return_all_iterations,
                    epoch=epoch,
                    batch_idx=i,
                    all_iterations_data=all_iterations_data,
                    is_minibatch=is_minibatch,
                    all_x=all_x,
                    batch_size=batch_size,
                    batch_size_current=batch_size_current,
                    device=device,
                )

            # --- Standard optimization step ---
            batch_precision = get_batch_precision(
                previous_epoch_precision,
                i,
                batch_size,
                batch_size_current,
                all_x,
                batch_y,
                is_minibatch,
            )

            loss_item = _train_step(
                model,
                optimizer,
                loss_fn,
                batch_x,
                batch_y,
                batch_precision,
                batch_cov,
                batch_mask,
                clip_grad_norm,
            )

            # Track loss
            epoch_loss += loss_item * batch_size_current
            batch_count += batch_size_current
            global_step += 1

        # --- Epoch summary ---
        if batch_count > 0:  # Ensure we processed at least one batch
            avg_epoch_loss = epoch_loss / batch_count
            train_loss_history.append(avg_epoch_loss)

            if should_print_epoch:
                print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_epoch_loss:.6f}")

        # --- Validation ---
        epoch_val_loss: Optional[float] = None
        if val_loader is not None and (epoch + 1) % val_freq == 0:
            val_loss = validate_model(model, val_loader, loss_fn, device)
            val_loss_history.append(val_loss)
            epoch_val_loss = val_loss
            if should_print_epoch:
                print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.6f}")

        # --- Callbacks ---
        if callbacks:
            callback_payload = {
                "epoch": epoch,
                "model": model,
                "train_loss": train_loss_history[-1] if train_loss_history else None,
                "val_loss": epoch_val_loss,
                "global_step": global_step,
                "update_type": update_type,
                "return_all_iterations": return_all_iterations,
            }
            for callback in callbacks:
                callback(**callback_payload)

    # Final summary
    if verbose:
        print(f"Training completed. Final train loss: {train_loss_history[-1]:.6f}")
        if val_loss_history:
            print(f"Final validation loss: {val_loss_history[-1]:.6f}")

    # Return a dictionary with training results
    result = {
        "model": model,
        "train_loss_history": train_loss_history,
        "final_precision": previous_epoch_precision,
    }

    if val_loss_history:
        result["val_loss_history"] = val_loss_history

    if all_iterations_data:
        result["all_iterations"] = all_iterations_data

    return result
