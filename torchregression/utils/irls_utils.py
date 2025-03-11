"""
Utility functions for Iteratively Reweighted Least Squares (IRLS) algorithm.

This module provides helper functions for implementing IRLS,
including weighting functions, variance estimation, and data handling utilities.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, IterableDataset

import warnings
from typing import Optional, Tuple, List, Union, Iterator

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
    return torch.where(abs_res <= c, (1 - (scaled_residuals / c) ** 2) ** 2, torch.zeros_like(scaled_residuals))

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
    y_pred: torch.Tensor,
    covariance_matrices: Optional[torch.Tensor] = None,
    variance_type: str = 'predicted',
    loss_fn: Optional[nn.Module] = None
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
    if variance_type == 'predicted':
        # Handle covariance matrices case (full multivariate Gaussian)
        if covariance_matrices is not None:
            return torch.diagonal(covariance_matrices, dim1=-2, dim2=-1)
        
        # Handle DiagonalGaussianNLL case with learnable variances
        elif hasattr(loss_fn, 'log_variances'):
            variance = torch.exp(loss_fn.log_variances.data)  # Use .data to avoid gradient tracking
            return variance.unsqueeze(0).expand(residuals.shape[0], -1)
        
        # Handle heteroscedastic output case (mean and log_std outputs)
        elif isinstance(y_pred, tuple) and len(y_pred) == 2:
            _, log_std = y_pred
            return torch.exp(2 * log_std)
        
        # Handle heteroscedastic output case (concatenated outputs)
        elif y_pred.shape[-1] == 2 * residuals.shape[-1]:
            n_features = residuals.shape[-1]
            log_sigma = y_pred[..., n_features:]
            return torch.exp(2 * log_sigma)
        
        else:
            raise ValueError("Cannot determine predicted variance. Model output format not recognized.")

    elif variance_type == 'fixed':
        if not hasattr(loss_fn, 'fixed_variance'):
            raise ValueError("Fixed variance requested, but loss_fn has no 'fixed_variance' attribute.")
        variance = loss_fn.fixed_variance.to(residuals.device)
        return variance.expand_as(residuals) if variance.ndim < residuals.ndim else variance
        
    elif variance_type == 'robust':
        mad = calculate_mad(residuals)
        return (1.4826 * mad) ** 2  # Consistent estimator for Gaussian distribution
    
    else:
        raise ValueError(f"Invalid variance_type: {variance_type}. Must be 'predicted', 'fixed', or 'robust'.")

def extract_mean_and_residuals(
    y_pred: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    y_true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
    if isinstance(y_pred, tuple) and len(y_pred) == 2:
        mean, _ = y_pred
        return mean, y_true - mean
    
    # Handle heteroscedastic output case for bellshape-like loss
    elif y_pred.shape[-1] == 2 * y_true.shape[-1]:
        n_features = y_true.shape[-1]
        mean = y_pred[..., :n_features]
        return mean, y_true - mean
    
    # Standard case: direct prediction
    else:
        return y_pred, y_true - y_pred

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
    num_epochs: int
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
        return DataLoader(
            TensorDataset(x_train, y_train), 
            batch_size=len(x_train),
            shuffle=False
        ), False
    
    elif isinstance(train_data, IterableDataset):
        return DataLoader(
            train_data, 
            batch_size=batch_size
        ), True
    
    else:
        raise TypeError("train_data must be DataLoader, tuple of Tensors, or IterableDataset")

def setup_validation_loader(
    val_data: Union[DataLoader, Tuple[torch.Tensor, torch.Tensor]],
    device: Union[str, torch.device]
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
        return DataLoader(
            TensorDataset(x_val, y_val), 
            batch_size=len(x_val),
            shuffle=False
        )
    
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
    batch_data: Union[Tuple, List], 
    device: Union[str, torch.device]
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
    data_loader: DataLoader, 
    device: Union[str, torch.device]
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
    is_minibatch: bool
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
    model: nn.Module, 
    val_loader: DataLoader, 
    loss_fn: nn.Module, 
    device: Union[str, torch.device]
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
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch_data in val_loader:
            batch_x, batch_y, batch_cov, batch_mask = unpack_batch_data(batch_data, device)
            batch_size = batch_x.shape[0]
            
            # Forward pass
            y_pred = model(batch_x)
            
            # Calculate loss (without precision weighting for validation)
            if hasattr(loss_fn, 'forward'):
                if batch_cov is not None and 'covariance_matrices' in loss_fn.forward.__code__.co_varnames:
                    loss = loss_fn(y_pred, batch_y, covariance_matrices=batch_cov, mask=batch_mask)
                else:
                    loss = loss_fn(y_pred, batch_y, mask=batch_mask)
            else:
                # Fallback for standard PyTorch losses
                loss = loss_fn(y_pred, batch_y)
            
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    
    model.train()  # Set back to training mode
    return total_loss / max(1, total_samples)
