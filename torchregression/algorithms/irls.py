import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, IterableDataset

import warnings
from typing import Callable, Optional, Dict, Tuple, List, Union, Any, Iterator

from .gaussian import DiagonalGaussianNLL, GaussianNLLWithCovariance
from .robust import HuberLoss, L1Loss

# --- Constants ---
# Get machine epsilon for numerical stability
EPS = torch.finfo(torch.float32).eps

# --- Weighting Functions ---
def _huber_weights(scaled_residuals: torch.Tensor, delta: float) -> torch.Tensor:
    """Huber weighting function."""
    abs_res = torch.abs(scaled_residuals)
    return torch.where(abs_res <= delta, torch.ones_like(scaled_residuals), delta / (abs_res + EPS))

def _tukey_weights(scaled_residuals: torch.Tensor, c: float) -> torch.Tensor:
    """Tukey's biweight weighting function."""
    abs_res = torch.abs(scaled_residuals)
    return torch.where(abs_res <= c, (1 - (scaled_residuals / c) ** 2) ** 2, torch.zeros_like(scaled_residuals))

def _power_weights(scaled_residuals: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """Power-law weighting function (generalization of DAOPHOT-like weighting)."""
    return 1.0 / (1.0 + (torch.abs(scaled_residuals) / a) ** b)

def _calculate_mad(residuals: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Calculates the Median Absolute Deviation (MAD) along the specified dimension."""
    median = torch.median(residuals, dim=dim, keepdim=True)[0]
    return torch.median(torch.abs(residuals - median), dim=dim, keepdim=True)[0]

# --- Variance Estimation Functions ---
def _estimate_variance(
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
    if variance_type == 'predicted':
        # Handle covariance matrices case (full multivariate Gaussian)
        if covariance_matrices is not None:
            return torch.diagonal(covariance_matrices, dim1=-2, dim2=-1)
        
        # Handle DiagonalGaussianNLL case with learnable variances
        elif hasattr(loss_fn, 'log_variances'):
            variance = torch.exp(loss_fn.log_variances.data * 2)  # Use .data to avoid gradient tracking
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
        mad = _calculate_mad(residuals)
        return (1.4826 * mad) ** 2  # Consistent estimator for Gaussian distribution
    
    else:
        raise ValueError(f"Invalid variance_type: {variance_type}. Must be 'predicted', 'fixed', or 'robust'.")

def _extract_mean_and_residuals(
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

# --- Core IRLS Function ---
def iteratively_reweighted_least_squares(
    model: nn.Module,
    x: torch.Tensor,
    y_true: torch.Tensor,
    initial_precision: Optional[torch.Tensor] = None,
    covariance_matrices: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    base_loss: str = 'gaussian',
    max_iter: int = 10,
    tol: float = 1e-4,
    delta: float = 1.0,
    weight_fn: Union[str, Callable] = 'huber',
    weight_params: Optional[Dict[str, Any]] = None,
    variance_type: str = 'predicted',
    epsilon: float = EPS,
    return_all_predictions: bool = False,
) -> Union[Tuple[torch.Tensor, List[float], torch.Tensor], 
           Tuple[torch.Tensor, List[float], torch.Tensor, List[torch.Tensor]]]:
    """
    Applies iteratively reweighted least squares (IRLS) for robust regression.
    
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
        
    Returns:
        y_pred: Final predicted values
        loss_history: List of loss values over iterations
        final_precision: Final precision tensor
        [optional] all_predictions: List of predictions from all iterations
    """
    x = x.detach().clone()  # Don't modify original input
    device = x.device
    weight_params = {} if weight_params is None else weight_params
    
    # --- Loss Function Setup ---
    if base_loss == 'gaussian':
        if covariance_matrices is None:
            loss_fn = DiagonalGaussianNLL(y_true.shape[-1])
        else:
            loss_fn = GaussianNLLWithCovariance()
    elif base_loss == 'huber':
        loss_fn = HuberLoss(delta=delta)
    elif base_loss == 'l1':
        loss_fn = L1Loss()
    else:
        raise ValueError(f"Invalid base_loss: {base_loss}. Must be 'gaussian', 'huber', or 'l1'.")

    # --- Weight Function Setup ---
    if isinstance(weight_fn, str):
        if weight_fn == 'huber':
            _weight_fn = _huber_weights
            weight_params = {'delta': delta, **weight_params}
        elif weight_fn == 'tukey':
            _weight_fn = _tukey_weights
            weight_params = {'c': 4.685, **weight_params}  # Default robust against 95% efficiency for Gaussian
        elif weight_fn == 'power':
            _weight_fn = _power_weights
            weight_params = {'a': 1.0, 'b': 2.0, **weight_params}
        else:
            raise ValueError(f"Invalid weight_fn: {weight_fn}. Must be 'huber', 'tukey', or 'power'.")
    elif callable(weight_fn):
        _weight_fn = weight_fn
    else:
        raise TypeError("weight_fn must be a string or a callable")

    # --- Initial Precision ---
    if initial_precision is None:
        precision = torch.ones_like(y_true)  # Initialize equal precision weights
    else:
        if initial_precision.shape != y_true.shape:
            raise ValueError(f"initial_precision shape {initial_precision.shape} must match y_true shape {y_true.shape}")
        precision = initial_precision.clone().detach().to(device)

    loss_history = []
    all_predictions = [] if return_all_predictions else None

    # --- IRLS Iterations ---
    for iteration in range(max_iter):
        # Forward pass
        y_pred = model(x)
        if return_all_predictions:
            all_predictions.append(y_pred)
            
        # Extract mean predictions and calculate residuals
        mean, residuals = _extract_mean_and_residuals(y_pred, y_true)
        
        # Calculate current loss
        if base_loss == 'gaussian':
            if covariance_matrices is not None:
                current_loss = loss_fn(y_true, y_pred, covariance_matrices, mask)
            else:
                current_loss = loss_fn(y_true, y_pred, mask=mask)
        else:  # huber or l1
            current_loss = loss_fn(y_true, y_pred, mask=mask, weights=precision)
            
        loss_history.append(current_loss.item())
        
        # Check for early convergence
        if iteration > 0 and abs(loss_history[-1] - loss_history[-2]) < tol:
            break
            
        # Calculate weights (no gradients needed)
        with torch.no_grad():
            # Estimate variance based on current residuals
            variance = _estimate_variance(
                residuals, y_pred, covariance_matrices, variance_type, loss_fn
            )
            
            # Scale residuals for weighting
            scaled_residuals = residuals / (torch.sqrt(variance) + epsilon)
            
            # Calculate weights using the chosen weighting function
            iter_weights = _weight_fn(scaled_residuals, **weight_params)
            
            # Update precision weights
            precision = precision * iter_weights
            
            # Update model's variance parameters if using learnable variance
            if base_loss == 'gaussian' and isinstance(loss_fn, DiagonalGaussianNLL):
                # More stable approach: directly compute optimal log variances
                weighted_variance = (variance / (precision + epsilon)).mean(dim=0)
                loss_fn.log_variances.data = 0.5 * torch.log(weighted_variance + epsilon)
    
    final_y_pred = model(x) if iteration < max_iter - 1 else y_pred
    
    if return_all_predictions:
        return final_y_pred, loss_history, precision, all_predictions
    else:
        return final_y_pred, loss_history, precision
    


def IRLS(
    model: nn.Module,
    train_data: Union[DataLoader, Tuple[torch.Tensor, torch.Tensor], IterableDataset],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 1,
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 32,
    irls_max_iter: int = 10,
    irls_tol: float = 1e-4,
    delta: float = 1.0,
    weight_fn: Union[str, Callable] = 'huber',
    weight_params: Optional[Dict[str, Any]] = None,
    variance_type: str = 'predicted',
    initial_precision: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    covariance_matrices: Optional[torch.Tensor] = None,
    verbose: bool = True,
    verbose_epoch_freq: int = 10,   # New parameter for epoch output frequency
    verbose_batch_freq: int = 5,    # New parameter for batch output frequency
    update_weights: str = "epoch",  # "epoch", "batch", or "iter:N"
    val_data: Optional[Union[DataLoader, Tuple[torch.Tensor, torch.Tensor]]] = None,
    val_freq: int = 1,              # New parameter for validation frequency
    epsilon: float = EPS,
    return_all_iterations: bool = False,
    clip_grad_norm: Optional[float] = None,
    base_loss: Optional[str] = None, # New parameter for base loss type
) -> Dict[str, Any]:
    """
    Trains a PyTorch model using Iteratively Reweighted Least Squares (IRLS).

    Supports mini-batch, full-batch, and streaming data, with flexible
    weight update frequencies and optional validation set monitoring.

    Args:
        model: The PyTorch model to train.
        train_data: Training data. Can be:
            - A DataLoader (for mini-batch training).
            - A tuple of (x_train, y_train) Tensors (for full-batch training).
            - An IterableDataset (for streaming data).
        loss_fn: The instance of the loss function.
        optimizer: The PyTorch optimizer.
        num_epochs: Number of training epochs. Default is 1.
        device: 'cpu' or 'cuda'.
        batch_size: Batch size (used for DataLoader creation with IterableDataset).
        irls_max_iter: Maximum number of IRLS iterations per reweighting.
        irls_tol: IRLS convergence tolerance.
        delta: Delta parameter for Huber loss.
        weight_fn: Weighting function ('huber', 'tukey', 'power', or a callable).
        weight_params: Parameters for the weighting function.
        variance_type: 'predicted', 'fixed', or 'robust'.
        initial_precision: Initial precision tensor (optional).
        mask: Optional mask for ignoring certain values.
        covariance_matrices: Optional covariance matrices.
        verbose: Whether to print progress.
        verbose_epoch_freq: Print epoch info every N epochs when verbose=True.
        verbose_batch_freq: Print batch info every N batches when verbose=True.
        update_weights: How often to update the IRLS weights:
            - "epoch": Once per epoch (recommended for mini-batch and full-batch).
            - "batch": Every mini-batch.
            - "iter:N": Every N inner IRLS iterations.
        val_data: Validation data. Can be a DataLoader or tuple of tensors.
        val_freq: Run validation every N epochs.
        epsilon: Small value for numerical stability.
        return_all_iterations: Whether to return predictions from all IRLS iterations.
        clip_grad_norm: Optional maximum norm for gradient clipping.
        base_loss: Override for the base loss type in IRLS ('gaussian', 'huber', or 'l1').
                  If None, it will be inferred from loss_fn.

    Returns:
        A dictionary containing:
            - "model": The trained model.
            - "train_loss_history": List of training losses.
            - "val_loss_history": List of validation losses (if val_data provided).
            - "final_precision": The final precision tensor from IRLS.
            - "all_iterations": All iteration results (if return_all_iterations=True).
    """

    model.to(device)
    model.train()
    previous_epoch_precision = initial_precision
    train_loss_history = []
    val_loss_history = []
    all_iterations_data = [] if return_all_iterations else None
    
    # --- Determine base loss type ---
    if base_loss is None:
        # Infer base loss type from loss_fn
        if isinstance(loss_fn, (DiagonalGaussianNLL, GaussianNLLWithCovariance)):
            base_loss = 'gaussian'
        elif isinstance(loss_fn, HuberLoss):
            base_loss = 'huber'
        elif isinstance(loss_fn, L1Loss):
            base_loss = 'l1'
        else:
            # Default to gaussian if we can't determine
            base_loss = 'gaussian'
            warnings.warn(f"Could not determine base_loss type from {loss_fn.__class__.__name__}. "
                          f"Using default 'gaussian'. Specify base_loss explicitly if needed.")

    # --- Parse update_weights ---
    update_type, update_freq = _parse_update_frequency(update_weights)

    # --- Data Handling ---
    data_loader, is_minibatch = _setup_data_loader(
        train_data, device, batch_size, num_epochs
    )

    # --- Validation Data Setup ---
    val_loader = None
    if val_data is not None:
        val_loader = _setup_validation_loader(val_data, device)

    # Initialize global step counter for batch/iter updates
    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0

        # --- Print epoch start if verbose ---
        should_print_epoch = verbose and (epoch % verbose_epoch_freq == 0 or epoch == num_epochs - 1)
        if should_print_epoch:
            print(f"Epoch {epoch+1}/{num_epochs} started")

        # --- Buffer data for epoch-level IRLS if needed ---
        all_x = all_y_true = all_cov = all_masks = None
        if is_minibatch and update_type == "epoch":
            all_x, all_y_true, all_cov, all_masks = _buffer_data(data_loader, device)
            # Skip epoch if no data (possible with IterableDataset)
            if len(all_x) == 0:
                if verbose:
                    print(f"Epoch {epoch+1}: No data available, skipping.")
                continue

        # --- Perform IRLS reweighting at epoch level if required ---
        do_epoch_reweight = (update_type == "epoch" and epoch % update_freq == 0)
        if do_epoch_reweight:
            if should_print_epoch:
                print(f"Epoch {epoch+1}: Performing IRLS reweighting with {base_loss} base loss")
            
            # Use buffered data for full-dataset IRLS
            x_for_irls = all_x if is_minibatch else next(iter(data_loader))[0].to(device)
            y_for_irls = all_y_true if is_minibatch else next(iter(data_loader))[1].to(device)
            cov_for_irls = all_cov if is_minibatch else covariance_matrices
            mask_for_irls = all_masks if is_minibatch else mask
            
            with torch.no_grad():
                irls_kwargs = {
                    "model": model,
                    "x": x_for_irls,
                    "y_true": y_for_irls,
                    "initial_precision": previous_epoch_precision,
                    "covariance_matrices": cov_for_irls,
                    "mask": mask_for_irls,
                    "base_loss": base_loss,  # Use determined or user-provided base_loss
                    "max_iter": irls_max_iter,
                    "tol": irls_tol,
                    "delta": delta,
                    "weight_fn": weight_fn,
                    "weight_params": weight_params,
                    "variance_type": variance_type,
                    "epsilon": epsilon,
                    "return_all_predictions": return_all_iterations  # Match parameter name
                }
                
                if return_all_iterations:
                    _, epoch_loss_history, previous_epoch_precision, epoch_iterations = iteratively_reweighted_least_squares(**irls_kwargs)
                    if all_iterations_data is not None:
                        all_iterations_data.append({
                            "epoch": epoch,
                            "iterations": epoch_iterations
                        })
                else:
                    _, epoch_loss_history, previous_epoch_precision = iteratively_reweighted_least_squares(**irls_kwargs)
            
            if not is_minibatch:
                train_loss_history.extend(epoch_loss_history)

        # --- Training Loop ---
        for i, batch_data in enumerate(_iterate_batches(data_loader)):
            batch_x, batch_y, batch_cov, batch_mask = _unpack_batch_data(batch_data, device)
            batch_size_current = batch_x.shape[0]

            # --- Batch-level IRLS reweighting if requested ---
            if update_type == "batch" and global_step % update_freq == 0:
                # Determine if we should print batch reweighting info
                should_print_batch = verbose and (i % (len(data_loader) // verbose_batch_freq + 1) == 0)
                
                if should_print_batch:
                    print(f"Epoch {epoch+1}, Batch {i+1}/{len(data_loader)}: "
                          f"Performing IRLS reweighting with {base_loss} base loss")
                
                with torch.no_grad():
                    irls_kwargs = {
                        "model": model,
                        "x": batch_x,
                        "y_true": batch_y,
                        "initial_precision": previous_epoch_precision,
                        "covariance_matrices": batch_cov,
                        "mask": batch_mask,
                        "base_loss": base_loss,  # Use determined or user-provided base_loss
                        "max_iter": irls_max_iter,
                        "tol": irls_tol,
                        "delta": delta,
                        "weight_fn": weight_fn,
                        "weight_params": weight_params,
                        "variance_type": variance_type,
                        "epsilon": epsilon,
                        "return_all_predictions": return_all_iterations  # Match parameter name
                    }
                    
                    if return_all_iterations:
                        _, batch_loss_history, batch_precision, batch_iterations = iteratively_reweighted_least_squares(**irls_kwargs)
                        if all_iterations_data is not None:
                            all_iterations_data.append({
                                "epoch": epoch,
                                "batch": i,
                                "iterations": batch_iterations
                            })
                    else:
                        _, batch_loss_history, batch_precision = iteratively_reweighted_least_squares(**irls_kwargs)
                    
                    # If we're doing batch-level updates, update the previous_epoch_precision
                    # but only for the current batch indices
                    if previous_epoch_precision is not None and is_minibatch:
                        # Extract indices for current batch
                        start_idx = (i * batch_size) % (all_x.shape[0] if all_x is not None else 0)
                        end_idx = min(start_idx + batch_size_current, 
                                     all_x.shape[0] if all_x is not None else 0)
                        
                        # Create a new precision tensor if it doesn't exist yet
                        if previous_epoch_precision is None:
                            previous_epoch_precision = torch.ones_like(
                                batch_y if all_x is None else 
                                torch.zeros((all_x.shape[0], batch_y.shape[1]), device=device)
                            )
                        
                        # Update just the relevant portion
                        if all_x is not None and start_idx < end_idx:
                            previous_epoch_precision[start_idx:end_idx] = batch_precision
                    else:
                        previous_epoch_precision = batch_precision

            # --- Standard optimization step ---
            optimizer.zero_grad()
            y_pred = model(batch_x)

            # --- Get precision for current batch ---
            batch_precision = _get_batch_precision(
                previous_epoch_precision, 
                i, batch_size, batch_size_current, 
                all_x, batch_y, is_minibatch
            )

            # --- Calculate loss ---
            loss = _calculate_loss(
                loss_fn, y_pred, batch_y, batch_precision, batch_cov, batch_mask
            )

            # --- Backward and optimization step ---
            loss.backward()
            
            # Apply gradient clipping if specified
            if clip_grad_norm is not None and clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                
            optimizer.step()
            
            # Track loss
            epoch_loss += loss.item() * batch_size_current
            batch_count += batch_size_current
            global_step += 1

        # --- Epoch summary ---
        if batch_count > 0:  # Ensure we processed at least one batch
            avg_epoch_loss = epoch_loss / batch_count
            train_loss_history.append(avg_epoch_loss)
            
            if should_print_epoch:
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_epoch_loss:.6f}")

        # --- Validation ---
        if val_loader is not None and (epoch + 1) % val_freq == 0:
            val_loss = _validate(model, val_loader, loss_fn, device)
            val_loss_history.append(val_loss)
            if should_print_epoch:
                print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.6f}")

    # Final summary
    if verbose:
        print(f"Training completed. Final train loss: {train_loss_history[-1]:.6f}")
        if val_loss_history:
            print(f"Final validation loss: {val_loss_history[-1]:.6f}")

    # Return a dictionary with training results
    result = {
        "model": model,
        "train_loss_history": train_loss_history,
        "final_precision": previous_epoch_precision
    }
    
    if val_loss_history:
        result["val_loss_history"] = val_loss_history
    
    if all_iterations_data:
        result["all_iterations"] = all_iterations_data
        
    return result


# --- Helper Functions ---

def _parse_update_frequency(update_weights: str) -> Tuple[str, int]:
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


def _setup_data_loader(
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


def _setup_validation_loader(
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


def _iterate_batches(data_loader: DataLoader) -> Iterator:
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


def _unpack_batch_data(
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


def _get_batch_precision(
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


def _calculate_loss(
    loss_fn: nn.Module,
    y_pred: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    y_true: torch.Tensor,
    precision: Optional[torch.Tensor] = None,
    covariance_matrices: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None
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
        if isinstance(loss_fn, DiagonalGaussianNLL):
            # For DiagonalGaussianNLL, we concatenate mean and log_std
            y_pred_combined = torch.cat((mu, log_sigma), dim=-1)
            return loss_fn(y_true=y_true, y_pred=y_pred_combined, mask=mask)
        else:
            # For other losses, just use the mean prediction
            return loss_fn(y_true=y_true, y_pred=mu, mask=mask)
    
    # Handle GaussianNLLWithCovariance
    elif isinstance(loss_fn, GaussianNLLWithCovariance):
        return loss_fn(y_true=y_true, y_pred=y_pred, 
                     covariance_matrices=covariance_matrices, mask=mask)
    
    # Handle robust losses with weights
    elif isinstance(loss_fn, (HuberLoss, L1Loss)) and precision is not None:
        return loss_fn(y_true=y_true, y_pred=y_pred, mask=mask, weights=precision)
    
    # Standard case
    else:
        return loss_fn(y_true=y_true, y_pred=y_pred, mask=mask)


def _buffer_data(
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
            batch_x, batch_y, batch_cov, batch_mask = _unpack_batch_data(batch_data, device)
            
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


def _validate(
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
            batch_x, batch_y, batch_cov, batch_mask = _unpack_batch_data(batch_data, device)
            batch_size = batch_x.shape[0]
            
            # Forward pass
            y_pred = model(batch_x)
            
            # Calculate loss (without precision weighting for validation)
            loss = _calculate_loss(
                loss_fn, y_pred, batch_y, 
                precision=None,  # Don't use precision weights for validation
                covariance_matrices=batch_cov, 
                mask=batch_mask
            )
            
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    
    model.train()  # Set back to training mode
    return total_loss / max(1, total_samples)