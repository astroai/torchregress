"""
Iteratively Reweighted Least Squares (IRLS) implementation.

This module provides implementations of IRLS for robust regression,
with support for various weighting schemes and loss functions.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
import warnings
from typing import Callable, Optional, Dict, Tuple, List, Union, Any

try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    
# For typing
try:
    from typing import Protocol
    class CallbackFn(Protocol):
        def __call__(self, *, 
                    iteration: int, 
                    model: nn.Module,
                    y_pred: torch.Tensor, 
                    mean: torch.Tensor,
                    residuals: torch.Tensor,
                    precision: torch.Tensor,
                    loss: float,
                    **kwargs: Any) -> None: ...
except ImportError:
    # Fallback for Python < 3.8
    CallbackFn = Callable

from ..losses.robust import HuberLoss, L1Loss, TukeyBiweightLoss
from ..losses.gaussian import DiagonalGaussianNLL, GaussianNLLWithCovariance
from ..utils.irls_utils import (
    huber_weights, tukey_weights, power_weights, 
    estimate_variance, extract_mean_and_residuals, parse_update_frequency,
    setup_data_loader, setup_validation_loader, iterate_batches,
    unpack_batch_data, buffer_data, get_batch_precision, validate_model
)

# Get machine epsilon for numerical stability
EPS = torch.finfo(torch.float32).eps

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
    callbacks: Optional[List[CallbackFn]] = None,
    use_compile: bool = False,
    compile_kwargs: Optional[Dict[str, Any]] = None,
    use_tqdm: bool = False,
) -> Union[Tuple[torch.Tensor, List[float], torch.Tensor], 
           Tuple[torch.Tensor, List[float], torch.Tensor, List[torch.Tensor]]]:
    """
    Applies iteratively reweighted least squares (IRLS) for robust regression.
    
    Performance-optimized implementation supporting PyTorch's latest features.
    
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
        callbacks: List of callback functions for monitoring/custom behavior
        use_compile: Whether to use torch.compile for the model (PyTorch 2.0+)
        compile_kwargs: Additional kwargs for torch.compile
        use_tqdm: Whether to display a progress bar for iterations
        
    Returns:
        y_pred: Final predicted values
        loss_history: List of loss values over iterations
        final_precision: Final precision tensor
        [optional] all_predictions: List of predictions from all iterations
    """
    x = x.detach().clone()  # Don't modify original input
    device = x.device
    weight_params = {} if weight_params is None else weight_params
    callbacks = callbacks or []
    compile_kwargs = compile_kwargs or {}
    
    # --- Compile model if requested (PyTorch 2.0+) ---
    if use_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, **compile_kwargs)
        except Exception as e:
            warnings.warn(f"Failed to compile model: {e}. Continuing with uncompiled model.")
    
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
            _weight_fn = huber_weights
            weight_params = {'delta': delta, **weight_params}
        elif weight_fn == 'tukey':
            _weight_fn = tukey_weights
            weight_params = {'c': 4.685, **weight_params}  # Default robust against 95% efficiency for Gaussian
        elif weight_fn == 'power':
            _weight_fn = power_weights
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

    # --- Set up progress bar if requested ---
    iter_range = range(max_iter)
    if use_tqdm and TQDM_AVAILABLE:
        iter_range = tqdm(iter_range, desc="IRLS iterations", leave=False)

    # --- IRLS Iterations ---
    for iteration in iter_range:
        # Forward pass - without gradients for inference
        with torch.no_grad():
            y_pred = model(x)
            if return_all_predictions:
                all_predictions.append(y_pred)
                
            # Extract mean predictions and calculate residuals
            mean, residuals = extract_mean_and_residuals(y_pred, y_true)
            
            # Calculate current loss
            if base_loss == 'gaussian':
                if covariance_matrices is not None:
                    current_loss = loss_fn(y_pred, y_true, covariance_matrices=covariance_matrices, mask=mask)
                else:
                    current_loss = loss_fn(y_pred, y_true, mask=mask)
            else:  # huber or l1
                current_loss = loss_fn(y_pred, y_true, mask=mask, weights=precision)
                
            loss_value = current_loss.item()
            loss_history.append(loss_value)
            
            # Execute callbacks
            for callback in callbacks:
                callback(
                    iteration=iteration,
                    model=model,
                    y_pred=y_pred,
                    mean=mean,
                    residuals=residuals,
                    precision=precision,
                    loss=loss_value,
                    mask=mask
                )
            
            # Check for early convergence
            if iteration > 0 and abs(loss_history[-1] - loss_history[-2]) < tol:
                if use_tqdm and TQDM_AVAILABLE:
                    iter_range.set_postfix(loss=f"{loss_value:.6f}", converged=True)
                break
            elif use_tqdm and TQDM_AVAILABLE:
                iter_range.set_postfix(loss=f"{loss_value:.6f}")
                
            # Estimate variance based on current residuals
            variance = estimate_variance(
                residuals, y_pred, covariance_matrices, variance_type, loss_fn
            )
            
            # Scale residuals for weighting - use EPS consistently
            scaled_residuals = residuals / (torch.sqrt(variance) + EPS)
            
            # Calculate weights using the chosen weighting function
            iter_weights = _weight_fn(scaled_residuals, **weight_params)
            
            # Update precision weights
            precision = precision * iter_weights
            
            # Update model's variance parameters if using learnable variance
            if base_loss == 'gaussian' and isinstance(loss_fn, DiagonalGaussianNLL):
                # More stable approach: directly compute optimal log variances
                weighted_variance = (variance / (precision + EPS)).mean(dim=0)
                loss_fn.log_variances.data = 0.5 * torch.log(weighted_variance + EPS)
    
    # Get final predictions if we didn't run all iterations
    with torch.no_grad():
        final_y_pred = model(x) if iteration < max_iter - 1 else y_pred
    
    if return_all_predictions:
        return final_y_pred, loss_history, precision, all_predictions
    else:
        return final_y_pred, loss_history, precision

def calculate_loss(
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
            return loss_fn(y_pred=(mu, log_sigma), target=y_true, mask=mask)
        else:
            # For other losses, just use the mean prediction
            return loss_fn(y_pred=mu, target=y_true, mask=mask, weights=precision)
    
    # Handle GaussianNLLWithCovariance
    elif isinstance(loss_fn, GaussianNLLWithCovariance):
        return loss_fn(y_pred=y_pred, target=y_true, 
                     covariance_matrices=covariance_matrices, mask=mask)
    
    # Handle robust losses with weights
    elif isinstance(loss_fn, (HuberLoss, L1Loss, TukeyBiweightLoss)) and precision is not None:
        return loss_fn(y_pred=y_pred, target=y_true, mask=mask, weights=precision)
    
    # Standard case
    else:
        return loss_fn(y_pred=y_pred, target=y_true, mask=mask)

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
    weight_fn: Union[str, Callable] = 'huber',
    weight_params: Optional[Dict[str, Any]] = None,
    variance_type: str = 'predicted',
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
) -> Dict[str, Any]:
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
        weight_params: Parameters for weight function
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
    if optimizer is None and model.parameters():
        optimizer = torch.optim.Adam(model.parameters())
        if verbose:
            print(f"No optimizer provided. Using default Adam optimizer with learning_rate=0.001")

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
    update_type, update_freq = parse_update_frequency(update_weights)

    # --- Data Handling ---
    data_loader, is_minibatch = setup_data_loader(
        train_data, device, batch_size, num_epochs
    )

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
        should_print_epoch = verbose and (epoch % verbose_epoch_freq == 0 or epoch == num_epochs - 1)
        if should_print_epoch:
            print(f"Epoch {epoch+1}/{num_epochs} started")

        # --- Buffer data for epoch-level IRLS if needed ---
        all_x = all_y_true = all_cov = all_masks = None
        if is_minibatch and update_type == "epoch":
            all_x, all_y_true, all_cov, all_masks = buffer_data(data_loader, device)
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
                    "return_all_predictions": return_all_iterations
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
        for i, batch_data in enumerate(iterate_batches(data_loader)):
            batch_x, batch_y, batch_cov, batch_mask = unpack_batch_data(batch_data, device)
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
                        "base_loss": base_loss,
                        "max_iter": irls_max_iter,
                        "tol": irls_tol,
                        "delta": delta,
                        "weight_fn": weight_fn,
                        "weight_params": weight_params,
                        "variance_type": variance_type,
                        "epsilon": epsilon,
                        "return_all_predictions": return_all_iterations
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
            batch_precision = get_batch_precision(
                previous_epoch_precision, 
                i, batch_size, batch_size_current, 
                all_x, batch_y, is_minibatch
            )

            # --- Calculate loss ---
            loss = calculate_loss(
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
            val_loss = validate_model(model, val_loader, loss_fn, device)
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