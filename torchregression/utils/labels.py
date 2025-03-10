"""
Label handling utilities.

This module provides functions for encoding, decoding, and manipulating
label data with PyTorch tensors. While similar functionality exists in libraries 
like scikit-learn, these implementations are tensor-native and optimized for
integration with PyTorch regression models.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Tuple, List, Dict

# Basic encoding/decoding functions

def encode_onehot(labels: torch.Tensor, num_classes: Optional[int] = None) -> torch.Tensor:
    """
    Convert class indices to one-hot encodings.
    
    While similar to torch.nn.functional.one_hot, this implementation
    handles arbitrary input shapes and automatically determines the
    number of classes when not provided.
    
    Args:
        labels: Class indices of shape [...] with integer values in [0, num_classes-1]
        num_classes: Number of classes. If None, inferred from labels.
        
    Returns:
        One-hot encoded tensor of shape [..., num_classes]
    """
    if num_classes is None:
        num_classes = int(torch.max(labels).item()) + 1
        
    # Ensure labels are integers
    if not labels.dtype.is_integer:
        labels = labels.long()
    
    shape = labels.shape
    labels = labels.reshape(-1)
    
    # Create one-hot encoding
    onehot = torch.zeros(labels.shape[0], num_classes, dtype=torch.float32, device=labels.device)
    onehot.scatter_(1, labels.unsqueeze(1), 1)
    
    # Reshape back to original dimensions with added class dimension
    return onehot.reshape(*shape, num_classes)

def decode_onehot(onehot: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Convert one-hot encodings to class indices.
    
    Args:
        onehot: One-hot encoded tensor
        dim: Dimension containing the one-hot encoding
        
    Returns:
        Class indices tensor
    """
    return torch.argmax(onehot, dim=dim)

def label_smoothing(onehot: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """
    Apply label smoothing to one-hot encoded labels.
    
    While torch.nn.CrossEntropyLoss supports label_smoothing, this standalone
    function allows applying smoothing to any one-hot tensor for flexibility
    in custom loss functions.
    
    Args:
        onehot: One-hot encoded tensor
        alpha: Smoothing factor in [0, 1]
        
    Returns:
        Smoothed labels tensor
    """
    num_classes = onehot.shape[-1]
    return (1.0 - alpha) * onehot + alpha / num_classes

def soft_to_hard_labels(soft_labels: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Convert soft labels (probabilities) to hard labels (one-hot).
    
    Args:
        soft_labels: Soft labels tensor with probability distributions
        dim: Dimension containing the class probabilities
        
    Returns:
        One-hot encoded tensor
    """
    indices = torch.argmax(soft_labels, dim=dim)
    return F.one_hot(indices, num_classes=soft_labels.shape[dim]).float()


def combine_binary_average(labels: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Simple averaging of binary labels from multiple annotators.
    
    Args:
        labels: Binary labels tensor [annotators, samples] or [samples, annotators]
        dim: Dimension along which to average (annotator dimension)
        
    Returns:
        Average labels [samples]
    """
    return torch.mean(labels.float(), dim=dim)

def combine_binary_weighted_average(labels: torch.Tensor, weights: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Weighted averaging of binary labels from multiple annotators.

    Args:
        labels: Binary labels tensor [annotators, samples] or [samples, annotators]
        weights: Weights for each annotator [annotators]
        dim: Dimension along which to average (annotator dimension)

    Returns:
        Weighted average labels [samples]
    """
    # Normalize weights
    norm_weights = weights / torch.sum(weights)

    # Reshape weights for broadcasting
    if dim == 0:
        weights_expanded = norm_weights.unsqueeze(1)
    else:
        weights_expanded = norm_weights.unsqueeze(0)

    # Weighted sum
    return torch.sum(labels.float() * weights_expanded, dim=dim)

def combine_dawid_skene(
    annotations: torch.Tensor, 
    num_classes: int, 
    max_iter: int = 100, 
    tol: float = 1e-6, 
    init_pi: Optional[torch.Tensor] = None, 
    init_confusion: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Implements the Dawid-Skene model for aggregating annotations from multiple annotators.
    
    Note:
        While libraries like 'crowdkit' offer Dawid-Skene implementations,
        this version is PyTorch tensor-native for seamless integration with
        deep learning workflows without numpy conversions.

    This implementation uses an iterative Expectation-Maximization (EM) algorithm
    to estimate the true labels and the confusion matrices of the annotators.

    Args:
        annotations (torch.LongTensor): A tensor of shape (num_samples, num_annotators)
            containing the integer-encoded labels provided by each annotator for each sample.
            -1 indicates a missing annotation.
        num_classes (int): The number of classes.
        max_iter (int): Maximum number of EM iterations.
        tol (float): Convergence tolerance for the EM algorithm.
        init_pi (torch.Tensor, optional): Initial class priors.  Shape: (num_classes,). If None,
            uniform priors are used.
        init_confusion (torch.Tensor, optional): Initial confusion matrices.
            Shape: (num_annotators, num_classes, num_classes). If None, initialized to identity.

    Returns:
        tuple: A tuple containing:
            - pi (torch.Tensor): Estimated class priors (num_classes,).
            - confusion_matrices (torch.Tensor): Estimated confusion matrices for each
              annotator (num_annotators, num_classes, num_classes).
            - q_z (torch.Tensor):  Estimated posterior probabilities of the true labels
              (num_samples, num_classes).
    """
    if not isinstance(annotations, torch.Tensor):
        raise TypeError("annotations must be a torch.Tensor")
    if annotations.dim() != 2:
        raise ValueError("annotations must be a 2D tensor of shape (num_samples, num_annotators)")
        
    num_samples, num_annotators = annotations.shape
    device = annotations.device

    # Initialize q_z (responsibilities) uniformly.  This will be updated in the E-step.
    q_z = torch.ones(num_samples, num_classes, device=device) / num_classes

    # Initialize pi (class priors)
    if init_pi is None:
        pi = torch.ones(num_classes, device=device) / num_classes
    else:
        if init_pi.shape != (num_classes,):
            raise ValueError(f"Expected init_pi shape ({num_classes},), got {init_pi.shape}")
        pi = init_pi.to(device)
        pi = pi / torch.sum(pi)  # Ensure it sums to 1

    # Initialize confusion matrices (annotator error rates)
    if init_confusion is None:
        # Identity matrix + small noise to break symmetry
        noise = torch.rand(num_annotators, num_classes, num_classes, device=device) * 0.01
        confusion_matrices = torch.eye(num_classes, device=device).unsqueeze(0).expand(num_annotators, -1, -1) + noise
        confusion_matrices /= confusion_matrices.sum(dim=2, keepdim=True)  # Normalize rows
    else:
        if init_confusion.shape != (num_annotators, num_classes, num_classes):
            raise ValueError(f"Expected init_confusion shape ({num_annotators}, {num_classes}, {num_classes}), got {init_confusion.shape}")
        confusion_matrices = init_confusion.to(device)
        # Ensure rows sum to 1
        confusion_matrices = confusion_matrices / confusion_matrices.sum(dim=2, keepdim=True)

    # Create mask for missing values (-1)
    missing_mask = (annotations == -1)  # (N, R)
    
    # Convert annotations to one-hot encoding
    valid_annotations = annotations.clone()
    valid_annotations[missing_mask] = 0  # Temporarily set to 0 for one-hot encoding
    annotations_one_hot = F.one_hot(valid_annotations, num_classes=num_classes).float()  # (N, R, C)
    
    # Zero out the one-hot vectors for missing annotations
    for i in range(num_annotators):
        annotations_one_hot[:, i][missing_mask[:, i]] = 0

    prev_log_likelihood = -float('inf')

    for iteration in range(max_iter):
        # --- E-step: Update q_z (responsibilities) ---
        # Initialize log likelihood with log-priors
        log_likelihood = torch.log(pi + 1e-10).unsqueeze(0).expand(num_samples, -1)  # (N, C)

        for r in range(num_annotators):
            # Skip missing annotations
            r_mask = ~missing_mask[:, r]  # (N,)
            
            if not torch.any(r_mask):
                continue  # Skip if all annotations from this annotator are missing
                
            # For each class z, calculate log P(y_r | z)
            # confusion_matrices[r, z, y_r] = P(y_r | z)
            # We use batch matrix multiplication for efficiency
            # annotations_one_hot[:, r] = one-hot of annotator r's labels
            # shape: (N, C)
            log_py_given_z = torch.log(confusion_matrices[r] + 1e-10)  # (C, C)
            
            # For samples with valid annotations, update their log likelihood
            # by adding log P(y_r | z)
            # For batch operation, we need to batch select from log_py_given_z 
            # based on the observed annotations
            for c in range(num_classes):
                # For samples where annotator r assigned class c
                class_mask = annotations_one_hot[:, r, c] > 0
                if not torch.any(class_mask):
                    continue
                
                # Update log likelihood for these samples
                # Add log P(y_r = c | z) for all possible z values
                log_likelihood[class_mask] += log_py_given_z[:, c]

        # Normalize using log-sum-exp trick for numerical stability
        max_loglik = torch.max(log_likelihood, dim=1, keepdim=True)[0]
        log_likelihood_stable = log_likelihood - max_loglik
        q_z = torch.exp(log_likelihood_stable)
        q_z = q_z / torch.sum(q_z, dim=1, keepdim=True)

        # --- M-step: Update pi and confusion_matrices ---
        # Update pi (class priors)
        pi = torch.mean(q_z, dim=0)  # Mean over samples

        # Update confusion matrices
        for r in range(num_annotators):
            # Skip missing annotations
            r_mask = ~missing_mask[:, r]  # (N,)
            
            if not torch.any(r_mask):
                continue  # Skip if all annotations from this annotator are missing
                
            # For each true class z and observed class c
            # confusion_matrices[r, z, c] = P(y_r = c | z)
            for z in range(num_classes):
                # Numerator: sum_i q_z[i, z] * annotations_one_hot[i, r, c] for all c
                # This gives the expected count of samples with true class z that annotator r labeled as each class
                numerator = torch.sum(
                    q_z[:, z].unsqueeze(1) * annotations_one_hot[:, r],  # (N, C)
                    dim=0  # Sum over samples
                )
                
                # Denominator: sum_i q_z[i, z]
                # This gives the expected total count of samples with true class z
                denominator = torch.sum(q_z[:, z])
                
                # Update confusion matrix for this annotator and true class
                # Add small epsilon to avoid division by zero
                if denominator > 0:
                    confusion_matrices[r, z] = numerator / (denominator + 1e-10)
                else:
                    # If no samples are assigned to this class, keep the current estimates
                    pass

        # --- Check for Convergence ---
        current_log_likelihood = torch.mean(torch.logsumexp(log_likelihood, dim=1))
        if abs(current_log_likelihood - prev_log_likelihood) < tol:
            break
        prev_log_likelihood = current_log_likelihood

    return pi, confusion_matrices, q_z

def combine_continuous_blue_with_scaling(
    estimates: torch.Tensor, 
    covariance_matrix: Optional[torch.Tensor] = None, 
    variances: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combines continuous estimates using Best Linear Unbiased Estimator (BLUE), scaling uncertainty if inconsistent.

    Handles both full covariance matrices and diagonal-only variances. If the estimators are 
    inconsistent, scales the uncertainty by the chi-square per degree of freedom.

    Args:
        estimates: Tensor of shape (num_samples, num_estimators) containing the estimates.
        covariance_matrix: Covariance matrix (num_estimators, num_estimators).
                          If None, `variances` must be provided.
        variances: Tensor of shape (num_estimators,) containing variances.
                   Used only if `covariance_matrix` is None.

    Returns:
        Tuple: (combined_estimate, scaled_variance, scale_factor)
            - combined_estimate: Tensor of shape (num_samples,) containing the combined estimates.
            - scaled_variance: Tensor of shape (num_samples,) containing the scaled variances.
            - scale_factor: Scalar tensor containing the scaling factor.
    """
    if estimates.ndim != 2:
        raise ValueError("estimates must be a 2D tensor (num_samples, num_estimators)")

    num_samples, num_estimators = estimates.shape
    device = estimates.device

    if covariance_matrix is not None and variances is not None:
        raise ValueError("Cannot provide both covariance_matrix and variances")
    if covariance_matrix is None and variances is None:
        raise ValueError("Must provide either covariance_matrix or variances")

    if covariance_matrix is not None:
        # --- Full Covariance Case ---
        if covariance_matrix.ndim != 2:
            raise ValueError(f"covariance_matrix must be a 2D tensor, got shape {covariance_matrix.shape}")
        if covariance_matrix.shape != (num_estimators, num_estimators):
            raise ValueError(f"covariance_matrix shape {covariance_matrix.shape} must match (num_estimators, num_estimators) = ({num_estimators}, {num_estimators})")

        covariance_matrix = covariance_matrix.to(device)
        ones = torch.ones(num_estimators, 1, device=device)
        
        # Use stronger jitter for better stability
        jitter = 1e-6 * torch.eye(num_estimators, device=device) * torch.max(torch.diag(covariance_matrix))
        
        try:
            V_inv = torch.linalg.inv(covariance_matrix + jitter)
        except torch.linalg.LinAlgError:
            # Fall back to pseudoinverse with more jitter if inverse fails
            V_inv = torch.linalg.pinv(covariance_matrix + 1e-4 * torch.eye(num_estimators, device=device))

        denominator = torch.matmul(ones.T, torch.matmul(V_inv, ones))
        weights = torch.matmul(V_inv, ones) / denominator
        combined_estimate = torch.matmul(estimates, weights).squeeze(-1)
        
        # Handle potential shape issues
        if num_samples == 1:
            combined_estimate = combined_estimate.reshape(1)
            
        # Base variance is the same for all samples
        base_variance = (1.0 / denominator).item()
        combined_variance = torch.full((num_samples,), base_variance, device=device)

        # Calculate chi-squared to check consistency
        diff = estimates - combined_estimate.unsqueeze(1)
        weighted_diff = torch.matmul(diff, V_inv)
        chi2 = torch.sum(weighted_diff * diff, dim=1)
        total_chi2 = torch.sum(chi2)

    else:
        # --- Diagonal Variance Case ---
        if variances.ndim != 1:
            raise ValueError(f"variances must be a 1D tensor, got shape {variances.shape}")
        if variances.shape[0] != num_estimators:
            raise ValueError(f"variances shape {variances.shape} must match (num_estimators,) = ({num_estimators},)")

        variances = variances.to(device)
        weights = 1.0 / (variances + 1e-10)  # Better numeric stability
        denominator = torch.sum(weights)
        weights = weights / denominator
        
        combined_estimate = torch.sum(estimates * weights.unsqueeze(0), dim=1)
        base_variance = 1.0 / denominator
        combined_variance = torch.full((num_samples,), base_variance, device=device)

        # Calculate chi-squared (weighted sum of squared deviations)
        diff = estimates - combined_estimate.unsqueeze(1)
        chi2 = torch.sum((diff ** 2) / variances.unsqueeze(0), dim=1)
        total_chi2 = torch.sum(chi2)

    # --- Scale the Variance if necessary ---
    dof = num_samples * (num_estimators - 1)  # Total degrees of freedom
    scale_factor = torch.tensor(1.0, device=device)  # Initialize to 1
    
    if dof > 0:
        # Scale by chi-square per degree of freedom
        scale_factor = total_chi2 / dof
        # Clamp to reasonable range to avoid extreme scaling
        scale_factor = torch.clamp(scale_factor, min=0.1, max=10.0)
            
    scaled_variance = combined_variance * scale_factor

    return combined_estimate, scaled_variance, scale_factor


def combine_continuous_simple(labels: torch.Tensor, method: str = 'mean', 
                             mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Combines continuous labels using simple aggregation methods.
    
    Args:
        labels: Tensor of shape (num_annotators, num_samples) or (num_samples, num_annotators)
                containing continuous labels.
        method: Aggregation method, one of ['mean', 'median', 'min', 'max'].
        mask: Optional tensor of shape matching labels with True for valid labels
              and False for invalid/missing labels.
              
    Returns:
        Tensor of shape (num_samples,) containing the combined labels.
    """
    if labels.ndim != 2:
        raise ValueError(f"labels must be a 2D tensor, got shape {labels.shape}")
    
    # Ensure we're working with (num_samples, num_annotators)
    if method not in ['mean', 'median', 'min', 'max']:
        raise ValueError(f"method must be one of ['mean', 'median', 'min', 'max'], got {method}")
    
    # Handle missing values if mask is provided
    if mask is not None:
        if mask.shape != labels.shape:
            raise ValueError(f"mask shape {mask.shape} must match labels shape {labels.shape}")
        # Replace invalid values with NaN
        labels = labels.float().clone()
        labels[~mask] = float('nan')
    else:
        labels = labels.float()
    
    if method == 'mean':
        return torch.nanmean(labels, dim=1)
    elif method == 'median':
        return torch.nanmedian(labels, dim=1).values
    elif method == 'min':
        return torch.nanmin(labels, dim=1).values
    else:  # max
        return torch.nanmax(labels, dim=1).values


def combine_continuous_trimmed_mean(labels: torch.Tensor, trim_percentage: float = 0.2, 
                             mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Combines continuous labels using trimmed mean (removing extreme values).
    
    Args:
        labels: Tensor of shape (num_samples, num_annotators) containing continuous labels.
        trim_percentage: Percentage of values to trim from each end (0.0 to 0.5).
        mask: Optional tensor of shape matching labels with True for valid labels
              and False for invalid/missing labels.
              
    Returns:
        Tensor of shape (num_samples,) containing the trimmed mean labels.
    """
    if not 0.0 <= trim_percentage < 0.5:
        raise ValueError(f"trim_percentage must be between 0.0 and 0.5, got {trim_percentage}")
    
    if labels.ndim != 2:
        raise ValueError(f"labels must be a 2D tensor, got shape {labels.shape}")
    
    num_samples, num_annotators = labels.shape
    device = labels.device
    result = torch.zeros(num_samples, device=device)
    
    # Handle each sample separately
    for i in range(num_samples):
        # Get annotations for this sample
        sample_labels = labels[i]
        
        # Apply mask if provided
        if mask is not None:
            sample_mask = mask[i]
            valid_labels = sample_labels[sample_mask]
            if valid_labels.numel() == 0:
                # No valid labels, assign NaN
                result[i] = float('nan')
                continue
        else:
            valid_labels = sample_labels
            
        # Sort labels
        sorted_labels, _ = torch.sort(valid_labels)
        n_valid = sorted_labels.shape[0]
        
        # Calculate number of values to trim from each end
        n_trim = int(n_valid * trim_percentage)
        
        # Calculate trimmed mean
        if n_valid - 2 * n_trim > 0:
            result[i] = torch.mean(sorted_labels[n_trim:n_valid-n_trim])
        else:
            # Not enough values after trimming
            result[i] = torch.mean(sorted_labels)
    
    return result


def combine_continuous_robust_blue(estimates: torch.Tensor, initial_variances: Optional[torch.Tensor] = None,
                           huber_threshold: float = 1.345, max_iter: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Implements a robust version of BLUE using iteratively reweighted least squares (IRLS)
    with Huber weighting to handle outliers.
    
    Args:
        estimates: Tensor of shape (num_samples, num_estimators) containing the estimates.
        initial_variances: Optional tensor of shape (num_estimators,) with initial variance estimates.
                          If None, uses sample variances.
        huber_threshold: Threshold for the Huber function (typical values: 1.345 to 2.0).
        max_iter: Maximum number of IRLS iterations.
        
    Returns:
        Tuple: (combined_estimate, estimated_variance)
    """
    num_samples, num_estimators = estimates.shape
    device = estimates.device
    
    # Initialize weights
    if initial_variances is None:
        # Use variance across samples as initial estimate
        initial_variances = torch.var(estimates, dim=0, unbiased=True)
    
    weights = 1.0 / (initial_variances + 1e-10)
    
    # Initialize combined estimate using weighted mean
    norm_weights = weights / torch.sum(weights)
    combined = torch.sum(estimates * norm_weights.unsqueeze(0), dim=1)
    
    for _ in range(max_iter):
        # Calculate residuals
        residuals = estimates - combined.unsqueeze(1)
        
        # Calculate standardized residuals
        stds = torch.sqrt(initial_variances + 1e-10)
        standardized_residuals = residuals / stds.unsqueeze(0)
        
        # Apply Huber weighting
        abs_std_residuals = torch.abs(standardized_residuals)
        w = torch.where(
            abs_std_residuals <= huber_threshold,
            torch.ones_like(standardized_residuals),
            huber_threshold / abs_std_residuals
        )
        
        # Update weights
        weights = w / (initial_variances + 1e-10).unsqueeze(0)
        
        # Calculate normalized weights
        weights_sum = torch.sum(weights, dim=1, keepdim=True)
        norm_weights = weights / (weights_sum + 1e-10)
        
        # Update combined estimate
        combined = torch.sum(estimates * norm_weights, dim=1)
    
    # Estimate final variance
    residuals = estimates - combined.unsqueeze(1)
    squared_residuals = residuals**2
    estimated_variance = torch.mean(squared_residuals * norm_weights)
    variance = torch.full((num_samples,), estimated_variance.item(), device=device)
    
    return combined, variance


def combine_continuous_bayesian(estimates: torch.Tensor, variances: Optional[torch.Tensor] = None,
                         prior_mean: Optional[float] = None, prior_var: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combines continuous labels using Bayesian inference, treating each estimator as a noisy measurement.
    
    Args:
        estimates: Tensor of shape (num_samples, num_estimators) containing the estimates.
        variances: Optional tensor of shape (num_estimators,) with variance estimates for each annotator.
                  If None, uses sample variances.
        prior_mean: Optional prior mean. If None, uses the mean of all estimates.
        prior_var: Optional prior variance. If None, uses a weakly informative prior.
        
    Returns:
        Tuple: (posterior_mean, posterior_variance)
    """
    num_samples, num_estimators = estimates.shape
    device = estimates.device
    
    if variances is None:
        # Use variance across samples as estimate
        variances = torch.var(estimates, dim=0, unbiased=True)
    
    # Set prior if not provided
    if prior_mean is None:
        prior_mean = torch.mean(estimates)
    
    if prior_var is None:
        # Use a weakly informative prior: 10x the average variance
        prior_var = 10.0 * torch.mean(variances)
    
    # Convert to precision (inverse variance)
    prior_precision = 1.0 / prior_var
    precisions = 1.0 / variances
    
    # Initialize result tensors
    posterior_mean = torch.zeros(num_samples, device=device)
    posterior_var = torch.zeros(num_samples, device=device)
    
    # Process each sample
    for i in range(num_samples):
        # Bayesian update formula
        # posterior_precision = prior_precision + sum(precisions)
        # posterior_mean = (prior_precision * prior_mean + sum(precisions * estimates)) / posterior_precision
        
        sample_estimates = estimates[i]
        
        # Calculate posterior precision
        posterior_precision = prior_precision + torch.sum(precisions)
        
        # Calculate posterior mean
        weighted_sum = prior_precision * prior_mean + torch.sum(precisions * sample_estimates)
        posterior_mean[i] = weighted_sum / posterior_precision
        
        # Calculate posterior variance
        posterior_var[i] = 1.0 / posterior_precision
    
    return posterior_mean, posterior_var