import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

def combine_binary_average(labels: List[torch.Tensor]) -> torch.Tensor:
    """
    Averages categorical labels from multiple annotators.

    Args:
        labels: A list of tensors, where each tensor represents the labels
                from one annotator.  Each tensor should have shape (num_samples,)
                and contain 0/1 (or boolean) values.

    Returns:
        A tensor of shape (num_samples,) containing the average probabilities.
    """
    if not labels:
        raise ValueError("Input list of labels cannot be empty")
    if not all(isinstance(label, torch.Tensor) for label in labels):
        raise TypeError('labels must be a list of tensors')

    # Convert to float and stack along a new dimension (annotator dimension)
    labels_tensor = torch.stack([label.float() for label in labels]) #shape (num_annotators, num_samples)

    # Average across the annotators
    return torch.mean(labels_tensor, dim=0) #along annotator dimension

def combine_binary_weighted_average(labels: List[torch.Tensor], weights: Union[List[float], torch.Tensor]) -> torch.Tensor:
    """
    Calculates weighted average of binary labels from multiple annotators.

    Args:
        labels: List of label tensors. Each tensor: (num_samples,), 0/1 or boolean.
        weights: List or tensor of weights for each annotator.

    Returns:
        Tensor of shape (num_samples,) with weighted average probabilities.
    """
    if not labels:
        raise ValueError("Input list of labels cannot be empty")
    if not all(isinstance(label, torch.Tensor) for label in labels):
        raise TypeError('labels must be a list of tensors')

    # Ensure all tensors are on the same device
    device = labels[0].device
    for i in range(1, len(labels)):
        if labels[i].device != device:
            labels[i] = labels[i].to(device)

    num_annotators = len(labels)
    if isinstance(weights, list):
        if len(weights) != num_annotators:
            raise ValueError("Number of weights must match the number of annotators.")
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
    elif isinstance(weights, torch.Tensor):
        if weights.ndim != 1 or weights.size(0) != num_annotators:
            raise ValueError("Weights must be a 1D tensor with length equal to the number of annotators.")
        weights = weights.to(device)
    else:
        raise TypeError("Weights must be a list or a torch.Tensor")

    # Normalize weights to sum to 1
    weights = weights / torch.sum(weights)
    # Stack labels and convert to float
    labels_tensor = torch.stack([label.float() for label in labels])

    # Calculate weighted average.  Use einsum for clarity and efficiency.
    return torch.einsum('i,i...->...', weights, labels_tensor)

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
            raise ValueError("init_pi must have shape (num_classes,)")
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
            raise ValueError("init_confusion must have shape (num_annotators, num_classes, num_classes)")
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
            annotator_mask = ~missing_mask[:, r]  # (N,)
            if not annotator_mask.any():
                continue  # Skip if all annotations are missing for this annotator
                
            annotator_confusion = confusion_matrices[r]  # (C, C)
            log_confusion = torch.log(annotator_confusion + 1e-10)  # (C, C)
            
            # For each sample with valid annotation, add log probability from this annotator
            for c in range(num_classes):  # true class
                # For samples with annotations from this annotator
                mask = annotator_mask
                if not mask.any():
                    continue
                    
                # Get observed class probabilities from confusion matrix
                # P(annotation=j | true_class=c) for all possible j
                class_probs = log_confusion[c]  # (C,)
                
                # Apply these probabilities to the one-hot encoded annotations
                # Multiply each sample's one-hot vector by the appropriate confusion matrix row
                contribution = torch.matmul(annotations_one_hot[mask, r], class_probs)  # (N_valid,)
                
                # Add to the log likelihood for true class c
                log_likelihood[mask, c] += contribution

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
            annotator_mask = ~missing_mask[:, r]  # (N,)
            if not annotator_mask.any():
                continue  # Skip if all annotations are missing
                
            # Get mask for samples with annotations from this annotator
            masked_q_z = q_z[annotator_mask]  # (N_valid, C)
            masked_annotations = annotations_one_hot[annotator_mask, r]  # (N_valid, C)
            
            # For each true class c, compute P(annotation=j | true_class=c)
            for c in range(num_classes):
                # Weight by q_z (probability that sample has true class c)
                weights = masked_q_z[:, c].unsqueeze(1)  # (N_valid, 1)
                
                # Weighted sum of one-hot annotations
                numerator = torch.sum(weights * masked_annotations, dim=0)  # (C,)
                denominator = torch.sum(weights) + 1e-10
                
                confusion_matrices[r, c] = numerator / denominator

        # --- Check for Convergence ---
        current_log_likelihood = torch.mean(torch.logsumexp(log_likelihood, dim=1))
        if abs(current_log_likelihood - prev_log_likelihood) < tol:
            break
        prev_log_likelihood = current_log_likelihood

    return pi, confusion_matrices, q_z

def comine_continuous_blue_with_scaling(estimates: torch.Tensor, covariance_matrix: Optional[torch.Tensor] = None, variances: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            raise ValueError("covariance_matrix must be a 2D tensor (num_estimators, num_estimators)")
        if covariance_matrix.shape != (num_estimators, num_estimators):
            raise ValueError("covariance_matrix shape must be (num_estimators, num_estimators)")

        covariance_matrix = covariance_matrix.to(device)
        ones = torch.ones(num_estimators, 1, device=device)
        
        # Use stronger jitter for better stability
        jitter = 1e-6 * torch.eye(num_estimators, device=device) * torch.max(torch.diag(covariance_matrix))
        
        try:
            V_inv = torch.linalg.inv(covariance_matrix + jitter)
        except torch.linalg.LinAlgError:
            V_inv = torch.linalg.pinv(covariance_matrix + jitter)

        denominator = torch.matmul(ones.T, torch.matmul(V_inv, ones))
        weights = torch.matmul(V_inv, ones) / denominator
        combined_estimate = torch.matmul(estimates, weights).squeeze(-1)
        
        # Handle potential shape issues
        if num_samples == 1:
            combined_estimate = combined_estimate.unsqueeze(0)
            
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
            raise ValueError("variances must be a 1D tensor (num_estimators,)")
        if variances.shape[0] != num_estimators:
            raise ValueError("variances must have shape (num_estimators,)")

        variances = variances.to(device)
        weights = 1.0 / (variances + 1e-10)  # Better numeric stability
        denominator = torch.sum(weights)
        weights = weights / denominator
        
        combined_estimate = torch.sum(estimates * weights.unsqueeze(0), dim=1)
        base_variance = 1.0 / denominator
        combined_variance = torch.full((num_samples,), base_variance, device=device)

        # Calculate chi-squared (weighted sum of squared deviations)
        diff = estimates - combined_estimate.unsqueeze(1)
        chi2 = torch.sum(diff**2 / variances.unsqueeze(0), dim=1)
        total_chi2 = torch.sum(chi2)

    # --- Scale the Variance if necessary ---
    dof = num_samples * (num_estimators - 1)  # Total degrees of freedom
    scale_factor = torch.tensor(1.0, device=device)  # Initialize to 1
    
    if dof > 0:
        chi2_per_dof = total_chi2 / dof
        if chi2_per_dof > 1.0:
            scale_factor = chi2_per_dof
            
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
        raise ValueError("labels must be a 2D tensor")
    
    # Ensure we're working with (num_samples, num_annotators)
    if method not in ['mean', 'median', 'min', 'max']:
        raise ValueError(f"Unknown method: {method}. Use one of ['mean', 'median', 'min', 'max']")
    
    # Handle missing values if mask is provided
    if mask is not None:
        if mask.shape != labels.shape:
            raise ValueError("mask must have the same shape as labels")
        # Fill missing values with NaN
        masked_labels = torch.where(mask, labels, torch.tensor(float('nan'), device=labels.device))
    else:
        masked_labels = labels
    
    if method == 'mean':
        return torch.nanmean(masked_labels, dim=1)
    elif method == 'median':
        return torch.nanmedian(masked_labels, dim=1).values
    elif method == 'min':
        return torch.nanmin(masked_labels, dim=1).values
    else:  # max
        return torch.nanmax(masked_labels, dim=1).values


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
        raise ValueError("trim_percentage must be between 0.0 and 0.5")
    
    if labels.ndim != 2:
        raise ValueError("labels must be a 2D tensor")
    
    num_samples, num_annotators = labels.shape
    device = labels.device
    result = torch.zeros(num_samples, device=device)
    
    # Handle each sample separately
    for i in range(num_samples):
        sample_labels = labels[i]
        
        # Apply mask if provided
        if mask is not None:
            sample_mask = mask[i]
            if not torch.any(sample_mask):
                result[i] = float('nan')
                continue
            sample_labels = sample_labels[sample_mask]
        
        # Sort values
        sorted_values, _ = torch.sort(sample_labels)
        n = len(sorted_values)
        
        if n <= 2:  # Not enough data to trim
            result[i] = torch.mean(sorted_values)
        else:
            # Calculate how many values to trim from each end
            k = int(n * trim_percentage)
            # Use values after trimming
            result[i] = torch.mean(sorted_values[k:n-k])
    
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
        
        # Apply Huber weighting to handle outliers
        abs_residuals = torch.abs(residuals)
        huber_weights = torch.ones_like(residuals)
        outlier_mask = abs_residuals > huber_threshold
        huber_weights[outlier_mask] = huber_threshold / abs_residuals[outlier_mask]
        
        # Update weights
        iteration_weights = weights.unsqueeze(0) * huber_weights
        norm_weights = iteration_weights / torch.sum(iteration_weights, dim=1, keepdim=True)
        
        # Update combined estimate
        combined = torch.sum(estimates * norm_weights, dim=1)
    
    # Estimate final variance
    residuals = estimates - combined.unsqueeze(1)
    squared_residuals = residuals**2
    estimated_variance = torch.mean(squared_residuals * norm_weights)
    variance = torch.full((num_samples,), estimated_variance.item(), device=device)
    
    return combined, variance


def combine_continuousbayesian(estimates: torch.Tensor, variances: Optional[torch.Tensor] = None,
                         prior_mean: Optional[float] = None, prior_var: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combines continuous labels using Bayesian inference, treating each estimator as a noisy measurement.
    Booooo Bayesian and their priors. 
    
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
        # Estimate variances from data, with a small floor
        variances = torch.var(estimates, dim=0, unbiased=True).clamp(min=1e-5)
    
    # Set prior if not provided
    if prior_mean is None:
        prior_mean = torch.mean(estimates).item()
    
    if prior_var is None:
        # Use a weakly informative prior
        data_var = torch.var(estimates).item()
        prior_var = max(data_var * 10, 1.0)  # Weakly informative
    
    # Convert to precision (inverse variance)
    prior_precision = 1.0 / prior_var
    precisions = 1.0 / variances
    
    # Initialize result tensors
    posterior_mean = torch.zeros(num_samples, device=device)
    posterior_var = torch.zeros(num_samples, device=device)
    
    # Process each sample
    for i in range(num_samples):
        # Calculate posterior precision
        post_precision = prior_precision + torch.sum(precisions)
        
        # Calculate posterior variance
        post_var = 1.0 / post_precision
        
        # Calculate weighted mean
        weighted_data = torch.sum(estimates[i] * precisions)
        weighted_prior = prior_mean * prior_precision
        
        # Calculate posterior mean
        post_mean = (weighted_data + weighted_prior) * post_var
        
        posterior_mean[i] = post_mean
        posterior_var[i] = post_var
    
    return posterior_mean, posterior_var