"""
Out-of-distribution (OOD) detection metrics for regression models.
"""

import torch
import numpy as np
from typing import Union, Optional, Dict, Tuple, List
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal

from torchregression.metrics.utils import convert_to_tensor, apply_reduction, ensure_batch_dim

def mahalanobis_distance(
    x: Union[torch.Tensor, np.ndarray],
    mean: Union[torch.Tensor, np.ndarray],
    cov: Union[torch.Tensor, np.ndarray],
    reduction: str = "none"
) -> Union[torch.Tensor, float]:
    """
    Calculate Mahalanobis distance for OOD detection.
    
    The Mahalanobis distance measures how many standard deviations away
    a point is from the mean of a distribution.
    
    Args:
        x: Input samples [batch_size, n_features]
        mean: Mean vector of the distribution [n_features]
        cov: Covariance matrix [n_features, n_features]
        reduction: How to reduce the distances ("none", "mean", "sum")
        
    Returns:
        Mahalanobis distances
    """
    x = ensure_batch_dim(convert_to_tensor(x))
    mean = convert_to_tensor(mean)
    cov = convert_to_tensor(cov)
    
    # Calculate inverse covariance matrix
    try:
        inv_cov = torch.linalg.inv(cov)
    except:
        # Add small regularization if matrix is singular
        inv_cov = torch.linalg.inv(cov + torch.eye(cov.shape[0], device=cov.device) * 1e-6)
    
    # Calculate Mahalanobis distance for each sample
    diff = x - mean
    md_squared = torch.sum(torch.matmul(diff, inv_cov) * diff, dim=1)
    md = torch.sqrt(md_squared)
    
    # Apply reduction
    return apply_reduction(md, reduction)

def typicality_score(
    x: Union[torch.Tensor, np.ndarray],
    model: torch.nn.Module,
    n_samples: int = 100,
    reduction: str = "none"
) -> Union[torch.Tensor, float]:
    """
    Calculate typicality score for OOD detection using predictive uncertainty.
    
    The typicality score measures how typical a test sample is under the model's 
    predicted distribution, useful for detecting OOD samples.
    
    Args:
        x: Input features [batch_size, n_features]
        model: Model that outputs mean and variance of a predictive distribution
        n_samples: Number of Monte Carlo samples to draw
        reduction: How to reduce the scores ("none", "mean", "sum")
        
    Returns:
        Typicality scores (lower values indicate OOD samples)
    """
    x = ensure_batch_dim(convert_to_tensor(x))
    
    # Ensure model is in eval mode
    model.eval()
    
    # Get predictive distribution parameters
    with torch.no_grad():
        pred = model(x)  # Assumed to return mean and variance
    
    if isinstance(pred, tuple):
        mean, var = pred
    else:
        # Assume the model returns a dictionary
        mean, var = pred['mean'], pred['variance']
    
    # Create normal distribution
    dist = Normal(mean, torch.sqrt(var))
    
    # Sample from the predictive distribution
    samples = dist.sample((n_samples,))  # [n_samples, batch_size, output_dim]
    
    # Calculate log probabilities
    log_probs = dist.log_prob(samples)
    
    if log_probs.dim() > 2:
        log_probs = log_probs.sum(dim=-1)  # Sum over output dimensions
    
    # Calculate typicality: average log probability across samples
    typicality = torch.mean(log_probs, dim=0)  # [batch_size]
    
    # Apply reduction
    return apply_reduction(typicality, reduction)

def entropy_score(
    samples: Union[torch.Tensor, np.ndarray],
    n_bins: int = 10,
    reduction: str = "none"
) -> Union[torch.Tensor, float]:
    """
    Calculate entropy of predictive distribution for OOD detection.
    
    Higher entropy indicates higher uncertainty, which may suggest OOD samples.
    
    Args:
        samples: Samples from predictive distribution [n_samples, batch_size, ...]
        n_bins: Number of bins for histogram estimation of entropy
        reduction: How to reduce the scores ("none", "mean", "sum")
        
    Returns:
        Entropy scores
    """
    samples = convert_to_tensor(samples)
    
    n_samples = samples.shape[0]
    batch_size = samples.shape[1]
    
    # Flatten the samples for each instance in the batch
    samples = samples.reshape(n_samples, batch_size, -1)
    output_dim = samples.shape[-1]
    
    # Calculate entropy for each dimension and instance
    entropies = torch.zeros(batch_size, output_dim, device=samples.device)
    
    for i in range(batch_size):
        for j in range(output_dim):
            # Get samples for this instance and dimension
            inst_samples = samples[:, i, j]
            
            # Calculate histogram
            hist = torch.histogram(inst_samples, n_bins)
            bin_counts = hist.hist
            
            # Calculate probabilities
            probs = bin_counts / n_samples
            
            # Remove zeros for log calculation
            probs = probs[probs > 0]
            
            # Calculate entropy
            entropies[i, j] = -torch.sum(probs * torch.log(probs))
    
    # Sum entropy across dimensions
    total_entropy = torch.sum(entropies, dim=1)
    
    # Apply reduction
    return apply_reduction(total_entropy, reduction)

def kernel_density_score(
    x: Union[torch.Tensor, np.ndarray],
    reference_data: Union[torch.Tensor, np.ndarray],
    bandwidth: float = 1.0,
    reduction: str = "none"
) -> Union[torch.Tensor, float]:
    """
    Calculate kernel density score for OOD detection.
    
    The kernel density score measures how similar a test sample is to 
    a set of reference samples (often training data).
    
    Args:
        x: Test samples [batch_size, n_features]
        reference_data: Reference samples [n_reference, n_features]
        bandwidth: Bandwidth for RBF kernel
        reduction: How to reduce the scores ("none", "mean", "sum")
        
    Returns:
        Kernel density scores (lower values indicate OOD samples)
    """
    x = ensure_batch_dim(convert_to_tensor(x))
    reference_data = ensure_batch_dim(convert_to_tensor(reference_data))
    
    batch_size = x.shape[0]
    n_reference = reference_data.shape[0]
    
    # Calculate pairwise distances efficiently
    # Expand dimensions for broadcasting
    x_expanded = x.unsqueeze(1)  # [batch_size, 1, n_features]
    ref_expanded = reference_data.unsqueeze(0)  # [1, n_reference, n_features]
    
    # Calculate squared distances
    dist_sq = torch.sum((x_expanded - ref_expanded)**2, dim=2)  # [batch_size, n_reference]
    
    # Apply RBF kernel
    kernel_values = torch.exp(-dist_sq / (2 * bandwidth**2))
    
    # Average over reference points
    density_scores = torch.mean(kernel_values, dim=1)
    
    # Apply reduction
    return apply_reduction(density_scores, reduction)

def ood_metrics_report(
    x: Union[torch.Tensor, np.ndarray],
    model: Optional[torch.nn.Module] = None,
    reference_data: Optional[Union[torch.Tensor, np.ndarray]] = None,
    mean: Optional[Union[torch.Tensor, np.ndarray]] = None,
    cov: Optional[Union[torch.Tensor, np.ndarray]] = None,
    samples: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> Dict[str, float]:
    """
    Generate a comprehensive report on OOD detection metrics.
    
    Args:
        x: Test samples to evaluate for OOD
        model: Model for typicality score (optional)
        reference_data: Reference data for kernel density (optional)
        mean: Mean for Mahalanobis distance (optional)
        cov: Covariance for Mahalanobis distance (optional)
        samples: Predictive samples for entropy calculation (optional)
        
    Returns:
        Dictionary with OOD detection metrics
    """
    metrics = {}
    
    # Calculate Mahalanobis distance if mean and covariance provided
    if mean is not None and cov is not None:
        metrics['mahalanobis_distance'] = mahalanobis_distance(x, mean, cov, reduction="mean").item()
    
    # Calculate typicality score if model provided
    if model is not None:
        metrics['typicality_score'] = typicality_score(x, model, reduction="mean").item()
    
    # Calculate kernel density if reference data provided
    if reference_data is not None:
        metrics['kernel_density'] = kernel_density_score(x, reference_data, reduction="mean").item()
    
    # Calculate entropy if samples provided
    if samples is not None:
        metrics['entropy'] = entropy_score(samples, reduction="mean").item()
    
    return metrics
