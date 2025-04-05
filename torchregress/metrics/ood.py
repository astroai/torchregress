"""
Out-of-distribution (OOD) detection metrics for regression models.
"""

import torch
import numpy as np
from typing import Union, Optional, Dict, Tuple
from torch.distributions import Normal

from torchregress.metrics.utils import convert_to_tensor, apply_reduction, ensure_batch_dim


def mahalanobis_distance(
    x: Union[torch.Tensor, np.ndarray],
    mean: Union[torch.Tensor, np.ndarray],
    cov: Union[torch.Tensor, np.ndarray],
    reduction: str = "none",
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

    # Ensure compatible devices
    if x.device != mean.device:
        mean = mean.to(x.device)
    if x.device != cov.device:
        cov = cov.to(x.device)

    # Calculate Cholesky decomposition instead of inverse for better numerical stability
    try:
        # Add small regularization to ensure positive definiteness
        L = torch.linalg.cholesky(cov + torch.eye(cov.shape[0], device=cov.device) * 1e-6)
        diff = x - mean
        # Solve the linear system instead of explicit inverse
        y = torch.linalg.solve_triangular(L, diff.T, upper=False)
        md_squared = torch.sum(y**2, dim=0)
        md = torch.sqrt(md_squared)
    except RuntimeError:
        # Fallback to eigendecomposition for extremely ill-conditioned matrices
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        # Regularize small eigenvalues
        eigenvalues = torch.clamp(eigenvalues, min=1e-6)
        # Compute Mahalanobis distance using eigendecomposition
        diff = x - mean
        scaled_diff = (
            diff @ eigenvectors @ torch.diag(1.0 / torch.sqrt(eigenvalues)) @ eigenvectors.T
        )
        md_squared = torch.sum(scaled_diff**2, dim=1)
        md = torch.sqrt(md_squared)

    # Apply reduction
    return apply_reduction(md, reduction)


def typicality_score(
    model_output: Union[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    x: Union[torch.Tensor, np.ndarray],
    n_samples: int = 100,
    reduction: str = "none",
) -> Union[torch.Tensor, float]:
    """
    Calculate typicality score for OOD detection using predictive uncertainty.

    The typicality score measures how typical a test sample is under the model's
    predicted distribution, useful for detecting OOD samples.

    Args:
        model_output: Predicted distribution parameters (mean, variance) or a dict with these keys
        x: Input features [batch_size, n_features]
        n_samples: Number of Monte Carlo samples to draw
        reduction: How to reduce the scores ("none", "mean", "sum")

    Returns:
        Typicality scores (lower values indicate OOD samples)
    """
    x = ensure_batch_dim(convert_to_tensor(x))

    # Extract predictive distribution parameters
    if isinstance(model_output, tuple):
        mean, var = model_output
    elif isinstance(model_output, dict):
        mean = model_output.get("mean", model_output.get("loc"))
        var = model_output.get("variance", model_output.get("var"))
    else:
        raise ValueError(
            "model_output must be a tuple (mean, var) or a dict with 'mean'/'variance' keys"
        )

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
    samples: Union[torch.Tensor, np.ndarray], n_bins: int = 10, reduction: str = "none"
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
    x_test: Union[torch.Tensor, np.ndarray],
    x_reference: Union[torch.Tensor, np.ndarray],
    bandwidth: float = 1.0,
    reduction: str = "none",
) -> Union[torch.Tensor, float]:
    """
    Calculate kernel density score for OOD detection.

    The kernel density score measures how similar a test sample is to
    a set of reference samples (often training data).

    Args:
        x_test: Test samples [batch_size, n_features]
        x_reference: Reference samples [n_reference, n_features]
        bandwidth: Bandwidth for RBF kernel
        reduction: How to reduce the scores ("none", "mean", "sum")

    Returns:
        Kernel density scores (lower values indicate OOD samples)
    """
    x_test = ensure_batch_dim(convert_to_tensor(x_test))
    x_reference = ensure_batch_dim(convert_to_tensor(x_reference))

    batch_size = x_test.shape[0]
    n_reference = x_reference.shape[0]

    # Calculate pairwise distances efficiently
    # Expand dimensions for broadcasting
    x_expanded = x_test.unsqueeze(1)  # [batch_size, 1, n_features]
    ref_expanded = x_reference.unsqueeze(0)  # [1, n_reference, n_features]

    # Calculate squared distances
    dist_sq = torch.sum((x_expanded - ref_expanded) ** 2, dim=2)  # [batch_size, n_reference]

    # Apply RBF kernel
    kernel_values = torch.exp(-dist_sq / (2 * bandwidth**2))

    # Average over reference points
    density_scores = torch.mean(kernel_values, dim=1)

    # Apply reduction
    return apply_reduction(density_scores, reduction)


def ood_metrics_report(
    model_output: Optional[
        Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
    ] = None,
    x_test: Optional[Union[torch.Tensor, np.ndarray]] = None,
    x_reference: Optional[Union[torch.Tensor, np.ndarray]] = None,
    mean: Optional[Union[torch.Tensor, np.ndarray]] = None,
    cov: Optional[Union[torch.Tensor, np.ndarray]] = None,
    samples: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Generate a comprehensive report on OOD detection metrics.

    Args:
        model_output: Model predictions with distribution parameters (optional)
        x_test: Test samples to evaluate for OOD
        x_reference: Reference data for kernel density (optional)
        mean: Mean for Mahalanobis distance (optional)
        cov: Covariance for Mahalanobis distance (optional)
        samples: Predictive samples for entropy calculation (optional)

    Returns:
        Dictionary with OOD detection metrics
    """
    metrics = {}

    # Calculate Mahalanobis distance if mean and covariance provided
    if mean is not None and cov is not None and x_test is not None:
        metrics["mahalanobis_distance"] = mahalanobis_distance(
            x_test, mean, cov, reduction="mean"
        ).item()

    # Calculate typicality score if model provided
    if model_output is not None and x_test is not None:
        metrics["typicality_score"] = typicality_score(
            model_output, x_test, reduction="mean"
        ).item()

    # Calculate kernel density if reference data provided
    if x_reference is not None and x_test is not None:
        metrics["kernel_density"] = kernel_density_score(
            x_test, x_reference, reduction="mean"
        ).item()

    # Calculate entropy if samples provided
    if samples is not None:
        metrics["entropy"] = entropy_score(samples, reduction="mean").item()

    return metrics
