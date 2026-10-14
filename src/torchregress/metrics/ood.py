"""
Out-of-distribution (OOD) detection metrics for regression models.
"""

from typing import Any, Dict, Optional, Tuple, Union, cast

import numpy as np
import torch
from torch.distributions import Normal
from torchmetrics import Metric

from torchregress.metrics.utils import (
    convert_to_tensor,
    ensure_batch_dim,
    float_dtype,
    metric_state_list,
)

_BLOCK_ELEMS = 1 << 24


def _kernel_density_blocked(
    x_test: torch.Tensor, x_reference: torch.Tensor, bandwidth: float
) -> torch.Tensor:
    """Per-row mean Gaussian kernel value, processing query rows in memory-bounded blocks."""
    n_ref = x_reference.shape[0]
    block = max(1, _BLOCK_ELEMS // max(1, n_ref))
    out = []
    for start in range(0, max(1, x_test.shape[0]), block):
        # M-MET-003: direct differences; the matmul path loses float32 precision.
        dists = torch.cdist(
            x_test[start : start + block],
            x_reference,
            p=2,
            compute_mode="donot_use_mm_for_euclid_dist",
        )
        kernel_values = torch.exp(-(dists**2) / (2 * bandwidth**2))
        out.append(torch.mean(kernel_values, dim=1))
    return torch.cat(out)


def _mahalanobis(
    x: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor, jitter: float
) -> torch.Tensor:
    """Per-row Mahalanobis distance with a covariance-relative jitter (M-MET-011)."""
    dtype = torch.promote_types(torch.promote_types(x.dtype, mean.dtype), cov.dtype)
    if not dtype.is_floating_point:
        dtype = torch.get_default_dtype()
    x, mean, cov = x.to(dtype), mean.to(dtype), cov.to(dtype)
    eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    level = jitter * torch.diagonal(cov).mean().abs()
    level = torch.clamp(level, min=torch.finfo(cov.dtype).tiny)
    diff = x - mean
    try:
        L = torch.linalg.cholesky(cov + level * eye)
        y = torch.linalg.solve_triangular(L, diff.T, upper=False)
        return torch.sqrt(torch.sum(y**2, dim=0))
    except RuntimeError:
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        eigenvalues = torch.clamp(eigenvalues, min=level)
        # Coverage invariants (TOR003): chain .to() on torch.diag because
        # torch.diag does not accept device=/dtype= kwargs natively.
        inv_sqrt = torch.diag(1.0 / torch.sqrt(eigenvalues)).to(
            device=eigenvalues.device, dtype=eigenvalues.dtype
        )
        scaled_diff = diff @ eigenvectors @ inv_sqrt @ eigenvectors.T
        return torch.sqrt(torch.sum(scaled_diff**2, dim=1))


class MahalanobisDistance(Metric):
    """Mean Mahalanobis distance for OOD detection.

    Parameters
    ----------
    jitter : float
        Relative diagonal regulariser: ``jitter * mean(diag(cov)) * I`` is added
        to the covariance before the Cholesky factorisation, so the distance is
        invariant to rescaling the features (see :func:`mahalanobis_distance`).
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, jitter: float = 1e-6, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.jitter = jitter
        self.add_state("distances", default=[], dist_reduce_fx="cat")

    def update(self, x: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        x = ensure_batch_dim(convert_to_tensor(x))
        mean = convert_to_tensor(mean)
        cov = convert_to_tensor(cov)

        if x.device != mean.device:
            mean = mean.to(x.device)
        if x.device != cov.device:
            cov = cov.to(x.device)

        md = _mahalanobis(x, mean, cov, self.jitter)
        metric_state_list[torch.Tensor](self.distances).append(md)

    def compute(self) -> torch.Tensor:
        """Compute Mahalanobis distance."""
        return torch.mean(torch.cat(metric_state_list[torch.Tensor](self.distances)))


class TypicalityScore(Metric):
    """
    Calculate typicality score for OOD detection using predictive uncertainty.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, n_samples: int = 100, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.add_state("scores", default=[], dist_reduce_fx="cat")

    def update(
        self, model_output: Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        """Update state with predictions and targets."""
        if isinstance(model_output, tuple):
            mean, var = model_output
        elif isinstance(model_output, dict):
            mean_opt = model_output.get("mean", model_output.get("loc"))
            var_opt = model_output.get("variance", model_output.get("var"))
            if mean_opt is None or var_opt is None:
                raise ValueError("model_output dict must contain mean/loc and variance/var entries")
            mean = mean_opt
            var = var_opt
        else:
            raise ValueError(
                "model_output must be a tuple (mean, var) or a dict with 'mean'/'variance' keys"
            )
        # TR-MET-16: guard against non-finite inputs and var <= 0 (silent NaN).
        mean = torch.as_tensor(mean)
        var = torch.as_tensor(var)
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            raise ValueError("mean contains NaN or infinite values")
        if torch.isnan(var).any() or torch.isinf(var).any():
            raise ValueError("variance contains NaN or infinite values")
        var = var.clamp_min(1e-12)

        dist = Normal(mean, torch.sqrt(var))

        samples = dist.sample((self.n_samples,))
        log_probs = dist.log_prob(samples)

        if log_probs.dim() > 2:
            log_probs = log_probs.sum(dim=-1)

        typicality = torch.mean(log_probs, dim=0)
        metric_state_list[torch.Tensor](self.scores).append(typicality)

    def compute(self) -> torch.Tensor:
        """Compute typicality score."""
        return torch.mean(torch.cat(metric_state_list[torch.Tensor](self.scores)))


class EntropyScore(Metric):
    """
    Calculate entropy of predictive distribution for OOD detection.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, n_bins: int = 10, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.n_bins = n_bins
        self.add_state("entropies", default=[], dist_reduce_fx="cat")

    def update(self, samples: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        samples = convert_to_tensor(samples)

        # Vectorized calculation
        entropies = _batched_entropy(samples, self.n_bins)

        total_entropy = torch.sum(entropies, dim=1)
        metric_state_list[torch.Tensor](self.entropies).append(total_entropy)

    def compute(self) -> torch.Tensor:
        """Compute entropy score."""
        return torch.mean(torch.cat(metric_state_list[torch.Tensor](self.entropies)))


class KernelDensityScore(Metric):
    """
    Calculate kernel density score for OOD detection.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, bandwidth: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.bandwidth = bandwidth
        self.add_state("scores", default=[], dist_reduce_fx="cat")

    def update(self, x_test: torch.Tensor, x_reference: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        x_test = ensure_batch_dim(convert_to_tensor(x_test))
        x_reference = ensure_batch_dim(convert_to_tensor(x_reference))

        # Memory optimization: query rows are processed in blocks so the distance
        # matrix never exceeds ~_BLOCK_ELEMS elements.
        density_scores = _kernel_density_blocked(x_test, x_reference, self.bandwidth)
        metric_state_list[torch.Tensor](self.scores).append(density_scores)

    def compute(self) -> torch.Tensor:
        """Compute kernel density score."""
        return torch.mean(torch.cat(metric_state_list[torch.Tensor](self.scores)))


def mahalanobis_distance(
    x: Union[torch.Tensor, np.ndarray],
    mean: Union[torch.Tensor, np.ndarray],
    cov: Union[torch.Tensor, np.ndarray],
    reduction: str = "none",
    jitter: float = 1e-6,
) -> torch.Tensor:
    """Functional Mahalanobis distance for OOD detection.

    ``sqrt((x - mean)^T (cov + jitter * mean(diag(cov)) * I)^{-1} (x - mean))``.

    Parameters
    ----------
    x : Tensor or ndarray
        Points ``[N, D]`` (or a single point ``[D]``).
    mean : Tensor or ndarray
        Reference mean ``[D]``.
    cov : Tensor or ndarray
        Reference covariance ``[D, D]``.
    reduction : {"none", "mean", "sum"}
        Reduction over points.
    jitter : float
        *Relative* diagonal regulariser, scaled by the mean variance so the
        distance is invariant to rescaling the features (an absolute ``1e-6``
        dominated covariances with variances ~1e-8). If the jittered matrix is
        still not positive definite, eigenvalues are floored at the same level.

    Returns
    -------
    Tensor
        Distances ``[N]`` or their reduction.
    """
    x_t = ensure_batch_dim(convert_to_tensor(x))
    mean_t = convert_to_tensor(mean).to(x_t.device)
    cov_t = convert_to_tensor(cov).to(x_t.device)

    md = _mahalanobis(x_t, mean_t, cov_t, jitter)

    if reduction == "mean":
        return torch.mean(md)
    if reduction == "sum":
        return torch.sum(md)
    return md


def typicality_score(
    model_output: Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    x: Optional[Union[torch.Tensor, np.ndarray]] = None,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Functional typicality score from predictive mean/variance.
    """
    if isinstance(model_output, tuple):
        mean, var = model_output
    elif isinstance(model_output, dict):
        mean_opt = model_output.get("mean", model_output.get("loc"))
        var_opt = model_output.get("variance", model_output.get("var"))
        if mean_opt is None or var_opt is None:
            raise ValueError("model_output dict must contain mean/loc and variance/var entries")
        mean = mean_opt
        var = var_opt
    else:
        raise ValueError(
            "model_output must be a tuple (mean, var) or a dict with 'mean'/'variance' keys"
        )

    # TR-MET-16: guard against non-finite inputs and var <= 0 (silent NaN).
    mean = convert_to_tensor(mean)
    var = convert_to_tensor(var)
    if torch.isnan(mean).any() or torch.isinf(mean).any():
        raise ValueError("mean contains NaN or infinite values")
    if torch.isnan(var).any() or torch.isinf(var).any():
        raise ValueError("variance contains NaN or infinite values")
    var = var.clamp_min(1e-12)
    dist = Normal(mean, torch.sqrt(var))

    if x is None:
        samples = dist.sample((100,))
        log_prob = dist.log_prob(samples).mean(dim=0)
    else:
        x_t = convert_to_tensor(x).to(mean.device)
        if mean.dim() == 1 and x_t.dim() > 1 and mean.shape[0] == x_t.shape[0]:
            mean = mean.unsqueeze(-1)
            var = var.unsqueeze(-1)
            dist = Normal(mean, torch.sqrt(var))
        log_prob = dist.log_prob(x_t)
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(dim=-1)

    if reduction == "mean":
        return torch.mean(log_prob)
    if reduction == "sum":
        return torch.sum(log_prob)
    return cast(torch.Tensor, log_prob)


def entropy_score(
    samples: Union[torch.Tensor, np.ndarray],
    n_bins: int = 10,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Functional entropy score from predictive samples.
    """
    samples_t = convert_to_tensor(samples)

    # Vectorized calculation
    entropies = _batched_entropy(samples_t, n_bins)

    total_entropy = torch.sum(entropies, dim=1)

    if reduction == "mean":
        return torch.mean(total_entropy)
    if reduction == "sum":
        return torch.sum(total_entropy)
    return total_entropy


def _batched_entropy(samples: torch.Tensor, n_bins: int) -> torch.Tensor:
    """
    Compute entropy for batched samples using vectorized histogram.

    Args:
        samples: Tensor of shape [n_samples, batch_size, output_dim]
        n_bins: Number of bins for histogram

    Returns:
        Tensor of shape [batch_size, output_dim]
    """
    n_samples, batch_size, output_dim = samples.shape

    # 1. Compute min/max per distribution
    min_vals = samples.min(dim=0, keepdim=True).values
    max_vals = samples.max(dim=0, keepdim=True).values
    ranges = max_vals - min_vals

    # Handle zero range (all samples equal) -> entropy is 0
    # Set range to 1 to avoid division by zero
    ranges = torch.where(ranges == 0, torch.ones_like(ranges), ranges)

    # 2. Normalize to [0, 1)
    norm_samples = (samples - min_vals) / ranges
    # Clamp to handle numerical instability or max value being exactly at edge
    norm_samples = torch.clamp(norm_samples, 0.0, 0.999999)

    # 3. Bin indices
    bin_idx = (norm_samples * n_bins).long()  # [n_samples, batch_size, output_dim]

    # 4. Count bins
    # Memory optimization: Use bincount on flattened tensor instead of one_hot
    # This avoids creating [n_samples, batch_size, output_dim, n_bins] tensor

    # Flatten batch and output dims: [n_samples, M] where M = batch_size * output_dim
    M = batch_size * output_dim
    bin_idx_flat = bin_idx.reshape(n_samples, M)

    # Add offsets to bin indices to separate histograms
    # offsets: [1, M]
    offsets = (torch.arange(M, device=samples.device) * n_bins).unsqueeze(0)

    # Global bin indices: [n_samples, M]
    global_bin_idx = bin_idx_flat + offsets

    # Flatten everything: [n_samples * M]
    global_bin_idx_flat = global_bin_idx.reshape(-1)

    # Compute counts using bincount
    # Result size: M * n_bins
    counts_flat = torch.bincount(global_bin_idx_flat, minlength=M * n_bins).to(float_dtype(samples))

    # Reshape back to [batch_size, output_dim, n_bins]
    counts = counts_flat.reshape(batch_size, output_dim, n_bins)

    # 5. Compute probs and entropy
    probs = counts / n_samples

    # Calculate -sum(p * log(p))
    # Handle p=0 case where p*log(p) should be 0
    # We use a mask for positive probabilities
    positive_probs = probs > 0
    entropy_per_bin = torch.zeros_like(probs)
    entropy_per_bin[positive_probs] = probs[positive_probs] * torch.log(probs[positive_probs])

    entropies = -torch.sum(entropy_per_bin, dim=-1)  # [batch_size, output_dim]
    # ponytail: discrete bin entropy (depends on bin width), not differential entropy;
    # fine for relative OOD ranking but absolute values depend on bin count/range.
    return entropies


def kernel_density_score(
    x_test: Union[torch.Tensor, np.ndarray],
    x_reference: Union[torch.Tensor, np.ndarray],
    bandwidth: float = 1.0,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Functional kernel density score for OOD detection.
    """
    x_test_t = ensure_batch_dim(convert_to_tensor(x_test))
    x_ref_t = ensure_batch_dim(convert_to_tensor(x_reference))

    density_scores = _kernel_density_blocked(x_test_t, x_ref_t, bandwidth)

    if reduction == "mean":
        return torch.mean(density_scores)
    if reduction == "sum":
        return torch.sum(density_scores)
    return density_scores


def ood_metrics_report(
    model_output: Optional[
        Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
    ] = None,
    x_test: Optional[Union[torch.Tensor, np.ndarray]] = None,
    x_reference: Optional[Union[torch.Tensor, np.ndarray]] = None,
    mean: Optional[Union[torch.Tensor, np.ndarray]] = None,
    cov: Optional[Union[torch.Tensor, np.ndarray]] = None,
    samples: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Generate an OOD metrics report based on available inputs.
    """
    results: Dict[str, torch.Tensor] = {}

    if model_output is not None and x_test is not None:
        results["typicality_score"] = typicality_score(model_output, x_test, reduction="mean")

    if mean is not None and cov is not None and x_test is not None:
        results["mahalanobis_distance"] = mahalanobis_distance(x_test, mean, cov, reduction="mean")

    if x_reference is not None and x_test is not None:
        results["kernel_density"] = kernel_density_score(x_test, x_reference, reduction="mean")

    if samples is not None:
        results["entropy"] = entropy_score(samples, reduction="mean")

    return results
