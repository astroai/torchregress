"""
Out-of-distribution (OOD) detection metrics for regression models.
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.distributions import Normal
from torchmetrics import Metric

from torchregress.metrics.utils import convert_to_tensor, ensure_batch_dim


class MahalanobisDistance(Metric):
    """
    Calculate Mahalanobis distance for OOD detection.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("distances", default=[], dist_reduce_fx="cat")

    def update(
        self, x: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor
    ) -> None:
        """Update state with predictions and targets."""
        x = ensure_batch_dim(convert_to_tensor(x))
        mean = convert_to_tensor(mean)
        cov = convert_to_tensor(cov)

        if x.device != mean.device:
            mean = mean.to(x.device)
        if x.device != cov.device:
            cov = cov.to(x.device)

        try:
            L = torch.linalg.cholesky(cov + torch.eye(cov.shape[0], device=cov.device) * 1e-6)
            diff = x - mean
            y = torch.linalg.solve_triangular(L, diff.T, upper=False)
            md_squared = torch.sum(y**2, dim=0)
            md = torch.sqrt(md_squared)
        except RuntimeError:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            eigenvalues = torch.clamp(eigenvalues, min=1e-6)
            diff = x - mean
            scaled_diff = (
                diff @ eigenvectors @ torch.diag(1.0 / torch.sqrt(eigenvalues)) @ eigenvectors.T
            )
            md_squared = torch.sum(scaled_diff**2, dim=1)
            md = torch.sqrt(md_squared)

        self.distances.append(md)

    def compute(self) -> torch.Tensor:
        """Compute Mahalanobis distance."""
        return torch.mean(torch.cat(self.distances))


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
            mean = model_output.get("mean", model_output.get("loc"))
            var = model_output.get("variance", model_output.get("var"))
        else:
            raise ValueError(
                "model_output must be a tuple (mean, var) or a dict with 'mean'/'variance' keys"
            )

        dist = Normal(mean, torch.sqrt(var))
        samples = dist.sample((self.n_samples,))
        log_probs = dist.log_prob(samples)

        if log_probs.dim() > 2:
            log_probs = log_probs.sum(dim=-1)

        typicality = torch.mean(log_probs, dim=0)
        self.scores.append(typicality)

    def compute(self) -> torch.Tensor:
        """Compute typicality score."""
        return torch.mean(torch.cat(self.scores))


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
        n_samples, batch_size, output_dim = samples.shape

        entropies = torch.zeros(batch_size, output_dim, device=samples.device)

        for i in range(batch_size):
            for j in range(output_dim):
                inst_samples = samples[:, i, j]
                counts, _ = torch.histogram(inst_samples, bins=self.n_bins)
                probs = counts / n_samples
                probs = probs[probs > 0]
                entropies[i, j] = -torch.sum(probs * torch.log(probs))

        total_entropy = torch.sum(entropies, dim=1)
        self.entropies.append(total_entropy)

    def compute(self) -> torch.Tensor:
        """Compute entropy score."""
        return torch.mean(torch.cat(self.entropies))


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

        x_expanded = x_test.unsqueeze(1)
        ref_expanded = x_reference.unsqueeze(0)

        dist_sq = torch.sum((x_expanded - ref_expanded) ** 2, dim=2)
        kernel_values = torch.exp(-dist_sq / (2 * self.bandwidth**2))
        density_scores = torch.mean(kernel_values, dim=1)
        self.scores.append(density_scores)

    def compute(self) -> torch.Tensor:
        """Compute kernel density score."""
        return torch.mean(torch.cat(self.scores))


def mahalanobis_distance(
    x: Union[torch.Tensor, np.ndarray],
    mean: Union[torch.Tensor, np.ndarray],
    cov: Union[torch.Tensor, np.ndarray],
    reduction: str = "none",
) -> torch.Tensor:
    """
    Functional Mahalanobis distance for OOD detection.
    """
    x_t = ensure_batch_dim(convert_to_tensor(x))
    mean_t = convert_to_tensor(mean).to(x_t.device)
    cov_t = convert_to_tensor(cov).to(x_t.device)

    try:
        L = torch.linalg.cholesky(cov_t + torch.eye(cov_t.shape[0], device=cov_t.device) * 1e-6)
        diff = x_t - mean_t
        y = torch.linalg.solve_triangular(L, diff.T, upper=False)
        md_squared = torch.sum(y**2, dim=0)
        md = torch.sqrt(md_squared)
    except RuntimeError:
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_t)
        eigenvalues = torch.clamp(eigenvalues, min=1e-6)
        diff = x_t - mean_t
        scaled_diff = diff @ eigenvectors @ torch.diag(1.0 / torch.sqrt(eigenvalues)) @ eigenvectors.T
        md_squared = torch.sum(scaled_diff**2, dim=1)
        md = torch.sqrt(md_squared)

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
        mean = model_output.get("mean", model_output.get("loc"))
        var = model_output.get("variance", model_output.get("var"))
    else:
        raise ValueError(
            "model_output must be a tuple (mean, var) or a dict with 'mean'/'variance' keys"
        )

    mean = convert_to_tensor(mean)
    var = convert_to_tensor(var)
    dist = Normal(mean, torch.sqrt(var))

    if x is None:
        log_prob = dist.sample((100,)).log_prob(dist.sample((100,))).mean(dim=0)
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
    return log_prob


def entropy_score(
    samples: Union[torch.Tensor, np.ndarray],
    n_bins: int = 10,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Functional entropy score from predictive samples.
    """
    samples_t = convert_to_tensor(samples)
    n_samples, batch_size, output_dim = samples_t.shape

    entropies = torch.zeros(batch_size, output_dim, device=samples_t.device)
    for i in range(batch_size):
        for j in range(output_dim):
            inst_samples = samples_t[:, i, j]
            counts, _ = torch.histogram(inst_samples, bins=n_bins)
            probs = counts / n_samples
            probs = probs[probs > 0]
            entropies[i, j] = -torch.sum(probs * torch.log(probs))

    total_entropy = torch.sum(entropies, dim=1)

    if reduction == "mean":
        return torch.mean(total_entropy)
    if reduction == "sum":
        return torch.sum(total_entropy)
    return total_entropy


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

    x_expanded = x_test_t.unsqueeze(1)
    ref_expanded = x_ref_t.unsqueeze(0)

    dist_sq = torch.sum((x_expanded - ref_expanded) ** 2, dim=2)
    kernel_values = torch.exp(-dist_sq / (2 * bandwidth**2))
    density_scores = torch.mean(kernel_values, dim=1)

    if reduction == "mean":
        return torch.mean(density_scores)
    if reduction == "sum":
        return torch.sum(density_scores)
    return density_scores


def ood_metrics_report(
    model_output: Optional[Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
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
        results["mahalanobis_distance"] = mahalanobis_distance(
            x_test, mean, cov, reduction="mean"
        )

    if x_reference is not None and x_test is not None:
        results["kernel_density"] = kernel_density_score(x_test, x_reference, reduction="mean")

    if samples is not None:
        results["entropy"] = entropy_score(samples, reduction="mean")

    return results
