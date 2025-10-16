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