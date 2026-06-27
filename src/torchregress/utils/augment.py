"""
Data augmentation utilities.

This module provides various data augmentation techniques for enhancing
regression and classification models.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


class Augmentation(nn.Module):
    """
    Base class for regression-specific data augmentation techniques.

    Args:
        probability: Probability of applying the augmentation
    """

    def __init__(self, probability: float = 0.5):
        super().__init__()
        if not 0 <= probability <= 1:
            raise ValueError(f"Probability must be between 0 and 1, got {probability}")
        self.probability = probability

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply augmentation to input data.
        """
        if torch.rand(1) < self.probability:
            return self.augment(x, y)
        return x, y

    def augment(
        self, x: torch.Tensor, y: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply the actual augmentation. Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")


class Adversarial(Augmentation):
    """
    Generate adversarial examples for regression.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        epsilon: float = 0.05,
        steps: int = 3,
        alpha: Optional[float] = None,
        probability: float = 0.5,
        random_start: bool = False,
    ):
        super().__init__(probability)
        if epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {epsilon}")
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        self.model = model
        self.loss_fn = loss_fn
        self.epsilon = float(epsilon)
        self.steps = int(steps)
        self.alpha = float(alpha) if alpha is not None else float(epsilon) / float(steps)
        self.random_start = random_start

    def augment(
        self, x: torch.Tensor, y: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate and apply adversarial perturbation.
        """
        if y is None:
            raise ValueError("Target values must be provided for adversarial augmentation")

        x_base = x.detach()
        if self.random_start and self.epsilon > 0.0:
            noise = torch.empty_like(x_base).uniform_(-self.epsilon, self.epsilon)
            x_adv = (x_base + noise).detach()
        else:
            x_adv = x_base.clone()

        if self.model.training:
            for _ in range(self.steps):
                x_adv = x_adv.detach().requires_grad_(True)
                output = self.model(x_adv)
                loss = self.loss_fn(output, y)
                grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
                with torch.no_grad():
                    grad_sign = torch.sign(grad)
                    x_adv = x_adv + self.alpha * grad_sign
                    delta = torch.clamp(x_adv - x_base, min=-self.epsilon, max=self.epsilon)
                    x_adv = x_base + delta

        return x_adv.detach(), y


class EnsemblePerturbationAugmenter(nn.Module):
    """
    Generate multiple perturbed versions of inputs for ensemble-style EIV losses.
    """

    def __init__(
        self,
        n_samples: int = 20,
        perturb_method: str = "gaussian",
        sigma: Union[float, torch.Tensor] = 0.1,
        feature_wise: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")
        if perturb_method not in {"gaussian", "uniform"}:
            raise ValueError("perturb_method must be 'gaussian' or 'uniform'")
        self.n_samples = n_samples
        self.perturb_method = perturb_method
        self.sigma = sigma
        self.feature_wise = feature_wise
        self.device = device

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        batch_size, n_features = x.shape
        device = self.device if self.device is not None else x.device

        if isinstance(self.sigma, (int, float)):
            if self.feature_wise:
                sigma_tensor = torch.ones(n_features, device=device, dtype=x.dtype) * self.sigma
            else:
                sigma_tensor = torch.tensor(self.sigma, device=device, dtype=x.dtype)
        else:
            sigma_tensor = self.sigma.to(device=device, dtype=x.dtype)
            if sigma_tensor.ndim == 0 and self.feature_wise:
                sigma_tensor = sigma_tensor.expand(n_features)

            if sigma_tensor.ndim == 1 and sigma_tensor.shape[0] != n_features:
                raise ValueError(
                    f"Sigma vector shape {tuple(sigma_tensor.shape)} doesn't match "
                    f"feature dimension {n_features}"
                )
            if sigma_tensor.ndim == 2 and sigma_tensor.shape != (n_features, n_features):
                raise ValueError(
                    f"Sigma matrix shape {tuple(sigma_tensor.shape)} doesn't match "
                    f"expected shape ({n_features}, {n_features})"
                )

        if self.perturb_method == "gaussian":
            if sigma_tensor.ndim <= 1:
                sigma_vec = (
                    sigma_tensor if sigma_tensor.ndim == 1 else sigma_tensor.expand(n_features)
                )
                noise = torch.randn(
                    self.n_samples, batch_size, n_features, device=device, dtype=x.dtype
                ) * sigma_vec.view(1, 1, -1)
                return list((x.unsqueeze(0) + noise).unbind(0))

            # Full covariance Gaussian
            try:
                mvn = MultivariateNormal(
                    torch.zeros(n_features, device=device, dtype=x.dtype), sigma_tensor
                )
                noise = mvn.sample((self.n_samples, batch_size))
                return list((x.unsqueeze(0) + noise).unbind(0))
            except (RuntimeError, ValueError):
                diag = torch.diagonal(sigma_tensor, dim1=-2, dim2=-1)
                noise = torch.randn(
                    self.n_samples, batch_size, n_features, device=device, dtype=x.dtype
                ) * torch.sqrt(diag).view(1, 1, -1)
                return list((x.unsqueeze(0) + noise).unbind(0))

        # uniform
        scale_factor = 1.732  # sqrt(3)
        if sigma_tensor.ndim <= 1:
            sigma_vec = sigma_tensor if sigma_tensor.ndim == 1 else sigma_tensor.expand(n_features)
            half_range = sigma_vec.view(1, 1, -1) * scale_factor
            noise = (
                torch.rand(self.n_samples, batch_size, n_features, device=device, dtype=x.dtype) * 2
                - 1
            ) * half_range
            return list((x.unsqueeze(0) + noise).unbind(0))

        diag = torch.diagonal(sigma_tensor, dim1=-2, dim2=-1)
        half_range = torch.sqrt(diag).view(1, 1, -1) * scale_factor
        noise = (
            torch.rand(self.n_samples, batch_size, n_features, device=device, dtype=x.dtype) * 2 - 1
        ) * half_range
        return list((x.unsqueeze(0) + noise).unbind(0))
