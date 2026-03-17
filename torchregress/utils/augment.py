"""
Data augmentation utilities.

This module provides various data augmentation techniques for enhancing
regression and classification models.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
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

    def apply(  # type: ignore[override]
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compatibility shim for older torchregress augmentation usage."""
        return self.augment(x, y)


class GaussianNoise(Augmentation):
    """
    Add Gaussian noise to input features.
    """

    def __init__(
        self,
        std: Union[float, torch.Tensor] = 0.1,
        probability: float = 0.5,
    ):
        super().__init__(probability)
        self.std = std

    def augment(
        self, x: torch.Tensor, y: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Add Gaussian noise to input features.
        """
        noise = torch.randn_like(x) * self.std
        return x + noise, y


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
        alpha: float = 0.01,
        probability: float = 0.5,
    ):
        super().__init__(probability)
        self.model = model
        self.loss_fn = loss_fn
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = alpha

    def augment(
        self, x: torch.Tensor, y: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate and apply adversarial perturbation.
        """
        if y is None:
            raise ValueError("Target values must be provided for adversarial augmentation")

        x_adv = x.clone().detach().requires_grad_(True)

        if self.model.training:
            for _ in range(self.steps):
                output = self.model(x_adv)
                loss = self.loss_fn(output, y)
                self.model.zero_grad()
                loss.backward()
                with torch.no_grad():
                    grad = x_adv.grad
                    if grad is None:
                        raise RuntimeError(
                            "Adversarial augmentation expected gradients but got None"
                        )
                    grad_sign = torch.sign(grad)
                    x_adv.data = x_adv.data + self.alpha * grad_sign
                    x_adv.data = torch.min(
                        torch.max(x_adv.data, x - self.epsilon), x + self.epsilon
                    )
                grad = x_adv.grad
                if grad is not None:
                    grad.zero_()

        return x_adv.detach(), y


class MixUp(Augmentation):
    """
    MixUp augmentation for regression.
    """

    def __init__(self, alpha: float = 0.2, probability: float = 0.5):
        super().__init__(probability)
        self.alpha = alpha

    def augment(
        self, x: torch.Tensor, y: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply MixUp augmentation.
        """
        if y is None:
            raise ValueError("Target values must be provided for MixUp")

        lam = np.random.beta(self.alpha, self.alpha)
        indices = torch.randperm(x.shape[0], device=x.device)

        mixed_x = lam * x + (1 - lam) * x[indices]
        mixed_y = lam * y + (1 - lam) * y[indices]

        return mixed_x, mixed_y


class FeatureMask(Augmentation):
    """
    Randomly mask (set to zero) some features.
    """

    def __init__(self, mask_ratio: float = 0.1, probability: float = 0.5):
        super().__init__(probability)
        if not 0 <= mask_ratio < 1:
            raise ValueError(f"Mask ratio must be between 0 and 1, got {mask_ratio}")
        self.mask_ratio = mask_ratio

    def augment(
        self, x: torch.Tensor, y: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Randomly mask features.
        """
        x_aug = x.clone()
        n_mask = int(x.shape[1] * self.mask_ratio)
        if n_mask == 0:
            return x, y

        for i in range(x.shape[0]):
            mask_indices = torch.randperm(x.shape[1], device=x.device)[:n_mask]
            x_aug[i, mask_indices] = 0.0

        return x_aug, y


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
                return list((x + noise).unbind(0))

            # Full covariance Gaussian
            try:
                mvn = MultivariateNormal(
                    torch.zeros(n_features, device=device, dtype=x.dtype), sigma_tensor
                )
                noise = mvn.sample((self.n_samples, batch_size))
                return list((x + noise).unbind(0))
            except (RuntimeError, ValueError):
                diag = torch.diagonal(sigma_tensor, dim1=-2, dim2=-1)
                noise = torch.randn(
                    self.n_samples, batch_size, n_features, device=device, dtype=x.dtype
                ) * torch.sqrt(diag).view(1, 1, -1)
                return list((x + noise).unbind(0))

        # uniform
        scale_factor = 1.732  # sqrt(3)
        if sigma_tensor.ndim <= 1:
            sigma_vec = sigma_tensor if sigma_tensor.ndim == 1 else sigma_tensor.expand(n_features)
            half_range = sigma_vec.view(1, 1, -1) * scale_factor
            noise = (
                torch.rand(self.n_samples, batch_size, n_features, device=device, dtype=x.dtype) * 2
                - 1
            ) * half_range
            return list((x + noise).unbind(0))

        diag = torch.diagonal(sigma_tensor, dim1=-2, dim2=-1)
        half_range = torch.sqrt(diag).view(1, 1, -1) * scale_factor
        noise = (
            torch.rand(self.n_samples, batch_size, n_features, device=device, dtype=x.dtype) * 2 - 1
        ) * half_range
        return list((x + noise).unbind(0))

    def generate_and_stack(self, x: torch.Tensor) -> torch.Tensor:
        samples = self.forward(x)
        return torch.stack(samples)
