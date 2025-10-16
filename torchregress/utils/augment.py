"""
Data augmentation utilities.

This module provides various data augmentation techniques for enhancing
regression and classification models.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


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
            return self.apply(x, y)
        return x, y

    def apply(
        self, x: torch.Tensor, y: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply the actual augmentation. Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")


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

    def apply(
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

    def apply(
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
                    grad_sign = torch.sign(x_adv.grad)
                    x_adv.data = x_adv.data + self.alpha * grad_sign
                    x_adv.data = torch.min(
                        torch.max(x_adv.data, x - self.epsilon), x + self.epsilon
                    )
                x_adv.grad.zero_()

        return x_adv.detach(), y


class MixUp(Augmentation):
    """
    MixUp augmentation for regression.
    """

    def __init__(self, alpha: float = 0.2, probability: float = 0.5):
        super().__init__(probability)
        self.alpha = alpha

    def apply(
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

    def apply(
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
            mask_indices = torch.randperm(x.shape[1])[:n_mask]
            x_aug[i, mask_indices] = 0.0

        return x_aug, y