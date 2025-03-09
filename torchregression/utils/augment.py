"""
Data augmentation utilities for regression.

This module provides data augmentation techniques specifically designed
for regression tasks to improve model robustness and accuracy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict, Tuple, List, Callable

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
    
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply augmentation to input data.
        
        Args:
            x: Input features [batch_size, n_features]
            y: Optional target values [batch_size, n_outputs]
            
        Returns:
            Tuple of (augmented_x, augmented_y)
        """
        if torch.rand(1) < self.probability:
            return self.apply(x, y)
        return x, y
    
    def apply(self, x: torch.Tensor, y: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply the actual augmentation. Must be implemented by subclasses.
        
        Args:
            x: Input features [batch_size, n_features]
            y: Optional target values [batch_size, n_outputs]
            
        Returns:
            Tuple of (augmented_x, augmented_y)
        """
        raise NotImplementedError("Subclasses must implement this method")


class GaussianNoiseAugmentation(Augmentation):
    """
    Add Gaussian noise to input features.
    
    Args:
        std: Standard deviation of the noise
        probability: Probability of applying the augmentation
        per_feature: Whether to use different noise for each feature
    """
    def __init__(self, std: Union[float, torch.Tensor] = 0.1, probability: float = 0.5, per_feature: bool = False):
        super().__init__(probability)
        self.std = std
        self.per_feature = per_feature
    
    def apply(self, x: torch.Tensor, y: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Add Gaussian noise to input features.
        
        Args:
            x: Input features [batch_size, n_features]
            y: Optional target values [batch_size, n_outputs]
            
        Returns:
            Tuple of (augmented_x, y)
        """
        if self.per_feature and isinstance(self.std, torch.Tensor):
            # Different noise per feature
            if self.std.shape[0] != x.shape[1]:
                raise ValueError(f"std shape {self.std.shape} doesn't match feature dim {x.shape[1]}")
            noise = torch.randn_like(x) * self.std.unsqueeze(0).expand_as(x)
        else:
            # Same noise for all features
            noise = torch.randn_like(x) * self.std
        
        return x + noise, y


class AdversarialAugmentation(Augmentation):
    """
    Generate adversarial examples for regression.
    
    This creates perturbed examples that maximize the loss while
    keeping the perturbation small, helping improve robustness.
    
    Args:
        model: The regression model to generate adversarial examples for
        loss_fn: Loss function to maximize
        epsilon: Maximum perturbation magnitude
        steps: Number of optimization steps
        alpha: Step size for optimization
        probability: Probability of applying the augmentation
        targeted: Whether to use targeted adversarial examples
    """
    def __init__(
        self, 
        model: nn.Module,
        loss_fn: nn.Module,
        epsilon: float = 0.05,
        steps: int = 3,
        alpha: float = 0.01,
        probability: float = 0.5,
        targeted: bool = False
    ):
        super().__init__(probability)
        self.model = model
        self.loss_fn = loss_fn
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = alpha
        self.targeted = targeted
    
    def apply(self, x: torch.Tensor, y: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate and apply adversarial perturbation.
        
        Args:
            x: Input features [batch_size, n_features]
            y: Target values [batch_size, n_outputs]
            
        Returns:
            Tuple of (augmented_x, y)
        """
        if y is None:
            raise ValueError("Target values must be provided for adversarial augmentation")
        
        # Make a copy of input that requires gradients
        x_adv = x.clone().detach().requires_grad_(True)
        
        # Only augment if in training mode
        if self.model.training:
            for _ in range(self.steps):
                # Forward pass
                output = self.model(x_adv)
                
                # Calculate loss
                if self.targeted:
                    # For targeted attacks, minimize loss to move toward target
                    loss = -self.loss_fn(output, y)
                else:
                    # For untargeted attacks, maximize loss to move away from ground truth
                    loss = self.loss_fn(output, y)
                
                # Backward pass
                self.model.zero_grad()
                loss.backward()
                
                # Update adversarial example
                with torch.no_grad():
                    # Use the sign of the gradient (FGSM-like)
                    grad_sign = torch.sign(x_adv.grad)
                    x_adv.data = x_adv.data + self.alpha * grad_sign
                    
                    # Project back to valid perturbation region
                    x_adv.data = torch.min(torch.max(x_adv.data, x - self.epsilon), x + self.epsilon)
                
                # Reset gradients
                x_adv.grad.zero_()
        
        # Return the adversarial example without requiring gradients
        return x_adv.detach(), y


class MixtureAugmentation(Augmentation):
    """
    Apply a mixture of different augmentation techniques.
    
    Args:
        augmentations: List of augmentation objects
        probabilities: Optional probability for each augmentation (must sum to 1)
        sequential: Whether to apply augmentations sequentially or choose one
    """
    def __init__(
        self, 
        augmentations: List[Augmentation],
        probabilities: Optional[List[float]] = None,
        sequential: bool = False,
        probability: float = 0.5
    ):
        super().__init__(probability)
        
        self.augmentations = nn.ModuleList(augmentations)
        self.sequential = sequential
        
        # Validate and normalize probabilities
        if probabilities is not None:
            if len(probabilities) != len(augmentations):
                raise ValueError("Number of probabilities must match number of augmentations")
            if abs(sum(probabilities) - 1.0) > 1e-5:
                raise ValueError("Probabilities must sum to 1")
            self.probabilities = probabilities
        else:
            # Equal probabilities
            self.probabilities = [1.0 / len(augmentations) for _ in augmentations]
    
    def apply(self, x: torch.Tensor, y: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply mixture of augmentations.
        
        Args:
            x: Input features [batch_size, n_features]
            y: Optional target values [batch_size, n_outputs]
            
        Returns:
            Tuple of (augmented_x, augmented_y)
        """
        if self.sequential:
            # Apply augmentations in sequence
            for aug in self.augmentations:
                x, y = aug.forward(x, y)
            return x, y
        else:
            # Choose one augmentation based on probabilities
            choice = torch.multinomial(torch.tensor(self.probabilities), 1).item()
            return self.augmentations[choice].forward(x, y)


# Specialized augmentations for regression

class FeatureJitter(Augmentation):
    """
    Add jitter to specific features.
    
    Args:
        feature_indices: Indices of features to jitter
        magnitude: Magnitude of the jitter
        probability: Probability of applying the augmentation
    """
    def __init__(
        self, 
        feature_indices: Union[List[int], torch.Tensor], 
        magnitude: Union[float, torch.Tensor] = 0.1,
        probability: float = 0.5
    ):
        super().__init__(probability)
        self.feature_indices = feature_indices
        self.magnitude = magnitude
    
    def apply(self, x: torch.Tensor, y: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Add jitter to selected features.
        
        Args:
            x: Input features [batch_size, n_features]
            y: Optional target values [batch_size, n_outputs]
            
        Returns:
            Tuple of (augmented_x, y)
        """
        x_aug = x.clone()
        
        # Get noise for selected features
        if isinstance(self.magnitude, torch.Tensor):
            if self.magnitude.shape[0] != len(self.feature_indices):
                raise ValueError("Magnitude tensor must match length of feature_indices")
            
            for i, idx in enumerate(self.feature_indices):
                noise = torch.randn_like(x[:, idx]) * self.magnitude[i]
                x_aug[:, idx] = x[:, idx] + noise
        else:
            for idx in self.feature_indices:
                noise = torch.randn_like(x[:, idx]) * self.magnitude
                x_aug[:, idx] = x[:, idx] + noise
        
        return x_aug, y


class MixUp(Augmentation):
    """
    MixUp augmentation for regression.
    
    Linearly interpolates between batches of examples and targets.
    
    Args:
        alpha: Parameter for the beta distribution
        probability: Probability of applying the augmentation
    """
    def __init__(self, alpha: float = 0.2, probability: float = 0.5):
        super().__init__(probability)
        self.alpha = alpha
    
    def apply(self, x: torch.Tensor, y: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply MixUp augmentation.
        
        Args:
            x: Input features [batch_size, n_features]
            y: Target values [batch_size, n_outputs]
            
        Returns:
            Tuple of (mixed_x, mixed_y)
        """
        if y is None:
            raise ValueError("Target values must be provided for MixUp")
        
        batch_size = x.shape[0]
        device = x.device
        
        # Generate mixing parameter from Beta distribution
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample((batch_size,)).to(device)
        else:
            lam = torch.ones(batch_size, device=device)
            
        # Reshape for broadcasting
        lam = lam.view(-1, 1)
        
        # Generate random indices for mixing
        indices = torch.randperm(batch_size, device=device)
        
        # Mix data points
        mixed_x = lam * x + (1 - lam) * x[indices]
        mixed_y = lam * y + (1 - lam) * y[indices]
        
        return mixed_x, mixed_y


class FeatureMask(Augmentation):
    """
    Randomly mask (set to zero) some features.
    
    Args:
        mask_ratio: Proportion of features to mask
        probability: Probability of applying the augmentation
    """
    def __init__(self, mask_ratio: float = 0.1, probability: float = 0.5):
        super().__init__(probability)
        if not 0 <= mask_ratio < 1:
            raise ValueError(f"Mask ratio must be between 0 and 1, got {mask_ratio}")
        self.mask_ratio = mask_ratio
    
    def apply(self, x: torch.Tensor, y: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Randomly mask features.
        
        Args:
            x: Input features [batch_size, n_features]
            y: Optional target values [batch_size, n_outputs]
            
        Returns:
            Tuple of (augmented_x, y)
        """
        x_aug = x.clone()
        batch_size, n_features = x.shape
        
        # Determine how many features to mask
        n_mask = int(n_features * self.mask_ratio)
        if n_mask == 0:
            n_mask = 1  # Mask at least one feature
            
        # Generate different masks for each sample in batch
        for i in range(batch_size):
            # Select random features to mask
            mask_indices = torch.randperm(n_features)[:n_mask]
            x_aug[i, mask_indices] = 0.0
        
        return x_aug, y