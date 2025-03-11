"""
Data augmentation utilities.

This module provides various data augmentation techniques for enhancing 
regression and classification models.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, Tuple, List

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


class GaussianNoiseAugmenter(nn.Module):
    """
    Add Gaussian noise to input data for augmentation.
    
    Args:
        std: Standard deviation of the Gaussian noise
        per_feature: Whether to use different noise levels per feature
        learn_std: Whether to make std a learnable parameter
        constant_noise: Whether to use the same noise pattern for all batch samples
    """
    def __init__(
        self, 
        std: float = 0.1, 
        per_feature: bool = False, 
        learn_std: bool = False,
        constant_noise: bool = False
    ):
        super().__init__()
        self.constant_noise = constant_noise
        self.per_feature = per_feature
        self.learn_std = learn_std
        
        # Initialize learnable parameters if requested
        if learn_std:
            self.log_std = nn.Parameter(torch.ones(1) * np.log(std))
        else:
            self.std = std
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian noise augmentation.
        
        Args:
            x: Input tensor of shape [batch_size, features]
            
        Returns:
            Augmented tensor with same shape as input
        """
        if self.training:
            # Get the effective standard deviation
            if self.learn_std:
                std = torch.exp(self.log_std)
            else:
                std = self.std
                
            # Generate noise
            if self.constant_noise:
                # Same noise pattern for all samples in batch
                if self.per_feature:
                    # Different noise per feature
                    noise_shape = (1, x.shape[1])
                else:
                    # Same noise for all features
                    noise_shape = (1, 1)
                    
                noise = torch.randn(noise_shape, device=x.device) * std
                noise = noise.expand_as(x)
            else:
                # Different noise for each sample
                if self.per_feature and x.dim() > 1:
                    # Different noise per sample and feature
                    noise = torch.randn_like(x) * std
                else:
                    # Same noise across features
                    noise = torch.randn(x.shape[0], 1, device=x.device) * std
                    if x.dim() > 1:
                        noise = noise.expand_as(x)
                        
            return x + noise
        else:
            return x


class MixupAugmenter(nn.Module):
    """
    Apply Mixup augmentation (https://arxiv.org/abs/1710.09412).
    
    Args:
        alpha: Parameter for Beta distribution
        apply_to_targets: Whether to apply mixup to targets
    """
    def __init__(self, alpha: float = 1.0, apply_to_targets: bool = True):
        super().__init__()
        self.alpha = alpha
        self.apply_to_targets = apply_to_targets
        
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply Mixup augmentation.
        
        Args:
            x: Input features [batch_size, ...]
            y: Optional targets [batch_size, ...]
            
        Returns:
            Tuple of augmented features and targets
        """
        if not self.training or self.alpha <= 0:
            return x, y
            
        batch_size = x.shape[0]
        
        # Sample from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Create random permutation of indices
        indices = torch.randperm(batch_size, device=x.device)
        
        # Mix the features
        mixed_x = lam * x + (1 - lam) * x[indices]
        
        # Mix targets if provided and requested
        mixed_y = None
        if y is not None and self.apply_to_targets:
            mixed_y = lam * y + (1 - lam) * y[indices]
        else:
            mixed_y = y
            
        return mixed_x, mixed_y


class CutoutAugmenter(nn.Module):
    """
    Apply Cutout augmentation (https://arxiv.org/abs/1708.04552).
    
    Args:
        size: Size of the cutout as a fraction of the input size
        n_holes: Number of holes to cut out
    """
    def __init__(self, size: float = 0.2, n_holes: int = 1):
        super().__init__()
        self.size = size
        self.n_holes = n_holes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Cutout augmentation.
        
        Args:
            x: Input features [batch_size, ...]
            
        Returns:
            Augmented features
        """
        if not self.training or self.size <= 0:
            return x
            
        # Handle only 2D+ inputs (batch_size, channels, height, width, ...)
        if x.dim() < 3:
            return x
            
        batch_size = x.shape[0]
        x_aug = x.clone()
        
        # Apply cutout
        for i in range(batch_size):
            for _ in range(self.n_holes):
                # Determine cutout dimensions
                h, w = x.shape[2], x.shape[3] if x.dim() > 3 else x.shape[2]
                cutout_h = int(h * self.size)
                cutout_w = int(w * self.size) if x.dim() > 3 else cutout_h
                
                # Random position
                top = np.random.randint(0, h - cutout_h + 1)
                left = np.random.randint(0, w - cutout_w + 1) if x.dim() > 3 else 0
                
                # Apply the cutout
                if x.dim() > 3:
                    x_aug[i, :, top:top+cutout_h, left:left+cutout_w] = 0
                else:
                    x_aug[i, :, top:top+cutout_h] = 0
                    
        return x_aug


class CutMixAugmenter(nn.Module):
    """
    Apply CutMix augmentation (https://arxiv.org/abs/1905.04899).
    
    Args:
        alpha: Parameter for Beta distribution
        apply_to_targets: Whether to apply mixup to targets
    """
    def __init__(self, alpha: float = 1.0, apply_to_targets: bool = True):
        super().__init__()
        self.alpha = alpha
        self.apply_to_targets = apply_to_targets
        
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply CutMix augmentation.
        
        Args:
            x: Input features [batch_size, channels, height, width]
            y: Optional targets [batch_size, ...]
            
        Returns:
            Tuple of augmented features and targets
        """
        if not self.training or self.alpha <= 0:
            return x, y
            
        # Only for images with at least 2D spatial dimensions
        if x.dim() < 4:
            return x, y
            
        batch_size = x.shape[0]
        
        # Sample mixing parameter
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Create random permutation of indices
        indices = torch.randperm(batch_size, device=x.device)
        
        # Get dimensions
        h, w = x.shape[2], x.shape[3]
        
        # Calculate cut size
        cut_ratio = np.sqrt(1. - lam)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)
        
        # Get random center
        cx = np.random.randint(0, h)
        cy = np.random.randint(0, w)
        
        # Calculate box boundaries
        top = max(cx - cut_h // 2, 0)
        left = max(cy - cut_w // 2, 0)
        bottom = min(cx + cut_h // 2, h)
        right = min(cy + cut_w // 2, w)
        
        # Create mixed image
        mixed_x = x.clone()
        mixed_x[:, :, top:bottom, left:right] = x[indices, :, top:bottom, left:right]
        
        # Calculate effective lambda
        mixed_area = (bottom - top) * (right - left)
        lam_effective = 1 - mixed_area / (h * w)
        
        # Mix targets if provided and requested
        mixed_y = None
        if y is not None and self.apply_to_targets:
            mixed_y = lam_effective * y + (1 - lam_effective) * y[indices]
        else:
            mixed_y = y
            
        return mixed_x, mixed_y


class EnsemblePerturbationAugmenter(nn.Module):
    """
    Generate multiple perturbed versions of inputs for ensemble prediction.
    
    This augmenter is designed specifically for uncertainty estimation methods 
    like EnsembleEIVLoss that need to generate multiple perturbed versions
    of the same inputs.
    
    Args:
        n_samples: Number of perturbed samples to generate
        perturb_method: Method for perturbation ('gaussian', 'uniform')
        sigma: Standard deviation or covariance matrix for perturbation
        feature_wise: Whether to use different noise for each feature
        device: Device for tensor operations
    """
    def __init__(
        self, 
        n_samples: int = 20, 
        perturb_method: str = 'gaussian',
        sigma: Union[float, torch.Tensor] = 0.1,
        feature_wise: bool = True,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.n_samples = n_samples
        self.perturb_method = perturb_method
        self.sigma = sigma
        self.feature_wise = feature_wise
        self.device = device
        
        if perturb_method not in ['gaussian', 'uniform']:
            raise ValueError(f"Unknown perturbation method: {perturb_method}. "
                            f"Must be one of ['gaussian', 'uniform']")
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate multiple perturbed versions of the input.
        
        Args:
            x: Input tensor [batch_size, n_features]
            
        Returns:
            List of perturbed samples, each with shape [batch_size, n_features]
        """
        batch_size, n_features = x.shape
        device = self.device if self.device is not None else x.device
        
        # Prepare sigma as covariance tensor
        if isinstance(self.sigma, (int, float)):
            if self.feature_wise:
                sigma_tensor = torch.ones(n_features, device=device) * self.sigma
            else:
                sigma_tensor = torch.tensor(self.sigma, device=device)
        else:
            sigma_tensor = self.sigma.to(device)
            
            # Validate shape
            if sigma_tensor.ndim == 1 and sigma_tensor.shape[0] != n_features:
                raise ValueError(f"Sigma vector shape {sigma_tensor.shape} doesn't match "
                                f"feature dimension {n_features}")
            elif sigma_tensor.ndim == 2 and sigma_tensor.shape != (n_features, n_features):
                raise ValueError(f"Sigma matrix shape {sigma_tensor.shape} doesn't match "
                                f"expected shape ({n_features}, {n_features})")
        
        perturbed_samples = []
        for _ in range(self.n_samples):
            if self.perturb_method == 'gaussian':
                if sigma_tensor.ndim <= 1:
                    # Diagonal covariance - different noise per feature
                    noise = torch.randn(batch_size, n_features, device=device) * sigma_tensor.view(1, -1)
                else:
                    # Full covariance - use multivariate normal
                    try:
                        dist = torch.distributions.MultivariateNormal(
                            torch.zeros(n_features, device=device),
                            sigma_tensor
                        )
                        noise = dist.sample((batch_size,))
                    except:
                        # Fallback to diagonal approximation
                        diag = torch.diagonal(sigma_tensor, dim1=-2, dim2=-1)
                        noise = torch.randn(batch_size, n_features, device=device) * torch.sqrt(diag).view(1, -1)
            else:  # uniform
                # Scale factor to match standard deviation between uniform and normal
                scale_factor = 1.732  # sqrt(3)
                if sigma_tensor.ndim <= 1:
                    half_range = sigma_tensor.view(1, -1) * scale_factor
                    noise = (torch.rand(batch_size, n_features, device=device) * 2 - 1) * half_range
                else:
                    # Use diagonal approximation for uniform with full covariance
                    diag = torch.diagonal(sigma_tensor, dim1=-2, dim2=-1)
                    half_range = torch.sqrt(diag).view(1, -1) * scale_factor
                    noise = (torch.rand(batch_size, n_features, device=device) * 2 - 1) * half_range
                
            perturbed_samples.append(x + noise)
            
        return perturbed_samples

    def generate_and_stack(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate perturbed samples and stack them into a single tensor.
        
        Args:
            x: Input tensor [batch_size, n_features]
            
        Returns:
            Stacked tensor of shape [n_samples, batch_size, n_features]
        """
        samples = self.forward(x)
        return torch.stack(samples)