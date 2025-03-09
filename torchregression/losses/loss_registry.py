"""
Registry of available loss functions.

This module provides a registry of loss functions that can be accessed by name,
making it easier to instantiate losses from configuration.
"""
from typing import Dict, Any, Callable, Optional, List, Union

import torch
import torch.nn as nn

# Dictionary to store registered loss functions
_LOSS_REGISTRY = {}

def register_loss(name: str, loss_fn: Callable, **default_kwargs) -> None:
    """
    Register a loss function in the global registry.
    
    Args:
        name: Name to register the loss function under
        loss_fn: The loss function class or callable
        **default_kwargs: Default arguments for the loss function
    """
    global _LOSS_REGISTRY
    if name in _LOSS_REGISTRY:
        raise ValueError(f"Loss function '{name}' is already registered")
    
    _LOSS_REGISTRY[name] = (loss_fn, default_kwargs)


def get_loss_function(name: str, **kwargs) -> nn.Module:
    """
    Get a loss function by name.
    
    Args:
        name: Name of the registered loss function
        **kwargs: Arguments to pass to the loss function
        
    Returns:
        An instance of the requested loss function
        
    Raises:
        ValueError: If the loss function name is not registered
    """
    global _LOSS_REGISTRY
    
    # Normalize loss name for case-insensitive lookup
    name_lower = name.lower()
    
    # Try direct name match first
    if name_lower in _LOSS_REGISTRY:
        loss_fn, default_kwargs = _LOSS_REGISTRY[name_lower]
        
        # Merge default kwargs with provided kwargs
        merged_kwargs = {**default_kwargs, **kwargs}
        
        return loss_fn(**merged_kwargs)
    
    # Try to find a partial match
    matching_keys = [key for key in _LOSS_REGISTRY if name_lower in key]
    if len(matching_keys) == 1:
        loss_fn, default_kwargs = _LOSS_REGISTRY[matching_keys[0]]
        merged_kwargs = {**default_kwargs, **kwargs}
        return loss_fn(**merged_kwargs)
    elif len(matching_keys) > 1:
        matches = ", ".join(matching_keys)
        raise ValueError(f"Multiple losses match '{name}': {matches}. Please use a more specific name.")
    
    # No matches found
    raise ValueError(f"Loss function '{name}' not found in registry. "
                   f"Available losses: {list(_LOSS_REGISTRY.keys())}")


def list_available_losses() -> List[str]:
    """
    List all available registered loss functions.
    
    Returns:
        List of loss function names
    """
    return list(_LOSS_REGISTRY.keys())


# Register common loss functions

# Import loss functions
from .robust import (
    L1Loss, HuberLoss, PseudoHuberLoss, LogCoshLoss, 
    CharbonnierLoss, LqLoss, LogBarrierLoss, TukeyBiweightLoss,
    AdaptiveHuberLoss, WinsorizedLoss, ClippedLoss,
    FairLoss, CauchyLoss
)

from .quantile import (
    QuantileLoss, MultiQuantileLoss, QuantileCrossover, TiltedLoss
)

from .expectile import (
    ExpectileLoss, MultiExpectileLoss, AsymmetricLeastSquaresLoss, 
    ExpectileCrossover
)

from .torch_extensions import MSELoss, BCELoss
from .poisson import PoissonNLLLoss
from .tweedie import TweedieLoss, GammaLoss, InverseGaussianLoss, CompoundPoissonLoss
from .mdn import MixtureDensityNetworkLoss, mdn_loss

# Register robust losses
register_loss('l1', L1Loss)
register_loss('mae', L1Loss)
register_loss('huber', HuberLoss, delta=1.0)
register_loss('pseudo_huber', PseudoHuberLoss)
register_loss('log_cosh', LogCoshLoss)
register_loss('charbonnier', CharbonnierLoss)
register_loss('lq', LqLoss, q=1.5)
register_loss('log_barrier', LogBarrierLoss)
register_loss('tukey', TukeyBiweightLoss)
register_loss('adaptive_huber', AdaptiveHuberLoss)
register_loss('winsorized', WinsorizedLoss)
register_loss('clipped', ClippedLoss)
register_loss('fair', FairLoss)
register_loss('cauchy', CauchyLoss)

# Register quantile and expectile losses
register_loss('quantile', QuantileLoss, quantile=0.5)
register_loss('multi_quantile', MultiQuantileLoss)
register_loss('quantile_crossover', QuantileCrossover)
register_loss('tilted', TiltedLoss, tau=0.5)
register_loss('expectile', ExpectileLoss, expectile=0.5)
register_loss('multi_expectile', MultiExpectileLoss)
register_loss('asymmetric_least_squares', AsymmetricLeastSquaresLoss)
register_loss('expectile_crossover', ExpectileCrossover)

# Register common losses
register_loss('mse', MSELoss)
register_loss('bce', BCELoss)
register_loss('poisson', PoissonNLLLoss)

# Register Tweedie distribution losses
register_loss('tweedie', TweedieLoss, p=1.5)
register_loss('gamma', GammaLoss)
register_loss('inverse_gaussian', InverseGaussianLoss)
register_loss('compound_poisson', CompoundPoissonLoss)

# Register mixture model losses
register_loss('mdn', MixtureDensityNetworkLoss, num_components=5, n_features=1)
