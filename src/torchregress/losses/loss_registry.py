"""
Registry and helpers for regression loss classes.

Use @register_regression_loss(name) to register custom loss classes.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Type

# Global registry dictionary
loss_registry: Dict[str, Type] = {}


def register_regression_loss(name: str) -> Callable:
    """
    Decorator to register a regression loss class under a given name.

    Args:
        name: Unique identifier for the loss class.

    Returns:
        Decorator that registers the class.
    """

    def decorator(cls: Type) -> Type:
        if name in loss_registry:
            raise KeyError(f"Loss '{name}' is already registered.")
        loss_registry[name] = cls
        return cls

    return decorator


def get_regression_loss(name: str) -> Type:
    """
    Retrieve a registered loss class by name.

    Args:
        name: Identifier of the loss class.

    Returns:
        The loss class type.

    Raises:
        KeyError if no loss is registered under the given name.
    """
    if name not in loss_registry:
        raise KeyError(f"Loss '{name}' is not registered.")
    return loss_registry[name]


def list_regression_losses() -> List[str]:
    """
    List all registered regression loss names.

    Returns:
        A list of registered loss identifiers.
    """
    return list(loss_registry.keys())


def create_loss_from_config(config: Dict[str, Any]) -> Any:
    """
    Create a loss instance from a lightweight config dictionary.

    Examples:
        >>> create_loss_from_config({"type": "mse"})
        >>> create_loss_from_config({"type": "huber", "delta": 1.0})
        >>> create_loss_from_config({"type": "gaussian_nll"})
    """
    if "type" not in config:
        raise KeyError("config must include a 'type' key")

    cfg = dict(config)
    loss_type = str(cfg.pop("type")).lower()

    # Local imports avoid circular dependencies with losses.__init__.
    from .balanced_mse import BalancedMSELoss, BMCLoss
    from .base import WeightedHuberLoss, WeightedL1Loss, WeightedMSELoss
    from .beta_nll import BetaNLLLoss
    from .faithful_gaussian import FaithfulGaussianLoss
    from .gaussian import GaussianNLLLoss
    from .gaussian_wasserstein import GaussianWassersteinBoundLoss

    aliases: Dict[str, Any] = {
        "mse": WeightedMSELoss,
        "l2": WeightedMSELoss,
        "l1": WeightedL1Loss,
        "mae": WeightedL1Loss,
        "huber": WeightedHuberLoss,
        "gaussian_nll": GaussianNLLLoss,
        "gaussian": GaussianNLLLoss,
        "beta_nll": BetaNLLLoss,
        "faithful_gaussian": FaithfulGaussianLoss,
        "gaussian_wasserstein_bound": GaussianWassersteinBoundLoss,
        "balanced_mse": BalancedMSELoss,
        "bmc": BMCLoss,
    }

    if loss_type in aliases:
        cls = aliases[loss_type]
        return cls(**cfg)

    if loss_type in loss_registry:
        return loss_registry[loss_type](**cfg)

    available = sorted(set(list(aliases) + list(loss_registry)))
    raise KeyError(f"Unknown loss type '{loss_type}'. Available types: {available}")
