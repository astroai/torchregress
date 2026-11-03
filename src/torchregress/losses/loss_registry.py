"""
Registry and helpers for regression loss classes.

Use @register_regression_loss(name) to register custom loss classes.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Type

# Global registry dictionary
loss_registry: Dict[str, Type] = {}

# Deprecated registry key -> replacement key. Lookup by a deprecated key warns.
deprecated_loss_keys: Dict[str, str] = {}


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


def register_deprecated_loss_key(old: str, new: str) -> None:
    """Mark registry key ``old`` as deprecated in favour of ``new``."""
    deprecated_loss_keys[old] = new


def _warn_deprecated_key(name: str) -> None:
    if name in deprecated_loss_keys:
        warnings.warn(
            f"Loss registry key '{name}' is deprecated and will be removed in the next minor "
            f"release; use '{deprecated_loss_keys[name]}'.",
            DeprecationWarning,
            stacklevel=3,
        )


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
    _warn_deprecated_key(name)
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

    # ponytail: registry is already populated by @register_regression_loss decorators.
    if loss_type in loss_registry:
        _warn_deprecated_key(loss_type)
        return loss_registry[loss_type](**cfg)

    available = sorted(loss_registry)
    raise KeyError(f"Unknown loss type '{loss_type}'. Available types: {available}")
