"""
Registry for regression loss classes.

Use @register_regression_loss(name) to register custom loss classes.

"""

from typing import Type, Callable, Dict, List

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
