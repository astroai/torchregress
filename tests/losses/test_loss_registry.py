"""Unit tests for torchregress.losses.loss_registry."""

from __future__ import annotations

import pytest
import torch.nn as nn

from torchregress.losses.loss_registry import (
    create_loss_from_config,
    get_regression_loss,
    list_regression_losses,
    loss_registry,
    register_regression_loss,
)

# ── register_regression_loss decorator ───────────────────────────────────────


def test_register_adds_class_to_registry() -> None:
    """Decorator adds the class to the global loss_registry dict."""

    @register_regression_loss("__test_dummy__")
    class DummyLoss(nn.Module):
        pass

    try:
        assert "__test_dummy__" in loss_registry
        assert loss_registry["__test_dummy__"] is DummyLoss
    finally:
        loss_registry.pop("__test_dummy__", None)


def test_register_returns_class_unchanged() -> None:
    """Decorator returns the same class it received."""

    @register_regression_loss("__test_returned__")
    class ReturnedLoss(nn.Module):
        pass

    try:
        assert ReturnedLoss.__name__ == "ReturnedLoss"
        assert issubclass(ReturnedLoss, nn.Module)
    finally:
        loss_registry.pop("__test_returned__", None)


def test_register_duplicate_name_raises_keyerror() -> None:
    """Registering the same name twice raises KeyError."""

    @register_regression_loss("__test_dup__")
    class FirstLoss(nn.Module):
        pass

    try:
        with pytest.raises(KeyError, match="already registered"):

            @register_regression_loss("__test_dup__")
            class SecondLoss(nn.Module):
                pass

    finally:
        loss_registry.pop("__test_dup__", None)


def test_register_stores_name_as_string_key() -> None:
    """Registry key is the exact string passed to the decorator."""
    name = "__test_key_type__"

    @register_regression_loss(name)
    class KeyedLoss(nn.Module):
        pass

    try:
        assert name in loss_registry
        assert loss_registry[name] is KeyedLoss
    finally:
        loss_registry.pop(name, None)


# ── get_regression_loss ──────────────────────────────────────────────────────


def test_get_returns_registered_class() -> None:
    """get_regression_loss returns the class object from the registry."""

    @register_regression_loss("__test_get__")
    class GettableLoss(nn.Module):
        pass

    try:
        cls = get_regression_loss("__test_get__")
        assert cls is GettableLoss
    finally:
        loss_registry.pop("__test_get__", None)


def test_get_unregistered_name_raises_keyerror() -> None:
    """Looking up an unknown name raises KeyError."""
    with pytest.raises(KeyError, match="not registered"):
        get_regression_loss("__definitely_not_registered__")


def test_get_populated_registry_contains_known_losses() -> None:
    """The global registry (populated by imports) contains standard loss names."""
    # These are auto-registered by importing the loss modules.
    known = {"gaussian_nll", "faithful_gaussian", "mdn", "beta_nll"}
    for name in known:
        cls = get_regression_loss(name)
        assert cls is not None
        assert isinstance(cls, type)


# ── list_regression_losses ───────────────────────────────────────────────────


def test_list_returns_all_registered_names() -> None:
    """list_regression_losses returns a list containing known entries."""
    names = list_regression_losses()
    assert isinstance(names, list)
    # The registry is populated by imports — at least these should be present.
    assert "gaussian_nll" in names
    assert "mdn" in names
    assert "faithful_gaussian" in names


def test_list_includes_newly_registered_names() -> None:
    """Newly registered names appear in list output."""

    @register_regression_loss("__test_list_new__")
    class NewListLoss(nn.Module):
        pass

    try:
        assert "__test_list_new__" in list_regression_losses()
    finally:
        loss_registry.pop("__test_list_new__", None)


def test_list_after_removal_excludes_name() -> None:
    """Directly removing from loss_registry is reflected in list output."""

    @register_regression_loss("__test_list_removed__")
    class RemovedLoss(nn.Module):
        pass

    try:
        assert "__test_list_removed__" in list_regression_losses()
        loss_registry.pop("__test_list_removed__")
        assert "__test_list_removed__" not in list_regression_losses()
    finally:
        loss_registry.pop("__test_list_removed__", None)


# ── loss_registry dict ───────────────────────────────────────────────────────


def test_registry_is_dict() -> None:
    """loss_registry is a plain dict."""
    assert isinstance(loss_registry, dict)


def test_registry_direct_write_and_read() -> None:
    """The dict supports direct read/write like any dict."""

    class DirectLoss(nn.Module):
        pass

    loss_registry["__test_direct__"] = DirectLoss
    try:
        assert loss_registry["__test_direct__"] is DirectLoss
    finally:
        loss_registry.pop("__test_direct__", None)


# ── create_loss_from_config ──────────────────────────────────────────────────


def test_create_loss_missing_type_key_raises() -> None:
    """Config dict without 'type' raises KeyError."""
    with pytest.raises(KeyError, match="must include a 'type' key"):
        create_loss_from_config({})


def test_create_loss_unknown_type_raises_keyerror_with_available() -> None:
    """Unknown loss type raises KeyError listing available types."""
    with pytest.raises(KeyError, match="Unknown loss type"):
        create_loss_from_config({"type": "__completely_unknown_loss__"})


def test_create_loss_mse_alias() -> None:
    """'mse' alias creates WeightedMSELoss."""
    loss = create_loss_from_config({"type": "mse"})
    from torchregress.losses.base import WeightedMSELoss

    assert isinstance(loss, WeightedMSELoss)


def test_create_loss_l1_alias() -> None:
    """'l1' alias creates WeightedL1Loss."""
    loss = create_loss_from_config({"type": "l1"})
    from torchregress.losses.base import WeightedL1Loss

    assert isinstance(loss, WeightedL1Loss)


def test_create_loss_mae_alias() -> None:
    """'mae' alias is WeightedL1Loss."""
    loss = create_loss_from_config({"type": "mae"})
    from torchregress.losses.base import WeightedL1Loss

    assert isinstance(loss, WeightedL1Loss)


def test_create_loss_huber_alias_with_delta() -> None:
    """'huber' alias creates WeightedHuberLoss with delta kwarg passed to inner loss."""
    loss = create_loss_from_config({"type": "huber", "delta": 1.0})
    from torchregress.losses.base import WeightedHuberLoss

    assert isinstance(loss, WeightedHuberLoss)
    assert loss.torch_loss.delta == 1.0


def test_create_loss_gaussian_nll_alias() -> None:
    """'gaussian_nll' alias creates GaussianNLLLoss."""
    loss = create_loss_from_config({"type": "gaussian_nll"})
    from torchregress.losses.gaussian import GaussianNLLLoss

    assert isinstance(loss, GaussianNLLLoss)


def test_create_loss_beta_nll_alias() -> None:
    """'beta_nll' alias creates BetaNLLLoss."""
    loss = create_loss_from_config({"type": "beta_nll"})
    from torchregress.losses.beta_nll import BetaNLLLoss

    assert isinstance(loss, BetaNLLLoss)


def test_create_loss_balanced_mse_alias() -> None:
    """'balanced_mse' alias creates BalancedMSELoss (needs bin_edges as tensor)."""
    import torch

    loss = create_loss_from_config({"type": "balanced_mse", "bin_edges": torch.tensor([0.0, 10.0])})
    from torchregress.losses.balanced_mse import BalancedMSELoss

    assert isinstance(loss, BalancedMSELoss)


def test_create_loss_extra_kwargs_passed_through() -> None:
    """Extra keys beyond 'type' are passed as kwargs to the constructor."""
    loss = create_loss_from_config({"type": "huber", "delta": 2.5, "reduction": "sum"})
    from torchregress.losses.base import WeightedHuberLoss

    assert isinstance(loss, WeightedHuberLoss)
    assert loss.torch_loss.delta == 2.5
    assert loss.reduction == "sum"


def test_create_loss_lowercases_type() -> None:
    """Type name is lowercased before lookup (case insensitive)."""
    loss_upper = create_loss_from_config({"type": "MSE"})
    loss_lower = create_loss_from_config({"type": "mse"})
    from torchregress.losses.base import WeightedMSELoss

    assert isinstance(loss_upper, WeightedMSELoss)
    assert isinstance(loss_lower, WeightedMSELoss)


def test_create_loss_from_registered_class() -> None:
    """A class registered with @register_regression_loss can be created via config."""

    @register_regression_loss("__test_config_create__")
    class ConfigCreated(nn.Module):
        def __init__(self, alpha: float = 1.0):
            super().__init__()
            self.alpha = alpha

    try:
        loss = create_loss_from_config({"type": "__test_config_create__", "alpha": 3.0})
        assert isinstance(loss, ConfigCreated)
        assert loss.alpha == 3.0
    finally:
        loss_registry.pop("__test_config_create__", None)


def test_create_loss_from_registry_fallback() -> None:
    """When type is not an alias, the registry is searched next."""

    @register_regression_loss("__test_registry_fallback__")
    class FallbackLoss(nn.Module):
        def __init__(self):
            super().__init__()

    try:
        loss = create_loss_from_config({"type": "__test_registry_fallback__"})
        assert isinstance(loss, FallbackLoss)
    finally:
        loss_registry.pop("__test_registry_fallback__", None)
