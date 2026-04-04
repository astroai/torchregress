"""Public loss forward-signature contracts (API consistency guardrails)."""

from __future__ import annotations

import inspect
from inspect import Parameter

import torch

import torchregress.losses as losses
from torchregress.losses.base import BaseLoss

ABSTRACT_OR_META = {
    "BaseLoss",
    "RegressionLoss",
    "DistributionLoss",
    "BaseEIVLoss",
    "WeightedLossWrapper",
}

# Temporary compatibility exceptions where an optional extra parameter appears
# after `weights` in the public signature. Keep this list small and explicit.
POST_WEIGHTS_EXTRA_EXCEPTIONS: dict[str, set[str]] = {}


def _iter_public_loss_classes() -> list[tuple[str, type[BaseLoss]]]:
    out: list[tuple[str, type[BaseLoss]]] = []
    for name in losses.__all__:
        obj = getattr(losses, name, None)
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseLoss)
            and obj.__name__ not in ABSTRACT_OR_META
            and not name.startswith("_")
        ):
            out.append((name, obj))
    return out


def test_public_loss_forward_signatures_follow_core_ordering() -> None:
    for export_name, cls in _iter_public_loss_classes():
        sig = inspect.signature(cls.forward)
        params = list(sig.parameters.values())
        assert params[0].name == "self", export_name

        user_params = params[1:]
        names = [p.name for p in user_params]
        assert len(user_params) >= 2, export_name
        assert names[0] == "y_pred", f"{export_name}.forward first arg must be y_pred"
        assert names[1] == "target", f"{export_name}.forward second arg must be target"

        if "mask" in names:
            mask = sig.parameters["mask"]
            assert mask.default is None, f"{export_name}.forward mask default should be None"
        if "weights" in names:
            weights = sig.parameters["weights"]
            assert weights.default is None, f"{export_name}.forward weights default should be None"

        if "mask" in names and "weights" in names:
            assert names.index("mask") < names.index("weights"), (
                f"{export_name}.forward should place mask before weights"
            )


def test_public_loss_forward_signatures_accept_kwargs_or_are_explicitly_exempt() -> None:
    # Keep empty by default; test documents intentional exceptions if added later.
    exempt: set[str] = set()
    for export_name, cls in _iter_public_loss_classes():
        if export_name in exempt:
            continue
        sig = inspect.signature(cls.forward)
        has_kwargs = any(p.kind is Parameter.VAR_KEYWORD for p in sig.parameters.values())
        assert has_kwargs, f"{export_name}.forward should accept **kwargs for API consistency"


def test_no_unexpected_optional_parameters_after_weights() -> None:
    for export_name, cls in _iter_public_loss_classes():
        sig = inspect.signature(cls.forward)
        params = list(sig.parameters.values())[1:]  # skip self
        names = [p.name for p in params]
        if "weights" not in names:
            continue
        weights_idx = names.index("weights")
        trailing_named = [
            p
            for p in params[weights_idx + 1 :]
            if p.kind
            in {
                Parameter.POSITIONAL_ONLY,
                Parameter.POSITIONAL_OR_KEYWORD,
                Parameter.KEYWORD_ONLY,
            }
            and p.name != "kwargs"
        ]
        if not trailing_named:
            continue
        allowed = POST_WEIGHTS_EXTRA_EXCEPTIONS.get(export_name, set())
        trailing_names = {p.name for p in trailing_named}
        assert trailing_names <= allowed, (
            f"{export_name}.forward has optional params after weights: {sorted(trailing_names)}; "
            "prefer extra params before mask/weights unless compatibility requires otherwise"
        )


def test_weighted_gaussian_nll_loss_legacy_positional_mask_weights_remain_supported() -> None:
    loss = losses.WeightedGaussianNLLLoss(log_variance=False)
    mean = torch.randn(8, 2)
    var = torch.rand(8, 2) + 0.1
    target = torch.randn(8, 2)
    mask = torch.tensor(
        [[1, 1], [1, 1], [0, 0], [1, 1], [1, 1], [0, 0], [1, 1], [1, 1]], dtype=torch.bool
    )
    weights = torch.rand(8, 2) + 0.5

    legacy = loss((mean, var), target, mask, weights)
    canonical = loss((mean, var), target, None, mask, weights)
    assert torch.allclose(legacy, canonical)


def test_gaussian_nll_loss_legacy_positional_mask_weights_remain_supported() -> None:
    loss = losses.GaussianNLLLoss()
    mean = torch.randn(8, 2)
    log_var = torch.randn(8, 2)
    target = torch.randn(8, 2)
    mask = torch.tensor(
        [[1, 1], [1, 1], [0, 0], [1, 1], [1, 1], [0, 0], [1, 1], [1, 1]], dtype=torch.bool
    )
    weights = torch.rand(8, 2) + 0.5

    legacy = loss((mean, log_var), target, mask, weights)
    canonical = loss((mean, log_var), target, None, mask, weights)
    assert torch.allclose(legacy, canonical)
