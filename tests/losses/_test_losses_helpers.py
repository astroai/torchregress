"""
Shared test helpers for Gaussian loss tests.

Provides data generators and loss-factory utilities used by
``test_gaussian_consistency.py`` (and available for future loss test files).
"""

from __future__ import annotations

import torch
from torch import nn

from torchregress.losses.beta_nll import BetaNLLLoss
from torchregress.losses.faithful_gaussian import FaithfulGaussianLoss
from torchregress.losses.gaussian import GaussianCRPSLoss, GaussianNLLLoss

# ── diagonal loss registry (for parametrized cross-class tests) ───────

DIAGONAL_LOSS_NAMES = ["GaussianNLL", "GaussianCRPS", "BetaNLL", "FaithfulGaussian"]


def build_diagonal_losses():
    """Return ``(name, loss_fn)`` pairs for all diagonal Gaussian losses."""
    factories = [
        lambda: GaussianNLLLoss(reduction="mean"),
        lambda: GaussianCRPSLoss(reduction="mean"),
        lambda: BetaNLLLoss(beta=0.3, reduction="mean"),
        lambda: FaithfulGaussianLoss(mean_weight=1.0, variance_weight=1.0, reduction="mean"),
    ]
    return list(zip(DIAGONAL_LOSS_NAMES, [f() for f in factories]))


# ── test data generator ────────────────────────────────────────────────


def make_test_data(
    batch: int = 8,
    dim: int = 3,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(mean, log_var, target)`` tensors for diagonal Gaussian tests.

    Args:
        batch: Number of samples.
        dim: Number of output dimensions.
        seed: RNG seed for reproducibility.
    """
    torch.manual_seed(seed)
    mean = torch.randn(batch, dim)
    log_var = torch.randn(batch, dim) * 0.5
    target = torch.randn(batch, dim)
    return mean, log_var, target


# ── loss construction helpers ──────────────────────────────────────────


def make_base_kwargs(loss_fn: nn.Module) -> dict:
    """Return the constructor kwargs (excluding *reduction*) for a loss instance.

    Handles ``BetaNLLLoss`` (picks up ``beta``) and ``FaithfulGaussianLoss``
    (picks up ``mean_weight``, ``variance_weight``).
    """
    kwargs: dict = {}
    if hasattr(loss_fn, "beta"):
        kwargs["beta"] = loss_fn.beta
    if hasattr(loss_fn, "mean_weight"):
        kwargs["mean_weight"] = loss_fn.mean_weight
        kwargs["variance_weight"] = loss_fn.variance_weight
    return kwargs


def make_none_reduction(loss_fn: nn.Module) -> nn.Module:
    """Return a new instance of the same loss class with ``reduction='none'``."""
    return type(loss_fn)(**make_base_kwargs(loss_fn), reduction="none")
