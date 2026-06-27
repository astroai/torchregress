"""Unit tests for torchregress.utils.pytorch_compat."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torchregress.utils.pytorch_compat import (
    get_device,
    set_all_seeds,
)

# ── set_all_seeds ───────────────────────────────────────────────────────────


def test_set_all_seeds_accepts_integer() -> None:
    result = set_all_seeds(42)
    assert result is None


def test_set_all_seeds_reproducibility() -> None:
    """Same seed produces identical random tensors."""
    set_all_seeds(99)
    t1 = torch.randn(3)
    set_all_seeds(99)
    t2 = torch.randn(3)
    torch.testing.assert_close(t1, t2)
    # Scalar check
    set_all_seeds(42)
    a = torch.randn(1).item()
    set_all_seeds(42)
    b = torch.randn(1).item()
    assert a == b


def test_set_all_seeds_different_seed_different_output() -> None:
    set_all_seeds(1)
    a = torch.randn(1).item()
    set_all_seeds(2)
    b = torch.randn(1).item()
    assert a != b


def test_set_all_seeds_sets_cudnn_flags() -> None:
    set_all_seeds(7)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


# ── get_device ──────────────────────────────────────────────────────────────


def test_get_device_default() -> None:
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ("cpu", "cuda")


def test_get_device_explicit_string() -> None:
    device = get_device("cpu")
    assert device == torch.device("cpu")
    assert isinstance(device, torch.device)
