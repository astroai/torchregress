"""Unit tests for torchregress.utils.pytorch_compat."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torchregress.utils.pytorch_compat import (
    convert_reduction_type,
    convert_to_pytorch_loss,
    extract_output_size,
    get_device,
    set_all_seeds,
    set_seed,
)

# ── convert_reduction_type ──────────────────────────────────────────────────


def test_convert_reduction_none() -> None:
    assert convert_reduction_type("none") == "none"


def test_convert_reduction_mean() -> None:
    assert convert_reduction_type("mean") == "mean"


def test_convert_reduction_sum() -> None:
    assert convert_reduction_type("sum") == "sum"


def test_convert_reduction_batch_mean() -> None:
    assert convert_reduction_type("batch_mean") == "mean"


def test_convert_reduction_average() -> None:
    assert convert_reduction_type("average") == "mean"


def test_convert_reduction_case_insensitive() -> None:
    assert convert_reduction_type("MEAN") == "mean"
    assert convert_reduction_type("NoNe") == "none"
    assert convert_reduction_type("SUM") == "sum"


def test_convert_reduction_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported reduction type"):
        convert_reduction_type("median")


# ── convert_to_pytorch_loss ─────────────────────────────────────────────────


def test_convert_module_returns_same() -> None:
    mse = nn.MSELoss()
    result = convert_to_pytorch_loss(mse)
    assert result is mse


def test_convert_callable_wraps() -> None:
    def my_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (output - target).abs().mean()

    wrapped = convert_to_pytorch_loss(my_loss)
    assert isinstance(wrapped, nn.Module)
    assert not isinstance(wrapped, type(my_loss))


def test_convert_wrapped_callable_forwards_correctly() -> None:
    def my_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (output - target).abs().mean()

    wrapped = convert_to_pytorch_loss(my_loss)
    out = torch.tensor([1.0, 2.0, 3.0])
    tgt = torch.tensor([0.0, 2.0, 4.0])
    expected = my_loss(out, tgt)
    result = wrapped(out, tgt)
    torch.testing.assert_close(result, expected)


def test_convert_callable_with_kwargs() -> None:
    def weighted_loss(
        output: torch.Tensor, target: torch.Tensor, *, alpha: float = 1.0
    ) -> torch.Tensor:
        return alpha * (output - target).abs().mean()

    wrapped = convert_to_pytorch_loss(weighted_loss, alpha=2.0)
    out = torch.tensor([1.0, 2.0])
    tgt = torch.tensor([0.0, 0.0])
    result = wrapped(out, tgt)
    expected = weighted_loss(out, tgt, alpha=2.0)
    torch.testing.assert_close(result, expected)


# ── extract_output_size ─────────────────────────────────────────────────────


def test_extract_output_size_tensor() -> None:
    """Linear model with int input_size (auto-wrapped to tuple)."""
    model = nn.Linear(4, 3)
    result = extract_output_size(model, 4)
    assert result == (3,)
    # Also test explicit tuple input
    result2 = extract_output_size(model, (4,))
    assert result2 == (3,)


def test_extract_output_size_tuple_input() -> None:
    model = nn.Conv2d(3, 8, 3)
    result = extract_output_size(model, (3, 32, 32))
    assert len(result) == 3


def test_extract_output_size_multi_output() -> None:
    class MultiOut(nn.Module):
        def forward(self, x):
            return x, x * 2

    model = MultiOut()
    result = extract_output_size(model, 4)
    assert isinstance(result, tuple)
    assert result == ((4,), (4,))


def test_extract_output_size_non_tensor_output_raises() -> None:
    class BadOutput(nn.Module):
        def forward(self, x):
            return "not a tensor"

    model = BadOutput()
    with pytest.raises(ValueError, match="not a tensor"):
        extract_output_size(model, 1)


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


# ── set_seed ────────────────────────────────────────────────────────────────


def test_set_seed_is_alias_for_set_all_seeds() -> None:
    """set_seed delegates to set_all_seeds."""
    set_all_seeds(42)
    a = torch.randn(1).item()
    set_seed(42)
    b = torch.randn(1).item()
    assert a == b


# ── get_device ──────────────────────────────────────────────────────────────


def test_get_device_default() -> None:
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ("cpu", "cuda")


def test_get_device_explicit_string() -> None:
    device = get_device("cpu")
    assert device == torch.device("cpu")
    assert isinstance(device, torch.device)
