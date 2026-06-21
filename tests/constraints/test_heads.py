"""
Unit tests for torchregress.constraints.heads.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torchregress.constraints.heads import (
    BoundedHead,
    NonCrossingSort,
    NonNegativeHead,
    SimplexHead,
    SpectralNormWrapper,
)

# ═══════════════════════════════════════════════════════════════════════════════
# NonNegativeHead
# ═══════════════════════════════════════════════════════════════════════════════


class TestNonNegativeHead:
    def test_outputs_are_nonnegative(self) -> None:
        """All outputs are >= 0 via softplus."""
        base = nn.Linear(3, 2)
        head = NonNegativeHead(base)
        x = torch.randn(8, 3)
        out = head(x)
        assert (out >= 0).all()

    def test_default_beta(self) -> None:
        """Default beta=1.0 (standard softplus)."""
        base = nn.Linear(3, 1)
        head = NonNegativeHead(base)
        assert head.beta == 1.0

    def test_custom_beta(self) -> None:
        """Custom beta changes softplus steepness."""
        base = nn.Linear(3, 1)
        head = NonNegativeHead(base, beta=5.0)
        assert head.beta == 5.0
        x = torch.randn(4, 3)
        out = head(x)
        assert out.shape == (4, 1)
        assert (out >= 0).all()


# ═══════════════════════════════════════════════════════════════════════════════
# BoundedHead
# ═══════════════════════════════════════════════════════════════════════════════


class TestBoundedHead:
    def test_outputs_in_range(self) -> None:
        """Outputs are clamped to [low, high] via sigmoid."""
        base = nn.Linear(3, 2)
        head = BoundedHead(base, low=0.0, high=1.0)
        x = torch.randn(16, 3)
        out = head(x)
        assert (out >= 0.0).all()
        assert (out <= 1.0).all()

    def test_custom_range(self) -> None:
        """Custom [low, high] range is respected."""
        base = nn.Linear(3, 1)
        head = BoundedHead(base, low=-2.0, high=3.0)
        x = torch.randn(8, 3)
        out = head(x)
        assert (out >= -2.0).all()
        assert (out <= 3.0).all()

    def test_low_equals_high_raises(self) -> None:
        """low >= high raises ValueError."""
        base = nn.Linear(3, 1)
        with pytest.raises(ValueError, match="low < high"):
            BoundedHead(base, low=1.0, high=1.0)

    def test_low_greater_than_high_raises(self) -> None:
        """low > high raises ValueError."""
        base = nn.Linear(3, 1)
        with pytest.raises(ValueError, match="low < high"):
            BoundedHead(base, low=2.0, high=1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# SimplexHead
# ═══════════════════════════════════════════════════════════════════════════════


class TestSimplexHead:
    def test_outputs_sum_to_one(self) -> None:
        """Softmax outputs sum to 1 along the specified dim."""
        base = nn.Linear(3, 4)
        head = SimplexHead(base, dim=-1)
        x = torch.randn(8, 3)
        out = head(x)
        assert out.shape == (8, 4)
        assert torch.allclose(out.sum(dim=-1), torch.ones(8), atol=1e-6)

    def test_custom_dim(self) -> None:
        """Softmax along a custom dimension."""
        base = nn.Linear(3, 4)
        head = SimplexHead(base, dim=0)
        x = torch.randn(8, 3)
        out = head(x)
        # Softmax along dim=0 on (8, 4) → sum along dim=0 gives ones of shape (4,)
        assert torch.allclose(out.sum(dim=0), torch.ones(4), atol=1e-6)

    def test_outputs_nonnegative(self) -> None:
        """All outputs are >= 0."""
        base = nn.Linear(3, 5)
        head = SimplexHead(base)
        x = torch.randn(8, 3)
        out = head(x)
        assert (out >= 0).all()


# ═══════════════════════════════════════════════════════════════════════════════
# NonCrossingSort
# ═══════════════════════════════════════════════════════════════════════════════


class TestNonCrossingSort:
    def test_outputs_are_sorted(self) -> None:
        """Outputs are sorted along the specified dimension."""
        head = NonCrossingSort(dim=-1)
        x = torch.tensor([[3.0, 1.0, 2.0], [5.0, 0.0, 4.0]])
        out = head(x)
        assert torch.equal(out, torch.tensor([[1.0, 2.0, 3.0], [0.0, 4.0, 5.0]]))

    def test_custom_dim(self) -> None:
        """Sort along dim=0."""
        head = NonCrossingSort(dim=0)
        x = torch.tensor([[3.0, 1.0], [1.0, 4.0], [2.0, 0.0]])
        out = head(x)
        assert (out[0] <= out[1]).all()
        assert (out[1] <= out[2]).all()  # noqa: RUF015

    def test_default_dim(self) -> None:
        """Default dim=-1."""
        head = NonCrossingSort()
        assert head.dim == -1


# ═══════════════════════════════════════════════════════════════════════════════
# SpectralNormWrapper
# ═══════════════════════════════════════════════════════════════════════════════


class TestSpectralNormWrapper:
    def test_forward_returns_tensor(self) -> None:
        """Forward pass returns a tensor."""
        base = nn.Linear(3, 2)
        wrapper = SpectralNormWrapper(base)
        x = torch.randn(4, 3)
        out = wrapper(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (4, 2)

    def test_spectral_norm_applied(self) -> None:
        """The wrapped module has spectral_norm parametrization."""
        base = nn.Linear(3, 2)
        wrapper = SpectralNormWrapper(base, name="weight")
        # The module weight should have parametrizations registered
        assert hasattr(wrapper.module, "parametrizations")
        assert "weight" in wrapper.module.parametrizations

    def test_not_tensor_raises(self) -> None:
        """Non-tensor output from wrapped module raises TypeError."""

        class BadModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(2, 2))

            def forward(self, x):  # noqa: ANN001, ANN201
                return "not a tensor"

        wrapper = SpectralNormWrapper(BadModule())
        with pytest.raises(TypeError, match="return a Tensor"):
            wrapper(torch.randn(2, 2))
