"""Deep unit tests for torchregress.constraints.heads constraint wrappers.

Extends the basic happy-path coverage in test_constraints_calibration.py with
edge cases, error paths, parameter variants, and gradient-flow checks.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torchregress.constraints.heads import (
    BoundedHead,
    NonCrossingSort,
    NonNegativeHead,
    SimplexHead,
    SpectralNormWrapper,
)

# ── NonNegativeHead ──────────────────────────────────────────────────


def test_non_negative_head_softplus_enforces_non_negativity():
    head = NonNegativeHead(nn.Linear(8, 5))
    x = torch.randn(32, 8)
    out = head(x)
    assert out.shape == (32, 5)
    assert (out >= 0).all()


def test_non_negative_head_beta_parameter():
    """beta controls softplus sharpness; different betas produce different outputs."""
    x = torch.randn(16, 4)
    h1 = NonNegativeHead(nn.Linear(4, 3), beta=1.0)
    h2 = NonNegativeHead(nn.Linear(4, 3), beta=5.0)
    h2.load_state_dict(h1.state_dict())  # same weights
    out1 = h1(x)
    out2 = h2(x)
    assert not torch.allclose(out1, out2)  # different betas, different outputs
    assert (out2 >= 0).all()


def test_non_negative_head_gradients_flow():
    head = NonNegativeHead(nn.Linear(4, 2))
    x = torch.randn(8, 4, requires_grad=True)
    loss = head(x).sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for p in head.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


# ── BoundedHead ──────────────────────────────────────────────────────


def test_bounded_head_outputs_in_range():
    head = BoundedHead(nn.Linear(4, 3), low=-1.0, high=2.0)
    x = torch.randn(16, 4)
    out = head(x)
    assert (out >= -1.0).all()
    assert (out <= 2.0).all()


def test_bounded_head_extreme_values_stay_bounded():
    """With large inputs, sigmoid saturates but outputs stay in bounds."""
    head = BoundedHead(nn.Linear(4, 1), low=10.0, high=20.0)
    # Very large weights → sigmoid saturates at 0 or 1
    nn.init.constant_(head.module.weight, 1e6)
    nn.init.constant_(head.module.bias, 0.0)
    x = torch.randn(16, 4)
    out = head(x)
    assert (out >= 10.0 - 1e-4).all()
    assert (out <= 20.0 + 1e-4).all()


def test_bounded_head_low_greater_equal_high_raises():
    with pytest.raises(ValueError, match="low < high"):
        BoundedHead(nn.Linear(4, 1), low=5.0, high=5.0)
    with pytest.raises(ValueError, match="low < high"):
        BoundedHead(nn.Linear(4, 1), low=5.0, high=3.0)


def test_bounded_head_gradients_flow():
    head = BoundedHead(nn.Linear(4, 2), low=0.0, high=1.0)
    x = torch.randn(8, 4, requires_grad=True)
    loss = head(x).sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_bounded_head_unit_interval():
    """Default [0, 1] bounds — outputs are in unit interval."""
    head = BoundedHead(nn.Linear(8, 3))
    x = torch.randn(64, 8)
    out = head(x)
    assert (out >= 0.0).all()
    assert (out <= 1.0).all()


# ── SimplexHead ──────────────────────────────────────────────────────


def test_simplex_head_sums_to_one():
    head = SimplexHead(nn.Linear(6, 5))
    x = torch.randn(10, 6)
    out = head(x)
    assert out.shape == (10, 5)
    assert (out >= 0).all()
    torch.testing.assert_close(out.sum(dim=-1), torch.ones(x.shape[0]), atol=1e-5, rtol=0.0)


def test_simplex_head_custom_dim():
    head = SimplexHead(nn.Linear(6, 5, bias=False), dim=0)
    x = torch.randn(3, 6)
    out = head(x)
    assert out.shape == (3, 5)
    torch.testing.assert_close(out.sum(dim=0), torch.ones(5), atol=1e-5, rtol=0.0)


def test_simplex_head_gradients_flow():
    head = SimplexHead(nn.Linear(4, 3))
    x = torch.randn(8, 4, requires_grad=True)
    loss = head(x).sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


# ── NonCrossingSort ──────────────────────────────────────────────────


def test_non_crossing_sort_enforces_order():
    sorter = NonCrossingSort(dim=-1)
    x = torch.randn(32, 10)
    out = sorter(x)
    assert out.shape == x.shape
    diffs = out[:, 1:] - out[:, :-1]
    assert (diffs >= 0).all()


def test_non_crossing_sort_custom_dim():
    sorter = NonCrossingSort(dim=0)
    x = torch.randn(10, 8)
    out = sorter(x)
    diffs = out[1:] - out[:-1]
    assert (diffs >= 0).all()


def test_non_crossing_sort_preserves_gradients():
    """Sort is differentiable through the values (though not the indices)."""
    x = torch.randn(10, 5, requires_grad=True)
    sorter = NonCrossingSort(dim=-1)
    out = sorter(x).sum()
    out.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_non_crossing_sort_already_sorted_unchanged():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 0.5, 1.0, 2.0]])
    sorter = NonCrossingSort(dim=-1)
    out = sorter(x)
    torch.testing.assert_close(out, x)


# ── SpectralNormWrapper ──────────────────────────────────────────────


def test_spectral_norm_wrapper_forward_shape():
    layer = SpectralNormWrapper(nn.Linear(4, 3))
    x = torch.randn(16, 4)
    out = layer(x)
    assert out.shape == (16, 3)


def test_spectral_norm_wrapper_non_tensor_output_raises():
    """If the wrapped module returns a non-tensor, TypeError is raised."""

    class NonTensorModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(2, 4))

        def forward(self, x):
            return [x.sum()]  # list, not tensor

    wrapped = SpectralNormWrapper(NonTensorModule())
    with pytest.raises(TypeError, match="expects wrapped module to return a Tensor"):
        wrapped(torch.randn(4))


def test_spectral_norm_wrapper_gradients_flow():
    layer = SpectralNormWrapper(nn.Linear(4, 2))
    x = torch.randn(8, 4, requires_grad=True)
    loss = layer(x).sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_spectral_norm_wrapper_custom_name():
    """SpectralNormWrapper supports a custom parameter name."""
    layer = SpectralNormWrapper(nn.Linear(4, 2), name="weight")
    x = torch.randn(8, 4)
    out = layer(x)
    assert torch.isfinite(out).all()
