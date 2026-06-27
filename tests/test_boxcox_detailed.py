from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.special import boxcox, inv_boxcox

from torchregress.utils.transform import (
    BoxCoxTransform,
    make_target_transform,
)


class TestBoxCoxDetailed:
    @pytest.mark.parametrize("lam", [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    def test_consistency_with_scipy(self, lam: float) -> None:
        eps = 1e-6
        x = torch.linspace(0.1, 5.0, 10).unsqueeze(-1)

        # Torch implementation (shifts by eps)
        transform = BoxCoxTransform(lam=lam, eps=eps)
        y_torch = transform.forward(x)

        # Scipy comparison (manually shift x by eps)
        x_np = x.numpy() + eps
        y_scipy = boxcox(x_np, lam)

        assert np.allclose(y_torch.numpy(), y_scipy, atol=1e-6)

        # Inverse consistency
        x_restored_torch = transform.inverse(y_torch)
        y_np = y_torch.numpy()
        x_restored_scipy = inv_boxcox(y_np, lam) - eps

        assert np.allclose(x_restored_torch.numpy(), x_restored_scipy, atol=1e-6)

    def test_lambda_zero_is_log(self) -> None:
        eps = 1e-6
        x = torch.linspace(0.1, 5.0, 10).unsqueeze(-1)
        transform = BoxCoxTransform(lam=0.0, eps=eps)

        y = transform.forward(x)
        expected = torch.log(x + eps)
        assert torch.allclose(y, expected)

        x_restored = transform.inverse(y)
        assert torch.allclose(x_restored, x)

    def test_near_zero_lambda(self) -> None:
        # Test that lam=1e-9 (very small but non-zero) behaves like lam=0
        eps = 1e-6
        x = torch.tensor([1.0, 2.0, 3.0])
        t_zero = BoxCoxTransform(lam=0.0, eps=eps)
        t_near_zero = BoxCoxTransform(lam=1e-9, eps=eps)

        assert torch.allclose(t_near_zero.forward(x), t_zero.forward(x), atol=1e-7)
        assert torch.allclose(t_near_zero.inverse(t_near_zero.forward(x)), x, atol=1e-6)

    def test_out_of_support_raises_value_error(self) -> None:
        eps = 1e-6
        transform = BoxCoxTransform(eps=eps)

        # x < -eps should raise ValueError
        with pytest.raises(ValueError, match="BoxCoxTransform requires inputs >= -1e-06"):
            transform.forward(torch.tensor([-2e-6]))

    def test_make_target_transform_integration(self) -> None:
        transform = make_target_transform("boxcox", lam=0.7, eps=1e-3)
        assert isinstance(transform, BoxCoxTransform)
        assert transform.lam == 0.7
        assert transform.eps == 1e-3

        # Case insensitive and hyphen/underscore handling
        transform2 = make_target_transform("Box-Cox", lam=0.3)
        assert isinstance(transform2, BoxCoxTransform)
        assert transform2.lam == 0.3

    def test_inverse_clamping(self) -> None:
        # Test that inverse clamps values to 0 if they would be negative due to eps shift
        transform = BoxCoxTransform(lam=1.0, eps=1.0)  # y = (x+1)-1 = x
        # forward(0.0) -> (0+1-1)/1 = 0
        # inverse(0.0) -> (0*1+1)**(1/1)-1 = 0
        # inverse(-0.5) -> (-0.5*1+1)**1 - 1 = -0.5 -> clamped to 0
        assert transform.inverse(torch.tensor([-0.5])).item() == 0.0

        # For lam=0
        transform_log = BoxCoxTransform(lam=0.0, eps=1.0)
        # forward(x) = log(x+1)
        # inverse(y) = exp(y)-1
        # inverse(-1.0) = exp(-1)-1 approx 0.36-1 = -0.64 -> clamped to 0
        assert transform_log.inverse(torch.tensor([-1.0])).item() == 0.0
