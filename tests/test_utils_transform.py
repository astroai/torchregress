from __future__ import annotations

import torch

from torchregress.utils.transform import (
    BoxCoxTransform,
    LogTransform,
    SqrtTransform,
    YeoJohnsonTransform,
    make_target_transform,
)


def test_positive_support_transforms_roundtrip() -> None:
    x = torch.linspace(0.05, 5.0, 64).unsqueeze(-1)
    for transform in (LogTransform(eps=1e-6), BoxCoxTransform(lam=0.25), SqrtTransform()):
        restored = transform.inverse(transform.forward(x))
        assert torch.allclose(restored, x, atol=1e-5, rtol=1e-5)


def test_yeojohnson_roundtrip_for_signed_targets() -> None:
    x = torch.linspace(-2.0, 3.0, 128).unsqueeze(-1)
    transform = YeoJohnsonTransform(lam=0.5)
    restored = transform.inverse(transform.forward(x))
    assert torch.allclose(restored, x, atol=1e-5, rtol=1e-5)


def test_make_target_transform_resolves_names() -> None:
    assert isinstance(make_target_transform("log"), LogTransform)
    assert isinstance(make_target_transform("boxcox", lam=0.5), BoxCoxTransform)
    assert isinstance(make_target_transform("sqrt"), SqrtTransform)
    assert isinstance(make_target_transform("yeo_johnson"), YeoJohnsonTransform)
