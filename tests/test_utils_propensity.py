from __future__ import annotations

import torch

from torchregress.utils import ipw_weights


def test_ipw_weights_basic_behavior() -> None:
    p = torch.tensor([0.2, 0.5, 0.8])
    w = ipw_weights(p, normalize=False)
    expected = torch.tensor([5.0, 2.0, 1.25])
    assert torch.allclose(w, expected)


def test_ipw_weights_with_observed() -> None:
    p = torch.tensor([0.2, 0.7, 0.8])
    obs = torch.tensor([1.0, 0.0, 1.0])
    w = ipw_weights(p, observed=obs, normalize=False)
    expected = torch.tensor([5.0, 1.0 / 0.3, 1.25])
    assert torch.allclose(w, expected)
