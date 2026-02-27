from __future__ import annotations

import torch

from torchregress.losses import AFTLoss, CensoredGaussianNLLLoss, CensoredQuantileLoss


def test_censored_gaussian_nll_observed_and_censored_paths() -> None:
    mean = torch.tensor([0.0, 1.0, -0.5])
    log_var = torch.zeros_like(mean)
    target = torch.tensor([0.2, 0.8, -0.3])
    censoring = torch.tensor([0, 1, -1])

    loss_fn = CensoredGaussianNLLLoss(reduction="none")
    loss = loss_fn((mean, log_var), target, censoring=censoring)

    assert loss.shape == target.shape
    assert torch.all(torch.isfinite(loss))
    assert torch.all(loss >= 0.0)


def test_censored_gaussian_nll_interval_path() -> None:
    mean = torch.tensor([0.0, 0.0])
    log_var = torch.zeros_like(mean)
    target = torch.tensor([0.0, 0.0])
    lower = torch.tensor([-0.2, float("nan")])
    upper = torch.tensor([0.2, float("nan")])

    loss_fn = CensoredGaussianNLLLoss()
    loss = loss_fn((mean, log_var), target, lower_bound=lower, upper_bound=upper)
    assert torch.isfinite(loss)


def test_censored_quantile_loss_prefers_inside_interval() -> None:
    y_pred = torch.tensor([0.5, 1.5, 2.5])
    target = torch.tensor([0.5, 1.0, 2.0])
    censoring = torch.tensor([0, 1, -1])

    loss_fn = CensoredQuantileLoss(quantile=0.5, reduction="none")
    loss = loss_fn(y_pred, target, censoring=censoring)
    assert loss.shape == target.shape
    assert torch.all(loss >= 0.0)


def test_aft_loss_forward_and_backward() -> None:
    loc = torch.randn(10, requires_grad=True)
    log_scale = torch.randn(10, requires_grad=True)
    target = torch.exp(torch.randn(10))
    censoring = torch.randint(low=-1, high=2, size=(10,))

    loss_fn = AFTLoss()
    loss = loss_fn((loc, log_scale), target, censoring=censoring)
    assert torch.isfinite(loss)
    loss.backward()
    assert loc.grad is not None
    assert log_scale.grad is not None
