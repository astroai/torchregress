import torch

from torchregress.losses import CVaRLoss


def test_cvar_mse_topk_mean():
    y_pred = torch.tensor([[0.0], [1.0], [3.0]])
    target = torch.zeros_like(y_pred)
    loss_fn = CVaRLoss(alpha=0.5, base_loss="mse", reduction="mean")

    # Per-sample MSE: [0, 1, 9] -> top 2 mean = 5
    loss = loss_fn(y_pred, target)
    assert torch.isclose(loss, torch.tensor(5.0))


def test_cvar_reduction_none_returns_per_sample():
    y_pred = torch.tensor([[0.0], [2.0]])
    target = torch.zeros_like(y_pred)
    loss_fn = CVaRLoss(alpha=1.0, base_loss="mae", reduction="none")

    loss = loss_fn(y_pred, target)
    assert loss.shape == (2,)
    assert torch.allclose(loss, torch.tensor([0.0, 2.0]))


def test_cvar_with_mask():
    y_pred = torch.tensor([[0.0, 2.0], [1.0, 3.0]])
    target = torch.zeros_like(y_pred)
    mask = torch.tensor([[True, False], [True, True]])

    loss_fn = CVaRLoss(alpha=1.0, base_loss="mse", reduction="mean")
    loss = loss_fn(y_pred, target, mask=mask)

    # Per-sample masked means: [0.0, (1^2 + 3^2)/2 = 5.0] -> mean = 2.5
    assert torch.isclose(loss, torch.tensor(2.5))
