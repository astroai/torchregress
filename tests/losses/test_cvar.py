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


def test_cvar_multi_output_per_sample_semantics():
    """CVaR selects the worst α fraction of *samples*, not elements.

    Regression test for the case where a naive :code:`numel()` on a
    multi-dimensional per-sample tensor would silently multiply by the
    output dimension and select more elements than samples exist.
    """
    torch.manual_seed(42)
    # 5 samples × 3 output dimensions
    y_pred = torch.randn(5, 3)
    target = torch.randn(5, 3)

    loss_fn = CVaRLoss(alpha=0.4, base_loss="mse", reduction="mean")
    loss = loss_fn(y_pred, target)

    # Manual reference: per-sample MSE then top 2 worst samples
    per_sample_mse = ((y_pred - target) ** 2).mean(dim=1)
    # alpha=0.4 → ceil(0.4*5) = 2 worst samples
    topk_manual = torch.topk(per_sample_mse, k=2, largest=True).values.mean()
    assert torch.allclose(loss, topk_manual), f"{loss} != {topk_manual}"


def test_cvar_multi_output_all_base_losses():
    """All supported base losses work correctly with multi-output targets."""
    torch.manual_seed(42)
    y_pred = torch.randn(5, 3)
    target = torch.randn(5, 3)

    for base in ("mse", "mae", "huber", "log_cosh", "cauchy", "tukey"):
        loss_fn = CVaRLoss(alpha=0.4, base_loss=base, reduction="mean")
        loss = loss_fn(y_pred, target)
        assert loss.dim() == 0, f"{base}: expected scalar, got shape {loss.shape}"
        assert torch.isfinite(loss), f"{base}: expected finite loss"


def test_cvar_multi_output_with_mask():
    """Masked multi-output CVaR correctly aggregates per-sample."""
    torch.manual_seed(42)
    y_pred = torch.randn(4, 3)
    target = torch.randn(4, 3)
    # Mask out last sample entirely
    mask = torch.ones(4, 3, dtype=torch.bool)
    mask[3, :] = False

    loss_fn = CVaRLoss(alpha=1.0, base_loss="mse", reduction="mean")
    loss = loss_fn(y_pred, target, mask=mask)

    # CVaRLoss computes per-sample MSE via masked mean:
    #   per_sample[i] = mean over valid dims of (y_pred - target)^2
    # Fully masked samples get loss 0 (valid clamped to 1).
    # alpha=1.0 → all 4 samples included in top-k.
    per_sample_mse = ((y_pred - target) ** 2 * mask.float()).sum(dim=1) / mask.float().sum(
        dim=1
    ).clamp(min=1)
    # Last sample is fully masked → loss 0, included in the mean
    expected = per_sample_mse.mean()
    assert torch.allclose(loss, expected), f"{loss} != {expected}"
