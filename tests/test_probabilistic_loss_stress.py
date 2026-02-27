import torch

from torchregress.losses import LowRankGaussianLoss, MDNLoss, MultivariateGaussianLoss
from torchregress.metrics import ensemble_variance_decomposition


def test_multivariate_gaussian_loss_handles_non_pd_covariance_with_mask_and_weights() -> None:
    torch.manual_seed(0)
    batch, dim = 5, 2
    y_pred = torch.randn(batch, dim)
    target = torch.randn(batch, dim)

    # Non-PD covariance (eigenvalues 3 and -1) should trigger the fallback path.
    cov = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    cov_batch = cov.unsqueeze(0).expand(batch, -1, -1).clone()

    mask = torch.tensor(
        [
            [True, True],
            [True, False],
            [True, True],
            [False, False],
            [True, True],
        ]
    )
    weights = torch.tensor(
        [
            [1.0, 1.0],
            [0.5, 0.5],
            [2.0, 2.0],
            [1.0, 1.0],
            [1.5, 1.5],
        ]
    )

    loss_none = MultivariateGaussianLoss(reduction="none")(
        y_pred, target, covariance_matrices=cov_batch, mask=mask, weights=weights
    )
    loss_mean = MultivariateGaussianLoss(reduction="mean")(
        y_pred, target, covariance_matrices=cov_batch, mask=mask, weights=weights
    )

    assert loss_none.shape == (batch,)
    assert torch.all(torch.isfinite(loss_none))
    assert torch.isfinite(loss_mean)


def test_low_rank_gaussian_loss_shape_stress_is_finite() -> None:
    torch.manual_seed(0)
    batch, dim, rank = 8, 16, 3
    y_pred = torch.randn(batch, dim)
    target = torch.randn(batch, dim)
    cov_factor = torch.randn(batch, dim, rank) * 0.1
    cov_diag = torch.zeros(batch, dim)  # exercise min_variance clamp

    loss = LowRankGaussianLoss(reduction="none")(y_pred, target, cov_factor, cov_diag)
    assert loss.shape == (batch,)
    assert torch.all(torch.isfinite(loss))


def _build_extreme_mdn_predictions(loss_fn: MDNLoss, batch: int) -> torch.Tensor:
    y_pred = torch.randn(batch, loss_fn.expected_output_size)
    # Extreme logits to test softmax stability.
    y_pred[:, : loss_fn.n_components] = torch.tensor([40.0, -40.0, 0.0])[: loss_fn.n_components]

    if loss_fn.covariance_type == "diagonal":
        # Very negative/positive values in log-std slots.
        y_pred[:, -loss_fn.n_components * loss_fn.n_features :] = torch.linspace(
            -50.0, 50.0, loss_fn.n_components * loss_fn.n_features
        )
    else:
        # Large negative diagonal Cholesky entries should remain valid after softplus.
        y_pred[:, -1] = -80.0
    return y_pred


def test_mdn_losses_diagonal_and_full_remain_finite_under_extreme_parameters() -> None:
    torch.manual_seed(0)
    batch = 6

    mdn_diag = MDNLoss(n_components=3, n_features=2, covariance_type="diagonal", reduction="none")
    mdn_full = MDNLoss(n_components=3, n_features=2, covariance_type="full", reduction="mean")

    target = torch.randn(batch, 2)
    y_pred_diag = _build_extreme_mdn_predictions(mdn_diag, batch)
    y_pred_full = _build_extreme_mdn_predictions(mdn_full, batch)

    diag_loss = mdn_diag(y_pred_diag, target)
    full_loss = mdn_full(y_pred_full, target)

    assert diag_loss.shape == (batch,)
    assert torch.all(torch.isfinite(diag_loss))
    assert torch.isfinite(full_loss)


def test_mdn_mask_and_weights_and_ensemble_decomposition_shape_stress() -> None:
    torch.manual_seed(0)
    batch = 7
    mdn = MDNLoss(n_components=2, n_features=3, covariance_type="diagonal", reduction="none")
    y_pred = torch.randn(batch, mdn.expected_output_size)
    target = torch.randn(batch, 3)
    mask = torch.tensor(
        [
            [True, True, True],
            [True, False, True],
            [True, True, True],
            [False, False, False],
            [True, True, True],
            [True, True, False],
            [True, True, True],
        ]
    )
    weights = torch.rand(batch, 3) + 0.1

    loss = mdn(y_pred, target, mask=mask, weights=weights)
    assert loss.shape == (batch,)
    assert torch.all(torch.isfinite(loss))

    means = torch.randn(5, batch, 3)
    variances = torch.rand(5, batch, 3) + 1e-6
    epistemic, aleatoric = ensemble_variance_decomposition(means, variances, dim=0)
    assert epistemic.shape == (batch, 3)
    assert aleatoric.shape == (batch, 3)
    assert torch.all(epistemic >= 0)
    assert torch.all(aleatoric >= 0)
