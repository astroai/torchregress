"""Scientific and API tests for split-measurement and corrected-score EIV losses.

The scientific core is:

* Monte-Carlo unbiasedness of both objectives for a linear model (the split
  loss needs no noise covariance, the corrected score subtracts the propagated
  variance);
* recovery of the un-attenuated slope in a small full-batch training run, where
  plain MSE on the noisy view is attenuated by the reliability ratio;
* finite, differentiable behaviour of the split loss for a non-linear model and
  a decreasing corrected-score objective for a small MLP.

Everything is CPU-only, seeded, and kept well below 30 s.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from torchregress.losses.split_eiv import (
    CorrectedScoreEIVLoss,
    SplitMeasurementLoss,
    corrected_score_eiv_loss,
    split_measurement_loss,
)

A_TRUE = 1.5
B_TRUE = 2.0
SIGMA_X = 0.5


def _linear_leaf(w: torch.Tensor) -> torch.Tensor:
    """Exact linear model ``a + b * w`` (a leaf function: no trainable params)."""
    return A_TRUE + B_TRUE * w


# ---------------------------------------------------------------------------
# 1. Monte-Carlo unbiasedness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target_noise_std", [0.0, 0.4])
def test_monte_carlo_unbiasedness_for_linear_model(target_noise_std: float) -> None:
    torch.manual_seed(0)
    n = 20_000
    x = torch.randn(n, 1)
    y = A_TRUE + B_TRUE * x + target_noise_std * torch.randn(n, 1)
    x1 = x + SIGMA_X * torch.randn(n, 1)
    x2 = x + SIGMA_X * torch.randn(n, 1)

    noiseless_mse = ((y - _linear_leaf(x)) ** 2).mean()
    plain_mse = ((y - _linear_leaf(x1)) ** 2).mean()
    split = split_measurement_loss(_linear_leaf, x1, y, x2, reduction="mean")
    corrected = corrected_score_eiv_loss(_linear_leaf, x1, y, sigma_x=SIGMA_X)

    # The noiseless risk is exactly the irreducible target noise.
    assert abs(noiseless_mse.item() - target_noise_std**2) < 0.03

    # Both corrections recover the noiseless risk (unbiased estimator).
    assert abs(split.item() - noiseless_mse.item()) < 0.05
    assert abs(corrected.item() - noiseless_mse.item()) < 0.05

    # Plain MSE on either noisy view is inflated by the propagated input noise.
    attenuation_inflation = (B_TRUE * SIGMA_X) ** 2
    assert abs(plain_mse.item() - (noiseless_mse.item() + attenuation_inflation)) < 0.1
    assert plain_mse > split + 0.5 * attenuation_inflation
    assert plain_mse > corrected + 0.5 * attenuation_inflation


def test_split_loss_matches_explicit_residual_product() -> None:
    torch.manual_seed(3)
    x1 = torch.randn(9, 2)
    x2 = x1 + 0.05 * torch.randn(9, 2)
    y = torch.randn(9, 2)

    def model(w: torch.Tensor) -> torch.Tensor:
        return 0.5 * w + 0.1

    expected_per_sample = ((y - model(x1)) * (y - model(x2))).mean(dim=1)
    got = SplitMeasurementLoss(model, reduction="none")(x1, y, y_pred_second=x2)
    assert torch.allclose(got, expected_per_sample, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. Attenuation-recovery training test
# ---------------------------------------------------------------------------


def _fit_slope(loss_closure, epochs: int = 300, lr: float = 0.05) -> float:
    """Fit ``nn.Linear(1, 1)`` full-batch with Adam and return the learned slope."""
    torch.manual_seed(0)
    model = nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_closure(model)
        loss.backward()
        optimizer.step()
    return float(model.weight.item())


def test_training_recovers_unattenuated_slope() -> None:
    torch.manual_seed(0)
    n = 2_048
    x = torch.randn(n, 1)
    y = A_TRUE + B_TRUE * x
    x1 = x + SIGMA_X * torch.randn(n, 1)
    x2 = x + SIGMA_X * torch.randn(n, 1)

    plain_slope = _fit_slope(lambda m: ((m(x1) - y) ** 2).mean())
    split_slope = _fit_slope(lambda m: SplitMeasurementLoss(m)(x1, y, y_pred_second=x2))
    corrected_slope = _fit_slope(lambda m: CorrectedScoreEIVLoss(m, sigma_x=SIGMA_X)(x1, y))

    # Theory: with var(x) = 1 the OLS slope is attenuated by 1 / (1 + sigma^2) = 0.8.
    theoretical_attenuation = B_TRUE / (1.0 + SIGMA_X**2)
    assert abs(plain_slope - theoretical_attenuation) < 0.15
    assert plain_slope < 0.9 * B_TRUE

    assert abs(split_slope - B_TRUE) < 0.15
    assert abs(corrected_slope - B_TRUE) < 0.15
    # Both corrections remove most of the attenuation bias.
    assert abs(split_slope - B_TRUE) < 0.5 * abs(plain_slope - B_TRUE)
    assert abs(corrected_slope - B_TRUE) < 0.5 * abs(plain_slope - B_TRUE)


# ---------------------------------------------------------------------------
# 3. Non-linear behaviour: gradients/HVP and decreasing corrected score
# ---------------------------------------------------------------------------


def test_split_loss_nonlinear_is_finite_and_differentiable_to_second_order() -> None:
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
    x1 = torch.randn(16, 2)
    x2 = x1 + 0.1 * torch.randn(16, 2)
    y = torch.randn(16, 1)

    loss = SplitMeasurementLoss(model)(x1, y, y_pred_second=x2)
    assert torch.isfinite(loss)

    params = list(model.parameters())
    grads = torch.autograd.grad(loss, params, create_graph=True)
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)

    # Hessian-vector product through the product-of-residuals objective: the
    # split loss keeps the squared-smoothing bias, but remains fully smooth
    # and differentiable for tanh networks.
    hvp = torch.autograd.grad(sum((g**2).sum() for g in grads), params)
    assert all(torch.isfinite(h).all() for h in hvp)


def test_corrected_score_decreases_on_small_mlp() -> None:
    torch.manual_seed(2)
    n = 200
    x = torch.randn(n, 1)
    y = torch.sin(2.0 * x)
    noisy_x = x + 0.1 * torch.randn(n, 1)

    model = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1))
    loss_fn = CorrectedScoreEIVLoss(model, sigma_x=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []
    for _ in range(120):
        optimizer.zero_grad()
        loss = loss_fn(noisy_x, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    assert all(math.isfinite(value) for value in losses)
    # No strict recovery claim for non-linear models: only that the objective
    # is optimised successfully.
    assert losses[-1] < losses[0]


# ---------------------------------------------------------------------------
# 4. API edge cases and reduction semantics
# ---------------------------------------------------------------------------


def test_missing_second_view_raises() -> None:
    model = nn.Linear(1, 1)
    loss_fn = SplitMeasurementLoss(model)
    with pytest.raises(ValueError, match="y_pred_second"):
        loss_fn(torch.randn(4, 1), torch.randn(4, 1))


def test_shape_validation_raises() -> None:
    model = nn.Linear(2, 1)
    split = SplitMeasurementLoss(model)
    corrected = CorrectedScoreEIVLoss(model, sigma_x=0.1)

    x1 = torch.randn(5, 2)
    x2 = torch.randn(5, 2)
    y = torch.randn(5, 1)

    with pytest.raises(ValueError, match="same shape"):
        split(x1, y, y_pred_second=torch.randn(5, 3))
    with pytest.raises(ValueError, match="same shape"):
        split(x1, y, y_pred_second=torch.randn(4, 2))
    with pytest.raises(ValueError, match="batch size mismatch"):
        split(x1, torch.randn(4, 1), y_pred_second=x2)
    with pytest.raises(ValueError, match="batch size mismatch"):
        corrected(x1, torch.randn(4, 1))
    with pytest.raises(ValueError, match=r"\[N, D\]"):
        split(torch.randn(5), y, y_pred_second=torch.randn(5))
    with pytest.raises(ValueError, match=r"\[N, D\]"):
        corrected(torch.randn(5), torch.randn(5))
    with pytest.raises(ValueError, match="mask"):
        corrected(x1, y, mask=torch.ones(3, dtype=torch.bool))
    with pytest.raises(ValueError, match="weights"):
        corrected(x1, y, weights=torch.ones(3))
    with pytest.raises(ValueError, match="weights must be non-negative"):
        corrected(x1, y, weights=-torch.ones(5))


def test_reduction_mask_and_weights_semantics() -> None:
    torch.manual_seed(4)
    model = nn.Linear(2, 1)
    x = torch.randn(8, 2)
    y = torch.randn(8, 1)

    per_sample = CorrectedScoreEIVLoss(model, sigma_x=0.2, reduction="none")(x, y)
    assert per_sample.shape == (8,)
    mean_loss = CorrectedScoreEIVLoss(model, sigma_x=0.2)(x, y)
    sum_loss = CorrectedScoreEIVLoss(model, sigma_x=0.2, reduction="sum")(x, y)
    assert torch.allclose(mean_loss, per_sample.mean())
    assert torch.allclose(sum_loss, per_sample.sum())

    # Samples gated off by the mask are dropped from 'mean' and contribute 0 to 'sum'.
    mask = torch.tensor([True, True, False, True, True, True, False, True])
    masked_mean = CorrectedScoreEIVLoss(model, sigma_x=0.2)(x, y, mask=mask)
    assert torch.allclose(masked_mean, per_sample[mask].mean())

    # Sample weights: all-ones must reproduce the unweighted mean; 'sum' with
    # weights is the weighted sum.
    ones = torch.ones(8)
    assert torch.allclose(CorrectedScoreEIVLoss(model, sigma_x=0.2)(x, y, weights=ones), mean_loss)
    weights = torch.linspace(0.5, 1.5, 8)
    weighted_sum = CorrectedScoreEIVLoss(model, sigma_x=0.2, reduction="sum")(x, y, weights=weights)
    assert torch.allclose(weighted_sum, (per_sample * weights).sum())

    # Split loss: masked mean equals mean over kept samples, and a target-shaped
    # mask drops a row when any output element of that row is invalid.
    x1 = x
    x2 = x + 0.05 * torch.randn(8, 2)
    y2 = torch.randn(8, 2)
    model2 = nn.Linear(2, 2)
    split_none = SplitMeasurementLoss(model2, reduction="none")(x1, y2, y_pred_second=x2)
    split_mask = torch.ones(8, 2, dtype=torch.bool)
    split_mask[2, 1] = False
    split_masked = SplitMeasurementLoss(model2)(x1, y2, y_pred_second=x2, mask=split_mask)
    keep = torch.tensor([True, True, False, True, True, True, True, True])
    assert torch.allclose(split_masked, split_none[keep].mean())


def test_multi_output_targets() -> None:
    torch.manual_seed(5)
    # Split: per-sample loss is the mean over output dimensions.
    split_model = nn.Linear(3, 2)
    x1 = torch.randn(6, 3)
    x2 = x1 + 0.1 * torch.randn(6, 3)
    y = torch.randn(6, 2)
    residuals_first = y - split_model(x1)
    residuals_second = y - split_model(x2)
    expected = (residuals_first * residuals_second).mean(dim=1)
    got = SplitMeasurementLoss(split_model, reduction="none")(x1, y, y_pred_second=x2)
    assert torch.allclose(got, expected, atol=1e-6)

    # Corrected score: per-sample loss sums over output dimensions, and the trace
    # correction uses the full M x D Jacobian.  For a linear model J = W.
    corrected_model = nn.Linear(3, 2)
    sigma = torch.tensor([0.1, 0.2, 0.3])
    w = torch.randn(6, 3)
    target = torch.randn(6, 2)
    residual_sq = ((target - corrected_model(w)) ** 2).sum(dim=1)
    weight_matrix = corrected_model.weight  # [M, D] = [2, 3]
    correction = ((weight_matrix**2) * (sigma**2).unsqueeze(0)).sum()
    expected_corrected = residual_sq - correction
    got_corrected = CorrectedScoreEIVLoss(corrected_model, sigma_x=sigma, reduction="none")(
        w, target
    )
    assert got_corrected.shape == (6,)
    assert torch.allclose(got_corrected, expected_corrected, atol=1e-6)

    # A 1-D model output (scalar-per-sample) is accepted too.
    scalar_model = nn.Sequential(nn.Linear(3, 1))
    scalar_pred = scalar_model(w)[:, 0]
    got_scalar = CorrectedScoreEIVLoss(scalar_model, sigma_x=sigma, reduction="none")(
        w, target[:, 0]
    )
    expected_scalar = (target[:, 0] - scalar_pred) ** 2 - (
        sigma**2 * scalar_model[0].weight ** 2
    ).sum()
    assert torch.allclose(got_scalar, expected_scalar, atol=1e-6)


def test_sigma_x_forms_agree() -> None:
    torch.manual_seed(6)
    model = nn.Linear(3, 2)
    w = torch.randn(7, 3)
    y = torch.randn(7, 2)
    std_vector = torch.tensor([0.1, 0.2, 0.3])

    scalar_loss = CorrectedScoreEIVLoss(model, sigma_x=0.2)(w, y)
    scalar_full_cov_loss = CorrectedScoreEIVLoss(
        model, sigma_x=torch.eye(3, dtype=w.dtype, device=w.device) * 0.04
    )(w, y)
    assert torch.allclose(scalar_loss, scalar_full_cov_loss, atol=1e-6)

    vector_loss = CorrectedScoreEIVLoss(model, sigma_x=std_vector)(w, y)
    full_cov_loss = CorrectedScoreEIVLoss(
        model, sigma_x=torch.diag(std_vector**2).to(device=w.device, dtype=w.dtype)
    )(w, y)
    assert torch.allclose(vector_loss, full_cov_loss, atol=1e-6)

    # Per-sample diagonal standard deviations agree with per-sample diagonal covariances.
    per_sample_std = torch.rand(7, 3) * 0.2
    per_sample_std_loss = CorrectedScoreEIVLoss(model, sigma_x=per_sample_std)(w, y)
    per_sample_cov_loss = CorrectedScoreEIVLoss(
        model,
        sigma_x=torch.diag_embed(per_sample_std**2).to(device=w.device, dtype=w.dtype),
    )(w, y)
    assert torch.allclose(per_sample_std_loss, per_sample_cov_loss, atol=1e-6)

    # Full (non-diagonal) covariance: compare against the explicit trace formula.
    factor = 0.1 * torch.randn(3, 3)
    covariance = factor @ factor.T + 0.01 * torch.eye(3, dtype=w.dtype, device=w.device)
    full_loss = CorrectedScoreEIVLoss(model, sigma_x=covariance, reduction="none")(w, y)
    residual_sq = ((y - model(w)) ** 2).sum(dim=1)
    weight_matrix = model.weight
    correction = torch.einsum("md,de,me->", weight_matrix, covariance, weight_matrix)
    assert torch.allclose(full_loss, residual_sq - correction, atol=1e-6)


def test_jacobian_chunking_is_exact() -> None:
    torch.manual_seed(7)
    model = nn.Linear(4, 2)
    w = torch.randn(11, 4)
    y = torch.randn(11, 2)

    unchunked = CorrectedScoreEIVLoss(model, sigma_x=0.3)(w, y)
    for chunk_size in (1, 3, 11):
        chunked = CorrectedScoreEIVLoss(model, sigma_x=0.3, jacobian_chunk_size=chunk_size)(w, y)
        assert torch.allclose(chunked, unchunked, atol=1e-6), chunk_size


def test_stabilize_and_gradient_contract() -> None:
    torch.manual_seed(8)
    # A large noise covariance makes the unbiased corrected score negative.
    model = nn.Linear(2, 1)
    w = torch.randn(16, 2)
    y = torch.randn(16, 1)

    per_sample = CorrectedScoreEIVLoss(model, sigma_x=5.0, reduction="none")(w, y)
    assert (per_sample < 0).any()
    clamped_per_sample = CorrectedScoreEIVLoss(
        model, sigma_x=5.0, reduction="none", stabilize="clamp"
    )(w, y)
    assert torch.allclose(clamped_per_sample, per_sample.clamp_min(0.0))

    # With a constant model (zero weight) the corrected-score curvature term is
    # zero, so each per-sample loss is a non-negative residual square and
    # 'clamp' is the identity map: gradients must be identical.
    model_a = nn.Linear(2, 1)
    model_b = nn.Linear(2, 1)
    with torch.no_grad():
        model_a.weight.zero_()
        model_b.weight.zero_()
        model_a.bias.fill_(0.3)
        model_b.bias.fill_(0.3)
    per_sample_small = CorrectedScoreEIVLoss(model_a, sigma_x=0.2, reduction="none")(w, y)
    assert not (per_sample_small < 0).any()
    loss_a = CorrectedScoreEIVLoss(model_a, sigma_x=0.2)(w, y)
    loss_b = CorrectedScoreEIVLoss(model_b, sigma_x=0.2, stabilize="clamp")(w, y)
    grad_a = torch.autograd.grad(loss_a, model_a.bias, retain_graph=True)[0]
    grad_b = torch.autograd.grad(loss_b, model_b.bias, retain_graph=True)[0]
    assert torch.allclose(grad_a, grad_b)

    with pytest.raises(ValueError, match="stabilize"):
        CorrectedScoreEIVLoss(model, sigma_x=0.2, stabilize="bogus")
    with pytest.raises(ValueError, match="jacobian_chunk_size"):
        CorrectedScoreEIVLoss(model, sigma_x=0.2, jacobian_chunk_size=0)
    with pytest.raises(ValueError, match="jacobian_chunk_size"):
        CorrectedScoreEIVLoss(model, sigma_x=0.2, jacobian_chunk_size=2.5)  # type: ignore[arg-type]


def test_helpers_match_classes_and_forward_sigma_override() -> None:
    torch.manual_seed(9)
    model = nn.Linear(2, 1)
    w = torch.randn(6, 2)
    y = torch.randn(6, 1)
    x2 = w + 0.1 * torch.randn(6, 2)

    split_class = SplitMeasurementLoss(model)(w, y, y_pred_second=x2)
    assert torch.allclose(split_measurement_loss(model, w, y, x2), split_class)

    corrected_class = CorrectedScoreEIVLoss(model, sigma_x=0.2)(w, y)
    assert torch.allclose(corrected_score_eiv_loss(model, w, y, 0.2), corrected_class)

    # sigma_x can be overridden per forward call (same convention as other EIV losses).
    overridden = CorrectedScoreEIVLoss(model, sigma_x=0.2)(w, y, sigma_x=0.5)
    explicit = CorrectedScoreEIVLoss(model, sigma_x=0.5)(w, y)
    assert torch.allclose(overridden, explicit)


def test_module_exports_are_unique() -> None:
    import torchregress.losses.split_eiv as module

    assert module.__all__ == [
        "CorrectedScoreEIVLoss",
        "SplitMeasurementLoss",
        "corrected_score_eiv_loss",
        "split_measurement_loss",
    ]
    assert len(module.__all__) == len(set(module.__all__))
    for name in module.__all__:
        assert hasattr(module, name)
