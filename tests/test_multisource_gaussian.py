"""Independent covariance and gradient checks for shared latent sources."""

import math

import pytest
import torch

from torchregress.losses.multisource import (
    MarginalizedMultiSourceGaussianNLLLoss,
    PackedObservations,
)


def problem():
    torch.manual_seed(71)
    mean = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
    lower = torch.eye(3, dtype=torch.double).expand(4, -1, -1).clone()
    lower[:, 1, 0] = 0.3
    lower[:, 2, 0] = -0.2
    lower.requires_grad_()
    h = torch.randn(4, 5, 3, dtype=torch.double)
    bias = torch.randn(4, 5, dtype=torch.double, requires_grad=True)
    mapped = torch.einsum("bkd,bd->bk", h, mean) + bias
    values = torch.randn(4, 5, dtype=torch.double)
    raw = torch.randn(4, 5, 5, dtype=torch.double)
    r = raw @ raw.transpose(-1, -2) + torch.eye(5, dtype=raw.dtype, device=raw.device) * 0.2
    mask = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 1, 1]], dtype=torch.bool
    )
    return mean, lower, bias, PackedObservations(values, mapped, h, r, mask)


def test_direct_information_and_independent_distribution_match():
    mean, lower, _, obs = problem()
    results = []
    for method in ("direct", "information", "auto"):
        results.append(MarginalizedMultiSourceGaussianNLLLoss("none", method)(mean, lower, obs))
    torch.testing.assert_close(results[0], results[1], atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(results[0], results[2], atol=2e-12, rtol=2e-12)
    for i in (0, 1, 3):
        idx = obs.mask[i].nonzero(as_tuple=True)[0]
        h = obs.jacobian[i, idx]
        cov = obs.measurement_covariance[i][idx][:, idx] + h @ lower[i] @ lower[i].T @ h.T
        law = torch.distributions.MultivariateNormal(obs.mapped_mean[i, idx], covariance_matrix=cov)
        torch.testing.assert_close(results[0][i], -law.log_prob(obs.values[i, idx]))
    assert results[0][2] == 0


def test_gradients_reach_latent_and_calibration():
    mean, lower, bias, obs = problem()
    loss = MarginalizedMultiSourceGaussianNLLLoss(method="information")(mean, lower, obs)
    loss.backward()
    assert all(
        x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
        for x in (mean, lower, bias)
    )


def test_missing_values_do_not_enter_and_empty_star_not_averaged():
    mean, lower, _, obs = problem()
    values = obs.values.clone()
    values[~obs.mask] = float("nan")
    changed = PackedObservations(
        values, obs.mapped_mean, obs.jacobian, obs.measurement_covariance, obs.mask
    )
    loss = MarginalizedMultiSourceGaussianNLLLoss()
    per = loss.per_star(mean, lower, changed)
    torch.testing.assert_close(loss(mean, lower, changed), per[[0, 1, 3]].mean())
    assert torch.isfinite(per).all()


def test_shared_latent_is_not_product_of_separate_marginals():
    mean = torch.zeros(1, 1, dtype=torch.double)
    chol = torch.ones(1, 1, 1, dtype=torch.double)
    obs = PackedObservations(
        torch.tensor([[1.0, 1.0]], dtype=torch.double),
        torch.zeros(1, 2, dtype=torch.double),
        torch.ones(1, 2, 1, dtype=torch.double),
        torch.eye(2, dtype=torch.double)[None] * 0.1,
        torch.ones(1, 2, dtype=torch.bool),
    )
    actual = MarginalizedMultiSourceGaussianNLLLoss()(mean, chol, obs)
    independent = -2 * torch.distributions.Normal(0.0, math.sqrt(1.1)).log_prob(torch.tensor(1.0))
    assert abs(float(actual) - float(independent)) > 0.2


def test_invalid_covariance_is_not_silently_repaired():
    mean, lower, _, obs = problem()
    covariance = obs.measurement_covariance.clone()
    covariance[0] = -torch.eye(5, dtype=covariance.dtype, device=covariance.device)
    bad = PackedObservations(obs.values, obs.mapped_mean, obs.jacobian, covariance, obs.mask)
    with pytest.raises(ValueError, match="positive definite"):
        MarginalizedMultiSourceGaussianNLLLoss()(mean, lower, bad)


def test_outer_autocast_does_not_demote_likelihood_math():
    mean, lower, _, obs = problem()
    mean, lower = mean.float(), lower.float()
    obs = PackedObservations(
        obs.values.float(),
        obs.mapped_mean.float(),
        obs.jacobian.float(),
        obs.measurement_covariance.float(),
        obs.mask,
    )
    fn = MarginalizedMultiSourceGaussianNLLLoss(method="information")
    expected = fn(mean, lower, obs)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = fn(mean, lower, obs)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_half_precision_residual_is_promoted_before_subtraction():
    mean = torch.zeros(1, 1, dtype=torch.float16)
    lower = torch.ones(1, 1, 1, dtype=torch.float16)
    values = torch.full((1, 1), 60000.0, dtype=torch.float16)
    mapped = -values
    obs = PackedObservations(
        values,
        mapped,
        torch.ones(1, 1, 1, dtype=torch.float16),
        torch.ones(1, 1, 1, dtype=torch.float16),
        torch.ones(1, 1, dtype=torch.bool),
    )
    result = MarginalizedMultiSourceGaussianNLLLoss()(mean, lower, obs)
    assert result.dtype == torch.float32 and torch.isfinite(result)
