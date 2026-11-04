"""Independent normalized-mixture checks, with synthetic observations only."""

import itertools
import math

import pytest
import torch
from torch.distributions import MultivariateNormal, Normal

from torchregress.losses.multisource import (
    MarginalizedMultiSourceGaussianNLLLoss,
    PackedObservations,
)
from torchregress.losses.multisource_robust import RobustMultiSourceGaussianNLLLoss


def test_public_loss_exports_include_multisource_contract():
    import torchregress.losses as losses

    for name in (
        "PackedObservations",
        "MarginalizedMultiSourceGaussianNLLLoss",
        "RobustMultiSourceGaussianNLLLoss",
        "GaussianLatentForwardObservationNLLLoss",
    ):
        assert name in losses.__all__
        assert getattr(losses, name) is not None


def fixture():
    mean = torch.tensor([[0.2, -0.3]], dtype=torch.double)
    lower = torch.tensor([[[0.7, 0.0], [0.3, 0.6]]], dtype=torch.double)
    h = torch.tensor([[[1.0, 0.2], [0.1, 1.0], [1.0, -0.5]]], dtype=torch.double)
    obs = PackedObservations(
        torch.tensor([[0.4, -0.7, 1.4]], dtype=torch.double),
        (h @ mean[..., None]).squeeze(-1),
        h,
        torch.tensor([[[0.3, 0.08, 0.0], [0.08, 0.2, 0.0], [0.0, 0.0, 0.4]]], dtype=torch.double),
        torch.ones(1, 3, dtype=torch.bool),
        torch.tensor([[11, 11, 24]]),
    )
    return mean, lower, obs


def test_joint_assignments_match_independent_distribution_and_not_separate_latents():
    mean, lower, obs = fixture()
    ps = torch.tensor([[0.2, 0.2, 0.35]], dtype=torch.double)
    scales = torch.tensor([[3.0, 3.0, 5.0]], dtype=torch.double)
    actual = RobustMultiSourceGaussianNLLLoss()(
        mean, lower, obs, outlier_probabilities=ps, outlier_scales=scales
    )
    joint, wrong = [], []
    latent_cov = obs.jacobian[0] @ lower[0] @ lower[0].T @ obs.jacobian[0].T
    for a, b in itertools.product((False, True), repeat=2):
        multiplier = torch.tensor(
            [3.0 if a else 1.0, 3.0 if a else 1.0, 5.0 if b else 1.0], dtype=torch.double
        )
        r = obs.measurement_covariance[0] * multiplier[:, None] * multiplier[None, :]
        log_weight = math.log(0.2 if a else 0.8) + math.log(0.35 if b else 0.65)
        dist = MultivariateNormal(obs.mapped_mean[0], covariance_matrix=r + latent_cov)
        joint.append(dist.log_prob(obs.values[0]) + log_weight)
        wrong.append(
            MultivariateNormal(
                obs.mapped_mean[0, :2], covariance_matrix=(r + latent_cov)[:2, :2]
            ).log_prob(obs.values[0, :2])
            + Normal(obs.mapped_mean[0, 2], (r + latent_cov)[2, 2].sqrt()).log_prob(
                obs.values[0, 2]
            )
            + log_weight
        )
    torch.testing.assert_close(
        actual, -torch.logsumexp(torch.stack(joint), 0), rtol=1e-12, atol=1e-12
    )
    assert abs(actual + torch.logsumexp(torch.stack(wrong), 0)) > 1e-3


def test_density_integrates_to_one_with_full_normalizers():
    grid = torch.linspace(-45, 45, 12001, dtype=torch.double)
    count = len(grid)
    obs = PackedObservations(
        grid[:, None],
        torch.zeros(count, 1, dtype=torch.double),
        torch.ones(count, 1, 1, dtype=torch.double),
        torch.full((count, 1, 1), 0.4, dtype=torch.double),
        torch.ones(count, 1, dtype=torch.bool),
        torch.zeros(count, 1, dtype=torch.long),
    )
    loss = RobustMultiSourceGaussianNLLLoss(
        outlier_probability=0.25, outlier_scale=4, reduction="none"
    )
    # One batch object here would enumerate thousands of tiny Cholesky solves;
    # vectorized independent reference first establishes the normalized target.
    variance = 0.6**2
    reference = (
        0.75
        * Normal(grid.new_tensor(0.0), grid.new_tensor(math.sqrt(variance + 0.4)))
        .log_prob(grid)
        .exp()
        + 0.25
        * Normal(grid.new_tensor(0.0), grid.new_tensor(math.sqrt(variance + 16 * 0.4)))
        .log_prob(grid)
        .exp()
    )
    assert abs(torch.trapezoid(reference, grid) - 1) < 1e-12
    pick = torch.arange(0, count, 120)
    selected = PackedObservations(
        *(
            x[pick]
            for x in (
                obs.values,
                obs.mapped_mean,
                obs.jacobian,
                obs.measurement_covariance,
                obs.mask,
                obs.block_ids,
            )
        )
    )
    actual = loss(
        torch.zeros(len(pick), 1, dtype=torch.double),
        torch.full((len(pick), 1, 1), 0.6, dtype=torch.double),
        selected,
    )
    torch.testing.assert_close(actual.exp().reciprocal(), reference[pick], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("probability", [0.0, 1.0, 0.3])
def test_endpoints_and_unit_scale_reduce_to_expected_gaussian(probability):
    mean, lower, obs = fixture()
    loss = RobustMultiSourceGaussianNLLLoss(outlier_probability=probability, outlier_scale=1)
    expected = MarginalizedMultiSourceGaussianNLLLoss()(mean, lower, obs)
    torch.testing.assert_close(loss(mean, lower, obs), expected)
    if probability in (0.0, 1.0):
        multiplier = 9 if probability else 1
        other = PackedObservations(
            obs.values,
            obs.mapped_mean,
            obs.jacobian,
            obs.measurement_covariance * multiplier,
            obs.mask,
            obs.block_ids,
        )
        torch.testing.assert_close(
            RobustMultiSourceGaussianNLLLoss(outlier_probability=probability, outlier_scale=3)(
                mean, lower, obs
            ),
            MarginalizedMultiSourceGaussianNLLLoss()(mean, lower, other),
        )


def test_gradcheck_all_statistical_parameters_and_encoder_chain():
    mean, lower, obs = fixture()
    loss = RobustMultiSourceGaussianNLLLoss(method="information")
    x = torch.tensor([[0.2, -0.1]], dtype=torch.double, requires_grad=True)
    packed = torch.tensor([0.7, 0.2, 0.6], dtype=torch.double, requires_grad=True)
    h = obs.jacobian.clone().requires_grad_()
    offset = torch.zeros_like(obs.values, requires_grad=True)
    noise = torch.tensor(0.8, dtype=torch.double, requires_grad=True)
    p = torch.tensor(0.2, dtype=torch.double, requires_grad=True)
    s = torch.tensor(3.0, dtype=torch.double, requires_grad=True)

    def objective(x, packed, h, offset, noise, p, s):
        mu = torch.sin(x)
        factor = torch.stack((packed[0], packed[0] * 0, packed[1], packed[2])).reshape(1, 2, 2)
        observations = PackedObservations(
            obs.values,
            (h @ mu[..., None]).squeeze(-1) + offset,
            h,
            obs.measurement_covariance * noise,
            obs.mask,
            obs.block_ids,
        )
        return loss(mu, factor, observations, outlier_probabilities=p, outlier_scales=s)

    assert torch.autograd.gradcheck(
        objective, (x, packed, h, offset, noise, p, s), atol=1e-5, rtol=1e-4
    )
    grads = torch.autograd.grad(
        objective(x, packed, h, offset, noise, p, s), (x, packed, h, offset, noise, p, s)
    )
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_missing_block_padding_empty_star_and_weighted_reduction():
    mean, lower, obs = fixture()
    values = obs.values.repeat(2, 1)
    values[0, 2] = float("nan")
    values[1] = float("nan")
    mask = torch.tensor([[1, 1, 0], [0, 0, 0]], dtype=torch.bool)
    ids = torch.tensor([[11, 11, -1], [-1, -1, -1]])
    observations = PackedObservations(
        values,
        obs.mapped_mean.repeat(2, 1),
        obs.jacobian.repeat(2, 1, 1),
        obs.measurement_covariance.repeat(2, 1, 1),
        mask,
        ids,
    )
    p = torch.tensor([[0.2, 0.2, float("nan")], [float("nan")] * 3], dtype=torch.double)
    m = mean.repeat(2, 1).requires_grad_()
    loss = RobustMultiSourceGaussianNLLLoss()
    result = loss(
        m,
        lower.repeat(2, 1, 1),
        observations,
        weights=torch.tensor([2.0, 99.0]),
        outlier_probabilities=p,
    )
    selected = PackedObservations(
        obs.values[:, :2],
        obs.mapped_mean[:, :2],
        obs.jacobian[:, :2],
        obs.measurement_covariance[:, :2, :2],
        mask[:1, :2],
        ids[:1, :2],
    )
    torch.testing.assert_close(result, loss(mean, lower, selected, outlier_probabilities=0.2))
    assert loss.last_diagnostics["assignment_counts"] == [2]
    result.backward()
    assert torch.equal(m.grad[1], torch.zeros_like(m.grad[1]))


def test_probability_endpoint_gradients_and_extreme_zero_weight_components():
    mean, lower, obs = fixture()
    for endpoint in (0.0, 1.0):
        p = torch.tensor(endpoint, dtype=torch.double, requires_grad=True)
        result = RobustMultiSourceGaussianNLLLoss()(mean, lower, obs, outlier_probabilities=p)
        grad = torch.autograd.grad(result, p)[0]
        step = 1e-6
        other = endpoint + step if endpoint == 0 else endpoint - step
        finite = RobustMultiSourceGaussianNLLLoss()(mean, lower, obs, outlier_probabilities=other)
        numerical = (finite - result) / step if endpoint == 0 else (result - finite) / step
        torch.testing.assert_close(grad, numerical, atol=2e-4, rtol=2e-4)
    # At p=0 an unused broad component can have vastly higher density; it must
    # not underflow the only nonzero mixture component through a bad shift.
    extreme = PackedObservations(
        torch.tensor([[100.0]], dtype=torch.double),
        torch.zeros(1, 1, dtype=torch.double),
        torch.ones(1, 1, 1, dtype=torch.double),
        torch.ones(1, 1, 1, dtype=torch.double),
        torch.ones(1, 1, dtype=torch.bool),
        torch.zeros(1, 1, dtype=torch.long),
    )
    m = torch.zeros(1, 1, dtype=torch.double)
    factor = torch.ones(1, 1, 1, dtype=torch.double)
    torch.testing.assert_close(
        RobustMultiSourceGaussianNLLLoss(outlier_probability=0)(m, factor, extreme),
        MarginalizedMultiSourceGaussianNLLLoss()(m, factor, extreme),
    )


def test_invalid_blocks_probabilities_covariance_and_enumeration_bounds():
    mean, lower, obs = fixture()

    def replace(**kw):
        return PackedObservations(**{**obs.__dict__, **kw})

    with pytest.raises(ValueError, match="block_ids"):
        RobustMultiSourceGaussianNLLLoss()(mean, lower, replace(block_ids=None))
    with pytest.raises(ValueError, match="nonnegative"):
        RobustMultiSourceGaussianNLLLoss()(mean, lower, replace(block_ids=-obs.block_ids))
    with pytest.raises(ValueError, match="constant"):
        RobustMultiSourceGaussianNLLLoss()(
            mean, lower, obs, outlier_probabilities=torch.tensor([[0.1, 0.2, 0.3]])
        )
    for p in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="probabilit"):
            RobustMultiSourceGaussianNLLLoss()(mean, lower, obs, outlier_probabilities=p)
    with pytest.raises(ValueError, match="scales"):
        RobustMultiSourceGaussianNLLLoss()(mean, lower, obs, outlier_scales=0.5)
    r = obs.measurement_covariance.clone()
    r[0, 0, 2] = r[0, 2, 0] = 0.01
    with pytest.raises(ValueError, match="Cross-block"):
        RobustMultiSourceGaussianNLLLoss()(mean, lower, replace(measurement_covariance=r))
    for kw in ({"max_blocks": 1}, {"max_assignments": 3}):
        with pytest.raises(ValueError, match="bound"):
            RobustMultiSourceGaussianNLLLoss(**kw)(mean, lower, obs)


def test_half_precision_uses_full_precision_mixture_products_and_inflation():
    count = 8
    mean = torch.zeros(1, 1, dtype=torch.float16)
    lower = torch.ones(1, 1, 1, dtype=torch.float16)
    obs = PackedObservations(
        torch.zeros(1, count, dtype=torch.float16),
        torch.zeros(1, count, dtype=torch.float16),
        torch.ones(1, count, 1, dtype=torch.float16),
        1000 * torch.eye(count, dtype=torch.float16)[None],
        torch.ones(1, count, dtype=torch.bool),
        torch.arange(count)[None],
    )
    loss = RobustMultiSourceGaussianNLLLoss(outlier_probability=0.05, outlier_scale=10.0)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = loss(mean, lower, obs)
    assert result.dtype == torch.float32 and torch.isfinite(result)
    reference = PackedObservations(
        obs.values.float(),
        obs.mapped_mean.float(),
        obs.jacobian.float(),
        obs.measurement_covariance.float(),
        obs.mask,
        obs.block_ids,
    )
    torch.testing.assert_close(
        result,
        loss(
            mean.float(),
            lower.float(),
            reference,
            outlier_probabilities=0.05,
        ),
    )
