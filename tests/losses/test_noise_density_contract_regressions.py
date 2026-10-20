"""Independent probability/reduction limits for the noise-aware density APIs."""

import math

import pytest
import torch

from torchregress.losses import (
    BinnedDensityLoss,
    ConR,
    NoiseAwareDensityLoss,
    NormalizingFlowLoss,
    UncertaintyAwareConR,
    create_flow_model,
)
from torchregress.losses.base import BaseLoss
from torchregress.losses.prime import conr_loss


class GaussianScore(BaseLoss):
    def forward(self, y_pred, target, mask=None, weights=None):
        score = 0.5 * (target - y_pred).square() + 0.5 * math.log(2 * math.pi)
        return self._reduce(score, mask, weights)


def test_noise_wrapper_applies_weights_once_without_sigma():
    prediction = torch.tensor([[0.0], [1.0]])
    target = torch.tensor([[1.0], [3.0]])
    weights = torch.tensor([2.0, 7.0])
    loss = NoiseAwareDensityLoss(GaussianScore())
    expected = (
        (0.5 * (target - prediction).square().flatten() + 0.5 * math.log(2 * math.pi)) * weights
    ).sum() / weights.sum()
    torch.testing.assert_close(loss(prediction, target, weights=weights), expected)


def test_joint_density_mask_excludes_nan_rows_and_rejects_partial_coordinates():
    prediction = torch.tensor([[0.1, 0.2], [0.4, 0.5]], requires_grad=True)
    target = torch.tensor([[0.3, 0.7], [float("nan"), float("nan")]])
    loss = NoiseAwareDensityLoss(GaussianScore(), mode="mc", n_samples=8)
    actual = loss(
        prediction,
        target,
        sigma=torch.tensor([0.0, float("nan")]),
        mask=torch.tensor([True, False]),
    )
    expected = GaussianScore()(prediction[:1], target[:1]) * 2
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(prediction.grad).all()
    assert torch.equal(prediction.grad[1], torch.zeros(2))
    with pytest.raises(ValueError, match="partial target masks"):
        loss(
            prediction,
            torch.zeros_like(prediction),
            sigma=0.0,
            mask=torch.tensor([[True, False], [True, True]]),
        )


@pytest.mark.parametrize("sigma", [-0.1, float("nan"), float("inf")])
def test_invalid_measurement_errors_raise(sigma):
    with pytest.raises(ValueError, match="finite nonnegative"):
        NoiseAwareDensityLoss(GaussianScore())(torch.zeros(2, 1), torch.zeros(2, 1), sigma=sigma)


def test_binned_uniform_sampling_density_and_cdf_agree():
    loss = BinnedDensityLoss([0.0, 1.0, 2.0])
    logits = torch.zeros(5, 2)
    queries = torch.tensor([-1.0, 0.0, 0.5, 1.5, 3.0])
    torch.testing.assert_close(loss.cdf(logits, queries), torch.tensor([0.0, 0.0, 0.25, 0.75, 1.0]))
    with pytest.raises(ValueError, match="scalar target"):
        loss(logits, torch.ones(5, 3))


def test_uncertainty_conr_zero_noise_exactly_matches_hard_objective():
    torch.manual_seed(19)
    features = torch.randn(5, 3, requires_grad=True)
    labels = torch.tensor([0.0, 0.05, 0.15, 0.6, 0.8])
    prediction = torch.zeros(5)
    hard = ConR(label_threshold=0.2)(features, labels, prediction=prediction)
    noisy = UncertaintyAwareConR(label_sigma=0.0, label_threshold=0.2)(
        features, labels, prediction=prediction
    )
    torch.testing.assert_close(noisy, hard)
    torch.testing.assert_close(
        torch.autograd.grad(noisy, features, retain_graph=True)[0],
        torch.autograd.grad(hard, features)[0],
    )


def test_uncertainty_conr_matches_independent_gaussian_interval_probability():
    torch.manual_seed(20)
    f = torch.randn(4, 3, dtype=torch.float64)
    y = torch.tensor([0.0, 0.1, 0.5, 0.8], dtype=torch.float64)
    sigma = torch.tensor([0.1, 0.2, 0.3, 0.1], dtype=torch.float64)
    distance = (y[:, None] - y[None, :]).abs()
    spread = (sigma[:, None].square() + sigma[None, :].square()).sqrt()
    normal = torch.distributions.Normal(0.0, 1.0)
    positive = normal.cdf((0.2 - distance) / spread) - normal.cdf((-0.2 - distance) / spread)
    positive.fill_diagonal_(0)
    similarity = (
        torch.nn.functional.normalize(f, dim=-1) @ torch.nn.functional.normalize(f, dim=-1).T / 0.07
    )
    negative = (1 - positive) * distance * 0.2
    negative.fill_diagonal_(0)
    # Direct exponentials are safe for this normalized, bounded toy.
    denominator = ((positive + negative) * similarity.exp()).sum(1)
    expected = denominator.log() - (positive * similarity).sum(1) / positive.sum(1)
    actual = conr_loss(f, y, torch.zeros(4), label_threshold=0.2, label_sigma=sigma)
    torch.testing.assert_close(actual, expected)
    aware = UncertaintyAwareConR(label_sigma=sigma, label_threshold=0.2, reduction="none")
    torch.testing.assert_close(aware(f, y, prediction=torch.zeros(4)), expected)
    keep = torch.tensor([True, False, True, True])
    torch.testing.assert_close(
        aware(f, y, prediction=torch.zeros(4), mask=keep)[keep],
        UncertaintyAwareConR(label_sigma=sigma[keep], label_threshold=0.2, reduction="none")(
            f[keep], y[keep], prediction=torch.zeros(3)
        ),
    )


@pytest.mark.parametrize("family", ["nsf", "cinn", "bpf"])
def test_three_output_flow_has_true_joint_density_and_finite_gradients(family):
    torch.manual_seed(21)
    extra = {"degree": 8} if family == "bpf" else {}
    model = create_flow_model(
        n_features=3,
        context_dim=4,
        flow_type=family,
        n_transforms=2,
        hidden_features=[16, 16],
        **extra,
    )
    density = NormalizingFlowLoss(model)
    context = torch.randn(6, 4, requires_grad=True)
    target = torch.randn(6, 3)
    logp = density.log_prob(context, target)
    assert logp.shape == (6,) and torch.isfinite(logp).all()
    (-logp.mean()).backward()
    assert torch.isfinite(context.grad).all()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_noise_wrapper_contamination_applies_without_measurement_sigma():
    y = torch.tensor([[4.0]], dtype=torch.float64)
    clean = torch.distributions.Normal(0.0, 1.0).log_prob(y).sum(-1)
    bad = torch.distributions.StudentT(3.0, 0.0, 2.0).log_prob(y).sum(-1)
    expected = -torch.logaddexp(math.log(0.9) + clean, math.log(0.1) + bad)
    actual = NoiseAwareDensityLoss(
        GaussianScore(), contamination_fraction=0.1, contamination_scale=2.0
    )(torch.zeros_like(y), y)
    torch.testing.assert_close(actual, expected.mean())


def test_binned_missing_sigma_and_label_have_zero_finite_gradient():
    logits = torch.randn(3, 4, requires_grad=True)
    labels = torch.tensor([0.2, float("nan"), 0.7])
    sigma = torch.tensor([[0.1], [float("nan")], [0.2]])
    keep = torch.tensor([[True], [False], [True]])
    loss = BinnedDensityLoss(torch.linspace(0, 1, 5))(logits, labels, sigma=sigma, mask=keep)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and torch.equal(logits.grad[1], torch.zeros(4))


def test_binned_uniform_distribution_rejects_infinite_edges():
    with pytest.raises(ValueError, match="finite and strictly increasing"):
        BinnedDensityLoss([-float("inf"), 0.0, float("inf")])
