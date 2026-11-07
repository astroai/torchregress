"""Tests for DensityEnsemble (equal-weight mixtures of conditional densities)."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor, nn

from torchregress.ensemble import DensityEnsemble
from torchregress.metrics import model_hpd_level, sample_hpd_coverage, sample_hpd_level


class GaussianHead(nn.Module):
    """Isotropic Gaussian head ``y | c ~ N(c + shift, scale^2 I)`` with the flow-head API."""

    def __init__(self, shift: float = 0.0, scale: float = 1.0) -> None:
        super().__init__()
        self.shift = shift
        self.scale = scale

    def _dist(self, context: Tensor) -> torch.distributions.Distribution:
        base = torch.distributions.Normal(context + self.shift, self.scale)
        return torch.distributions.Independent(base, 1)

    def log_prob(self, context: Tensor, target: Tensor) -> Tensor:
        return self._dist(context).log_prob(target)

    def sample(self, context: Tensor, n_samples: int = 1) -> Tensor:
        draws = self._dist(context).sample((n_samples,)).transpose(0, 1)
        return draws.squeeze(1) if n_samples == 1 else draws


def _gaussian_entropy(dim: int, scale: float) -> float:
    return 0.5 * dim * math.log(2 * math.pi * math.e * scale**2)


def test_log_prob_is_equal_weight_mixture() -> None:
    torch.manual_seed(0)
    heads = [GaussianHead(-1.0, 0.5), GaussianHead(1.0, 2.0), GaussianHead(0.0, 1.0)]
    ensemble = DensityEnsemble(heads)
    x, y = torch.randn(16, 2), torch.randn(16, 2)
    members = torch.stack([head.log_prob(x, y) for head in heads])
    expected = torch.log(members.exp().mean(dim=0))
    torch.testing.assert_close(ensemble.log_prob(x, y), expected)
    torch.testing.assert_close(ensemble.member_log_prob(x, y), members)
    torch.testing.assert_close(ensemble(x, y), -expected)


def test_encoders_are_applied_per_member() -> None:
    heads = [GaussianHead(), GaussianHead()]
    encoders = [nn.Identity(), nn.Linear(2, 2, bias=False)]
    with torch.no_grad():
        encoders[1].weight.copy_(2.0 * torch.eye(2, dtype=torch.float32))
    ensemble = DensityEnsemble(heads, encoders)
    x, y = torch.randn(4, 2), torch.randn(4, 2)
    expected = torch.stack([heads[0].log_prob(x, y), heads[1].log_prob(2.0 * x, y)])
    torch.testing.assert_close(ensemble.member_log_prob(x, y), expected)


def test_sample_shapes_and_member_mixing() -> None:
    torch.manual_seed(1)
    ensemble = DensityEnsemble([GaussianHead(-5.0, 0.1), GaussianHead(5.0, 0.1)])
    x = torch.zeros(3, 2)
    assert ensemble.sample(x).shape == (3, 2)
    draws = ensemble.sample(x, n_samples=4000)
    assert draws.shape == (3, 4000, 2)
    # Each draw comes from one member (no averaging of draws), about half each.
    first = draws[..., 0]
    assert bool(((first + 5.0).abs() < 1.0).logical_or((first - 5.0).abs() < 1.0).all())
    fraction = float((first > 0).to(torch.float64).mean())
    assert abs(fraction - 0.5) < 0.03


def test_identical_members_have_no_epistemic_uncertainty() -> None:
    torch.manual_seed(2)
    ensemble = DensityEnsemble([GaussianHead(0.0, 0.7) for _ in range(3)])
    parts = ensemble.entropy_decomposition(torch.randn(8, 2), n_samples=512)
    expected = _gaussian_entropy(2, 0.7)
    torch.testing.assert_close(parts["epistemic"], torch.zeros(8), rtol=0, atol=1e-6)
    assert float((parts["aleatoric"] - expected).abs().max()) < 0.15
    torch.testing.assert_close(parts["total"], parts["aleatoric"] + parts["epistemic"])


def test_disjoint_members_have_log_m_epistemic_information() -> None:
    # Well-separated members: the mutual information between y and the member
    # index is log M, and the aleatoric part is the member entropy.
    torch.manual_seed(3)
    ensemble = DensityEnsemble([GaussianHead(-20.0, 1.0), GaussianHead(20.0, 1.0)])
    parts = ensemble.entropy_decomposition(torch.zeros(6, 1), n_samples=1024)
    torch.testing.assert_close(
        parts["epistemic"], torch.full((6,), math.log(2.0)), rtol=0, atol=1e-3
    )
    assert float((parts["aleatoric"] - _gaussian_entropy(1, 1.0)).abs().max()) < 0.1


def test_epistemic_grows_with_member_disagreement() -> None:
    torch.manual_seed(4)
    x = torch.zeros(32, 1)
    values = []
    for spread in (0.0, 0.5, 1.5):
        ensemble = DensityEnsemble([GaussianHead(-spread), GaussianHead(spread)])
        values.append(float(ensemble.entropy_decomposition(x, n_samples=1024)["epistemic"].mean()))
    assert values[0] < values[1] < values[2]


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="at least two"):
        DensityEnsemble([GaussianHead()])
    with pytest.raises(TypeError, match="nn.Module"):
        DensityEnsemble([GaussianHead(), "head"])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="log_prob"):
        DensityEnsemble([GaussianHead(), nn.Linear(1, 1)])
    with pytest.raises(ValueError, match="encoders"):
        DensityEnsemble([GaussianHead(), GaussianHead()], [nn.Identity()])
    ensemble = DensityEnsemble([GaussianHead(), GaussianHead()])
    with pytest.raises(ValueError, match="n_samples"):
        ensemble.sample(torch.zeros(2, 1), n_samples=0)
    with pytest.raises(ValueError, match="n_samples"):
        ensemble.entropy_decomposition(torch.zeros(2, 1), n_samples=1)


def test_calibrated_ensemble_has_uniform_joint_hpd_levels() -> None:
    # Data from the mixture itself: joint HPD coverage matches nominal.
    torch.manual_seed(5)
    ensemble = DensityEnsemble([GaussianHead(-1.0, 0.8), GaussianHead(1.5, 0.5)])
    x = torch.randn(3000, 3)
    y = ensemble.sample(x)
    levels = model_hpd_level(ensemble, x, y, n_samples=200)
    nominal = torch.tensor([0.5, 0.68, 0.9, 0.95])
    coverage = (levels.unsqueeze(0) <= nominal.unsqueeze(1)).to(torch.float32).mean(dim=1)
    assert float((coverage - nominal).abs().max()) < 0.03


def test_overconfident_member_fails_joint_hpd_coverage() -> None:
    # A single (too-narrow) member as the predictive under-covers.
    torch.manual_seed(6)
    truth = DensityEnsemble([GaussianHead(-1.0, 0.8), GaussianHead(1.5, 0.5)])
    narrow = DensityEnsemble([GaussianHead(1.5, 0.5), GaussianHead(1.5, 0.5)])
    x = torch.randn(2000, 2)
    y = truth.sample(x)
    with torch.no_grad():
        draws = narrow.sample(x, n_samples=200)
        lp_samples = narrow.log_prob(x.repeat_interleave(200, dim=0), draws.reshape(-1, 2))
        lp_target = narrow.log_prob(x, y)
    coverage = sample_hpd_coverage(lp_target, lp_samples.reshape(2000, 200), credibility=0.9)
    assert float(coverage) < 0.7
    assert float(sample_hpd_level(lp_target, lp_samples.reshape(2000, 200)).mean()) > 0.6


def test_ensemble_is_a_conformal_density_score() -> None:
    # An over-confident ensemble regains nominal joint coverage once conformalised.
    from torchregress.losses import ConformalizedDensity

    torch.manual_seed(8)
    ensemble = DensityEnsemble([GaussianHead(0.0, 0.5), GaussianHead(0.2, 0.5)])
    x = torch.randn(6000, 2)
    y = x + torch.randn(6000, 2)
    conformal = ConformalizedDensity(alpha=0.1)
    conformal.calibrate(ensemble, x[:3000], y[:3000])
    assert conformal.threshold is not None
    with torch.no_grad():
        inside = ensemble(x[3000:], y[3000:]) <= conformal.threshold
    assert abs(float(inside.to(torch.float32).mean()) - 0.9) < 0.02


def test_flow_heads_integrate() -> None:
    pytest.importorskip("zuko")
    from torchregress.losses import create_flow_loss

    torch.manual_seed(7)
    heads = [create_flow_loss(n_features=2, context_dim=3, flow_type="nsf") for _ in range(2)]
    ensemble = DensityEnsemble(heads)
    x, y = torch.randn(5, 3), torch.randn(5, 2)
    assert ensemble.log_prob(x, y).shape == (5,)
    assert ensemble.sample(x, n_samples=7).shape == (5, 7, 2)
    parts = ensemble.entropy_decomposition(x, n_samples=8)
    assert set(parts) == {"total", "aleatoric", "epistemic"}
    assert all(value.shape == (5,) and torch.isfinite(value).all() for value in parts.values())
    assert model_hpd_level(ensemble, x, y, n_samples=16).shape == (5,)


# ---------------------------------------------------------------------------
# Fitted mixture weights (wifi-style frequentist uncertainty)
# ---------------------------------------------------------------------------

TRUE_WEIGHTS = torch.tensor([0.3, 0.7])


def _two_member_ensemble() -> DensityEnsemble:
    return DensityEnsemble([GaussianHead(-1.0, 0.6), GaussianHead(1.0, 0.8)])


def _mixture_data(generator: torch.Generator, n: int) -> tuple[Tensor, Tensor]:
    x = torch.randn(n, 1, generator=generator)
    first = torch.rand(n, generator=generator) < TRUE_WEIGHTS[0]
    noise = torch.randn(n, 1, generator=generator)
    y = torch.where(first.unsqueeze(1), x - 1.0 + 0.6 * noise, x + 1.0 + 0.8 * noise)
    return x, y


def test_fit_weights_recovers_mixture_and_matches_scipy() -> None:
    import numpy as np
    import scipy.optimize as so

    generator = torch.Generator().manual_seed(20)
    x, y = _mixture_data(generator, 20_000)
    ensemble = _two_member_ensemble().fit_weights(x, y)
    weights = ensemble.mixture_weights
    assert abs(float(weights[0]) - 0.3) < 0.02
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0))

    member = ensemble.member_log_prob(x, y).double().exp().numpy()

    def negative_log_likelihood(first: float) -> float:
        return -float(np.mean(np.log(first * member[0] + (1 - first) * member[1])))

    reference = so.minimize_scalar(
        negative_log_likelihood, bounds=(1e-6, 1 - 1e-6), method="bounded"
    )
    assert float(weights[0]) == pytest.approx(reference.x, abs=1e-5)
    covariance = ensemble.weight_covariance
    assert covariance is not None and covariance.shape == (2, 2)
    torch.testing.assert_close(covariance.sum(dim=1), torch.zeros(2, dtype=torch.float64))


def test_fitted_weight_intervals_have_nominal_coverage() -> None:
    generator = torch.Generator().manual_seed(21)
    x0 = torch.tensor([[0.0], [0.5], [-1.0]])
    y0 = torch.tensor([[0.2], [1.8], [-2.5]])
    true_log_prob = torch.log(
        TRUE_WEIGHTS[0] * GaussianHead(-1.0, 0.6).log_prob(x0, y0).exp()
        + TRUE_WEIGHTS[1] * GaussianHead(1.0, 0.8).log_prob(x0, y0).exp()
    )
    weight_hits, density_hits = [], []
    for _ in range(300):
        x, y = _mixture_data(generator, 1500)
        ensemble = _two_member_ensemble().fit_weights(x, y)
        covariance = ensemble.weight_covariance
        assert covariance is not None
        weight_error = abs(float(ensemble.mixture_weights[0]) - 0.3)
        weight_hits.append(weight_error < math.sqrt(float(covariance[0, 0])))
        z = (ensemble.log_prob(x0, y0) - true_log_prob) / ensemble.log_prob_std(x0, y0)
        density_hits.append(z.abs() < 1.0)
    weight_coverage = sum(weight_hits) / len(weight_hits)
    density_coverage = torch.stack(density_hits).to(torch.float64).mean(dim=0)
    assert abs(weight_coverage - 0.6827) < 0.07
    assert float((density_coverage - 0.6827).abs().max()) < 0.07


def test_fitted_weights_drive_sampling_and_entropy() -> None:
    generator = torch.Generator().manual_seed(22)
    x, y = _mixture_data(generator, 20_000)
    ensemble = _two_member_ensemble().fit_weights(x, y)
    torch.manual_seed(23)
    draws = ensemble.sample(torch.zeros(1, 1), n_samples=20_000)
    fraction_first = float((draws[..., 0] < 0).to(torch.float64).mean())
    # P(y < 0) under the fitted mixture at x = 0.
    expected = 0.3 * 0.9522 + 0.7 * 0.1056
    assert abs(fraction_first - expected) < 0.02

    # A single member carrying all the weight has no epistemic term.
    ensemble.mixture_weights.copy_(torch.tensor([1.0, 0.0]))
    parts = ensemble.entropy_decomposition(torch.zeros(4, 1), n_samples=256)
    torch.testing.assert_close(parts["epistemic"], torch.zeros(4), rtol=0, atol=1e-6)


def test_log_prob_std_requires_fit_and_handles_identical_members() -> None:
    ensemble = DensityEnsemble([GaussianHead(), GaussianHead()])
    with pytest.raises(RuntimeError, match="fit_weights"):
        ensemble.log_prob_std(torch.zeros(2, 1), torch.zeros(2, 1))
    generator = torch.Generator().manual_seed(24)
    x = torch.randn(500, 1, generator=generator)
    ensemble.fit_weights(x, x + torch.randn(500, 1, generator=generator))
    # Identical members: weights unidentified, but log p and its error are well defined.
    std = ensemble.log_prob_std(torch.zeros(3, 1), torch.ones(3, 1))
    torch.testing.assert_close(std, torch.zeros(3, dtype=torch.float64), rtol=0, atol=1e-8)
    with pytest.raises(ValueError, match="max_iter"):
        ensemble.fit_weights(x, x, max_iter=0)
