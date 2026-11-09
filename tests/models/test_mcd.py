"""Marginal Contrastive Discrimination (MCD) conditional density."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from torchregress.losses import build_ratio_classifier
from torchregress.metrics import model_hpd_level, sample_hpd_coverage
from torchregress.models import MarginalContrastiveDensity

# y | x ~ N(A x, S), x ~ N(0, I): the marginal is y ~ N(0, A A^T + S).
A = torch.tensor([[1.0, 0.5], [-0.3, 0.8]], dtype=torch.float64)
S = torch.tensor([[0.5, 0.2], [0.2, 0.4]], dtype=torch.float64)
MARGINAL_COV = A @ A.T + S


def _conditional(x: torch.Tensor) -> torch.distributions.MultivariateNormal:
    return torch.distributions.MultivariateNormal(x @ A.T, covariance_matrix=S)


def _marginal() -> torch.distributions.MultivariateNormal:
    return torch.distributions.MultivariateNormal(
        torch.zeros(2, dtype=torch.float64), covariance_matrix=MARGINAL_COV
    )


class _OracleMarginal(nn.Module):
    def forward(self) -> torch.distributions.MultivariateNormal:
        return _marginal()


class _OracleRatio(nn.Module):
    """Exact log r(x, y) = log N(y; A x, S) - log N(y; 0, A A^T + S) on concat([x, y])."""

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        x, y = pair[:, :2], pair[:, 2:]
        return (_conditional(x).log_prob(y) - _marginal().log_prob(y)).unsqueeze(-1)


def _oracle() -> MarginalContrastiveDensity:
    return MarginalContrastiveDensity(_OracleMarginal(), _OracleRatio())


def _data(n: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(n, 2, generator=generator, dtype=torch.float64)
    noise = torch.randn(n, 2, generator=generator, dtype=torch.float64)
    return x, x @ A.T + noise @ torch.linalg.cholesky(S).T


def test_oracle_log_prob_is_the_true_conditional_density() -> None:
    generator = torch.Generator().manual_seed(0)
    x, y = _data(50, generator)
    model = _oracle()
    truth = _conditional(x).log_prob(y)
    torch.testing.assert_close(model.log_prob(x, y, normalize=False), truth)
    # E_{p(y)}[r(x, y)] = 1 for the exact ratio, so the Monte-Carlo log-normalizer
    # is zero up to its sampling error, sd(r(x, .)) / sqrt(n) per row.
    n = 20_000
    torch.manual_seed(1)
    log_z = model.log_normalizer(x, n_samples=n)
    torch.manual_seed(2)
    y_ref = _marginal().sample((200_000,))
    ratio = torch.exp(
        _conditional(x.unsqueeze(1)).log_prob(y_ref.unsqueeze(0)) - _marginal().log_prob(y_ref)
    )  # [50, 200_000]
    standard_error = ratio.std(dim=1) / math.sqrt(n)
    assert bool((log_z.abs() < 4 * standard_error + 1e-3).all())
    corrected = model.log_prob(x, y, n_samples=n)
    torch.testing.assert_close(corrected, truth, rtol=0, atol=float(4 * standard_error.max()))
    torch.manual_seed(3)
    nll = model(x, y)
    torch.manual_seed(3)
    torch.testing.assert_close(nll, -model.log_prob(x, y))


def test_oracle_sampling_matches_the_conditional_moments() -> None:
    torch.manual_seed(2)
    x = torch.tensor([[0.0, 0.0], [1.0, -1.0], [-1.5, 0.5]], dtype=torch.float64)
    samples = _oracle().sample(x, n_samples=4000, n_proposals=40_000)
    assert samples.shape == (3, 4000, 2)
    mean = x @ A.T
    torch.testing.assert_close(samples.mean(dim=1), mean, rtol=0, atol=0.06)
    for i in range(3):
        torch.testing.assert_close(torch.cov(samples[i].T), S, rtol=0, atol=0.06)
    ess = _oracle().effective_sample_size(x, n_proposals=40_000)
    assert ess.shape == (3,) and bool((ess > 0.1 * 40_000).all())


def test_effective_sample_size_falls_for_narrow_conditionals() -> None:
    # A context far in the tail moves p(y | x) away from the bulk of p(y).
    torch.manual_seed(3)
    ess = _oracle().effective_sample_size(
        torch.tensor([[0.0, 0.0], [4.0, 4.0]], dtype=torch.float64), n_proposals=10_000
    )
    assert float(ess[1]) < 0.2 * float(ess[0])


def test_oracle_works_with_model_hpd_level() -> None:
    generator = torch.Generator().manual_seed(4)
    x, y = _data(400, generator)
    torch.manual_seed(5)
    level = model_hpd_level(_oracle(), x, y, n_samples=256)
    assert level.shape == (400,)
    # Exact density: the HPD level of the observed target is uniform.
    coverage = torch.stack([(level <= c).double().mean() for c in (0.5, 0.9)])
    expected = torch.tensor([0.5, 0.9], dtype=torch.float64)
    torch.testing.assert_close(coverage, expected, rtol=0, atol=0.06)


def test_learned_model_approaches_the_true_density() -> None:
    zuko = pytest.importorskip("zuko")
    del zuko
    from torchregress.losses import create_flow_model

    generator = torch.Generator().manual_seed(6)
    torch.manual_seed(6)
    x, y = (t.float() for t in _data(3000, generator))
    x_test, y_test = (t.float() for t in _data(2000, generator))
    model = MarginalContrastiveDensity(
        create_flow_model(2, 0, n_transforms=3, hidden_features=[32, 32]),
        build_ratio_classifier(context_dim=2, observable_dim=2, hidden_dims=(64, 64)),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(30):
        for batch in torch.randperm(3000, generator=generator).split(256):
            optimizer.zero_grad()
            model.loss(x[batch], y[batch]).backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        log_z = model.log_normalizer(x_test, n_samples=4096)  # once per context
        log_q = model.log_prob(x_test, y_test, normalize=False) - log_z
        m, s_draws = 500, 200
        samples = model.sample(x_test[:m], n_samples=s_draws, n_proposals=8192)
        log_q_samples = (
            model.log_prob(
                x_test[:m].repeat_interleave(s_draws, dim=0),
                samples.reshape(-1, 2),
                normalize=False,
            ).view(m, s_draws)
            - log_z[:m, None]
        )
        coverage = sample_hpd_coverage(log_q[:m], log_q_samples, (0.5, 0.9))
    truth = _conditional(x_test.double()).log_prob(y_test.double()).float()
    marginal_gap = float((truth - _marginal().log_prob(y_test.double()).float()).mean())
    gap = float((truth - log_q).mean())
    # The marginal alone misses ~0.9 nats of conditional information here.
    assert marginal_gap > 0.5
    assert gap < 0.1
    torch.testing.assert_close(coverage, torch.tensor([0.5, 0.9]), rtol=0, atol=0.07)


def test_loss_uses_cyclic_batch_negatives_and_needs_two_rows() -> None:
    torch.manual_seed(7)
    model = MarginalContrastiveDensity(
        _OracleMarginal(), build_ratio_classifier(2, 2).double(), n_negatives=8
    )
    x, y = _data(2, torch.Generator().manual_seed(8))
    loss = model.loss(x, y)  # one negative per row when the batch has two rows
    assert loss.dim() == 0 and math.isfinite(float(loss))
    loss.backward()
    assert all(p.grad is not None for p in model.classifier.parameters())
    with pytest.raises(ValueError, match="two rows"):
        model.loss(x[:1], y[:1])


def test_validation_errors() -> None:
    with pytest.raises(TypeError, match="nn.Module"):
        MarginalContrastiveDensity(_marginal, _OracleRatio())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_negatives"):
        MarginalContrastiveDensity(_OracleMarginal(), _OracleRatio(), n_negatives=0)
    with pytest.raises(ValueError, match="1-D or 2-D"):
        _oracle().log_prob(torch.zeros(2, 2, 2), torch.zeros(2, 2))
    with pytest.raises(TypeError, match="torch.Tensor"):
        _oracle().log_ratio([[0.0, 0.0]], torch.zeros(1, 2))  # type: ignore[arg-type]


def test_mcd_heads_combine_in_a_density_ensemble() -> None:
    from torchregress.ensemble import DensityEnsemble

    generator = torch.Generator().manual_seed(9)
    x, y = _data(20, generator)
    ensemble = DensityEnsemble([_oracle(), _oracle()])
    torch.manual_seed(10)
    # Two exact heads: the mixture is the true conditional, up to the
    # Monte-Carlo normalizer of each head (bounded as in the first test).
    torch.testing.assert_close(
        ensemble.log_prob(x, y), _conditional(x).log_prob(y), rtol=0, atol=0.1
    )
    assert ensemble.sample(x, n_samples=7).shape == (20, 7, 2)


class _CountingRatio(_OracleRatio):
    """Oracle ratio that counts how many pairs it scores."""

    def __init__(self) -> None:
        super().__init__()
        self.pairs = 0

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        self.pairs += pair.shape[0]
        return super().forward(pair)


def test_eval_mode_reuses_one_bank_until_refreshed() -> None:
    generator = torch.Generator().manual_seed(11)
    x, y = _data(30, generator)
    model = _oracle().eval()
    first = model.log_prob(x, y, n_samples=512)
    torch.testing.assert_close(model.log_prob(x, y, n_samples=512), first, rtol=0, atol=0)
    # Target and draws scored in separate calls share the normalizer.
    torch.testing.assert_close(
        model.log_normalizer(x[:5], n_samples=512),
        model.log_normalizer(x, n_samples=512)[:5],
        rtol=0,
        atol=0,
    )
    model.refresh_bank()
    assert not torch.equal(model.log_prob(x, y, n_samples=512), first)
    model.train()
    a, b = model.log_normalizer(x, n_samples=512), model.log_normalizer(x, n_samples=512)
    assert not torch.equal(a, b)  # train mode draws a fresh bank per call


def test_normalizer_runs_once_per_unique_context_without_gradients() -> None:
    ratio = _CountingRatio()
    model = MarginalContrastiveDensity(_OracleMarginal(), ratio).eval()
    x = torch.randn(4, 2, dtype=torch.float64).repeat_interleave(100, dim=0)  # 400 rows
    log_z = model.log_normalizer(x, n_samples=256)
    assert ratio.pairs == 4 * 256
    assert log_z.shape == (400,) and not log_z.requires_grad
    torch.testing.assert_close(log_z[::100], model.log_normalizer(x[::100], n_samples=256))


def test_classifier_is_registered_once() -> None:
    model = MarginalContrastiveDensity(_OracleMarginal(), build_ratio_classifier(2, 2))
    keys = list(model.state_dict())
    assert keys and all(k.startswith("classifier.") for k in keys)


def test_float64_contexts_round_trip_through_a_float32_flow() -> None:
    zuko = pytest.importorskip("zuko")
    del zuko
    from torchregress.losses import create_flow_model

    torch.manual_seed(12)
    model = MarginalContrastiveDensity(
        create_flow_model(2, 0, n_transforms=2), build_ratio_classifier(2, 2).double()
    ).eval()
    x = torch.randn(6, 2, dtype=torch.float64)
    samples = model.sample(x, n_samples=3, n_proposals=256)
    assert samples.dtype == torch.float64
    scores = model.log_prob(x.repeat_interleave(3, dim=0), samples.reshape(-1, 2), n_samples=256)
    assert scores.dtype == torch.float64 and bool(torch.isfinite(scores).all())
    level = model_hpd_level(model, x, torch.randn(6, 2, dtype=torch.float64), n_samples=16)
    assert level.shape == (6,)
