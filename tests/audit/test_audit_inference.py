"""Regression tests for the 0.3.0 audit: inference / causal / semi-supervised /
prediction / ensemble findings (I-INF-*, I-CAU-*, I-ENS-*).

Each test reproduces one audit finding and pins the fixed behaviour. Coverage
simulations use fewer replications than the original reproductions; each one
states the binomial tolerance it relies on. All randomness is seeded, so the
tests are deterministic.
"""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, LogisticRegression
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import torchregress.inference.ppi as ppi_mod
from torchregress.causal import dr_ate
from torchregress.ensemble import (
    BaseEnsembleModel,
    BatchEnsembleRegressor,
    HeteroscedasticEnsembleModel,
)
from torchregress.ensemble._variance import (
    GAUSSIAN_NLL_MAX_LOGVAR,
    GAUSSIAN_NLL_MIN_LOGVAR,
)
from torchregress.inference import (
    PPIConfig,
    orthogonal_partially_linear,
    ppi_calibrated_mean_ci,
    ppi_mean_ci,
    ppi_ols_ci,
    ppi_pp_mean_ci,
    ppi_quantile_ci,
)
from torchregress.losses import GaussianNLLLoss
from torchregress.prediction import PredictiveBatch, bars_to_density_grid
from torchregress.semi_supervised import (
    SAGERegLoss,
    TeacherStudentTrainer,
    disagreement_to_weight,
    predictive_agreement_score,
)

DOCS = Path(__file__).resolve().parents[2] / "docs"


# ---------------------------------------------------------------------------
# I-INF-001: ppi_quantile_ci must invert the rectified CDF
# ---------------------------------------------------------------------------


def _quantile_draw(rng: np.random.Generator, n: int, big_n: int):
    x_l = rng.standard_normal(n)
    y_l = x_l + rng.standard_normal(n)
    x_u = rng.standard_normal(big_n)
    return torch.tensor(y_l), torch.tensor(x_l), torch.tensor(x_u)


def test_I_INF_001_ppi_quantile_point_estimate_consistent() -> None:
    # Y = X + eps, f(X) = X; true Q_0.9(Y) = z_0.9 * sqrt(2). The old shift
    # estimator converged to z_0.9 (bias ~0.53).
    rng = np.random.default_rng(0)
    q = 0.9
    truth = norm.ppf(q) * math.sqrt(2.0)
    y_l, f_l, f_u = _quantile_draw(rng, 20_000, 200_000)
    out = ppi_quantile_ci(y_l, f_l, f_u, q=q)
    assert abs(out["estimate"] - truth) < 0.05, (out["estimate"], truth)
    assert out["ci_lower"] <= out["estimate"] <= out["ci_upper"]
    assert out["bootstrap_samples"] == 0


def test_I_INF_001_ppi_quantile_matches_bruteforce_rectified_cdf() -> None:
    rng = np.random.default_rng(3)
    y_l, f_l, f_u = _quantile_draw(rng, 40, 300)
    q, alpha = 0.3, 0.1
    out = ppi_quantile_ci(y_l, f_l, f_u, q=q, config=PPIConfig(alpha=alpha))
    # Brute force on all jump points (all sizes < 4096, so the library grid is exact).
    grid = torch.unique(torch.cat([y_l, f_l, f_u]))
    ind = lambda v: (v[:, None] <= grid[None, :]).double()  # noqa: E731
    f_unl = ind(f_u).mean(0)
    rect = ind(y_l) - ind(f_l)
    cdf = f_unl + rect.mean(0)
    est = float(grid[torch.nonzero(cdf >= q)[0]])
    z = norm.ppf(1 - alpha / 2)
    half = z * torch.sqrt(f_unl * (1 - f_unl) / f_u.numel() + rect.var(0) / y_l.numel())
    inside = (cdf - q).abs() <= half
    assert out["estimate"] == pytest.approx(est)
    assert out["ci_lower"] == pytest.approx(min(float(grid[inside].min()), est))
    assert out["ci_upper"] == pytest.approx(max(float(grid[inside].max()), est))


def test_I_INF_001_ppi_quantile_ci_coverage() -> None:
    # Nominal 90%, 100 seeded replications: the binomial sd is 0.03, so the 99%
    # one-sided lower bound is ~0.83; the old estimator covered ~0.0 here.
    rng = np.random.default_rng(1)
    q = 0.9
    truth = norm.ppf(q) * math.sqrt(2.0)
    reps, hits = 100, 0
    for _ in range(reps):
        y_l, f_l, f_u = _quantile_draw(rng, 300, 3000)
        out = ppi_quantile_ci(y_l, f_l, f_u, q=q, config=PPIConfig(alpha=0.1))
        hits += out["ci_lower"] <= truth <= out["ci_upper"]
    assert hits / reps >= 0.82, hits / reps


# ---------------------------------------------------------------------------
# I-INF-002: PPI helpers must keep float64 inputs in float64
# ---------------------------------------------------------------------------


def _offset_data():
    rng = np.random.default_rng(0)
    n, big_n, offset = 500, 5000, 1.0e7
    y_l = offset + 0.1 * rng.standard_normal(n)
    f_l = y_l + 0.01 * rng.standard_normal(n)
    f_u = offset + 0.1 * rng.standard_normal(big_n)
    return y_l, f_l, f_u


def test_I_INF_002_ppi_mean_ci_float64_precision() -> None:
    y_l, f_l, f_u = _offset_data()
    est_ref = f_u.mean() + (y_l - f_l).mean()
    se_ref = np.sqrt((y_l - f_l).var(ddof=1) / y_l.size + f_u.var(ddof=1) / f_u.size)
    out = ppi_mean_ci(
        torch.tensor(y_l), torch.tensor(f_l), torch.tensor(f_u), config=PPIConfig(n_boot=50, seed=0)
    )
    assert abs(out["se"] / se_ref - 1.0) < 0.01, (out["se"], se_ref)
    assert abs(out["estimate"] - est_ref) < 0.1 * se_ref


def test_I_INF_002_ppi_pp_mean_ci_float64_precision() -> None:
    y_l, f_l, f_u = _offset_data()
    out = ppi_pp_mean_ci(torch.tensor(y_l), torch.tensor(f_l), torch.tensor(f_u), lambdas=[1.0])
    est_ref = y_l.mean() + (f_u.mean() - f_l.mean())
    se_ref = np.sqrt(
        y_l.var(ddof=1) / y_l.size
        + f_u.var(ddof=1) / f_u.size
        + f_l.var(ddof=1) / f_l.size
        - 2 * np.cov(y_l, f_l)[0, 1] / y_l.size
    )
    assert abs(out["se"] / se_ref - 1.0) < 0.01, (out["se"], se_ref)
    assert abs(out["estimate"] - est_ref) < 0.1 * se_ref


def test_I_INF_002_ppi_keeps_dtype_in_quantile_and_ols() -> None:
    y_l, f_l, f_u = (torch.tensor(a) for a in _offset_data())
    out = ppi_quantile_ci(y_l, f_l, f_u, q=0.5)
    # float32 would round every value near 1e7 to a multiple of 1.0
    assert out["estimate"] != round(out["estimate"])
    x_l = torch.randn(y_l.numel(), 1, dtype=torch.float64)
    x_u = torch.randn(f_u.numel(), 1, dtype=torch.float64)
    ols = ppi_ols_ci(x_l, y_l, x_u, f_l, f_u, config=PPIConfig(n_boot=10, seed=0))
    ref = ppi_mod._ols_beta(ppi_mod._add_intercept(x_u), f_u) + ppi_mod._ols_beta(
        ppi_mod._add_intercept(x_l), y_l - f_l
    )
    assert ols["coef"] == pytest.approx(ref.tolist(), rel=1e-12)


# ---------------------------------------------------------------------------
# I-INF-003: semi-supervised trust-weight knobs must be honoured
# ---------------------------------------------------------------------------


def _views() -> list[PredictiveBatch]:
    g = torch.Generator().manual_seed(0)
    mu = torch.randn(8, 1, generator=g)
    return [
        PredictiveBatch(
            mean=mu + 0.3 * k * torch.randn(8, 1, generator=g), std=torch.full((8, 1), 0.5)
        )
        for k in range(3)
    ]


def _trainer(**kw) -> TeacherStudentTrainer:
    p = nn.Parameter(torch.zeros(1))
    return TeacherStudentTrainer(
        optimizer=torch.optim.SGD([p], lr=0.1),
        supervised_loss_fn=lambda m, x, y: torch.zeros(()),
        predictive_batch_fn=lambda m, x: _views()[0],
        **kw,
    )


def test_I_INF_003_trainer_tau_is_used() -> None:
    views = _views()
    d = predictive_agreement_score(views)
    tr = _trainer(tau=5.0)
    w = tr._default_sample_weight(views, tr.compute_consensus(views))
    torch.testing.assert_close(w, torch.exp(-d / 5.0).clamp_max(1.0))


def test_I_INF_003_trainer_default_tau_preserves_previous_behaviour() -> None:
    views = _views()
    d = predictive_agreement_score(views)
    tr = _trainer()
    w = tr._default_sample_weight(views, tr.compute_consensus(views))
    torch.testing.assert_close(w, disagreement_to_weight(d, 0.2))


def test_I_INF_003_trainer_all_knobs_are_forwarded() -> None:
    views = _views()
    d = predictive_agreement_score(views)
    kw = dict(hard_weight_threshold=0.05, batch_relative_mode=True, batch_trust_top_k=5)
    tr = _trainer(tau=0.5, weight_power=2.0, **kw)
    w = tr._default_sample_weight(views, tr.compute_consensus(views))
    torch.testing.assert_close(w, disagreement_to_weight(d, 0.5, power=2.0, **kw))


def test_I_INF_003_trainer_rejects_unknown_kwargs() -> None:
    with pytest.raises(TypeError):
        _trainer(batch_trust_topk=3)
    with pytest.raises(TypeError):
        disagreement_to_weight(torch.zeros(3), 0.2, batch_trust_topk=3)  # type: ignore[call-arg]


def test_I_INF_003_disagreement_to_weight_top_k_and_batch_relative() -> None:
    d = torch.linspace(0.0, 1.0, 10)
    w = disagreement_to_weight(d, tau=0.2, batch_trust_top_k=3)
    assert torch.equal(w > 0, d <= d[2])  # the 3 most consistent samples
    torch.testing.assert_close(w[:3], torch.exp(-d[:3] / 0.2))
    # batch-relative weights are invariant to an affine rescaling of the disagreement
    w1 = disagreement_to_weight(d, tau=0.5, batch_relative_mode=True)
    w2 = disagreement_to_weight(10.0 * d + 3.0, tau=0.5, batch_relative_mode=True)
    torch.testing.assert_close(w1, w2)
    with pytest.raises(ValueError, match="batch_trust_top_k"):
        disagreement_to_weight(d, tau=0.2, batch_trust_top_k=0)


def test_I_INF_003_sageregloss_accepts_documented_options() -> None:
    views = _views()
    loss = SAGERegLoss(tau=0.2, hard_weight_threshold=0.5, batch_trust_top_k=2)
    agreement = loss.agreement(views)
    expected = disagreement_to_weight(
        agreement.disagreement, 0.2, hard_weight_threshold=0.5, batch_trust_top_k=2
    )
    torch.testing.assert_close(agreement.weights, expected)
    assert int((agreement.weights > 0).sum()) <= 2


# ---------------------------------------------------------------------------
# I-INF-004: PPI bootstrap memory must be O(N), not O(n_boot * N)
# ---------------------------------------------------------------------------

_BUDGET_BYTES = 1 * 1024**3


@pytest.fixture
def randint_budget(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Refuse any single bootstrap-index request above 1 GB (no real allocation)."""
    real_randint = torch.randint
    requested: list[int] = []

    class _Torch:
        def __getattr__(self, name: str):
            return getattr(torch, name)

        @staticmethod
        def randint(*args, size, **kwargs):
            nbytes = 8 * math.prod(int(s) for s in size)
            requested.append(nbytes)
            if nbytes > _BUDGET_BYTES:
                raise MemoryError(f"bootstrap index tensor of {nbytes / 1024**3:.1f} GB")
            return real_randint(*args, size=size, **kwargs)

    monkeypatch.setattr(ppi_mod, "torch", _Torch())
    return requested


def _million():
    g = torch.Generator().manual_seed(0)
    y = torch.randn(1000, generator=g)
    f_l = y + 0.1 * torch.randn(1000, generator=g)
    f_u = torch.randn(1_000_000, generator=g)
    return y, f_l, f_u


def test_I_INF_004_ppi_mean_ci_one_million_unlabeled(randint_budget: list[int]) -> None:
    y, f_l, f_u = _million()
    out = ppi_mean_ci(y, f_l, f_u, config=PPIConfig(seed=0))
    assert out["ci_lower"] < out["estimate"] < out["ci_upper"]
    assert max(randint_budget) <= _BUDGET_BYTES
    # bootstrap CI half-width agrees with the analytic SE
    z = norm.ppf(0.95)
    assert (out["ci_upper"] - out["ci_lower"]) / (2 * z * out["se"]) == pytest.approx(1, abs=0.1)


def test_I_INF_004_ppi_calibrated_mean_ci_one_million_unlabeled(
    randint_budget: list[int],
) -> None:
    y, f_l, f_u = _million()
    out = ppi_calibrated_mean_ci(y, f_l, f_u, config=PPIConfig(n_boot=200, seed=0))
    assert out["ci_lower"] < out["estimate"] < out["ci_upper"]


def test_I_INF_004_chunked_bootstrap_matches_clt() -> None:
    # Below the CLT threshold the exact bootstrap runs in chunks; its spread must
    # match std / sqrt(N).
    g = torch.Generator().manual_seed(0)
    v = torch.randn(50_000, generator=g, dtype=torch.float64)
    boot = ppi_mod._bootstrap_means(v, n_boot=400, generator=g)
    assert boot.dtype == torch.float64
    assert float(boot.std()) / (float(v.std()) / math.sqrt(v.numel())) == pytest.approx(
        1.0, abs=0.15
    )


# ---------------------------------------------------------------------------
# I-INF-005: DML identification check must be offset invariant
# ---------------------------------------------------------------------------


def _dml_data(offset: float):
    rng = np.random.default_rng(0)
    n = 500
    z = rng.standard_normal(n)
    v = 0.1 * rng.standard_normal(n)
    x = offset + 0.05 * z + v
    y = 2.0 * (x - offset) + np.sin(z) + 0.1 * rng.standard_normal(n)
    return torch.tensor(y), torch.tensor(x), torch.tensor(z)


def test_I_INF_005_offset_invariance() -> None:
    ref = orthogonal_partially_linear(*_dml_data(0.0), folds=5, seed=0)
    out = orthogonal_partially_linear(*_dml_data(1000.0), folds=5, seed=0)
    assert abs(out.theta - ref.theta) < 1e-3 * ref.sigma


def test_I_INF_005_constant_treatment_still_rejected() -> None:
    y, _, z = _dml_data(0.0)
    with pytest.raises(ValueError, match="not identified"):
        orthogonal_partially_linear(y, torch.full_like(y, 1000.0), z, folds=5, seed=0)


# ---------------------------------------------------------------------------
# I-CAU-001..003: doubly-robust ATE
# ---------------------------------------------------------------------------


def _dr_draw(rng: np.random.Generator, n: int):
    x = rng.standard_normal((n, 2))
    e = 1.0 / (1.0 + np.exp(-0.5 * x[:, 0]))
    t = (rng.random(n) < e).astype(float)
    y = x[:, 0] + 0.5 * x[:, 1] + 1.0 * t + rng.standard_normal(n)
    return torch.tensor(x), torch.tensor(t), torch.tensor(y)


_DR_KW = dict(
    outcome_model=LinearRegression, propensity_model=LogisticRegression, trim_threshold=0.0
)


@pytest.mark.parametrize("folds", [2, 5])
def test_I_CAU_001_fold_bootstrap_se_matches_analytic(folds: int) -> None:
    # Both SEs estimate sd(DR score)/sqrt(n); with B=500 the bootstrap SE has ~3%
    # relative noise. The old fold-mean bootstrap gave ~0.56x at K=2.
    x, t, y = _dr_draw(np.random.default_rng(0), 2000)
    an = dr_ate(x, t, y, folds=folds, seed=0, **_DR_KW)
    bs = dr_ate(x, t, y, folds=folds, seed=0, se_method="fold_bootstrap", **_DR_KW)
    assert bs["se"] / an["se"] == pytest.approx(1.0, abs=0.12)


def test_I_CAU_001_fold_bootstrap_ci_coverage() -> None:
    # Nominal 95%, 60 seeded replications: binomial sd ~0.028, 99% one-sided lower
    # bound ~0.885; we require 0.85. The old implementation covered ~0.56-0.65.
    rng = np.random.default_rng(0)
    reps, hits = 60, 0
    for r in range(reps):
        x, t, y = _dr_draw(rng, 300)
        out = dr_ate(x, t, y, folds=2, seed=r, se_method="fold_bootstrap", **_DR_KW)
        hits += out["ci_lower"] <= 1.0 <= out["ci_upper"]
    assert hits / reps >= 0.85, hits / reps


def test_I_CAU_002_dr_ate_shift_equivariance_float64() -> None:
    rng = np.random.default_rng(0)
    n = 1000
    x = rng.standard_normal((n, 2))
    e = 1.0 / (1.0 + np.exp(-0.5 * x[:, 0]))
    t = (rng.random(n) < e).astype(float)
    y = 0.3 * x[:, 0] + 0.05 * t + 0.1 * rng.standard_normal(n)
    xt, tt = torch.tensor(x), torch.tensor(t)
    base = dr_ate(xt, tt, torch.tensor(y), folds=2, seed=0, **_DR_KW)
    shifted = dr_ate(xt, tt, torch.tensor(y + 1.0e7), folds=2, seed=0, **_DR_KW)
    assert base["dr_scores"].dtype == torch.float64
    assert abs(shifted["se"] / base["se"] - 1.0) < 0.01
    assert abs(shifted["estimate"] - base["estimate"]) < 0.05 * base["se"]


def test_I_CAU_003_no_overlap_raises_instead_of_nan() -> None:
    rng = np.random.default_rng(0)
    n = 400
    x = rng.standard_normal((n, 2))
    x[:, 0] += np.sign(x[:, 0])  # separated covariate -> propensity exactly 0/1
    t = (x[:, 0] > 0).astype(float)
    y = x[:, 0] + t + rng.standard_normal(n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="propensity trimming"):
            dr_ate(
                torch.tensor(x),
                torch.tensor(t),
                torch.tensor(y),
                outcome_model=LinearRegression,
                propensity_model=LogisticRegression(C=1e6),
            )


# ---------------------------------------------------------------------------
# I-ENS-001: predict-time log-variance bounds must match GaussianNLLLoss
# ---------------------------------------------------------------------------


def _member(mean: float, log_var: float) -> nn.Module:
    lin = nn.Linear(1, 2)
    with torch.no_grad():
        lin.weight.zero_()
        lin.bias.copy_(torch.tensor([mean, log_var]))
    return lin


def _ensemble(*members: nn.Module) -> HeteroscedasticEnsembleModel:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # identical-members warning
        return HeteroscedasticEnsembleModel(
            member_factory=lambda i, s: members[i], ensemble_size=len(members)
        )


def test_I_ENS_001_bounds_match_gaussian_nll_loss() -> None:
    loss = GaussianNLLLoss()
    y = torch.zeros(4, 1)

    def nll(log_var: float) -> float:
        return float(loss(torch.cat([y, torch.full_like(y, log_var)], 1), y))

    assert math.exp(GAUSSIAN_NLL_MIN_LOGVAR) == pytest.approx(loss.min_variance)
    # the loss is flat beyond each bound and not flat just inside it
    assert nll(GAUSSIAN_NLL_MIN_LOGVAR - 3.0) == pytest.approx(nll(GAUSSIAN_NLL_MIN_LOGVAR))
    assert nll(GAUSSIAN_NLL_MAX_LOGVAR + 3.0) == pytest.approx(nll(GAUSSIAN_NLL_MAX_LOGVAR))
    assert nll(GAUSSIAN_NLL_MIN_LOGVAR + 1.0) != pytest.approx(nll(GAUSSIAN_NLL_MIN_LOGVAR))


@pytest.mark.parametrize("variance", [2500.0, 1e-6])
def test_I_ENS_001_heteroscedastic_ensemble_aleatoric_variance(variance: float) -> None:
    ens = _ensemble(_member(0.0, math.log(variance)), _member(1.0, math.log(variance)))
    out = ens.predict(torch.zeros(3, 1))
    torch.testing.assert_close(out["aleatoric_variance"], torch.full((3, 1), variance))
    torch.testing.assert_close(out["epistemic_variance"], torch.full((3, 1), 0.25))
    cov = ens.predict_full_covariance(torch.zeros(3, 1))
    torch.testing.assert_close(cov["aleatoric_covariance"], torch.full((3, 1, 1), variance))


def test_I_ENS_001_batch_ensemble_regressor_predict_output() -> None:
    class Ident(nn.Module):
        feature_dim = 1

        def forward(self, x):
            return x

    reg = BatchEnsembleRegressor(Ident(), feature_dim=1, output_dim=1, ensemble_size=2)
    layer = reg._model.output_layer
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.copy_(torch.tensor([0.0, math.log(2500.0)]))
    out = reg.predict_output(torch.zeros(3, 1))
    torch.testing.assert_close(out.aleatoric_variance, torch.full((3, 1), 2500.0))
    torch.testing.assert_close(
        reg.predict(torch.zeros(3, 1))["aleatoric_variance"], torch.full((3, 1), 2500.0)
    )


# ---------------------------------------------------------------------------
# I-ENS-002: predict must run members in eval mode and restore modes
# ---------------------------------------------------------------------------


class _DropBNNet(nn.Module):
    def __init__(self, out: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(2, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(0.3), nn.Linear(16, out)
        )

    def forward(self, x):
        return self.body(x)


def _loader() -> DataLoader:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 2, generator=g)
    return DataLoader(TensorDataset(x, x.sum(1, keepdim=True)), batch_size=16)


def _fitted_base(**kw) -> BaseEnsembleModel:
    torch.manual_seed(0)
    ens = BaseEnsembleModel(base_model=_DropBNNet, ensemble_size=2, base_seed=0, **kw)
    ens.fit(_loader(), nn.MSELoss(), epochs=1, verbose=False)
    return ens


def test_I_ENS_002_predict_deterministic_and_restores_modes() -> None:
    ens = _fitted_base()
    modes = [m.training for m in ens.modules()]
    x = torch.randn(32, 2) * 3.0 + 5.0
    torch.testing.assert_close(ens.predict(x)["mean"], ens.predict(x)["mean"])
    torch.testing.assert_close(
        ens.predict_full_covariance(x)["mean"], ens.predict_full_covariance(x)["mean"]
    )
    assert [m.training for m in ens.modules()] == modes


def test_I_ENS_002_predict_does_not_mutate_batchnorm_state() -> None:
    ens = _fitted_base()
    before = [m.body[1].running_mean.clone() for m in ens.models]
    ens.predict(torch.randn(32, 2) * 3.0 + 5.0)
    for m, b in zip(ens.models, before):
        torch.testing.assert_close(m.body[1].running_mean, b)


def test_I_ENS_002_heteroscedastic_predict_deterministic() -> None:
    torch.manual_seed(0)
    ens = HeteroscedasticEnsembleModel(base_model=_DropBNNet, ensemble_size=2, base_seed=0, out=2)
    crit = lambda pred, y: ((pred[:, :1] - y) ** 2).mean()  # noqa: E731
    ens.fit(_loader(), crit, epochs=1, verbose=False)
    x = torch.randn(32, 2)
    torch.testing.assert_close(
        ens.predict(x)["epistemic_variance"], ens.predict(x)["epistemic_variance"]
    )
    assert ens.training  # previous (train) mode restored


# ---------------------------------------------------------------------------
# I-ENS-003: bars_to_density_grid has no mass outside the bin edges
# ---------------------------------------------------------------------------


def _bars():
    probs = torch.tensor([[0.9, 0.1]])
    return probs.log(), torch.tensor([0.0, 1.0, 10.0])


def test_I_ENS_003_density_zero_outside_edges_and_mass_preserved() -> None:
    logits, edges = _bars()
    support, density = bars_to_density_grid(logits, edges, n_support=4001)
    outside = (support < 0.0) | (support > 10.0)
    assert outside.any() and float(density[outside].abs().max()) == 0.0
    s, d = support[0], density[0]
    m = s < 1.0
    assert float(torch.trapezoid(d[m], s[m])) == pytest.approx(0.9, abs=0.01)


def test_I_ENS_003_predictive_batch_with_density_mean() -> None:
    logits, edges = _bars()
    pb = PredictiveBatch(bar_logits=logits, bin_edges=edges).with_density(n_support=4001)
    s, d = torch.as_tensor(pb.support), torch.as_tensor(pb.density)[0]
    assert float(torch.trapezoid(d * s, s)) == pytest.approx(0.9 * 0.5 + 0.1 * 5.5, abs=0.02)


# ---------------------------------------------------------------------------
# I-ENS-004: base_seed must not reseed the global RNG
# ---------------------------------------------------------------------------


def _linear_ens() -> BaseEnsembleModel:
    return BaseEnsembleModel(
        base_model=nn.Linear, ensemble_size=2, base_seed=0, in_features=2, out_features=1
    )


def test_I_ENS_004_constructor_preserves_user_stream() -> None:
    torch.manual_seed(1)
    expected = torch.rand(4)
    torch.manual_seed(1)
    _linear_ens()
    torch.testing.assert_close(torch.rand(4), expected)


def test_I_ENS_004_base_seed_still_reproducible() -> None:
    torch.manual_seed(1)
    a = _linear_ens()
    torch.manual_seed(2)
    b = _linear_ens()
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb)
    assert not torch.equal(a.models[0].weight, a.models[1].weight)


# ---------------------------------------------------------------------------
# I-ENS-005: ensemble docs must not document removed / wrong APIs
# ---------------------------------------------------------------------------


def test_I_ENS_005_docs_do_not_document_missing_fit_kwargs() -> None:
    for page in ("methods/ensemble/methods.md", "api/ensemble.md"):
        text = (DOCS / page).read_text(encoding="utf-8")
        assert not re.search(r"adversarial_(training|epsilon|steps|loss_weight)\s*=", text), page
        assert "torch.stack(preds).mean(dim=0)" not in text, page
    with pytest.raises(TypeError):
        _linear_ens().fit(
            DataLoader(TensorDataset(torch.zeros(2, 2), torch.zeros(2, 1))),
            nn.MSELoss(),
            adversarial_training=True,  # type: ignore[call-arg]
        )


def test_I_ENS_005_documented_forward_returns_stacked_tensor() -> None:
    preds = _linear_ens().forward(torch.randn(5, 2))
    assert isinstance(preds, torch.Tensor) and preds.shape == (2, 5, 1)
    assert preds.mean(dim=0).shape == (5, 1)
