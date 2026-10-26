"""Regression tests for the 0.3.0 audit, batch 4: algorithms + test-time (ALG-*, TT-*).

Each test reproduces one audit finding and pins the fixed behaviour. The Monte Carlo
checks use smaller samples than the original reproductions, state the tolerance they
rely on, and are fully seeded.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch
import torch.nn as nn
from scipy.stats import norm
from torch.utils.data import DataLoader, TensorDataset

from torchregress.algorithms import (
    IVON,
    SIMEX,
    HeteroscedasticLaplaceRegressor,
    NaturalHeteroscedasticHead,
    NeighborhoodCovarianceConfig,
    NeighborhoodCovariancePseudoLabeler,
    RegressionCalibration,
)
from torchregress.algorithms.irls import huber_weights, reweight_fixed_predictions
from torchregress.prediction import PredictiveBatch
from torchregress.test_time import (
    BayesianLinearHead,
    DelayedLabelResidualAdapter,
    DomainClassifierRatioEstimator,
    ShiftFactoredPredictiveTransport,
    ShiftFactoredTransportConfig,
)
from torchregress.test_time.label_shift import (
    GaussianLabelShiftConfig,
    correct_gaussian_predictions_for_label_shift,
    gaussian_moments_from_binned_probabilities,
)
from torchregress.test_time.ot_conformal import WeightedConformalRegressionAdapter

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _lstsq_train(model: nn.Linear, X: torch.Tensor, Y: torch.Tensor) -> nn.Linear:
    """Closed-form least-squares 'training' for a ``nn.Linear`` (deterministic, fast)."""
    d = X.shape[1]
    A = torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], 1)
    sol = torch.linalg.lstsq(A, Y).solution
    with torch.no_grad():
        model.weight.copy_(sol[:d].T)
        model.bias.copy_(sol[d])
    return model


class _ZeroLinear(nn.Module):
    """``nn.Linear(1, n_out)`` with all parameters zero: mean 0 (and log_sigma 0)."""

    def __init__(self, n_out: int = 1) -> None:
        super().__init__()
        self.lin = nn.Linear(1, n_out)
        with torch.no_grad():
            self.lin.weight.zero_()
            self.lin.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class _TupleHead(nn.Module):
    """Returns a ``(mean, log_sigma)`` tuple, both identically zero."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))  # IRLS needs a parameter to find the device

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(x.shape[0], 1, dtype=x.dtype)
        return z, z.clone()


_Y6 = torch.tensor([[0.1], [0.5], [2.0], [4.0], [-3.0], [0.0]])

# ---------------------------------------------------------------------------
# algorithms: IRLS
# ---------------------------------------------------------------------------


def test_ALG_001_irls_weights_not_compounded() -> None:
    """Fixed model => residuals constant => every iteration yields the same weights."""
    x = torch.zeros(6, 1)
    for max_iter in (1, 3, 10):
        _, _, prec = reweight_fixed_predictions(
            _ZeroLinear(),
            x,
            _Y6,
            variance_type="fixed",
            weight_fn="huber",
            delta=1.0,
            max_iter=max_iter,
            tol=0.0,
        )
        # fixed_variance = 1 -> scaled residual = y: [1, 1, 0.5, 0.25, 0.333, 1]
        torch.testing.assert_close(prec, huber_weights(_Y6, 1.0), rtol=1e-4, atol=1e-6)


def test_ALG_001_irls_weights_scale_initial_precision() -> None:
    x = torch.zeros(6, 1)
    init = torch.linspace(1.0, 3.0, 6).unsqueeze(1)
    _, _, prec = reweight_fixed_predictions(
        _ZeroLinear(),
        x,
        _Y6,
        initial_precision=init,
        variance_type="fixed",
        weight_fn="huber",
        delta=1.0,
        max_iter=7,
        tol=0.0,
    )
    torch.testing.assert_close(prec, init * huber_weights(_Y6, 1.0), rtol=1e-4, atol=1e-6)


def test_ALG_002_irls_robust_scale_over_samples() -> None:
    """MAD must be taken over samples; over the single output column it was 0."""
    g = torch.Generator().manual_seed(0)
    n = 201
    y = torch.randn(n, 1, generator=g)
    x = torch.zeros(n, 1)
    _, _, prec = reweight_fixed_predictions(
        _ZeroLinear(),
        x,
        y,
        variance_type="robust",
        weight_fn="huber",
        delta=1.345,
        max_iter=1,
    )
    sigma = 1.4826 * (y - y.median()).abs().median()
    # Most standard-normal residuals are inside delta: mean weight is ~0.9 (was ~0).
    assert prec.mean().item() > 0.8
    torch.testing.assert_close(prec, huber_weights(y / sigma, 1.345), rtol=1e-3, atol=1e-5)


def test_ALG_002_irls_robust_scale_is_per_output_column() -> None:
    g = torch.Generator().manual_seed(1)
    y = torch.randn(301, 2, generator=g) * torch.tensor([1.0, 10.0])
    _, _, prec = reweight_fixed_predictions(
        _TwoOutputZero(),
        torch.zeros(301, 1),
        y,
        variance_type="robust",
        weight_fn="huber",
        delta=1.345,
        max_iter=1,
    )
    # Scale-free: both columns have the same standardized residuals, hence similar weights.
    assert abs(prec[:, 0].mean().item() - prec[:, 1].mean().item()) < 0.05


class _TwoOutputZero(nn.Module):
    """Mean-only model with two outputs (zero)."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], 2, dtype=x.dtype)


def test_ALG_003_irls_concatenated_output_default_config() -> None:
    """Default config (gaussian base loss, predicted variance) with ``[mean, log_sigma]``."""
    x = torch.zeros(6, 1)
    y_pred, hist, prec = reweight_fixed_predictions(_ZeroLinear(2), x, _Y6, max_iter=1)
    assert y_pred.shape == (6, 2)
    assert len(hist) == 1 and math.isfinite(hist[0])
    assert prec.shape == _Y6.shape
    torch.testing.assert_close(prec, huber_weights(_Y6, 1.0))


def test_ALG_003_irls_tuple_output_default_config() -> None:
    x = torch.zeros(6, 1)
    _, hist, prec = reweight_fixed_predictions(_TupleHead(), x, _Y6, max_iter=1)
    assert math.isfinite(hist[0])
    torch.testing.assert_close(prec, huber_weights(_Y6, 1.0))


# ---------------------------------------------------------------------------
# algorithms: SIMEX
# ---------------------------------------------------------------------------


def test_ALG_004_simex_float64_scalar_sigma() -> None:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(300, 1, dtype=torch.float64, generator=g)
    w = x + 0.5 * torch.randn(300, 1, dtype=torch.float64, generator=g)
    y = 2 * x
    for sigma in (0.5, torch.tensor(0.5)):
        s = SIMEX(lambda: nn.Linear(1, 1).double(), _lstsq_train, sigma, n_simulations=2)
        out = s.fit(w, y).predict(w[:5])
        assert out.dtype == torch.float64
        assert s.sigma_u is not None and s.sigma_u.dtype == torch.float64


def _simex_slopes(scale: float) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    n = 4000
    x = torch.randn(n, 2, dtype=torch.float64, generator=g)
    w = x + 0.5 * torch.randn(n, 2, dtype=torch.float64, generator=g)
    y = x @ torch.tensor([[2.0], [1.0]], dtype=torch.float64)
    torch.manual_seed(1)
    s = SIMEX(
        lambda: nn.Linear(2, 1).double(),
        _lstsq_train,
        torch.tensor([0.5 * scale] * 2, dtype=torch.float64),
        n_simulations=4,
    )
    s.fit(w * scale, y)
    return torch.tensor(
        [
            sum(m.weight[0, 0].item() for m in models) / len(models) * scale
            for models in s.models_by_lambda
        ]
    )


def test_ALG_005_simex_scale_equivariant() -> None:
    """Rescaling W and sigma_u by c leaves the attenuated slopes (times c) unchanged."""
    a = _simex_slopes(1.0)
    b = _simex_slopes(1e-3)  # sigma_u = 5e-4: the old absolute 1e-6 jitter was ~4x too noisy
    # same seed + equivariant noise => identical draws; tolerance only absorbs round-off
    torch.testing.assert_close(b, a, rtol=1e-3, atol=0.0)
    assert a[0] > a[-1]  # slopes attenuate with lambda


def test_ALG_005_simex_singular_psd_ok_and_indefinite_rejected() -> None:
    g = torch.Generator().manual_seed(0)
    w = torch.randn(200, 2, generator=g)
    y = w.sum(1, keepdim=True)
    singular = torch.tensor([[1.0, 1.0], [1.0, 1.0]])  # PSD, rank 1
    SIMEX(lambda: nn.Linear(2, 1), _lstsq_train, singular, n_simulations=1).fit(w, y)
    indefinite = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError, match="not PSD"):
        SIMEX(lambda: nn.Linear(2, 1), _lstsq_train, indefinite, n_simulations=1).fit(w, y)


def test_ALG_006_simex_predict_averages_independent_remeasurements() -> None:
    """Per-point MC sd must shrink like 1/sqrt(n_simulations) (1.9 -> ~0.42 for B=20)."""
    torch.manual_seed(0)
    n = 4000
    x = torch.randn(n, 1)
    w = x + 0.5 * torch.randn(n, 1)
    y = 2 * x + 0.1 * torch.randn(n, 1)
    s = SIMEX(lambda: nn.Linear(1, 1), _lstsq_train, 0.5, n_simulations=20).fit(w, y)
    p1 = s.predict(torch.ones(4000, 1))
    p0 = s.predict(torch.zeros(4000, 1))
    # quadratic extrapolant of 2 / (1 + 0.25 (1 + lambda)) at lambda = -1 is ~1.94
    assert abs((p1.mean() - p0.mean()).item() - 1.941) < 0.15
    assert p1.std().item() < 0.6, p1.std().item()


# ---------------------------------------------------------------------------
# algorithms: regression calibration
# ---------------------------------------------------------------------------


def test_ALG_007_rc_posterior_uses_sigma_override() -> None:
    g = torch.Generator().manual_seed(0)
    n = 2000
    x = torch.randn(n, 1, dtype=torch.float64, generator=g)
    w = x + 0.5 * torch.randn(n, 1, dtype=torch.float64, generator=g)
    rc = RegressionCalibration(0.5).fit(w)
    assert rc.signal_covariance is not None and rc.mu_w is not None
    sx = rc.signal_covariance
    su = torch.tensor([[0.01**2]], dtype=torch.float64)
    expected_cov = sx - sx @ torch.linalg.inv(sx + su) @ sx  # ~1e-4 (was ~0.2)
    expected_mean = rc.mu_w + (w[:3] - rc.mu_w) @ (sx @ torch.linalg.inv(sx + su)).T
    for override in (0.01, torch.tensor([0.01], dtype=torch.float64), torch.sqrt(su)):
        mean, cov = rc.posterior(w[:3], sigma_u=override)
        torch.testing.assert_close(cov, expected_cov, rtol=1e-3, atol=1e-8)
        torch.testing.assert_close(mean, expected_mean, rtol=1e-6, atol=1e-8)
    # the stored specification is untouched
    assert rc.sigma_u_input == 0.5
    default_mean, _ = rc.posterior(w[:3])
    assert not torch.allclose(default_mean, expected_mean)


def test_ALG_007_rc_posterior_matrix_and_vector_override() -> None:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1000, 2, dtype=torch.float64, generator=g)
    w = x + 0.5 * torch.randn(1000, 2, dtype=torch.float64, generator=g)
    rc = RegressionCalibration(0.5).fit(w)
    assert rc.signal_covariance is not None
    std = torch.tensor([0.01, 0.02], dtype=torch.float64)
    cov = torch.diag(std**2).to(dtype=torch.float64)
    m_vec, c_vec = rc.posterior(w[:4], sigma_u=std)  # (D,) = per-feature std
    m_mat, c_mat = rc.posterior(w[:4], sigma_u=cov)  # (D, D) = covariance
    sx = rc.signal_covariance
    torch.testing.assert_close(
        c_vec, sx - sx @ torch.linalg.inv(sx + cov) @ sx, rtol=1e-3, atol=1e-8
    )
    torch.testing.assert_close(c_mat, c_vec)
    torch.testing.assert_close(m_mat, m_vec)


def test_ALG_008_rc_reliability_scale_equivariant() -> None:
    g = torch.Generator().manual_seed(0)
    n = 5000
    x = torch.randn(n, 2, dtype=torch.float64, generator=g)
    w = x + 0.5 * torch.randn(n, 2, dtype=torch.float64, generator=g)
    ref = RegressionCalibration(0.5).fit(w).reliability_matrix
    assert ref is not None
    c = 1e-4
    scaled = RegressionCalibration(0.5 * c).fit(w * c).reliability_matrix
    assert scaled is not None
    # true reliability sigma_x^2 / (sigma_x^2 + sigma_u^2) = 1 / 1.25 = 0.8 in any units
    torch.testing.assert_close(
        torch.diagonal(ref), torch.full((2,), 0.8, dtype=torch.float64), atol=0.03, rtol=0
    )
    torch.testing.assert_close(scaled, ref, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# algorithms: heteroscedastic Laplace
# ---------------------------------------------------------------------------


def _laplace_loader() -> tuple[torch.Tensor, DataLoader]:
    g = torch.Generator().manual_seed(0)
    X = torch.randn(64, 3, generator=g)
    y = X[:, :1] + 0.1 * torch.randn(64, 1, generator=g)
    return X, DataLoader(TensorDataset(X, y), batch_size=16)


def test_ALG_009_laplace_plain_linear_head_trains() -> None:
    X, dl = _laplace_loader()
    torch.manual_seed(0)
    reg = HeteroscedasticLaplaceRegressor(nn.Identity(), nn.Linear(3, 2)).fit(dl, epochs=1)
    pred = reg.predict_distribution(X[:4], n_samples=8)
    assert pred.mean is not None and pred.mean.shape == (4, 1)


def test_ALG_010_laplace_single_sample_finite() -> None:
    X, dl = _laplace_loader()
    torch.manual_seed(0)
    reg = HeteroscedasticLaplaceRegressor(nn.Identity(), NaturalHeteroscedasticHead(3, 1))
    reg.fit(dl, epochs=1)
    pred = reg.predict_distribution(X[:4], n_samples=1)
    assert pred.std is not None and torch.isfinite(pred.std).all()
    assert pred.extra is not None
    assert (pred.extra["epistemic_variance"] == 0).all()
    # two or more draws keep a genuine (non-negative, finite) epistemic variance
    pred2 = reg.predict_distribution(X[:4], n_samples=2)
    assert pred2.extra is not None and torch.isfinite(pred2.extra["epistemic_variance"]).all()


# ---------------------------------------------------------------------------
# algorithms: IVON
# ---------------------------------------------------------------------------


def _ivon_run(model: nn.Module, opt: IVON, steps: int, seed: int) -> None:
    torch.manual_seed(seed)
    x = torch.randn(32, 3)
    y = x.sum(1, keepdim=True)
    for _ in range(steps):
        with opt.sampled_params(train=True):
            opt.zero_grad()
            ((model(x) - y) ** 2).mean().backward()
        opt.step()


def test_ALG_011_ivon_checkpoint_resume_matches_uninterrupted() -> None:
    torch.manual_seed(0)
    m = nn.Linear(3, 1)
    o = IVON(m.parameters(), lr=0.1, ess=32)
    _ivon_run(m, o, 3, 1)
    sd_m, sd_o = copy.deepcopy(m.state_dict()), copy.deepcopy(o.state_dict())
    assert sd_o["current_step"] == 3
    _ivon_run(m, o, 3, 2)

    m2 = nn.Linear(3, 1)
    m2.load_state_dict(sd_m)
    o2 = IVON(m2.parameters(), lr=0.1, ess=32)
    o2.load_state_dict(sd_o)
    assert o2.current_step == 3
    _ivon_run(m2, o2, 3, 2)
    torch.testing.assert_close(m2.weight, m.weight)
    torch.testing.assert_close(m2.bias, m.bias)


def test_ALG_011_ivon_loads_legacy_state_dict_without_current_step() -> None:
    m = nn.Linear(3, 1)
    o = IVON(m.parameters(), lr=0.1, ess=32)
    _ivon_run(m, o, 2, 0)
    legacy = copy.deepcopy(o.state_dict())
    del legacy["current_step"]
    o2 = IVON(nn.Linear(3, 1).parameters(), lr=0.1, ess=32)
    o2.load_state_dict(legacy)
    assert o2.current_step == 0


# ---------------------------------------------------------------------------
# algorithms: covariance pseudo-labels
# ---------------------------------------------------------------------------


def test_ALG_012_pseudo_cov_keeps_float64() -> None:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(40, 2, dtype=torch.float64, generator=g)
    y = 1e4 + torch.randn(40, 2, dtype=torch.float64, generator=g)  # large offset
    lab = NeighborhoodCovariancePseudoLabeler(
        NeighborhoodCovarianceConfig(n_neighbors=5, metric="euclidean", regularization=1e-8)
    )
    cov = lab.fit_predict(x, y)
    assert cov.dtype == torch.float64
    xn, yn = x.numpy(), y.numpy()
    d = ((xn[0] - xn) ** 2).sum(1)
    d[0] = np.inf
    idx = np.argsort(d)[:5]
    w = np.exp(-(d[idx] - d[idx].min()))
    w /= w.sum()
    mu = (w[:, None] * yn[idx]).sum(0)
    ref = ((yn[idx] - mu).T * w) @ (yn[idx] - mu) + 1e-8 * np.eye(2)
    np.testing.assert_allclose(cov[0].numpy(), ref, rtol=1e-8, atol=1e-10)
    # query path keeps float64 as well; float32 inputs stay float32
    cov_q = lab.predict_for_query(x[:3], x_reference=x, y_reference=y)
    assert cov_q.dtype == torch.float64
    assert lab.fit_predict(x.float(), y.float()).dtype == torch.float32


# ---------------------------------------------------------------------------
# test-time: label shift
# ---------------------------------------------------------------------------


def test_TT_001_binned_moments_keep_float64() -> None:
    edges = 3.0e5 + np.linspace(0.0, 1.0, 5)
    probs = np.array([[0.1, 0.2, 0.3, 0.4]])
    mean, std = gaussian_moments_from_binned_probabilities(probs, edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    assert mean.dtype == np.float64 and std.dtype == np.float64
    np.testing.assert_allclose(mean, probs @ centers, rtol=0, atol=1e-9)


def test_TT_001_correct_gaussian_predictions_keep_input_dtype() -> None:
    rng = np.random.default_rng(0)
    targets = 3.0e5 + rng.normal(size=200)
    mean = 3.0e5 + rng.normal(size=60)
    std = np.full(60, 0.5)
    cfg = GaussianLabelShiftConfig(n_bins=8)
    m64, s64, _ = correct_gaussian_predictions_for_label_shift(
        mean=mean, std=std, source_targets=targets, config=cfg
    )
    assert m64.dtype == np.float64 and s64.dtype == np.float64
    m32, s32, _ = correct_gaussian_predictions_for_label_shift(
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        source_targets=targets,
        config=cfg,
    )
    assert m32.dtype == np.float32 and s32.dtype == np.float32


# ---------------------------------------------------------------------------
# test-time: weighted conformal, density ratios
# ---------------------------------------------------------------------------


class _OracleRatio:
    """Domain 'classifier' returning the exact posterior for N(0,1) -> N(1.5,1)."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_OracleRatio":
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        r = norm.pdf(X[:, 0], 1.5, 1.0) / norm.pdf(X[:, 0], 0.0, 1.0)
        p = r / (1.0 + r)
        return np.stack([1 - p, p], axis=1)


class _ConstRatio:
    def fit(self, X: np.ndarray, y: np.ndarray) -> "_ConstRatio":
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.full((len(X), 2), 0.5)


def test_TT_002_weighted_conformal_interval_is_infinite_when_test_mass_needed() -> None:
    # uniform weights, n=10, alpha=0.05: ceil(0.95 * 11) = 11 > 10 -> unbounded interval
    a = WeightedConformalRegressionAdapter(alpha=0.05, classifier=_ConstRatio())
    Xc = np.zeros((10, 1))
    a.calibrate(
        torch.zeros(10, dtype=torch.float64),
        torch.arange(1.0, 11.0, dtype=torch.float64),
        Xc,
        Xc,
    )
    mean = torch.zeros(3, dtype=torch.float64)
    lo, hi = a.predict_interval(mean, np.zeros((3, 1)))
    assert torch.isneginf(lo).all() and torch.isposinf(hi).all()
    # (m, 1)-shaped predictions and a zero std do not produce NaN (inf * 0)
    pb = PredictiveBatch(
        point=mean.unsqueeze(-1),
        mean=mean.unsqueeze(-1),
        std=torch.zeros(3, 1, dtype=torch.float64),
    )
    lo2, hi2 = a.predict_interval(pb, np.zeros((3, 1)))
    assert torch.isneginf(lo2).all() and torch.isposinf(hi2).all()


def test_TT_002_weighted_conformal_finite_when_enough_calibration_mass() -> None:
    # n=19, alpha=0.1: k = ceil(0.9 * 20) = 18 <= 19 -> k-th smallest score (here 18)
    a = WeightedConformalRegressionAdapter(alpha=0.1, classifier=_ConstRatio())
    Xc = np.zeros((19, 1))
    a.calibrate(
        torch.zeros(19, dtype=torch.float64),
        torch.arange(1.0, 20.0, dtype=torch.float64),
        Xc,
        Xc,
    )
    lo, hi = a.predict_interval(torch.zeros(2, dtype=torch.float64), np.zeros((2, 1)))
    torch.testing.assert_close(hi, torch.full((2,), 18.0, dtype=torch.float64))
    torch.testing.assert_close(lo, torch.full((2,), -18.0, dtype=torch.float64))


def test_TT_002_weighted_conformal_coverage_under_covariate_shift() -> None:
    """Oracle density ratios: coverage must be >= 1 - alpha = 0.90 (was ~0.84).

    100 trials x 200 test points; SE of the mean coverage is ~0.01, tolerance 0.88.
    """
    rng = np.random.default_rng(0)
    cov = []
    for _ in range(100):
        n = 20
        Xc = rng.normal(0, 1, (n, 1))
        yc = Xc[:, 0] + rng.normal(0, 1, n) * (0.2 + np.abs(Xc[:, 0]))
        Xt = rng.normal(1.5, 1, (200, 1))
        yt = Xt[:, 0] + rng.normal(0, 1, 200) * (0.2 + np.abs(Xt[:, 0]))
        a = WeightedConformalRegressionAdapter(alpha=0.1, classifier=_OracleRatio())
        a.calibrate(torch.tensor(Xc[:, 0]), torch.tensor(yc), Xc, Xt)
        lo, hi = a.predict_interval(torch.tensor(Xt[:, 0]), Xt)
        cov.append(np.mean((yt >= lo.numpy()) & (yt <= hi.numpy())))
    assert float(np.mean(cov)) >= 0.88


def test_TT_003_domain_ratio_is_normalised_for_pool_sizes() -> None:
    """No shift, n_s=2000, n_t=200: E_source[w] must be ~1 (was 0.10)."""
    torch.manual_seed(0)
    xs = torch.randn(2000, 2)
    xt = torch.randn(200, 2)
    est = DomainClassifierRatioEstimator(hidden=(16,), epochs=100, lr=1e-2).fit(xs, xt)
    w = est.weights_for(xs)
    assert 0.7 < w.mean().item() < 1.4, w.mean().item()


# ---------------------------------------------------------------------------
# test-time: COSA
# ---------------------------------------------------------------------------


class _UnitBase:
    """Calibrated base model: mean 0, std 1."""

    def predict_distribution(self, X: torch.Tensor) -> PredictiveBatch:
        n = X.shape[0]
        z = torch.zeros(n, 1, dtype=torch.float64)
        return PredictiveBatch(point=z, mean=z, std=torch.ones(n, 1, dtype=torch.float64))


def test_TT_004_cosa_first_single_label_does_not_collapse_std() -> None:
    a = DelayedLabelResidualAdapter(_UnitBase())
    a.partial_fit(torch.zeros(1, 1), torch.tensor([[2.0]], dtype=torch.float64))
    assert a.variance_inflation_ is not None
    # z^2 = (2 - 0)^2 / 1^2: the first batch is measured against the uncorrected base
    # (previous correction 0), not centred on itself (which gave exactly 0 -> clamp 1e-5)
    torch.testing.assert_close(
        a.variance_inflation_, torch.tensor([4.0], dtype=torch.float64), rtol=0, atol=1e-12
    )
    assert a.residual_mean_ is not None
    torch.testing.assert_close(a.residual_mean_, torch.tensor([2.0], dtype=torch.float64))


def test_TT_004_cosa_calibrated_base_keeps_unit_inflation_on_average() -> None:
    g = torch.Generator().manual_seed(0)
    infl = []
    for _ in range(1000):
        a = DelayedLabelResidualAdapter(_UnitBase())
        y = torch.randn(1, 1, dtype=torch.float64, generator=g)
        a.partial_fit(torch.zeros(1, 1), y)
        assert a.variance_inflation_ is not None
        infl.append(a.variance_inflation_.item())
    # chi2_1 mean 1, SE ~ sqrt(2 / 1000) = 0.045; tolerance 0.15
    assert abs(sum(infl) / len(infl) - 1.0) < 0.15


# ---------------------------------------------------------------------------
# test-time: transport
# ---------------------------------------------------------------------------


def _transport_q_hat(n: int, alpha: float) -> tuple[float, float, PredictiveBatch]:
    cfg = ShiftFactoredTransportConfig(
        alpha=alpha, enable_alignment=False, enable_uncertainty_inflation=False
    )
    y = torch.arange(1.0, n + 1.0, dtype=torch.float64)
    tr = ShiftFactoredPredictiveTransport(cfg).fit_source(PredictiveBatch(point=y), y)
    zeros = torch.zeros(n, dtype=torch.float64)
    cal = PredictiveBatch(point=zeros, mean=zeros, std=torch.full((n,), 1e-3, dtype=torch.float64))
    tr.calibrate_target(cal, y, method="interval")
    one = torch.zeros(1, dtype=torch.float64)
    out = tr.apply_conformal(
        PredictiveBatch(point=one, mean=one, std=torch.full((1,), 1e-3, dtype=torch.float64))
    )
    assert out.extra is not None
    upper_native = 1e-3 * norm.ppf(0.95)  # native 90% interval half-width of N(0, 1e-3^2)
    return out.extra["conformal_q_hat"], upper_native, out


def test_TT_005_transport_conformal_quantile_is_kth_order_statistic() -> None:
    n, alpha = 19, 0.1
    k = math.ceil((n + 1) * (1 - alpha))  # 18
    q, u, _ = _transport_q_hat(n, alpha)
    assert abs(q - (k - u)) < 1e-9, (q, k - u)  # scores are y - u, y = 1..19


def test_TT_005_transport_conformal_quantile_inf_when_k_exceeds_n() -> None:
    q, _, out = _transport_q_hat(5, 0.1)  # k = ceil(5.4) = 6 > 5
    assert math.isinf(q)
    assert out.extra is not None
    assert out.extra["interval_lower"] == [-math.inf]
    assert out.extra["interval_upper"] == [math.inf]


def test_TT_007_adapt_keeps_float64() -> None:
    g = torch.Generator().manual_seed(0)
    n = 200
    y = 3.0e5 + torch.randn(n, dtype=torch.float64, generator=g)
    mean = y + 0.3 * torch.randn(n, dtype=torch.float64, generator=g)
    std = torch.full((n,), 0.3, dtype=torch.float64)
    cfg = ShiftFactoredTransportConfig(enable_alignment=False, enable_uncertainty_inflation=False)
    tr = ShiftFactoredPredictiveTransport(cfg).fit_source(PredictiveBatch(point=mean), y)
    out = tr.adapt_unlabeled_target(
        target_predictions=PredictiveBatch(point=mean, mean=mean, std=std)
    )
    assert out.mean is not None and out.std is not None
    assert out.mean.dtype == torch.float64 and out.std.dtype == torch.float64
    # float32 predictions keep float32
    mean32, std32 = mean.float(), std.float()
    out32 = tr.adapt_unlabeled_target(
        target_predictions=PredictiveBatch(point=mean32, mean=mean32, std=std32)
    )
    assert out32.mean is not None and out32.mean.dtype == torch.float32


def test_TT_007_adapt_density_family_keeps_float64() -> None:
    """Non-Gaussian path (density input) goes through ``_probability_moments`` / quantiles."""
    g = torch.Generator().manual_seed(1)
    n = 40
    y = torch.randn(n, dtype=torch.float64, generator=g)
    cfg = ShiftFactoredTransportConfig(enable_alignment=False, enable_uncertainty_inflation=False)
    tr = ShiftFactoredPredictiveTransport(cfg).fit_source(PredictiveBatch(point=y), y)
    support = torch.linspace(-4, 4, 81, dtype=torch.float64)
    density = torch.exp(-0.5 * (support[None, :] - y[:, None]) ** 2) / math.sqrt(2 * math.pi)
    out = tr.adapt_unlabeled_target(
        target_predictions=PredictiveBatch(
            point=y, mean=y, support=support, density=density.to(torch.float64)
        )
    )
    assert out.mean is not None and out.std is not None
    assert out.mean.dtype == torch.float64 and out.std.dtype == torch.float64
    assert out.density is not None and out.density.dtype == torch.float64


# ---------------------------------------------------------------------------
# test-time: Bayesian linear head
# ---------------------------------------------------------------------------


def test_TT_006_blr_rbf_fit_reproducible_given_generator() -> None:
    """The median-heuristic subsample (n > 1000) must use ``generator``, not the global RNG."""

    def fit(global_seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        g0 = torch.Generator().manual_seed(123)
        X = torch.randn(1500, 2, generator=g0)
        y = X.sum(1)
        torch.manual_seed(global_seed)
        h = BayesianLinearHead(2, rbf_centers=10).fit(
            X, y, generator=torch.Generator().manual_seed(1)
        )
        return h._rbf_gamma.clone(), h.predict(X[:5])["mean"]

    g_a, m_a = fit(0)
    g_b, m_b = fit(1)
    torch.testing.assert_close(g_a, g_b)
    torch.testing.assert_close(m_a, m_b)
    # and the global RNG state is not consumed by the fit
    torch.manual_seed(5)
    before = torch.get_rng_state()
    g0 = torch.Generator().manual_seed(123)
    X = torch.randn(1500, 2, generator=g0)
    torch.manual_seed(5)
    BayesianLinearHead(2, rbf_centers=10).fit(
        X, X.sum(1), generator=torch.Generator().manual_seed(1)
    )
    assert torch.equal(torch.get_rng_state(), before)
