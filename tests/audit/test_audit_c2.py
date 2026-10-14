"""Regression tests for audit batch C2 (post-audit additions) of the 0.3.0 release prep.

Covers ``models`` (``TabularPreprocessor``, ``fit_tabular``), ``estimators``,
``inference.orthogonal.median_heuristic_bandwidth`` and ``test_time.selection``.
Each test failed on the commit that introduced the code (``7167361b``).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.linear_model import LinearRegression

from torchregress.estimators import (
    CalibratedRegressor,
    ConformalRegressor,
    calibrated_deep_ensemble,
)
from torchregress.inference import median_heuristic_bandwidth
from torchregress.losses import GaussianNLLLoss
from torchregress.models import TabularMLP, TabularPreprocessor, fit_tabular
from torchregress.test_time.selection import select_high_confidence

torch.set_num_threads(1)


class _Gauss:
    """Prefit Gaussian base: mean x0, unit variance."""

    def predict(self, X):
        return X[:, 0], np.ones(len(X))


def test_C2_001_median_heuristic_translation_invariant():
    z = torch.randn(200, 1, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    ref = median_heuristic_bandwidth(z)
    assert ref == pytest.approx(0.938, abs=0.01)
    assert median_heuristic_bandwidth(z + 1e9) == pytest.approx(ref, rel=1e-6)


def test_C2_001_median_heuristic_rejects_nonfinite():
    with pytest.raises(ValueError):
        median_heuristic_bandwidth(torch.tensor([0.0, float("nan"), 1.0]))


def test_C2_002_smooth_clip_saturates_for_huge_and_infinite_inputs():
    X = np.random.default_rng(0).normal(size=(50, 1))
    z = TabularPreprocessor().fit(X).transform(np.array([[1e200], [-1e200], [np.inf], [-np.inf]]))
    assert np.isfinite(z).all()
    assert z[0, 0] > 2.9 and z[1, 0] < -2.9
    assert z[2, 0] > 2.9 and z[3, 0] < -2.9


def test_C2_003_variance_head_mapped_back_with_sigma_squared():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 2))
    y = 100.0 + 50.0 * X[:, 0]
    torch.manual_seed(0)
    model = TabularMLP(2, 2, hidden=(4,))
    fit = fit_tabular(model, GaussianNLLLoss(log_variance=False), X, y, epochs=2)
    sd = float(fit.target_std[0])
    with torch.no_grad():
        raw = model(fit.preprocessor.transform(torch.as_tensor(X, dtype=torch.float32)))
    expected_var = raw[:, 1].clamp_min(1e-6) * sd**2
    got = fit.predict(X)[:, 1]
    assert torch.allclose(got.clamp_min(0), expected_var, rtol=1e-3, atol=1e-3 * sd**2)


def _nan_target():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    y = X[:, 0] + rng.normal(size=40)
    y[0] = np.nan
    return X, y


def test_C2_004_conformal_regressor_rejects_nan_calibration_target():
    X, y = _nan_target()
    with pytest.raises(ValueError, match="finite"):
        ConformalRegressor(LinearRegression(), "split").fit(X[10:], y[10:], X[:10], y[:10])


def test_C2_004_conformal_regressor_prefit_rejects_nan_target():
    X, y = _nan_target()
    with pytest.raises(ValueError, match="finite"):
        ConformalRegressor(_Gauss(), prefit=True).fit(X, y)


def test_C2_004_calibrated_regressor_rejects_nan_target():
    X, y = _nan_target()
    with pytest.raises(ValueError, match="finite"):
        CalibratedRegressor(_Gauss()).fit(X, y)


def test_C2_005_nan_score_is_never_most_confident():
    probs = np.full((5, 2), 0.5)
    scores = np.array([0.1, np.nan, 0.9, 0.2, 0.3])
    mask = select_high_confidence(probs, top_fraction=0.2, scores=scores)
    assert mask.tolist() == [False, False, True, False, False]


def test_C2_006_gaussian_layout_kept_without_target_standardisation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    fit = fit_tabular(
        TabularMLP(2, 2, hidden=(4,)),
        GaussianNLLLoss(),
        X,
        X[:, 0],
        epochs=2,
        standardize_target=False,
    )
    assert fit.output_layout == "gaussian"


def test_C2_006_calibrated_deep_ensemble_without_standardisation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 2))
    y = X[:, 0] + 0.1 * rng.normal(size=80)
    m = calibrated_deep_ensemble(
        X, y, n_members=2, epochs=2, standardize_target=False, model_kwargs={"hidden": (4,)}
    )
    mean, std = m.predict_dist(X[:3])
    assert mean.shape == (3,) and np.all(std > 0)


@pytest.mark.parametrize("param", ["precision", "covariance"])
def test_mdn_full_covariance_sampling_matches_component_covariance(param):
    """Full-covariance MDN sampling (needed for multi-target density comparisons)."""
    from torchregress.losses import MixtureDensityLoss

    torch.manual_seed(0)
    mdn = MixtureDensityLoss(
        n_components=1, n_features=2, covariance_type="full", full_parameterization=param
    )
    out = torch.zeros(1, mdn.expected_output_size)
    out[0, 1:3] = torch.tensor([1.0, -2.0])
    out[0, 3:6] = torch.tensor([2.0, 0.8, -1.0])
    _, means, T = mdn._extract_distribution_parameters(out)
    s = mdn.sample(out, n_samples=200_000)
    assert s.shape == (200_000, 1, 2)
    T0 = T[0, 0]
    cov = T0 @ T0.T if param == "covariance" else torch.linalg.inv(T0 @ T0.T)
    torch.testing.assert_close(s[:, 0].mean(0), means[0, 0], atol=0.03, rtol=0)
    torch.testing.assert_close(torch.cov(s[:, 0].T), cov, atol=0.05, rtol=0.03)
    mean, std = mdn.predict_mean_std(out)
    torch.testing.assert_close(std[0], cov.diagonal().sqrt(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("param", ["precision", "covariance"])
def test_mdn_full_log_prob_matches_torch_multivariate_normal(param):
    from torchregress.losses import MixtureDensityLoss

    torch.manual_seed(1)
    D, K = 3, 2
    mdn = MixtureDensityLoss(
        n_components=K, n_features=D, covariance_type="full", full_parameterization=param
    )
    out = torch.randn(5, mdn.expected_output_size, dtype=torch.float64)
    y = torch.randn(5, D, dtype=torch.float64)
    logw, means, T = mdn._extract_distribution_parameters(out)
    if param == "covariance":
        comp = torch.distributions.MultivariateNormal(means, scale_tril=T)
    else:
        comp = torch.distributions.MultivariateNormal(
            means, precision_matrix=T @ T.transpose(-1, -2)
        )
    ref = torch.logsumexp(logw + comp.log_prob(y.unsqueeze(-2)), dim=-1)
    nll = mdn._calculate_nll(y, (logw, means, T))
    torch.testing.assert_close(-nll, ref, atol=1e-6, rtol=1e-6)
