"""Tests for conjugate Bayesian linear heads (batch and recursive)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchregress.test_time import BayesianLinearHead, RecursiveBayesianHead


def test_batch_matches_recursive_disjoint_batches():
    torch.manual_seed(0)
    n, d = 80, 4
    x = torch.randn(n, d)
    w_true = torch.tensor([1.5, -0.5, 0.25, 1.0])
    y = (x @ w_true).unsqueeze(-1) + 0.1 * torch.randn(n, 1)

    cfg = dict(in_features=d, fit_intercept=False, prior_precision=0.1, noise_variance=0.25**2)
    full = BayesianLinearHead(**cfg).fit(x, y)
    rec = RecursiveBayesianHead(**cfg, forgetting_factor=1.0)
    rec.partial_fit(x[:40], y[:40])
    rec.partial_fit(x[40:], y[40:])

    torch.testing.assert_close(full.posterior_mean, rec.posterior_mean, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        full.posterior_precision, rec.posterior_precision, rtol=1e-4, atol=1e-5
    )
    p_full = full.predict(x[:5], return_std=True)
    p_rec = rec.predict(x[:5], return_std=True)
    torch.testing.assert_close(p_full["mean"], p_rec["mean"], rtol=1e-4, atol=1e-5)


def test_one_d_slope_recovery():
    torch.manual_seed(1)
    n = 400
    x = torch.randn(n, 1)
    slope = 2.0
    y = slope * x + 0.3 * torch.randn(n, 1)
    head = BayesianLinearHead(
        in_features=1,
        fit_intercept=False,
        prior_precision=1e-2,
        noise_variance=0.3**2,
    ).fit(x, y)
    est = head.posterior_mean[0, 0].item()
    assert abs(est - slope) < 0.15


def test_sample_weights_mean_matches_posterior_mean():
    torch.manual_seed(2)
    n, d = 100, 3
    x = torch.randn(n, d)
    y = (x @ torch.tensor([0.5, -1.0, 0.2])).unsqueeze(-1) + 0.2 * torch.randn(n, 1)
    head = BayesianLinearHead(
        in_features=d,
        fit_intercept=True,
        prior_precision=0.5,
        noise_variance=0.2**2,
    ).fit(x, y)
    draws = head.sample_weights(20_000, generator=torch.Generator().manual_seed(3))
    mc_mean = draws.mean(dim=0)
    torch.testing.assert_close(mc_mean, head.posterior_mean, rtol=0.02, atol=0.02)


def test_predictive_batch_extra_fields():
    torch.manual_seed(4)
    n, d = 30, 2
    x = torch.randn(n, d)
    y = torch.randn(n, 1)
    head = BayesianLinearHead(in_features=d, fit_intercept=True, noise_variance=1.0).fit(x, y)
    pb = head.predictive_batch(x[:5], include_noise=True)
    assert pb.mean is not None and pb.std is not None and pb.point is not None
    assert pb.extra is not None
    for k in ("epistemic_variance", "aleatoric_variance", "posterior_trace", "n_observations_seen"):
        assert k in pb.extra
    assert pb.extra["n_observations_seen"].shape == pb.mean.shape


def test_weight_equivalent_to_replication():
    torch.manual_seed(5)
    x1 = torch.randn(5, 2)
    y1 = torch.randn(5, 1)
    x = torch.cat([x1, x1], dim=0)
    y = torch.cat([y1, y1], dim=0)
    w = torch.cat([torch.ones(5), 2.0 * torch.ones(5)])
    h_w = BayesianLinearHead(in_features=2, fit_intercept=False, noise_variance=1.0).fit(
        x, y, sample_weight=w
    )
    h_rep = BayesianLinearHead(in_features=2, fit_intercept=False, noise_variance=1.0).fit(
        torch.cat([x1, x1, x1], dim=0),
        torch.cat([y1, y1, y1], dim=0),
    )
    torch.testing.assert_close(h_w.posterior_mean, h_rep.posterior_mean, rtol=1e-4, atol=1e-5)


def test_multi_output_independent_rows():
    torch.manual_seed(6)
    n, d = 50, 2
    x = torch.randn(n, d)
    w1 = torch.tensor([1.0, -0.5])
    w2 = torch.tensor([0.0, 2.0])
    y = torch.stack([x @ w1, x @ w2], dim=1) + 0.05 * torch.randn(n, 2)
    head = BayesianLinearHead(
        in_features=d,
        out_features=2,
        fit_intercept=False,
        prior_precision=1.0,
        noise_variance=0.1**2,
    ).fit(x, y)
    h1 = BayesianLinearHead(
        in_features=d,
        out_features=1,
        fit_intercept=False,
        prior_precision=1.0,
        noise_variance=0.1**2,
    ).fit(x, y[:, :1])
    h2 = BayesianLinearHead(
        in_features=d,
        out_features=1,
        fit_intercept=False,
        prior_precision=1.0,
        noise_variance=0.1**2,
    ).fit(x, y[:, 1:])
    torch.testing.assert_close(head.posterior_mean[0], h1.posterior_mean[0], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(head.posterior_mean[1], h2.posterior_mean[0], rtol=1e-4, atol=1e-4)
    pr = head.predict(x[:3], return_std=True)
    assert pr["mean"].shape == (3, 2)


def test_negative_sample_weight_raises():
    x = torch.randn(4, 2)
    y = torch.randn(4, 1)
    head = BayesianLinearHead(in_features=2, fit_intercept=False)
    with pytest.raises(ValueError, match="non-negative"):
        head.fit(x, y, sample_weight=torch.tensor([1.0, 1.0, -0.1, 1.0]))


def test_predict_before_fit_raises():
    head = BayesianLinearHead(in_features=2, fit_intercept=False)
    with pytest.raises(RuntimeError, match="fit"):
        head.predict(torch.randn(2, 2))


def test_numpy_inputs():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((20, 2)).astype(np.float32)
    y = (x @ np.array([0.7, -0.3], dtype=np.float32)).reshape(-1, 1)
    head = BayesianLinearHead(in_features=2, fit_intercept=False, noise_variance=0.5)
    head.fit(x, y)
    out = head.predict(x[:2])
    assert out["mean"].shape == (2, 1)


def test_forgetting_partial_fit_runs():
    x = torch.randn(10, 2)
    y = torch.randn(10, 1)
    rec = RecursiveBayesianHead(in_features=2, forgetting_factor=0.9)
    rec.partial_fit(x[:5], y[:5])
    rec.partial_fit(x[5:], y[5:])
    assert rec.is_fitted
    assert rec.predict(x).keys() >= {"mean"}
