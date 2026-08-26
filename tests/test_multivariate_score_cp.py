"""Unit tests for multivariate scalar-score conformal regions (ICLR MTTA)."""

from __future__ import annotations

import math

import pytest
import torch

from torchregress.losses.conformal import MultivariateScoreConformal


def test_known_gaussian_joint_region() -> None:
    """d=2, known covariance, n=2000 calibration: joint coverage within 1% of 0.90."""
    torch.manual_seed(0)
    d = 2
    cov = torch.tensor([[1.0, 0.6], [0.6, 1.0]])
    n = 2000
    mu_cal = torch.zeros(n, d)
    y_cal = torch.distributions.MultivariateNormal(
        mu_cal, covariance_matrix=cov.expand(n, d, d)
    ).sample()
    mcp = MultivariateScoreConformal(0.10).calibrate(mu_cal, cov.expand(n, d, d), y_cal)

    n_test = 200_000
    y_test = torch.distributions.MultivariateNormal(
        torch.zeros(n_test, d), covariance_matrix=cov.expand(n_test, d, d)
    ).sample()
    covered = mcp.covers(torch.zeros(n_test, d), cov.expand(n_test, d, d), y_test)
    emp = float(covered.float().mean())
    assert abs(emp - 0.90) < 0.01, emp


def test_uniform_weights_reduce_to_split_cp() -> None:
    torch.manual_seed(1)
    n, d = 500, 3
    mu = torch.zeros(n, d)
    y = torch.randn(n, d)
    mcp_w = MultivariateScoreConformal(0.1).calibrate(mu, torch.ones(n, d), y, torch.ones(n))
    mcp_n = MultivariateScoreConformal(0.1).calibrate(mu, torch.ones(n, d), y)
    assert float(mcp_w.threshold_) == float(mcp_n.threshold_)


def test_diagonal_radius_matches_chi_square() -> None:
    """Identity covariance: radius^2 sits at the finite-sample chi2(2) quantile."""
    import scipy.stats as stats

    torch.manual_seed(2)
    n, d = 4000, 2
    y = torch.randn(n, d)
    mcp = MultivariateScoreConformal(0.1).calibrate(
        torch.zeros(n, d), torch.ones(n, d), y, torch.ones(n)
    )
    r = mcp.region_radius()
    k = min(math.ceil((n + 1) * 0.9), n)
    expected = float(stats.chi2.ppf((k - 1) / (n - 1), df=2))
    assert abs(r - expected) < 0.25, (r, expected)


def test_nll_score_is_affine_in_mahalanobis() -> None:
    """NLL = 0.5*(d log 2pi + logdet) + 0.5*maha, hence r_nll = c + r_maha/2."""
    torch.manual_seed(3)
    d = 2
    cov = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    n = 800
    mu = torch.zeros(n, d)
    y = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov.expand(n, d, d)).sample()
    r_maha = MultivariateScoreConformal(0.1).calibrate(mu, cov.expand(n, d, d), y).region_radius()
    r_nll = (
        MultivariateScoreConformal(0.1, score_fn="nll")
        .calibrate(mu, cov.expand(n, d, d), y)
        .region_radius()
    )
    const = 0.5 * (d * math.log(2.0 * math.pi) + float(torch.logdet(cov)))
    assert abs(r_nll - (const + 0.5 * r_maha)) < 1e-4


def test_covariance_shape_flexibility_and_errors() -> None:
    torch.manual_seed(4)
    n, d = 60, 2
    mu = torch.zeros(n, d)
    y = torch.randn(n, d)
    a = torch.randn(n, d, d)
    spd_batch = a @ a.transpose(-1, -2) + 0.5 * torch.eye(d, device=mu.device, dtype=mu.dtype)
    for cov in (
        torch.eye(d, device=mu.device, dtype=mu.dtype),
        torch.ones(n, d),
        spd_batch,
    ):
        MultivariateScoreConformal(0.1).calibrate(mu, cov, y)
    mcp = MultivariateScoreConformal(0.1).calibrate(mu, torch.ones(n, d), y)
    with pytest.raises(RuntimeError):
        MultivariateScoreConformal(0.1).region_radius()
    with pytest.raises(RuntimeError):
        MultivariateScoreConformal(0.1).covers(mu, torch.ones(n, d), y)
    with pytest.raises(ValueError):
        MultivariateScoreConformal(0.1, score_fn="bogus")
    with pytest.raises(ValueError):
        mcp.covers(mu, torch.ones(d + 1, d + 1), y)
    assert bool(mcp.covers(mu, torch.ones(n, d), y).dtype == torch.bool)
