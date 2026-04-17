"""Tests for neighbourhood covariance pseudo-labels."""

from __future__ import annotations

import pytest
import torch

from torchregress.algorithms.covariance_pseudo_labels import (
    NeighborhoodCovariancePseudoLabeler,
    mahalanobis_covariance_pseudo_labels,
)


def test_constant_targets_yield_near_diagonal_pseudo_cov() -> None:
    torch.manual_seed(0)
    n, p, d = 40, 3, 2
    x = torch.randn(n, p)
    y = torch.ones(n, d)
    cov = NeighborhoodCovariancePseudoLabeler(n_neighbors=8, regularization=0.01).fit_predict(x, y)
    evals = torch.linalg.eigvalsh(cov)
    assert evals.min() > 0
    assert float(cov.mean().item()) < 0.15


def test_functional_matches_class() -> None:
    torch.manual_seed(1)
    x = torch.randn(25, 4)
    y = torch.randn(25, 2)
    a = NeighborhoodCovariancePseudoLabeler(n_neighbors=5, metric="euclidean").fit_predict(x, y)
    b = mahalanobis_covariance_pseudo_labels(x, y, n_neighbors=5, metric="euclidean")
    torch.testing.assert_close(a, b)


def test_predict_for_query_runs_and_is_spd() -> None:
    torch.manual_seed(2)
    n, p, d = 30, 2, 2
    x = torch.randn(n, p)
    y = torch.randn(n, d)
    lab = NeighborhoodCovariancePseudoLabeler(n_neighbors=6, metric="mahalanobis", temperature=0.5)
    xq = torch.randn(4, p)
    q = lab.predict_for_query(xq, x_reference=x, y_reference=y)
    assert q.shape == (4, d, d)
    assert torch.all(torch.linalg.eigvalsh(q) > 0)


def test_too_few_rows_raises() -> None:
    x = torch.randn(4, 2)
    y = torch.randn(4, 1)
    lab = NeighborhoodCovariancePseudoLabeler(n_neighbors=8)
    with pytest.raises(ValueError, match="at least"):
        lab.fit_predict(x, y)


def test_outputs_symmetric_spd() -> None:
    torch.manual_seed(3)
    x = torch.randn(35, 5)
    y = torch.randn(35, 3)
    cov = NeighborhoodCovariancePseudoLabeler(n_neighbors=10).fit_predict(x, y)
    err = (cov - cov.transpose(-1, -2)).abs().max().item()
    assert err < 1e-5
    assert torch.all(torch.linalg.eigvalsh(cov) > 0)
