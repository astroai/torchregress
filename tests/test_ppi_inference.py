from __future__ import annotations

import torch

from torchregress.inference import ppi_diagnostics, ppi_mean_ci, ppi_ols_ci, ppi_quantile_ci


def _synthetic(seed: int = 123) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    n_l, n_u = 120, 600
    x_l = torch.randn(n_l, 3)
    x_u = torch.randn(n_u, 3)
    beta = torch.tensor([0.7, -0.4, 0.2])
    y_l = x_l @ beta + 0.2 * torch.randn(n_l)
    pred_l = x_l @ beta + 0.25 * torch.randn(n_l)
    pred_u = x_u @ beta + 0.25 * torch.randn(n_u)
    return x_l, x_u, y_l, pred_l, pred_u


def test_ppi_mean_ci_returns_expected_fields() -> None:
    _, _, y_l, pred_l, pred_u = _synthetic()
    out = ppi_mean_ci(y_l, pred_l, pred_u, alpha=0.1, n_boot=200, seed=1)
    assert out["method"] == "ppi_mean_ci"
    assert out["ci_lower"] <= out["estimate"] <= out["ci_upper"]
    assert out["n_labeled"] == y_l.numel()
    assert out["n_unlabeled"] == pred_u.numel()


def test_ppi_quantile_ci_returns_expected_fields() -> None:
    _, _, y_l, pred_l, pred_u = _synthetic()
    out = ppi_quantile_ci(y_l, pred_l, pred_u, q=0.9, alpha=0.1, n_boot=200, seed=1)
    assert out["method"] == "ppi_quantile_ci"
    assert out["ci_lower"] <= out["estimate"] <= out["ci_upper"]
    assert 0.0 < out["q"] < 1.0


def test_ppi_ols_ci_returns_vector_coefficients() -> None:
    x_l, x_u, y_l, pred_l, pred_u = _synthetic()
    out = ppi_ols_ci(x_l, y_l, x_u, pred_l, pred_u, n_boot=200, seed=1)
    coef = out["coef"]
    ci_lower = out["ci_lower"]
    ci_upper = out["ci_upper"]
    assert len(coef) == x_l.shape[1] + 1  # intercept + 3 covariates
    assert len(ci_lower) == len(coef)
    assert len(ci_upper) == len(coef)
    for lo, est, hi in zip(ci_lower, coef, ci_upper, strict=True):
        assert lo <= est <= hi


def test_ppi_diagnostics_fields_present() -> None:
    _, _, y_l, pred_l, pred_u = _synthetic()
    out = ppi_diagnostics(y_l, pred_l, pred_u)
    assert out["n_labeled"] == float(y_l.numel())
    assert out["n_unlabeled"] == float(pred_u.numel())
    assert -1.0 <= out["prediction_label_correlation"] <= 1.0
    assert 0.0 <= out["prediction_range_overlap_ratio"] <= 1.0

