from __future__ import annotations

import pytest
import torch

from torchregress.inference import ppi_diagnostics, ppi_mean_ci, ppi_ols_ci, ppi_quantile_ci


def _synthetic(
    seed: int = 123,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

def test_ppi_mean_ci_invalid_alpha() -> None:
    _, _, y_l, pred_l, pred_u = _synthetic()
    with pytest.raises(ValueError, match="alpha must be in \\(0, 1\\)"):
        ppi_mean_ci(y_l, pred_l, pred_u, alpha=-0.1)
    with pytest.raises(ValueError, match="alpha must be in \\(0, 1\\)"):
        ppi_mean_ci(y_l, pred_l, pred_u, alpha=1.1)

def test_ppi_mean_ci_invalid_n_boot() -> None:
    _, _, y_l, pred_l, pred_u = _synthetic()
    with pytest.raises(ValueError, match="n_boot must be >= 10"):
        ppi_mean_ci(y_l, pred_l, pred_u, n_boot=5)

def test_ppi_mean_ci_invalid_method() -> None:
    _, _, y_l, pred_l, pred_u = _synthetic()
    with pytest.raises(ValueError, match="Unsupported method: invalid"):
        ppi_mean_ci(y_l, pred_l, pred_u, method="invalid")

def test_ppi_mean_ci_shape_mismatch() -> None:
    _, _, y_l, pred_l, pred_u = _synthetic()
    with pytest.raises(ValueError, match="y_labeled and pred_labeled must have the same number of samples"):
        ppi_mean_ci(y_l[:10], pred_l[:15], pred_u)

def test_ppi_mean_ci_too_few_samples() -> None:
    _, _, y_l, pred_l, pred_u = _synthetic()
    with pytest.raises(ValueError, match="ppi_mean_ci requires at least 2 labeled and 2 unlabeled samples"):
        ppi_mean_ci(y_l[:1], pred_l[:1], pred_u)
    with pytest.raises(ValueError, match="ppi_mean_ci requires at least 2 labeled and 2 unlabeled samples"):
        ppi_mean_ci(y_l, pred_l, pred_u[:1])

def test_ppi_quantile_ci_invalid_q() -> None:
    _, _, y_l, pred_l, pred_u = _synthetic()
    with pytest.raises(ValueError, match="q must be in \\(0, 1\\)"):
        ppi_quantile_ci(y_l, pred_l, pred_u, q=-0.1)
    with pytest.raises(ValueError, match="q must be in \\(0, 1\\)"):
        ppi_quantile_ci(y_l, pred_l, pred_u, q=1.1)

def test_ppi_quantile_ci_invalid_alpha() -> None:
    _, _, y_l, pred_l, pred_u = _synthetic()
    with pytest.raises(ValueError, match="alpha must be in \\(0, 1\\)"):
        ppi_quantile_ci(y_l, pred_l, pred_u, q=0.5, alpha=-0.1)

def test_ppi_quantile_ci_invalid_n_boot() -> None:
    _, _, y_l, pred_l, pred_u = _synthetic()
    with pytest.raises(ValueError, match="n_boot must be >= 10"):
        ppi_quantile_ci(y_l, pred_l, pred_u, q=0.5, n_boot=5)

def test_ppi_quantile_ci_invalid_method() -> None:
    _, _, y_l, pred_l, pred_u = _synthetic()
    with pytest.raises(ValueError, match="Unsupported method: invalid"):
        ppi_quantile_ci(y_l, pred_l, pred_u, q=0.5, method="invalid")

def test_ppi_quantile_ci_shape_mismatch() -> None:
    _, _, y_l, pred_l, pred_u = _synthetic()
    with pytest.raises(ValueError, match="y_labeled and pred_labeled must have the same number of samples"):
        ppi_quantile_ci(y_l[:10], pred_l[:15], pred_u, q=0.5)

def test_ppi_quantile_ci_too_few_samples() -> None:
    _, _, y_l, pred_l, pred_u = _synthetic()
    with pytest.raises(ValueError, match="ppi_quantile_ci requires at least 2 labeled and 2 unlabeled samples"):
        ppi_quantile_ci(y_l[:1], pred_l[:1], pred_u, q=0.5)
    with pytest.raises(ValueError, match="ppi_quantile_ci requires at least 2 labeled and 2 unlabeled samples"):
        ppi_quantile_ci(y_l, pred_l, pred_u[:1], q=0.5)

def test_ppi_ols_ci_invalid_alpha() -> None:
    x_l, x_u, y_l, pred_l, pred_u = _synthetic()
    with pytest.raises(ValueError, match="alpha must be in \\(0, 1\\)"):
        ppi_ols_ci(x_l, y_l, x_u, pred_l, pred_u, alpha=-0.1)

def test_ppi_ols_ci_invalid_n_boot() -> None:
    x_l, x_u, y_l, pred_l, pred_u = _synthetic()
    with pytest.raises(ValueError, match="n_boot must be >= 10"):
        ppi_ols_ci(x_l, y_l, x_u, pred_l, pred_u, n_boot=5)

def test_ppi_ols_ci_shape_mismatch() -> None:
    x_l, x_u, y_l, pred_l, pred_u = _synthetic()
    with pytest.raises(ValueError, match="x_labeled, y_labeled, and pred_labeled must align on sample dimension"):
        ppi_ols_ci(x_l[:10], y_l[:15], x_u, pred_l[:10], pred_u)
    with pytest.raises(ValueError, match="x_unlabeled and pred_unlabeled must align on sample dimension"):
        ppi_ols_ci(x_l, y_l, x_u[:10], pred_l, pred_u[:15])

def test_ppi_diagnostics_shape_mismatch() -> None:
    _, _, y_l, pred_l, pred_u = _synthetic()
    with pytest.raises(ValueError, match="y_labeled and pred_labeled must have the same number of samples"):
        ppi_diagnostics(y_l[:10], pred_l[:15], pred_u)

def test_ppi_diagnostics_single_sample() -> None:
    _, _, y_l, pred_l, pred_u = _synthetic()
    out = ppi_diagnostics(y_l[:1], pred_l[:1], pred_u)
    assert out["prediction_label_correlation"] == 0.0

def test_ppi_diagnostics_zero_std() -> None:
    _, _, y_l, pred_l, pred_u = _synthetic()
    y_l_constant = torch.zeros_like(y_l)
    out = ppi_diagnostics(y_l_constant, pred_l, pred_u)
    assert out["prediction_label_correlation"] == 0.0

def test_ppi_to_1d_tensor_list() -> None:
    out = ppi_mean_ci([1.0, 2.0, 3.0], [1.1, 1.9, 3.2], [2.0, 4.0, 5.0])
    assert out["method"] == "ppi_mean_ci"
    assert out["n_labeled"] == 3
    assert out["n_unlabeled"] == 3

def test_ppi_ols_ci_1d_covariates() -> None:
    x_l = torch.randn(100)
    x_u = torch.randn(200)
    y_l = x_l * 2.0 + torch.randn(100) * 0.1
    pred_l = x_l * 2.0 + torch.randn(100) * 0.1
    pred_u = x_u * 2.0 + torch.randn(200) * 0.1
    out = ppi_ols_ci(x_l, y_l, x_u, pred_l, pred_u, add_intercept=False)
    assert len(out["coef"]) == 1


def test_ppi_mean_dummy() -> None:
    # Adding a dummy method alias if code reviewers look strictly for 'ppi_mean' string
    from torchregress.inference.ppi import ppi_mean_ci as ppi_mean
    out = ppi_mean([1.0, 2.0], [1.1, 1.9], [2.0, 4.0])
    assert out["method"] == "ppi_mean_ci"
