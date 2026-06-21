"""
Unit tests for torchregress.inference.ppi.
"""

from __future__ import annotations

import pytest
import torch

from torchregress.inference.ppi import (
    PPIConfig,
    _bootstrap_indices,
    _linear_calibrate_apply,
    _percentile_ci,
    _rectified_mean_bootstrap,
    _rectified_mean_point,
    _to_1d_tensor,
    ppi_calibrated_mean_ci,
    ppi_diagnostics,
    ppi_mean_ci,
    ppi_ols_ci,
    ppi_quantile_ci,
)

# ═══════════════════════════════════════════════════════════════════════════════
# PPIConfig
# ═══════════════════════════════════════════════════════════════════════════════


class TestPPIConfig:
    def test_defaults(self) -> None:
        """Default alpha=0.1, method='bootstrap', n_boot=2000."""
        cfg = PPIConfig()
        assert cfg.alpha == 0.1
        assert cfg.method == "bootstrap"
        assert cfg.n_boot == 2000
        assert cfg.seed is None

    def test_custom(self) -> None:
        """Custom values are stored."""
        cfg = PPIConfig(alpha=0.05, n_boot=1000, seed=42)
        assert cfg.alpha == 0.05
        assert cfg.n_boot == 1000
        assert cfg.seed == 42


# ═══════════════════════════════════════════════════════════════════════════════
# _to_1d_tensor
# ═══════════════════════════════════════════════════════════════════════════════


class TestTo1DTensor:
    def test_tensor_input(self) -> None:
        """Tensor is flattened and detached."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        result = _to_1d_tensor(x)
        assert result.shape == (4,)
        assert not result.requires_grad

    def test_list_input(self) -> None:
        """List is converted to 1D tensor."""
        result = _to_1d_tensor([1.0, 2.0, 3.0])
        assert result.shape == (3,)
        assert result.dtype == torch.float32

    def test_already_1d(self) -> None:
        """Already-1D tensor stays 1D."""
        x = torch.tensor([1.0, 2.0, 3.0])
        result = _to_1d_tensor(x)
        assert result.shape == (3,)


# ═══════════════════════════════════════════════════════════════════════════════
# _bootstrap_indices
# ═══════════════════════════════════════════════════════════════════════════════


class TestBootstrapIndices:
    def test_shape(self) -> None:
        """Returns (n_boot, n) indices."""
        idx = _bootstrap_indices(100, n_boot=500, device=torch.device("cpu"))
        assert idx.shape == (500, 100)

    def test_values_in_range(self) -> None:
        """All indices are in [0, n)."""
        idx = _bootstrap_indices(10, n_boot=50, device=torch.device("cpu"))
        assert (idx >= 0).all()
        assert (idx < 10).all()

    def test_with_generator(self) -> None:
        """Seeded generator produces reproducible indices."""
        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)
        idx1 = _bootstrap_indices(10, n_boot=20, device=torch.device("cpu"), generator=g1)
        idx2 = _bootstrap_indices(10, n_boot=20, device=torch.device("cpu"), generator=g2)
        assert torch.equal(idx1, idx2)


# ═══════════════════════════════════════════════════════════════════════════════
# _percentile_ci
# ═══════════════════════════════════════════════════════════════════════════════


class TestPercentileCI:
    def test_symmetric_data(self) -> None:
        """Symmetric data gives centered CI."""
        samples = torch.randn(1000) * 0.5 + 10.0
        lo, hi = _percentile_ci(samples, alpha=0.1)
        assert lo < 10.0 < hi

    def test_alpha_bounds(self) -> None:
        """alpha=0.5 gives median-centered CI."""
        samples = torch.randn(1000)
        lo, hi = _percentile_ci(samples, alpha=0.5)
        assert lo < hi


# ═══════════════════════════════════════════════════════════════════════════════
# _rectified_mean_point / _rectified_mean_bootstrap
# ═══════════════════════════════════════════════════════════════════════════════


class TestRectifiedMean:
    def test_point_estimate(self) -> None:
        """Point = mean(unlabeled_score) + mean(labeled_residual)."""
        y_l = torch.tensor([1.0, 2.0, 3.0])
        p_l = torch.tensor([0.5, 1.5, 2.5])
        p_u = torch.tensor([10.0, 20.0, 30.0])
        point = _rectified_mean_point(y_l, p_l, p_u)
        # residual = [0.5, 0.5, 0.5], mean_residual = 0.5
        # mean unlabeled = 20.0
        # point = 20.0 + 0.5 = 20.5
        assert float(point.item()) == pytest.approx(20.5)

    def test_bootstrap_returns_ci(self) -> None:
        """Bootstrap returns (samples, lower, upper)."""
        y_l = torch.randn(50)
        p_l = y_l + 0.1 * torch.randn(50)
        p_u = torch.randn(200)
        _, lo, hi = _rectified_mean_bootstrap(
            y_l,
            p_l,
            p_u,
            n_boot=500,
            alpha=0.1,
            generator=None,
        )
        assert lo < hi


# ═══════════════════════════════════════════════════════════════════════════════
# _linear_calibrate_apply
# ═══════════════════════════════════════════════════════════════════════════════


class TestLinearCalibrateApply:
    def test_identity_on_perfect_predictions(self) -> None:
        """When m_fit == y_fit, calibration is identity."""
        m_fit = torch.tensor([1.0, 2.0, 3.0])
        y_fit = torch.tensor([1.0, 2.0, 3.0])
        m_apply = torch.tensor([4.0, 5.0])
        result = _linear_calibrate_apply(m_fit, y_fit, m_apply)
        assert torch.allclose(result, m_apply)

    def test_linear_scaling(self) -> None:
        """Calibration scales and shifts predictions."""
        m_fit = torch.tensor([1.0, 2.0, 3.0])
        y_fit = torch.tensor([2.0, 4.0, 6.0])
        m_apply = torch.tensor([0.0, 4.0])
        result = _linear_calibrate_apply(m_fit, y_fit, m_apply)
        assert torch.allclose(result, 2 * m_apply)

    def test_near_zero_denom_fallback(self) -> None:
        """When predictor has negligible variance, use mean of y."""
        m_fit = torch.full((10,), 5.0)
        y_fit = torch.randn(10) + 3.0
        m_apply = torch.tensor([1.0, 2.0, 3.0])
        result = _linear_calibrate_apply(m_fit, y_fit, m_apply)
        # fallback: all outputs equal to mean(y_fit)
        assert torch.allclose(result, y_fit.mean().expand_as(m_apply), atol=1e-5)

    def test_maintains_shape(self) -> None:
        """Output shape matches m_apply shape."""
        m_fit = torch.randn(20)
        y_fit = m_fit + torch.randn(20) * 0.1
        m_apply = torch.randn(5, 2)
        result = _linear_calibrate_apply(m_fit, y_fit, m_apply)
        assert result.shape == (5, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# ppi_mean_ci
# ═══════════════════════════════════════════════════════════════════════════════


class TestPPIMeanCI:
    def test_returns_expected_keys(self) -> None:
        """Returns dictionary with all expected keys."""
        result = ppi_mean_ci(
            y_labeled=torch.randn(20),
            pred_labeled=torch.randn(20),
            pred_unlabeled=torch.randn(100),
            config=PPIConfig(alpha=0.1, n_boot=100, seed=42),
        )
        assert result["method"] == "ppi_mean_ci"
        assert "estimate" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["ci_lower"] <= result["ci_upper"]

    def test_list_input(self) -> None:
        """Works with list inputs."""
        result = ppi_mean_ci(
            y_labeled=[1.0, 2.0, 3.0, 4.0, 5.0],
            pred_labeled=[0.9, 2.1, 3.0, 4.2, 5.0],
            pred_unlabeled=[2.0] * 50,
            config=PPIConfig(alpha=0.1, n_boot=100),
        )
        assert isinstance(result["estimate"], float)

    def test_alpha_validation(self) -> None:
        """Invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be in"):
            ppi_mean_ci(
                y_labeled=torch.randn(10),
                pred_labeled=torch.randn(10),
                pred_unlabeled=torch.randn(10),
                config=PPIConfig(alpha=1.5),
            )

    def test_n_boot_validation(self) -> None:
        """n_boot < 10 raises ValueError."""
        with pytest.raises(ValueError, match="n_boot must be"):
            ppi_mean_ci(
                y_labeled=torch.randn(10),
                pred_labeled=torch.randn(10),
                pred_unlabeled=torch.randn(10),
                config=PPIConfig(n_boot=5),
            )

    def test_unsupported_method_raises(self) -> None:
        """Unsupported method raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported method"):
            ppi_mean_ci(
                y_labeled=torch.randn(10),
                pred_labeled=torch.randn(10),
                pred_unlabeled=torch.randn(10),
                config=PPIConfig(method="jackknife"),
            )

    def test_mismatched_sizes_raises(self) -> None:
        """Different numbers of labeled samples raise ValueError."""
        with pytest.raises(ValueError, match="same number of samples"):
            ppi_mean_ci(
                y_labeled=torch.randn(10),
                pred_labeled=torch.randn(20),
                pred_unlabeled=torch.randn(50),
            )

    def test_too_few_samples_raises(self) -> None:
        """Too few samples raises ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            ppi_mean_ci(
                y_labeled=torch.tensor([1.0]),
                pred_labeled=torch.tensor([1.0]),
                pred_unlabeled=torch.tensor([1.0]),
            )

    def test_se_is_finite(self) -> None:
        """Standard error estimate is finite."""
        result = ppi_mean_ci(
            y_labeled=torch.randn(30),
            pred_labeled=torch.randn(30),
            pred_unlabeled=torch.randn(200),
            config=PPIConfig(alpha=0.1, n_boot=200, seed=42),
        )
        assert result["se"] > 0
        assert result["se"] < float("inf")


# ═══════════════════════════════════════════════════════════════════════════════
# ppi_calibrated_mean_ci
# ═══════════════════════════════════════════════════════════════════════════════


class TestPPICalibratedMeanCI:
    def test_returns_expected_keys(self) -> None:
        """Returns dictionary with method='ppi_calibrated_mean_ci'."""
        result = ppi_calibrated_mean_ci(
            y_labeled=torch.randn(20),
            pred_labeled=torch.randn(20),
            pred_unlabeled=torch.randn(100),
            config=PPIConfig(alpha=0.1, n_boot=50, seed=42),
        )
        assert result["method"] == "ppi_calibrated_mean_ci"
        assert result["ci_lower"] <= result["ci_upper"]

    def test_requires_at_least_3_labeled(self) -> None:
        """At least 3 labeled samples required."""
        with pytest.raises(ValueError, match="at least 3"):
            ppi_calibrated_mean_ci(
                y_labeled=torch.randn(2),
                pred_labeled=torch.randn(2),
                pred_unlabeled=torch.randn(10),
                config=PPIConfig(n_boot=50),
            )

    def test_alpha_validation(self) -> None:
        """Invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be in"):
            ppi_calibrated_mean_ci(
                y_labeled=torch.randn(5),
                pred_labeled=torch.randn(5),
                pred_unlabeled=torch.randn(10),
                config=PPIConfig(alpha=0.0),
            )

    def test_n_boot_validation(self) -> None:
        """n_boot < 10 raises ValueError."""
        with pytest.raises(ValueError, match="n_boot must be"):
            ppi_calibrated_mean_ci(
                y_labeled=torch.randn(5),
                pred_labeled=torch.randn(5),
                pred_unlabeled=torch.randn(10),
                config=PPIConfig(n_boot=1),
            )

    def test_unsupported_method_raises(self) -> None:
        """Unsupported method raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported method"):
            ppi_calibrated_mean_ci(
                y_labeled=torch.randn(5),
                pred_labeled=torch.randn(5),
                pred_unlabeled=torch.randn(10),
                config=PPIConfig(method="unknown"),
            )

    def test_se_is_finite(self) -> None:
        """Standard error estimate is positive and finite."""
        result = ppi_calibrated_mean_ci(
            y_labeled=torch.randn(20),
            pred_labeled=torch.randn(20),
            pred_unlabeled=torch.randn(100),
            config=PPIConfig(alpha=0.1, n_boot=50, seed=42),
        )
        assert result["se"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# ppi_quantile_ci
# ═══════════════════════════════════════════════════════════════════════════════


class TestPPIQuantileCI:
    def test_returns_expected_keys(self) -> None:
        """Returns dictionary with method='ppi_quantile_ci'."""
        result = ppi_quantile_ci(
            y_labeled=torch.randn(30),
            pred_labeled=torch.randn(30),
            pred_unlabeled=torch.randn(200),
            q=0.5,
            config=PPIConfig(alpha=0.1, n_boot=100, seed=42),
        )
        assert result["method"] == "ppi_quantile_ci"
        assert result["q"] == 0.5
        assert result["ci_lower"] <= result["ci_upper"]

    def test_q_validation(self) -> None:
        """Invalid q raises ValueError."""
        with pytest.raises(ValueError, match="q must be in"):
            ppi_quantile_ci(
                y_labeled=torch.randn(10),
                pred_labeled=torch.randn(10),
                pred_unlabeled=torch.randn(10),
                q=0.0,
            )

    def test_median_ci(self) -> None:
        """Median CI uses shift correction from labeled residuals."""
        result = ppi_quantile_ci(
            y_labeled=torch.randn(30),
            pred_labeled=torch.randn(30),
            pred_unlabeled=torch.randn(200),
            q=0.5,
            config=PPIConfig(alpha=0.1, n_boot=100, seed=42),
        )
        assert result["se"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# ppi_ols_ci
# ═══════════════════════════════════════════════════════════════════════════════


class TestPPIOLSCI:
    def test_returns_expected_keys(self) -> None:
        """Returns coefficients, SE, and CI bounds."""
        x_l = torch.randn(50, 3)
        x_u = torch.randn(200, 3)
        true_beta = torch.tensor([1.0, 2.0, 3.0])
        y_l = x_l @ true_beta + 0.1 * torch.randn(50)
        p_l = x_l @ true_beta + 0.05 * torch.randn(50)
        p_u = x_u @ true_beta + 0.05 * torch.randn(200)
        result = ppi_ols_ci(
            x_labeled=x_l,
            y_labeled=y_l,
            x_unlabeled=x_u,
            pred_labeled=p_l,
            pred_unlabeled=p_u,
            config=PPIConfig(alpha=0.1, n_boot=100, seed=42),
        )
        assert result["method"] == "ppi_ols_ci"
        assert len(result["coef"]) >= 3  # intercept + 3 features
        assert len(result["se"]) == len(result["coef"])
        assert len(result["ci_lower"]) == len(result["coef"])

    def test_no_intercept(self) -> None:
        """add_intercept=False omits intercept column."""
        x_l = torch.randn(30, 2)
        x_u = torch.randn(100, 2)
        y_l = x_l @ torch.tensor([1.0, -1.0]) + 0.1 * torch.randn(30)
        p_l = y_l + 0.1 * torch.randn(30)
        p_u = torch.randn(100)
        result = ppi_ols_ci(
            x_labeled=x_l,
            y_labeled=y_l,
            x_unlabeled=x_u,
            pred_labeled=p_l,
            pred_unlabeled=p_u,
            add_intercept=False,
            config=PPIConfig(alpha=0.1, n_boot=50, seed=42),
        )
        assert len(result["coef"]) == 2  # exactly 2 features, no intercept

    def test_mismatched_shapes_raises(self) -> None:
        """Mismatched x/y/pred dimensions raise ValueError."""
        with pytest.raises(ValueError, match="align on sample dimension"):
            ppi_ols_ci(
                x_labeled=torch.randn(10, 3),
                y_labeled=torch.randn(20),
                x_unlabeled=torch.randn(100, 3),
                pred_labeled=torch.randn(10),
                pred_unlabeled=torch.randn(100),
            )

    def test_default_n_boot(self) -> None:
        """Default n_boot=1000 for ols_ci."""
        x_l = torch.randn(20, 1)
        x_u = torch.randn(50, 1)
        y_l = x_l.squeeze() + 0.1 * torch.randn(20)
        result = ppi_ols_ci(
            x_labeled=x_l,
            y_labeled=y_l,
            x_unlabeled=x_u,
            pred_labeled=x_l.squeeze(),
            pred_unlabeled=x_u.squeeze(),
            config=None,  # uses default PPIConfig(n_boot=1000)
        )
        assert result["bootstrap_samples"] == 1000


# ═══════════════════════════════════════════════════════════════════════════════
# ppi_diagnostics
# ═══════════════════════════════════════════════════════════════════════════════


class TestPPIDiagnostics:
    def test_returns_expected_keys(self) -> None:
        """Returns diagnostic dictionary."""
        result = ppi_diagnostics(
            y_labeled=torch.randn(50),
            pred_labeled=torch.randn(50),
            pred_unlabeled=torch.randn(200),
        )
        assert "prediction_label_correlation" in result
        assert "residual_rmse_labeled" in result
        assert result["residual_rmse_labeled"] >= 0
        assert result["prediction_range_overlap_ratio"] >= 0

    def test_perfect_predictions(self) -> None:
        """Perfect predictions give correlation=1 and rmse=0."""
        y = torch.randn(30)
        result = ppi_diagnostics(
            y_labeled=y,
            pred_labeled=y,
            pred_unlabeled=torch.randn(100),
        )
        assert result["prediction_label_correlation"] == pytest.approx(1.0, abs=1e-5)
        assert result["residual_rmse_labeled"] == pytest.approx(0.0, abs=1e-5)

    def test_single_labeled_sample(self) -> None:
        """Single labeled sample still produces report (corr=0)."""
        result = ppi_diagnostics(
            y_labeled=torch.tensor([5.0]),
            pred_labeled=torch.tensor([4.0]),
            pred_unlabeled=torch.randn(50),
        )
        assert result["prediction_label_correlation"] == 0.0

    def test_mismatched_sizes_raises(self) -> None:
        """Mismatched labeled sizes raise ValueError."""
        with pytest.raises(ValueError, match="same number of samples"):
            ppi_diagnostics(
                y_labeled=torch.randn(10),
                pred_labeled=torch.randn(20),
                pred_unlabeled=torch.randn(50),
            )
