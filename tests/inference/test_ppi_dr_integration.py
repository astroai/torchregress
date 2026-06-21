"""
Integration tests: end-to-end PPI + DR pipelines with simulated data.

These tests verify that the full inference stack works together correctly:
- PPI methods produce valid confidence intervals covering the true parameter
- DR methods recover treatment effects from simulated causal data
- Combined PPI+DR pipeline handles real-world workflows
"""

from __future__ import annotations

import numpy as np
import torch

from torchregress.causal.diagnostics import causal_overlap_report
from torchregress.causal.dr import dr_ate, dr_cate, dr_policy_value
from torchregress.inference.ppi import (
    PPIConfig,
    ppi_calibrated_mean_ci,
    ppi_diagnostics,
    ppi_mean_ci,
    ppi_ols_ci,
    ppi_quantile_ci,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Simple sklearn-style models for DR testing
# ═══════════════════════════════════════════════════════════════════════════════


class RidgeRegressor:
    """Ridge regression via numpy for clean, fast DR tests."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_features = X.shape[1]
        self.coef_ = np.linalg.solve(X.T @ X + self.alpha * np.eye(n_features), X.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return (X @ self.coef_).reshape(-1)


class LogisticClassifier:
    """Logistic regression via IRLS for clean DR tests."""

    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n, d = X.shape
        beta = np.zeros(d)
        for _ in range(20):
            eta = X @ beta
            mu = 1.0 / (1.0 + np.exp(-eta))
            mu = np.clip(mu, 1e-6, 1 - 1e-6)
            W = np.diag(mu * (1 - mu))
            z = eta + (y - mu) / (mu * (1 - mu))
            try:
                beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ z)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ z, rcond=None)[0]
        self.coef_ = beta
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        eta = X @ self.coef_
        p1 = 1.0 / (1.0 + np.exp(-eta))
        return np.column_stack([1.0 - p1, p1])


# ═══════════════════════════════════════════════════════════════════════════════
# PPI Integration: Mean Inference
# ═══════════════════════════════════════════════════════════════════════════════


class TestPPIMeanIntegration:
    """End-to-end PPI mean inference with known ground truth."""

    def test_mean_ci_covers_truth_with_good_predictor(self) -> None:
        """When predictions are good, PPI mean CI covers the true mean."""
        torch.manual_seed(42)
        np.random.seed(42)
        true_mean = 5.0
        n_labeled = 100
        n_unlabeled = 1000

        # Generate data with known mean
        y_labeled = torch.randn(n_labeled) * 2.0 + true_mean
        # Good predictor: noisy version of the truth
        pred_labeled = y_labeled + 0.2 * torch.randn(n_labeled)
        pred_unlabeled = torch.randn(n_unlabeled) * 2.0 + true_mean + 0.2 * torch.randn(n_unlabeled)

        result = ppi_mean_ci(
            y_labeled=y_labeled,
            pred_labeled=pred_labeled,
            pred_unlabeled=pred_unlabeled,
            config=PPIConfig(alpha=0.1, n_boot=500, seed=42),
        )

        # CI should cover the true mean
        assert result["ci_lower"] <= true_mean <= result["ci_upper"]
        # Estimate should be close to true mean
        assert abs(result["estimate"] - true_mean) < 2 * result["se"]
        # SE should be reasonable
        assert result["se"] < 1.0  # Good predictor → small SE

    def test_mean_ci_wider_with_poor_predictor(self) -> None:
        """Poor predictions give wider or more uncertain CIs."""
        torch.manual_seed(42)
        np.random.seed(42)
        true_mean = 5.0

        # Good predictor case
        y = torch.randn(100) * 2.0 + true_mean
        p_good = y + 0.1 * torch.randn(100)
        result_good = ppi_mean_ci(
            y_labeled=y,
            pred_labeled=p_good,
            pred_unlabeled=torch.randn(500) * 2.0 + true_mean + 0.1 * torch.randn(500),
            config=PPIConfig(alpha=0.1, n_boot=300, seed=42),
        )

        # Poor predictor case
        p_poor = torch.randn(100) * 5.0  # Unrelated to y
        result_poor = ppi_mean_ci(
            y_labeled=y,
            pred_labeled=p_poor,
            pred_unlabeled=torch.randn(500) * 5.0,
            config=PPIConfig(alpha=0.1, n_boot=300, seed=42),
        )

        # Poor predictor should have larger SE
        assert result_poor["se"] > result_good["se"]

    def test_calibrated_improves_poor_predictions(self) -> None:
        """Linearly calibrated PPI recovers truth even with biased predictions."""
        torch.manual_seed(42)
        np.random.seed(42)
        true_mean = 5.0

        y = torch.randn(50) * 2.0 + true_mean
        # Biased predictor: pred = 0.5 * y + 10
        p_labeled = 0.5 * y + 10.0
        p_unlabeled = 0.5 * (torch.randn(500) * 2.0 + true_mean) + 10.0

        result_cal = ppi_calibrated_mean_ci(
            y_labeled=y,
            pred_labeled=p_labeled,
            pred_unlabeled=p_unlabeled,
            config=PPIConfig(alpha=0.1, n_boot=200, seed=42),
        )

        # Calibrated CI should cover the truth despite bias
        assert result_cal["ci_lower"] <= true_mean <= result_cal["ci_upper"]
        assert result_cal["method"] == "ppi_calibrated_mean_ci"

    def test_diagnostics_on_end_to_end_data(self) -> None:
        """PPI diagnostics report meaningful values on real data."""
        torch.manual_seed(42)
        np.random.seed(42)

        y = torch.randn(80)
        p_labeled = y + 0.3 * torch.randn(80)
        p_unlabeled = torch.randn(300) + 0.3 * torch.randn(300)

        diag = ppi_diagnostics(y_labeled=y, pred_labeled=p_labeled, pred_unlabeled=p_unlabeled)

        # Correlation should be positive for good predictor
        assert diag["prediction_label_correlation"] > 0.5
        # RMSE should be reasonable
        assert diag["residual_rmse_labeled"] < 1.0
        # Overlap ratio should be reasonable
        assert 0 <= diag["prediction_range_overlap_ratio"] <= 1


# ═══════════════════════════════════════════════════════════════════════════════
# PPI Integration: Quantile + OLS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPPIQuantileOLSIntegration:
    """End-to-end PPI quantile and OLS inference."""

    def test_median_ci_covers_true_median(self) -> None:
        """PPI quantile CI covers the true median."""
        torch.manual_seed(42)
        true_median = 3.0

        y = torch.randn(100) + true_median
        p_labeled = y + 0.2 * torch.randn(100)
        p_unlabeled = torch.randn(500) + true_median + 0.2 * torch.randn(500)

        result = ppi_quantile_ci(
            y_labeled=y,
            pred_labeled=p_labeled,
            pred_unlabeled=p_unlabeled,
            q=0.5,
            config=PPIConfig(alpha=0.1, n_boot=300, seed=42),
        )

        # CI should cover median
        assert result["ci_lower"] <= true_median <= result["ci_upper"]
        assert result["q"] == 0.5

    def test_ols_ci_recovers_true_coefficients(self) -> None:
        """PPI OLS CI recovers true regression coefficients."""
        torch.manual_seed(42)
        np.random.seed(42)

        n_labeled = 100
        n_unlabeled = 500
        n_features = 3
        true_beta = torch.tensor([1.0, -2.0, 0.5])

        x_l = torch.randn(n_labeled, n_features)
        x_u = torch.randn(n_unlabeled, n_features)
        y_l = x_l @ true_beta + 1.0 * torch.randn(n_labeled)
        p_l = x_l @ true_beta + 0.3 * torch.randn(n_labeled)
        p_u = x_u @ true_beta + 0.3 * torch.randn(n_unlabeled)

        result = ppi_ols_ci(
            x_labeled=x_l,
            y_labeled=y_l,
            x_unlabeled=x_u,
            pred_labeled=p_l,
            pred_unlabeled=p_u,
            config=PPIConfig(alpha=0.1, n_boot=200, seed=42),
        )

        coef = result["coef"]
        ci_lo = result["ci_lower"]
        ci_hi = result["ci_upper"]
        # Intercept + 3 feature coefficients
        assert len(coef) == 4
        # Each true coefficient (feature 1..3) should be in its CI
        # Skip intercept check since it's not part of true_beta
        for j in range(3):
            assert ci_lo[j + 1] <= float(true_beta[j]) <= ci_hi[j + 1]

    def test_ols_no_intercept(self) -> None:
        """OLS without intercept returns correct number of coefficients."""
        torch.manual_seed(42)

        x_l = torch.randn(50, 2)
        x_u = torch.randn(200, 2)
        true_beta = torch.tensor([0.5, 1.5])
        y_l = x_l @ true_beta + 0.1 * torch.randn(50)

        result = ppi_ols_ci(
            x_labeled=x_l,
            y_labeled=y_l,
            x_unlabeled=x_u,
            pred_labeled=y_l,
            pred_unlabeled=x_u @ true_beta,
            add_intercept=False,
            config=PPIConfig(alpha=0.1, n_boot=100, seed=42),
        )

        assert len(result["coef"]) == 2
        assert result["add_intercept"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# DR Integration: ATE
# ═══════════════════════════════════════════════════════════════════════════════


class TestDRATEIntegration:
    """End-to-end doubly-robust ATE estimation."""

    def test_dr_ate_recovers_known_treatment_effect(self) -> None:
        """DR ATE recovers a ground-truth treatment effect from simulated data."""
        np.random.seed(42)
        torch.manual_seed(42)

        n = 200
        x = torch.randn(n, 3)
        true_ate = 2.0

        # Treatment depends on x (confounded)
        t_prob = torch.sigmoid(x[:, 0] * 0.5 + x[:, 1] * 0.3)
        t = (torch.rand(n) < t_prob).float()

        # Outcome: base + treatment effect + noise
        y = 1.0 * x[:, 0] + 0.5 * x[:, 1] + true_ate * t + 0.5 * torch.randn(n)

        result = dr_ate(
            x=x,
            t=t,
            y=y,
            outcome_model=RidgeRegressor(alpha=0.1),
            propensity_model=LogisticClassifier(),
            folds=3,
            alpha=0.05,
            seed=42,
        )

        # CI should cover true ATE
        assert result["ci_lower"] <= true_ate <= result["ci_upper"]
        # Estimate should be reasonably close
        assert abs(result["estimate"] - true_ate) < 3 * result["se"]
        # SE should be positive
        assert result["se"] > 0
        # Diagnostics present
        assert "diagnostics" in result
        assert result["diagnostics"]["overlap_rate"] > 0

    def test_dr_ate_zero_effect(self) -> None:
        """DR ATE correctly identifies zero treatment effect."""
        np.random.seed(42)
        torch.manual_seed(42)

        n = 150
        x = torch.randn(n, 2)
        t = (torch.rand(n) > 0.5).float()
        y = 1.0 * x[:, 0] + 0.3 * torch.randn(n)  # No treatment effect

        result = dr_ate(
            x=x,
            t=t,
            y=y,
            outcome_model=RidgeRegressor(),
            propensity_model=LogisticClassifier(),
            folds=2,
            alpha=0.05,
            seed=42,
        )

        # CI should contain zero
        assert result["ci_lower"] <= 0.0 <= result["ci_upper"]
        # Estimate should be near zero
        assert abs(result["estimate"]) < 2.0

    def test_dr_cate_produces_sensible_estimates(self) -> None:
        """DR CATE produces per-unit estimates that average to ATE."""
        np.random.seed(42)
        torch.manual_seed(42)

        n = 100
        x = torch.randn(n, 2)
        true_ate = 1.5
        t = (torch.rand(n) > 0.5).float()
        y = x[:, 0] + true_ate * t + 0.3 * torch.randn(n)

        result = dr_cate(
            x=x,
            t=t,
            y=y,
            cate_model=RidgeRegressor(),
            outcome_model=RidgeRegressor(),
            propensity_model=LogisticClassifier(),
            folds=2,
            alpha=0.05,
            seed=42,
        )

        # CATE estimates should average close to ATE estimate
        cate_mean = float(result["cate_hat"].mean().item())
        assert abs(cate_mean - result["ate_estimate"]) < 2.0

    def test_dr_policy_value_estimates_policy_quality(self) -> None:
        """DR policy value evaluates a treatment assignment policy."""
        np.random.seed(42)
        torch.manual_seed(42)

        n = 120
        x = torch.randn(n, 2)
        true_ate = 1.0
        t = (torch.rand(n) > 0.5).float()
        y = x[:, 0] + true_ate * t + 0.3 * torch.randn(n)

        # Optimal policy: treat when t=1 (the actual treatment)
        optimal_policy = t.clone()
        # Random policy
        random_policy = (torch.rand(n) > 0.5).float()

        result_optimal = dr_policy_value(
            x=x,
            t=t,
            y=y,
            policy=optimal_policy,
            outcome_model=RidgeRegressor(),
            propensity_model=LogisticClassifier(),
            folds=2,
            seed=42,
        )

        result_random = dr_policy_value(
            x=x,
            t=t,
            y=y,
            policy=random_policy,
            outcome_model=RidgeRegressor(),
            propensity_model=LogisticClassifier(),
            folds=2,
            seed=42,
        )

        # Optimal policy should have higher value
        assert result_optimal["estimate"] > result_random["estimate"]


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-module Integration: PPI → DR
# ═══════════════════════════════════════════════════════════════════════════════


class TestPPIDRCrossModuleIntegration:
    """PPI predictions fed into DR causal pipeline."""

    def test_ppi_and_dr_agree_on_effect_direction(self) -> None:
        """PPI and DR methods both identify the same direction of effect."""
        np.random.seed(42)
        torch.manual_seed(42)

        n = 200
        # Simulate data with known ATE
        x = torch.randn(n, 3)
        t = (torch.rand(n) > 0.5).float()
        true_ate = 3.0
        y = 1.0 * x[:, 0] + 0.5 * x[:, 1] + true_ate * t + 0.5 * torch.randn(n)

        # Split into labeled/unlabeled for PPI
        n_labeled = 80
        idx = torch.randperm(n)
        labeled_idx = idx[:n_labeled]
        unlabeled_idx = idx[n_labeled:]

        # Step 1: PPI mean CI on the full population
        ppi_result = ppi_mean_ci(
            y_labeled=y[labeled_idx],
            pred_labeled=y[labeled_idx],  # Perfect predictor on labeled
            pred_unlabeled=torch.randn(len(unlabeled_idx)) + y.mean(),  # Reasonable predictor
            config=PPIConfig(alpha=0.1, n_boot=300, seed=42),
        )
        assert ppi_result["ci_lower"] <= ppi_result["ci_upper"]

        # Step 2: DR ATE on the same data
        dr_result = dr_ate(
            x=x,
            t=t,
            y=y,
            outcome_model=RidgeRegressor(),
            propensity_model=LogisticClassifier(),
            folds=2,
            alpha=0.05,
            seed=42,
        )

        # Both methods should agree on the sign of the effect
        ppi_sign = 1 if ppi_result["estimate"] > 0 else -1
        dr_sign = 1 if dr_result["estimate"] > 0 else -1
        assert ppi_sign == dr_sign

        # DR should capture the ATE (treatment effect)
        assert dr_result["ci_lower"] <= true_ate <= dr_result["ci_upper"]

    def test_causal_overlap_diagnostics_integration(self) -> None:
        """Overlap diagnostics detect poor overlap conditions."""
        torch.manual_seed(42)

        n = 200
        # Extreme propensity: treatment nearly deterministic based on x
        x = torch.randn(n, 2)
        t = (x[:, 0] > 0).float()
        # Very extreme propensities → poor overlap after trimming
        propensity = t * 0.98 + (1 - t) * 0.02

        report = causal_overlap_report(propensity, t, trim_threshold=0.05)
        # With extreme propensities, many samples are trimmed
        assert report["overlap_rate"] < 0.5
        assert report["n_trimmed"] > 0
        # ESS should be much smaller than n due to trimming
        assert report["min_group_ess"] < n

    def test_full_pipeline_ppi_mean_then_dr_ate(self) -> None:
        """Complete pipeline: compute PPI diagnostics, then run DR ATE."""
        np.random.seed(42)
        torch.manual_seed(42)

        n = 200
        x = torch.randn(n, 3)
        t = (torch.rand(n) > 0.5).float()
        true_ate = 2.0
        y = 0.8 * x[:, 0] + 0.4 * x[:, 1] + true_ate * t + 0.4 * torch.randn(n)

        # PPI diagnostics to assess prediction quality
        n_labeled = 60
        idx = torch.randperm(n)
        l_idx = idx[:n_labeled]
        u_idx = idx[n_labeled:]

        diag = ppi_diagnostics(
            y_labeled=y[l_idx],
            pred_labeled=y[l_idx] + 0.1 * torch.randn(n_labeled),
            pred_unlabeled=y[u_idx] + 0.1 * torch.randn(len(u_idx)),
        )

        # Validate diagnostics
        assert diag["n_labeled"] == n_labeled
        assert diag["prediction_label_correlation"] > 0.8

        # PPI mean CI
        ppi_res = ppi_mean_ci(
            y_labeled=y[l_idx],
            pred_labeled=y[l_idx],
            pred_unlabeled=y[u_idx],
            config=PPIConfig(alpha=0.1, n_boot=200, seed=42),
        )
        assert ppi_res["ci_lower"] <= ppi_res["ci_upper"]

        # DR ATE (using all data)
        dr_res = dr_ate(
            x=x,
            t=t,
            y=y,
            outcome_model=RidgeRegressor(),
            propensity_model=LogisticClassifier(),
            folds=2,
            alpha=0.05,
            seed=42,
        )

        # Both CIs should be valid
        assert dr_res["ci_lower"] <= true_ate <= dr_res["ci_upper"]

        # Verify all expected keys across both results
        expected_ppi_keys = {"method", "estimate", "se", "ci_lower", "ci_upper", "alpha"}
        expected_dr_keys = {"estimate", "se", "ci_lower", "ci_upper", "diagnostics"}
        assert expected_ppi_keys.issubset(set(ppi_res.keys()))
        assert expected_dr_keys.issubset(set(dr_res.keys()))
