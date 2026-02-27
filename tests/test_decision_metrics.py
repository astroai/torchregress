"""
Tests for decision support metrics.
"""

import pytest
import torch

from torchregress.metrics.decision import RejectionPolicy, RiskCoverageCurve, risk_coverage_curve


class TestRiskCoverageMetrics:
    """Test Risk-Coverage Curve and Rejection Policy."""

    def setup_method(self):
        """Setup test data."""
        torch.manual_seed(42)
        self.n_samples = 100

        # Perfect calibration case: uncertainty is exactly the absolute error
        self.y_true = torch.randn(self.n_samples)
        self.y_pred = self.y_true + torch.randn(self.n_samples) * 0.1
        self.perfect_uncertainty = torch.abs(self.y_pred - self.y_true)

        # Random uncertainty case: noise unrelated to error
        self.random_uncertainty = torch.rand(self.n_samples)

    def test_risk_coverage_curve_perfect(self):
        """Test RCC with perfect uncertainty (risk should decrease as coverage decreases)."""
        metric = RiskCoverageCurve(n_points=10)
        metric.update(self.y_pred, self.y_true, self.perfect_uncertainty)
        result = metric.compute()

        assert "coverage" in result
        assert "risk" in result
        assert "aurc" in result
        assert len(result["coverage"]) == 10

        # Risk at 100% coverage should be the full MSE
        full_mse = torch.mean((self.y_pred - self.y_true) ** 2)
        assert torch.allclose(result["risk"][-1], full_mse, atol=1e-5)

        # With perfect uncertainty, risk at lower coverage MUST be lower or equal
        # (monotonicity check)
        risks = result["risk"]
        assert torch.all(risks[1:] >= risks[:-1] - 1e-7)

    def test_risk_coverage_curve_functional(self):
        """Test the functional interface."""
        result = risk_coverage_curve(self.y_pred, self.y_true, self.perfect_uncertainty, n_points=5)
        assert len(result["risk"]) == 5
        assert result["aurc"] > 0

    def test_risk_coverage_curve_class_functional_consistency(self):
        """Class and functional interfaces should agree on the same inputs."""
        metric = RiskCoverageCurve(n_points=7)
        metric.update(self.y_pred, self.y_true, self.perfect_uncertainty)
        class_res = metric.compute()
        func_res = risk_coverage_curve(
            self.y_pred, self.y_true, self.perfect_uncertainty, n_points=7
        )

        assert torch.allclose(class_res["coverage"], func_res["coverage"])
        assert torch.allclose(class_res["risk"], func_res["risk"])
        assert torch.allclose(class_res["aurc"], func_res["aurc"])

    def test_risk_coverage_curve_multitarget_reduces_feature_dims(self):
        """RCC should reduce non-batch dimensions when computing per-sample risk."""
        y_true = torch.randn(20, 3)
        y_pred = y_true + 0.1 * torch.randn(20, 3)
        uncertainty = torch.rand(20)

        result = risk_coverage_curve(y_pred, y_true, uncertainty, n_points=6)
        assert result["risk"].shape == (6,)

        full_mse = ((y_pred - y_true) ** 2).mean()
        assert torch.allclose(result["risk"][-1], full_mse, atol=1e-6)

    def test_rejection_policy_fraction(self):
        """Test rejection policy with fixed fraction."""
        # Reject 20%, should keep 80 samples
        policy = RejectionPolicy(fraction=0.2)
        policy.update(self.y_pred, self.y_true, self.perfect_uncertainty)
        result = policy.compute()

        assert result["coverage"] == pytest.approx(0.8)
        assert int(result["n_rejected"]) == 20

        # Mean risk of 80% best samples should be lower than full MSE
        full_mse = torch.mean((self.y_pred - self.y_true) ** 2)
        assert result["mean_risk"] < full_mse

    def test_rejection_policy_threshold(self):
        """Test rejection policy with fixed threshold."""
        # Setup threshold that should keep roughly half
        median_unc = torch.median(self.perfect_uncertainty)
        policy = RejectionPolicy(threshold=median_unc.item())
        policy.update(self.y_pred, self.y_true, self.perfect_uncertainty)
        result = policy.compute()

        assert result["coverage"] == pytest.approx(0.5, abs=0.02)
        assert result["mean_risk"] < torch.mean((self.y_pred - self.y_true) ** 2)

    def test_rejection_policy_fraction_precedence_over_threshold(self):
        """If both are set, fraction should take precedence."""
        policy = RejectionPolicy(fraction=0.3, threshold=-1.0)
        policy.update(self.y_pred, self.y_true, self.perfect_uncertainty)
        result = policy.compute()

        assert result["coverage"] == pytest.approx(0.7, abs=0.02)

    def test_rejection_policy_requires_fraction_or_threshold(self):
        """Policy should error if neither threshold nor fraction is provided."""
        policy = RejectionPolicy()
        policy.update(self.y_pred, self.y_true, self.perfect_uncertainty)
        with pytest.raises(ValueError):
            policy.compute()

    def test_rejection_policy_all_rejected_returns_nan_risk(self):
        """Threshold below all uncertainties should reject everything."""
        policy = RejectionPolicy(threshold=float(self.perfect_uncertainty.min().item() - 1.0))
        policy.update(self.y_pred, self.y_true, self.perfect_uncertainty)
        result = policy.compute()

        assert torch.isnan(result["mean_risk"])
        assert result["coverage"] == 0
        assert int(result["n_rejected"]) == self.n_samples

    def test_empty_case(self):
        """Test behavior with no updates."""
        metric = RiskCoverageCurve()
        result = metric.compute()
        assert result["aurc"] == 0
        assert result["coverage"].numel() == 0

    def test_input_validation(self):
        """Test validation of inputs."""
        metric = RiskCoverageCurve()
        with pytest.raises(ValueError):
            # Mismatched uncertainty shape
            metric.update(torch.randn(10), torch.randn(10), torch.randn(5))
