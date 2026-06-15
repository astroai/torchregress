import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from torchregress.viz import (
    plot_causal_uplift_qini,
    plot_censored_survival_curves,
    plot_conditional_density_slices,
    plot_residuals,
    plot_risk_coverage_curve,
    plot_simex_extrapolation,
    plot_target_density_error_overlap,
    plot_uncertainty_vs_error,
)


class TestVizExtended:
    """Test extended and new visualization functions."""

    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        torch.manual_seed(42)
        self.n_samples = 100
        self.y_true = np.random.randn(self.n_samples)
        self.y_pred = self.y_true + np.random.randn(self.n_samples) * 0.1
        self.y_pred_std = np.ones(self.n_samples) * 0.1
        self.y_true_std = np.ones(self.n_samples) * 0.05
        self.censoring_indicator = np.random.binomial(1, 0.8, self.n_samples)

    def teardown_method(self):
        """Close all figures."""
        plt.close("all")

    def test_plot_residuals_extended(self):
        """Test plot_residuals with extended inputs."""
        # 1. Test standard
        fig = plot_residuals(self.y_pred, self.y_true, return_figure=True)
        assert isinstance(fig, Figure)
        plt.close(fig)

        # 2. Test with std (standardized residuals)
        fig_std = plot_residuals(
            self.y_pred,
            self.y_true,
            y_pred_std=self.y_pred_std,
            y_true_std=self.y_true_std,
            return_figure=True,
        )
        assert isinstance(fig_std, Figure)
        plt.close(fig_std)

        # 3. Test with censoring indicators
        fig_cens = plot_residuals(
            self.y_pred,
            self.y_true,
            censoring_indicator=self.censoring_indicator,
            return_figure=True,
        )
        assert isinstance(fig_cens, Figure)
        plt.close(fig_cens)

        # 4. Test on existing axes
        fig, ax = plt.subplots()
        plot_residuals(self.y_pred, self.y_true, ax=ax)
        assert len(ax.collections) > 0 or len(ax.lines) > 0
        plt.close(fig)

    def test_plot_uncertainty_vs_error_extended(self):
        """Test plot_uncertainty_vs_error with decomposition components."""
        # 1. Standard
        fig = plot_uncertainty_vs_error(
            self.y_pred, self.y_pred_std, self.y_true, return_figure=True
        )
        if isinstance(fig, tuple):
            fig = fig[0]
        assert isinstance(fig, Figure)
        plt.close(fig)

        # 2. With decomposition
        aleatoric = np.random.uniform(0.01, 0.05, self.n_samples)
        epistemic = np.random.uniform(0.01, 0.05, self.n_samples)
        fig_decomp = plot_uncertainty_vs_error(
            self.y_pred,
            self.y_pred_std,
            self.y_true,
            aleatoric_var=aleatoric,
            epistemic_var=epistemic,
            sort_by="error",
            return_figure=True,
        )
        if isinstance(fig_decomp, tuple):
            fig_decomp = fig_decomp[0]
        assert isinstance(fig_decomp, Figure)
        plt.close(fig_decomp)

    def test_plot_target_density_error_overlap(self):
        """Test plot_target_density_error_overlap function."""
        # Test with return figure
        fig = plot_target_density_error_overlap(self.y_true, self.y_pred, return_figure=True)
        assert isinstance(fig, Figure)
        plt.close(fig)

        # Test on existing axis
        fig, ax = plt.subplots()
        plot_target_density_error_overlap(self.y_true, self.y_pred, ax=ax)
        # Verify twin axes was created by checking number of axes in figure
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_plot_conditional_density_slices(self):
        """Test plot_conditional_density_slices function."""

        # Define a mock density function: p(y | x) = Gaussian PDF
        def mock_density_fn(x, y):
            mean = np.sum(x)
            variance = 0.5
            return 1.0 / np.sqrt(2 * np.pi * variance) * np.exp(-((y - mean) ** 2) / (2 * variance))

        x_slices = np.array([[0.5, 0.5], [1.0, -1.0], [0.0, 2.0]])
        y_grid = np.linspace(-5, 5, 100)
        y_true_slices = np.array([1.0, 0.0, 2.0])

        fig = plot_conditional_density_slices(
            density_fn=mock_density_fn,
            x_slices=x_slices,
            y_grid=y_grid,
            y_true_slices=y_true_slices,
            return_figure=True,
        )
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_plot_censored_survival_curves(self):
        """Test plot_censored_survival_curves function."""
        time_grid = np.linspace(0, 10, 50)
        # predicted_survival shape [N, T]
        pred_survival = np.exp(
            -time_grid[None, :] * np.random.uniform(0.1, 0.5, (self.n_samples, 1))
        )
        observed_times = np.random.uniform(1, 10, self.n_samples)

        fig = plot_censored_survival_curves(
            predicted_survival=pred_survival,
            time_grid=time_grid,
            observed_times=observed_times,
            censoring_indicators=self.censoring_indicator,
            return_figure=True,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_simex_extrapolation(self):
        """Test plot_simex_extrapolation function."""
        lambdas = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        sim_vals = np.array([1.5, 1.8, 2.2, 2.7, 3.3])

        # Extrapolator is a simple polynomial
        poly_coeff = np.polyfit(lambdas, sim_vals, 2)

        fig = plot_simex_extrapolation(
            lambda_values=lambdas,
            simulated_values=sim_vals,
            extrapolator=poly_coeff,
            return_figure=True,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_risk_coverage_curve(self):
        """Test plot_risk_coverage_curve function."""
        rejection_scores = np.random.uniform(0, 1, self.n_samples)
        fig = plot_risk_coverage_curve(
            y_true=self.y_true,
            y_pred=self.y_pred,
            rejection_scores=rejection_scores,
            return_figure=True,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_causal_uplift_qini(self):
        """Test plot_causal_uplift_qini function."""
        uplift_scores = np.random.randn(self.n_samples)
        treatment = np.random.binomial(1, 0.5, self.n_samples)
        y_obs = treatment * 1.5 + np.random.randn(self.n_samples)

        fig = plot_causal_uplift_qini(
            uplift_scores=uplift_scores,
            treatment=treatment,
            y_obs=y_obs,
            return_figure=True,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_tensor_inputs(self):
        """Verify the functions correctly handle PyTorch tensors."""
        y_true_t = torch.tensor(self.y_true)
        y_pred_t = torch.tensor(self.y_pred)
        y_pred_std_t = torch.tensor(self.y_pred_std)

        # residuals
        fig = plot_residuals(y_pred_t, y_true_t, return_figure=True)
        assert isinstance(fig, Figure)
        plt.close(fig)

        # risk-coverage
        fig_rc = plot_risk_coverage_curve(
            y_true=y_true_t,
            y_pred=y_pred_t,
            rejection_scores=y_pred_std_t,
            return_figure=True,
        )
        assert isinstance(fig_rc, Figure)
        plt.close(fig_rc)
