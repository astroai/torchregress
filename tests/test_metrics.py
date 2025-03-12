"""
Tests for the metrics modules in torchregression.
"""

import torch
import numpy as np
import pytest

from torchregression.metrics.utils import convert_to_tensor, validate_inputs
from torchregression.metrics.calibration import (
    expected_calibration_error,
    marginal_calibration_error,
    calibration_metrics_report
)
from torchregression.metrics.distribution import (
    probability_integral_transform,
    continuous_ranked_probability_score,
    energy_score,
    distribution_metrics_report
)
from torchregression.metrics.ood import (
    mahalanobis_distance,
    typicality_score,
    entropy_score,
    kernel_density_score,
    ood_metrics_report
)
from torchregression.metrics.point import (
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    huber_loss,
    normalized_rmse,
    trimmed_mean_squared_error,
    median_absolute_deviation,
    regression_metrics_report
)
from torchregression.metrics.interval import (
    interval_score,
    prediction_interval_coverage_probability,
    interval_metrics_report
)


class TestMetricsUtils:
    """Test utility functions for metrics."""
    
    def test_convert_to_tensor(self):
        """Test conversion of different input types to torch tensors."""
        # Test numpy array conversion
        np_arr = np.array([1.0, 2.0, 3.0])
        tensor = convert_to_tensor(np_arr)
        assert isinstance(tensor, torch.Tensor)
        assert torch.allclose(tensor, torch.tensor([1.0, 2.0, 3.0]))
        
        # Test list conversion
        lst = [4.0, 5.0, 6.0]
        tensor = convert_to_tensor(lst)
        assert isinstance(tensor, torch.Tensor)
        assert torch.allclose(tensor, torch.tensor([4.0, 5.0, 6.0]))
        
        # Test scalar conversion
        scalar = 7.0
        tensor = convert_to_tensor(scalar)
        assert isinstance(tensor, torch.Tensor)
        assert torch.allclose(tensor, torch.tensor([7.0]))
        
        # Test tensor passthrough
        original = torch.tensor([8.0, 9.0])
        tensor = convert_to_tensor(original)
        assert tensor is original
        
        # Test invalid input
        with pytest.raises(TypeError):
            convert_to_tensor("not convertible")
    
    def test_validate_inputs(self):
        """Test validation of input tensors."""
        # Valid inputs
        y_pred = torch.randn(10, 2)
        y_true = torch.randn(10, 2)
        validate_inputs(y_pred, y_true)  # Should not raise exception
        
        # Mismatched batch sizes
        y_pred_bad = torch.randn(11, 2)
        with pytest.raises(ValueError):
            validate_inputs(y_pred_bad, y_true)
        
        # Scalar inputs (should be rejected)
        with pytest.raises(ValueError):
            validate_inputs(torch.tensor(1.0), torch.tensor(2.0))
        
        # Test NaN/inf detection
        y_pred_nan = torch.tensor([1.0, float('nan'), 3.0])
        y_true_ok = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            validate_inputs(y_pred_nan, y_true_ok)
        
        y_pred_ok = torch.tensor([1.0, 2.0, 3.0])
        y_true_inf = torch.tensor([1.0, float('inf'), 3.0])
        with pytest.raises(ValueError):
            validate_inputs(y_pred_ok, y_true_inf)


class TestCalibrationMetrics:
    """Test calibration metrics."""
    
    def setup_method(self):
        """Setup test data."""
        torch.manual_seed(42)
        self.batch_size = 100
        self.n_samples = 20
        
        # Generate ground truth
        self.y_true = torch.randn(self.batch_size)
        
        # Generate quantiles
        self.q_levels = [0.1, 0.5, 0.9]
        self.y_pred_quantiles = {}
        for q in self.q_levels:
            offset = torch.randn(self.batch_size) * 0.1  # Add some noise for imperfect calibration
            self.y_pred_quantiles[q] = torch.quantile(self.y_true, q) + offset
        
        # Generate samples
        self.y_pred_samples = torch.randn(self.n_samples, self.batch_size)
    
    def test_expected_calibration_error(self):
        """Test expected calibration error metric."""
        result = expected_calibration_error(self.y_pred_quantiles, self.y_true)
        
        # Check output structure
        assert 'mean_absolute_calibration_error' in result
        assert 'root_mean_squared_calibration_error' in result
        assert 'maximum_calibration_error' in result
        
        # Check values are reasonable
        assert 0 <= result['mean_absolute_calibration_error'] <= 1
        assert 0 <= result['root_mean_squared_calibration_error'] <= 1
        assert 0 <= result['maximum_calibration_error'] <= 1
        
        # Test with diagnostics
        result_diag = expected_calibration_error(self.y_pred_quantiles, self.y_true, return_diagnostics=True)
        assert 'bin_errors' in result_diag
        assert 'expected_proportions' in result_diag
        assert 'actual_proportions' in result_diag
        
        # Test numpy inputs
        y_true_np = self.y_true.numpy()
        y_pred_q_np = {q: self.y_pred_quantiles[q].numpy() for q in self.y_pred_quantiles}
        result_np = expected_calibration_error(y_pred_q_np, y_true_np)
        assert isinstance(result_np['mean_absolute_calibration_error'], float)
    
    def test_marginal_calibration_error(self):
        """Test marginal calibration error metric."""
        result = marginal_calibration_error(self.y_pred_samples, self.y_true)
        
        # Check output structure
        assert 'marginal_calibration_error' in result
        assert 'root_mean_squared_mce' in result
        assert 'maximum_marginal_calibration_error' in result
        
        # Check values are reasonable
        assert 0 <= result['marginal_calibration_error'] <= 1
        assert 0 <= result['root_mean_squared_mce'] <= 1
        assert 0 <= result['maximum_marginal_calibration_error'] <= 1
        
        # Test with diagnostics
        result_diag = marginal_calibration_error(self.y_pred_samples, self.y_true, return_diagnostics=True)
        assert 'bin_centers' in result_diag
        assert 'observed_cdf' in result_diag
        assert 'predicted_cdf' in result_diag
        assert 'abs_errors' in result_diag
        
        # Test numpy inputs
        y_true_np = self.y_true.numpy()
        y_pred_samples_np = self.y_pred_samples.numpy()
        result_np = marginal_calibration_error(y_pred_samples_np, y_true_np)
        assert isinstance(result_np['marginal_calibration_error'], float)
    
    def test_calibration_metrics_report(self):
        """Test comprehensive calibration report."""
        # Test with distribution
        from torch.distributions import Normal
        mean = torch.zeros(self.batch_size)
        std = torch.ones(self.batch_size)
        dist = Normal(mean, std)
        
        result = calibration_metrics_report(dist, self.y_true)
        assert len(result) > 0
        
        # Test with samples
        result = calibration_metrics_report(self.y_pred_samples, self.y_true)
        assert len(result) > 0
        
        # Test with quantiles
        result = calibration_metrics_report(None, self.y_true, self.y_pred_quantiles)
        assert len(result) > 0
        
        # Test with dictionary of distribution parameters
        dist_params = {'loc': mean, 'scale': std}
        result = calibration_metrics_report(dist_params, self.y_true)
        assert len(result) > 0


class TestDistributionMetrics:
    """Test distribution metrics."""
    
    def setup_method(self):
        """Setup test data."""
        torch.manual_seed(42)
        self.batch_size = 100
        self.n_samples = 20
        self.n_dims = 3
        
        # Generate ground truth
        self.y_true = torch.randn(self.batch_size)
        self.y_true_multi = torch.randn(self.batch_size, self.n_dims)
        
        # Generate quantiles
        self.q_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        self.y_pred_quantiles = {}
        for q in self.q_levels:
            offset = torch.randn(self.batch_size) * 0.1
            self.y_pred_quantiles[q] = torch.quantile(self.y_true, q) + offset
        
        # Generate samples
        self.y_pred_samples = torch.randn(self.n_samples, self.batch_size)
        self.y_pred_samples_multi = torch.randn(self.n_samples, self.batch_size, self.n_dims)
    
    def test_probability_integral_transform(self):
        """Test Probability Integral Transform (PIT) metric."""
        # Create a dummy CDF function
        def cdf_fn(x):
            # Simple normal CDF approximation
            return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))
        
        # Test basic PIT calculation
        pit_values = probability_integral_transform(cdf_fn, self.y_true)
        assert isinstance(pit_values, torch.Tensor)
        assert pit_values.shape == self.y_true.shape
        assert torch.all((pit_values >= 0) & (pit_values <= 1))
        
        # Test with histogram
        result = probability_integral_transform(cdf_fn, self.y_true, return_histogram=True)
        assert isinstance(result, dict)
        assert 'pit_values' in result
        assert 'histogram_counts' in result
        assert 'bin_edges' in result
        assert 'uniformity_chi2' in result
        
        # Test with numpy output
        y_true_np = self.y_true.numpy()
        result_np = probability_integral_transform(cdf_fn, y_true_np, return_histogram=True)
        assert isinstance(result_np['pit_values'], np.ndarray)
    
    def test_continuous_ranked_probability_score(self):
        """Test CRPS metric."""
        # Test basic CRPS calculation
        crps = continuous_ranked_probability_score(self.y_pred_quantiles, self.y_true)
        assert isinstance(crps, float)
        assert crps >= 0
        
        # Test with different reductions
        crps_none = continuous_ranked_probability_score(self.y_pred_quantiles, self.y_true, reduction="none")
        assert isinstance(crps_none, torch.Tensor)
        assert crps_none.shape == self.y_true.shape
        
        crps_sum = continuous_ranked_probability_score(self.y_pred_quantiles, self.y_true, reduction="sum")
        assert isinstance(crps_sum, float)
        
        # Test input validation
        with pytest.raises(ValueError):
            # Test with too few quantiles
            bad_quantiles = {0.5: self.y_pred_quantiles[0.5]}
            continuous_ranked_probability_score(bad_quantiles, self.y_true)
    
    def test_energy_score(self):
        """Test energy score metric."""
        # Test basic energy score for multivariate data
        es = energy_score(self.y_pred_samples_multi, self.y_true_multi)
        assert isinstance(es, float)
        assert es >= 0
        
        # Test with different beta values
        es_beta05 = energy_score(self.y_pred_samples_multi, self.y_true_multi, beta=0.5)
        assert isinstance(es_beta05, float)
        assert es_beta05 >= 0
        
        # Test with max_pairs limit
        es_max = energy_score(self.y_pred_samples_multi, self.y_true_multi, max_pairs=10)
        assert isinstance(es_max, float)
        assert es_max >= 0
        
        # Test with different reductions
        es_none = energy_score(self.y_pred_samples_multi, self.y_true_multi, reduction="none")
        assert isinstance(es_none, torch.Tensor)
        assert es_none.shape[0] == self.batch_size
    
    def test_distribution_metrics_report(self):
        """Test comprehensive distribution metrics report."""
        # Test with distribution
        from torch.distributions import Normal
        mean = torch.zeros(self.batch_size)
        std = torch.ones(self.batch_size)
        dist = Normal(mean, std)
        
        result = distribution_metrics_report(dist, self.y_true)
        assert 'log_prob' in result
        assert 'crps' in result
        
        # Test with samples and quantiles
        result = distribution_metrics_report(None, self.y_true, 
                                           self.y_pred_quantiles,
                                           self.y_pred_samples)
        assert 'crps' in result
        
        # Test with multi-dimensional data and samples
        result = distribution_metrics_report(None, self.y_true_multi, 
                                           samples=self.y_pred_samples_multi)
        assert 'energy_score' in result


class TestOODMetrics:
    """Test out-of-distribution detection metrics."""
    
    def setup_method(self):
        """Setup test data."""
        torch.manual_seed(42)
        self.batch_size = 100
        self.n_features = 5
        self.n_samples = 20
        
        # Generate feature data
        self.x_test = torch.randn(self.batch_size, self.n_features)
        self.x_ref = torch.randn(200, self.n_features)  # Reference set is larger
        
        # Generate distribution parameters
        self.mean = torch.zeros(self.n_features)
        self.cov = torch.eye(self.n_features)
        self.model_output = (torch.zeros(self.batch_size), torch.ones(self.batch_size))
        
        # Generate samples
        self.samples = torch.randn(self.n_samples, self.batch_size, 1)
    
    def test_mahalanobis_distance(self):
        """Test Mahalanobis distance metric."""
        # Test basic calculation
        md = mahalanobis_distance(self.x_test, self.mean, self.cov)
        assert isinstance(md, torch.Tensor)
        assert md.shape[0] == self.batch_size
        assert torch.all(md >= 0)
        
        # Test with reductions
        md_mean = mahalanobis_distance(self.x_test, self.mean, self.cov, reduction="mean")
        assert isinstance(md_mean, torch.Tensor)
        assert md_mean.ndim == 0
        
        md_sum = mahalanobis_distance(self.x_test, self.mean, self.cov, reduction="sum")
        assert isinstance(md_sum, torch.Tensor)
        assert md_sum.ndim == 0
        
        # Test with singular covariance matrix
        singular_cov = torch.ones((self.n_features, self.n_features))
        md_singular = mahalanobis_distance(self.x_test, self.mean, singular_cov)
        assert isinstance(md_singular, torch.Tensor)
        assert md_singular.shape[0] == self.batch_size
    
    def test_typicality_score(self):
        """Test typicality score metric."""
        # Test with tuple input (mean, var)
        ts = typicality_score(self.model_output, self.x_test)
        assert isinstance(ts, torch.Tensor)
        assert ts.shape[0] == self.batch_size
        
        # Test with dictionary input
        model_dict = {'mean': self.model_output[0], 'variance': self.model_output[1]}
        ts_dict = typicality_score(model_dict, self.x_test)
        assert isinstance(ts_dict, torch.Tensor)
        assert ts_dict.shape[0] == self.batch_size
        
        # Test with reductions
        ts_mean = typicality_score(self.model_output, self.x_test, reduction="mean")
        assert isinstance(ts_mean, torch.Tensor)
        assert ts_mean.ndim == 0
        
        # Test with invalid input
        with pytest.raises(ValueError):
            typicality_score(torch.randn(10), self.x_test)
    
    def test_entropy_score(self):
        """Test entropy score metric."""
        # Test basic calculation
        es = entropy_score(self.samples)
        assert isinstance(es, torch.Tensor)
        assert es.shape[0] == self.batch_size
        assert torch.all(es >= 0)
        
        # Test with reductions
        es_mean = entropy_score(self.samples, reduction="mean")
        assert isinstance(es_mean, torch.Tensor)
        assert es_mean.ndim == 0
        
        es_sum = entropy_score(self.samples, reduction="sum")
        assert isinstance(es_sum, torch.Tensor)
        assert es_sum.ndim == 0
    
    def test_kernel_density_score(self):
        """Test kernel density score metric."""
        # Test basic calculation
        kd = kernel_density_score(self.x_test, self.x_ref)
        assert isinstance(kd, torch.Tensor)
        assert kd.shape[0] == self.batch_size
        assert torch.all((kd >= 0) & (kd <= 1))
        
        # Test with different bandwidth
        kd_bw = kernel_density_score(self.x_test, self.x_ref, bandwidth=0.5)
        assert isinstance(kd_bw, torch.Tensor)
        assert kd_bw.shape[0] == self.batch_size
        
        # Test with reductions
        kd_mean = kernel_density_score(self.x_test, self.x_ref, reduction="mean")
        assert isinstance(kd_mean, torch.Tensor)
        assert kd_mean.ndim == 0
    
    def test_ood_metrics_report(self):
        """Test comprehensive OOD metrics report."""
        # Test with all inputs
        result = ood_metrics_report(
            model_output=self.model_output,
            x_test=self.x_test,
            x_reference=self.x_ref,
            mean=self.mean,
            cov=self.cov,
            samples=self.samples
        )
        
        assert 'mahalanobis_distance' in result
        assert 'typicality_score' in result
        assert 'kernel_density' in result
        assert 'entropy' in result
        
        # Test with partial inputs
        result_partial = ood_metrics_report(
            model_output=self.model_output,
            x_test=self.x_test
        )
        
        assert 'typicality_score' in result_partial
        assert 'mahalanobis_distance' not in result_partial


class TestPointMetrics:
    """Test point prediction metrics."""
    
    def setup_method(self):
        """Setup test data."""
        torch.manual_seed(42)
        self.batch_size = 100
        self.n_features = 2
        
        # Generate prediction and ground truth data
        self.y_pred = torch.randn(self.batch_size, self.n_features)
        self.y_true = torch.randn(self.batch_size, self.n_features)
        
        # Generate sample weights
        self.sample_weight = torch.ones(self.batch_size)
    
    def test_basic_metrics(self):
        """Test basic regression metrics."""
        # Test MSE
        mse = mean_squared_error(self.y_pred, self.y_true)
        assert mse > 0
        
        # Test MAE
        mae = mean_absolute_error(self.y_pred, self.y_true)
        assert mae > 0
        
        # Test Median AE
        median_ae = median_absolute_error(self.y_pred, self.y_true)
        assert median_ae > 0
    
    def test_robust_metrics(self):
        """Test robust regression metrics."""
        # Test huber loss
        huber = huber_loss(self.y_pred, self.y_true)
        assert huber > 0
        
        # Test trimmed MSE
        tmse = trimmed_mean_squared_error(self.y_pred, self.y_true)
        assert tmse > 0
        
        # Test MAD
        mad = median_absolute_deviation(self.y_pred, self.y_true)
        assert mad > 0
    
    def test_normalized_metrics(self):
        """Test normalized regression metrics."""
        # Test normalized RMSE
        nrmse = normalized_rmse(self.y_pred, self.y_true)
        assert nrmse > 0
        
        # Test with different normalizations
        nrmse_range = normalized_rmse(self.y_pred, self.y_true, normalization="range")
        assert nrmse_range > 0
        
        nrmse_mean = normalized_rmse(self.y_pred, self.y_true, normalization="mean")
        assert nrmse_mean > 0
        
        nrmse_iqr = normalized_rmse(self.y_pred, self.y_true, normalization="iqr")
        assert nrmse_iqr > 0
    
    def test_comprehensive_report(self):
        """Test comprehensive regression metrics report."""
        # Test basic report
        report = regression_metrics_report(self.y_pred, self.y_true)
        
        # Check basic metrics
        assert 'mse' in report
        assert 'rmse' in report
        assert 'mae' in report
        assert 'r2' in report
        
        # Check robust metrics
        assert 'huber_loss' in report
        assert 'mad' in report
        assert 'nmad' in report
        
        # Check outlier metrics
        assert 'outlier_fraction' in report


class TestIntervalMetrics:
    """Test prediction interval metrics."""
    
    def setup_method(self):
        """Setup test data."""
        torch.manual_seed(42)
        self.batch_size = 100
        
        # Generate ground truth
        self.y_true = torch.randn(self.batch_size)
        
        # Generate prediction intervals
        self.lower = self.y_true - 1 - 0.2 * torch.rand(self.batch_size)
        self.upper = self.y_true + 1 + 0.2 * torch.rand(self.batch_size)
        
        # Create predictions dictionary for multiple models
        self.predictions = {
            'model1': {'lower': self.lower, 'upper': self.upper},
            'model2': {'lower': self.lower - 0.5, 'upper': self.upper + 0.5}
        }
    
    def test_interval_score(self):
        """Test interval score (Winkler score) metric."""
        # Test basic calculation
        score = interval_score(self.lower, self.upper, self.y_true)
        assert score > 0
        
        # Test with full output
        result = interval_score(self.lower, self.upper, self.y_true, reduction="full")
        assert isinstance(result, dict)
        assert 'score' in result
        assert 'mean_width' in result
        assert 'mean_coverage' in result
        assert 'expected_coverage' in result
        assert 'coverage_error' in result
        
        # Test with invalid intervals
        with pytest.raises(ValueError):
            interval_score(self.upper, self.lower, self.y_true)  # Swapped bounds
    
    def test_prediction_interval_coverage(self):
        """Test prediction interval coverage probability."""
        # Test basic calculation
        picp = prediction_interval_coverage_probability(self.lower, self.upper, self.y_true)
        assert 0 <= picp <= 1
        
        # Test with diagnostics
        result = prediction_interval_coverage_probability(self.lower, self.upper, self.y_true, return_diagnostics=True)
        assert isinstance(result, dict)
        assert 'picp' in result
        assert 'expected_coverage' in result
        assert 'coverage_error' in result
        assert 'mpiw' in result
        assert 'miss_rate_low' in result
        assert 'miss_rate_high' in result
    
    def test_interval_metrics_report(self):
        """Test comprehensive interval metrics report."""
        report = interval_metrics_report(self.predictions, self.y_true)
        
        assert 'model1' in report
        assert 'model2' in report
        assert 'score' in report['model1']
        assert 'picp' in report['model1']
        assert 'mpiw' in report['model1']


if __name__ == "__main__":
    pytest.main(["-v"])
