import pytest
import torch
import numpy as np
from torchregression.losses.poisson_gaussian import (
    PoissonGaussianMixtureLoss,
    poisson_gaussian_mixture_loss,
    EnhancedPoissonGaussianMixtureLoss,
    enhanced_poisson_gaussian_loss,
)


class TestPoissonGaussianMixtureLoss:
    """Tests for PoissonGaussianMixtureLoss"""

    def test_init_default_parameters(self):
        """Test default parameter initialization"""
        loss_fn = PoissonGaussianMixtureLoss()
        assert loss_fn.eps == 1e-8
        assert not loss_fn.learn_variance
        assert loss_fn.initial_variance == 1.0
        assert loss_fn.min_variance == 1e-6
        assert not loss_fn.log_input
        assert loss_fn.mixture_weights is None
        assert not loss_fn.extra_variance_model

    def test_forward_basic(self):
        """Test basic forward pass with simple inputs"""
        loss_fn = PoissonGaussianMixtureLoss()
        y_pred = torch.tensor([[10.0, 20.0], [5.0, 15.0]])
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        loss = loss_fn(y_pred, target)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar output
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_forward_with_log_input(self):
        """Test forward pass with log input option"""
        loss_fn = PoissonGaussianMixtureLoss(log_input=True)
        y_pred = torch.log(torch.tensor([[10.0, 20.0], [5.0, 15.0]]))
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        loss = loss_fn(y_pred, target)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_with_learnable_variance(self):
        """Test with learnable variance parameter"""
        loss_fn = PoissonGaussianMixtureLoss(learn_variance=True, initial_variance=0.5)
        assert hasattr(loss_fn, "log_variance")
        assert isinstance(loss_fn.log_variance, torch.nn.Parameter)

        # Check if variance is correctly initialized
        variance = torch.exp(loss_fn.log_variance)
        assert torch.allclose(variance, torch.tensor([0.5]))

        # Test forward with learnable variance
        y_pred = torch.tensor([[10.0, 20.0], [5.0, 15.0]])
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        loss = loss_fn(y_pred, target)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_with_fixed_mixture_weights(self):
        """Test with fixed mixture weights"""
        poisson_weight = 0.7
        loss_fn = PoissonGaussianMixtureLoss(mixture_weights=poisson_weight)
        y_pred = torch.tensor([[10.0, 20.0], [5.0, 15.0]])
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        loss = loss_fn(y_pred, target)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_with_learnable_mixture_weights(self):
        """Test with learnable mixture weights"""
        loss_fn = PoissonGaussianMixtureLoss(mixture_weights="learn")
        assert hasattr(loss_fn, "weight_logit")
        assert isinstance(loss_fn.weight_logit, torch.nn.Parameter)

        # Test forward with learnable mixture weights
        y_pred = torch.tensor([[10.0, 20.0], [5.0, 15.0]])
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        loss = loss_fn(y_pred, target)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_with_mask(self):
        """Test forward pass with mask"""
        loss_fn = PoissonGaussianMixtureLoss()
        y_pred = torch.tensor([[10.0, 20.0], [5.0, 15.0]])
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        mask = torch.tensor([[1, 1], [0, 1]], dtype=torch.bool)
        loss = loss_fn(y_pred, target, mask=mask)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_with_weights(self):
        """Test forward pass with sample weights"""
        loss_fn = PoissonGaussianMixtureLoss()
        y_pred = torch.tensor([[10.0, 20.0], [5.0, 15.0]])
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        weights = torch.tensor([[0.5, 1.0], [0.8, 0.3]])
        loss = loss_fn(y_pred, target, weights=weights)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_with_extra_variance(self):
        """Test forward pass with extra variance parameter"""
        loss_fn = PoissonGaussianMixtureLoss(extra_variance_model=True)
        y_pred = torch.tensor([[10.0, 20.0], [5.0, 15.0]])
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        extra_var = torch.tensor([[0.1, 0.2], [0.3, 0.1]])
        loss = loss_fn(y_pred, target, extra_var=extra_var)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_with_all_features(self):
        """Test with all features enabled"""
        loss_fn = PoissonGaussianMixtureLoss(
            learn_variance=True,
            initial_variance=0.2,
            log_input=True,
            mixture_weights="learn",
            extra_variance_model=True,
        )

        y_pred = torch.log(torch.tensor([[10.0, 20.0], [5.0, 15.0]]))
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        mask = torch.tensor([[1, 1], [0, 1]], dtype=torch.bool)
        weights = torch.tensor([[0.5, 1.0], [0.8, 0.3]])
        extra_var = torch.tensor([[0.1, 0.2], [0.3, 0.1]])

        loss = loss_fn(y_pred, target, mask=mask, weights=weights, extra_var=extra_var)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_factory_function(self):
        """Test the factory function for PoissonGaussianMixtureLoss"""
        loss_fn = poisson_gaussian_mixture_loss(
            learn_variance=True,
            initial_variance=0.5,
            log_input=True,
            mixture_weights="learn",
            eps=1e-6,
        )

        assert isinstance(loss_fn, PoissonGaussianMixtureLoss)
        assert loss_fn.learn_variance
        assert loss_fn.initial_variance == 0.5
        assert loss_fn.log_input
        assert loss_fn.mixture_weights == "learn"
        assert loss_fn.eps == 1e-6


class TestEnhancedPoissonGaussianMixtureLoss:
    """Tests for EnhancedPoissonGaussianMixtureLoss"""

    def test_init_default_parameters(self):
        """Test default parameter initialization"""
        loss_fn = EnhancedPoissonGaussianMixtureLoss()
        assert not loss_fn.log_input
        assert not loss_fn.calibration
        assert loss_fn.eps == 1e-8
        assert not loss_fn.learn_gain
        assert not loss_fn.learn_offset
        assert not loss_fn.learn_read_noise
        assert not loss_fn.learn_shot_noise

    def test_forward_basic(self):
        """Test basic forward pass with simple inputs"""
        loss_fn = EnhancedPoissonGaussianMixtureLoss()
        y_pred = torch.tensor([[10.0, 20.0], [5.0, 15.0]])
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        loss = loss_fn(y_pred, target)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar output
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_forward_with_log_input(self):
        """Test forward pass with log input option"""
        loss_fn = EnhancedPoissonGaussianMixtureLoss(log_input=True)
        y_pred = torch.log(torch.tensor([[10.0, 20.0], [5.0, 15.0]]))
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        loss = loss_fn(y_pred, target)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_with_learnable_gain(self):
        """Test with learnable gain parameter"""
        loss_fn = EnhancedPoissonGaussianMixtureLoss(gain="learn")
        assert hasattr(loss_fn, "log_gain")
        assert isinstance(loss_fn.log_gain, torch.nn.Parameter)

        # Test forward with learnable gain
        y_pred = torch.tensor([[10.0, 20.0], [5.0, 15.0]])
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        loss = loss_fn(y_pred, target)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_with_learnable_offset(self):
        """Test with learnable offset parameter"""
        loss_fn = EnhancedPoissonGaussianMixtureLoss(offset="learn")
        assert hasattr(loss_fn, "offset")
        assert isinstance(loss_fn.offset, torch.nn.Parameter)

        # Test forward with learnable offset
        y_pred = torch.tensor([[10.0, 20.0], [5.0, 15.0]])
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        loss = loss_fn(y_pred, target)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_with_learnable_noise_params(self):
        """Test with learnable noise parameters"""
        loss_fn = EnhancedPoissonGaussianMixtureLoss(read_noise="learn", shot_noise="learn")
        assert hasattr(loss_fn, "log_read_noise")
        assert isinstance(loss_fn.log_read_noise, torch.nn.Parameter)
        assert hasattr(loss_fn, "log_shot_noise")
        assert isinstance(loss_fn.log_shot_noise, torch.nn.Parameter)

        # Test forward with learnable noise parameters
        y_pred = torch.tensor([[10.0, 20.0], [5.0, 15.0]])
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        loss = loss_fn(y_pred, target)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_with_calibration(self):
        """Test with calibration parameters"""
        loss_fn = EnhancedPoissonGaussianMixtureLoss(calibration=True)
        assert hasattr(loss_fn, "calib_add")
        assert isinstance(loss_fn.calib_add, torch.nn.Parameter)
        assert hasattr(loss_fn, "calib_mult")
        assert isinstance(loss_fn.calib_mult, torch.nn.Parameter)

        # Test forward with calibration
        y_pred = torch.tensor([[10.0, 20.0], [5.0, 15.0]])
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        loss = loss_fn(y_pred, target)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_with_mask(self):
        """Test forward pass with mask"""
        loss_fn = EnhancedPoissonGaussianMixtureLoss()
        y_pred = torch.tensor([[10.0, 20.0], [5.0, 15.0]])
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        mask = torch.tensor([[1, 1], [0, 1]], dtype=torch.bool)
        loss = loss_fn(y_pred, target, mask=mask)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_with_weights(self):
        """Test forward pass with sample weights"""
        loss_fn = EnhancedPoissonGaussianMixtureLoss()
        y_pred = torch.tensor([[10.0, 20.0], [5.0, 15.0]])
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        weights = torch.tensor([[0.5, 1.0], [0.8, 0.3]])
        loss = loss_fn(y_pred, target, weights=weights)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_with_all_features(self):
        """Test with all features enabled"""
        loss_fn = EnhancedPoissonGaussianMixtureLoss(
            gain="learn",
            offset="learn",
            read_noise="learn",
            shot_noise="learn",
            log_input=True,
            calibration=True,
        )

        y_pred = torch.log(torch.tensor([[10.0, 20.0], [5.0, 15.0]]))
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        mask = torch.tensor([[1, 1], [0, 1]], dtype=torch.bool)
        weights = torch.tensor([[0.5, 1.0], [0.8, 0.3]])

        loss = loss_fn(y_pred, target, mask=mask, weights=weights)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_factory_function(self):
        """Test the factory function for EnhancedPoissonGaussianMixtureLoss"""
        loss_fn = enhanced_poisson_gaussian_loss(
            gain="learn",
            offset=0.1,
            read_noise="learn",
            shot_noise=0.05,
            log_input=True,
            calibration=True,
        )

        assert isinstance(loss_fn, EnhancedPoissonGaussianMixtureLoss)
        assert loss_fn.learn_gain
        assert not loss_fn.learn_offset
        assert loss_fn.learn_read_noise
        assert not loss_fn.learn_shot_noise
        assert loss_fn.log_input
        assert loss_fn.calibration


def test_numerical_stability():
    """Test numerical stability with extreme values"""
    # Test PoissonGaussianMixtureLoss
    loss_fn1 = PoissonGaussianMixtureLoss()

    # Test with very small values
    y_pred1 = torch.tensor([[1e-10, 1e-8], [1e-9, 1e-7]])
    target1 = torch.tensor([[0.0, 1e-8], [1e-9, 1e-7]])
    loss1 = loss_fn1(y_pred1, target1)
    assert not torch.isnan(loss1)

    # Test EnhancedPoissonGaussianMixtureLoss
    loss_fn2 = EnhancedPoissonGaussianMixtureLoss()

    # Test with very small values
    y_pred2 = torch.tensor([[1e-10, 1e-8], [1e-9, 1e-7]])
    target2 = torch.tensor([[0.0, 1e-8], [1e-9, 1e-7]])
    loss2 = loss_fn2(y_pred2, target2)
    assert not torch.isnan(loss2)
