import pytest
import torch
from torch import optim
from torchregression.losses.poisson_gaussian import (
    PoissonGaussianMixtureLoss,
    EnhancedPoissonGaussianMixtureLoss
)


class SimpleModel(torch.nn.Module):
    """Simple model for testing gradient flow through the loss functions"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)


class TestGradients:
    """Tests for checking gradient flow through the loss functions"""
    
    def test_poisson_gaussian_mixture_loss_gradients(self):
        """Test that gradients flow correctly through PoissonGaussianMixtureLoss"""
        torch.manual_seed(42)
        
        # Create a simple model
        model = SimpleModel(5, 2)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Create inputs and targets
        x = torch.randn(10, 5)
        targets = torch.rand(10, 2) * 20  # Random values between 0 and 20
        
        # Create loss function with learnable parameters
        loss_fn = PoissonGaussianMixtureLoss(
            learn_variance=True,
            mixture_weights='learn'
        )
        
        # Store initial parameter values
        initial_weight = model.linear.weight.clone().detach()
        initial_bias = model.linear.bias.clone().detach()
        initial_variance = loss_fn.log_variance.clone().detach()
        initial_weight_logit = loss_fn.weight_logit.clone().detach()
        
        # Training loop
        for _ in range(5):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, targets)
            loss.backward()
            optimizer.step()
        
        # Check that parameters have changed
        assert not torch.allclose(model.linear.weight, initial_weight)
        assert not torch.allclose(model.linear.bias, initial_bias)
        assert not torch.allclose(loss_fn.log_variance, initial_variance)
        assert not torch.allclose(loss_fn.weight_logit, initial_weight_logit)
    
    def test_enhanced_poisson_gaussian_loss_gradients(self):
        """Test that gradients flow correctly through EnhancedPoissonGaussianMixtureLoss"""
        torch.manual_seed(42)
        
        # Create a simple model
        model = SimpleModel(5, 2)
        
        # Create loss function with all learnable parameters
        loss_fn = EnhancedPoissonGaussianMixtureLoss(
            gain='learn',
            offset='learn',
            read_noise='learn',
            shot_noise='learn',
            calibration=True
        )
        
        # Combine all parameters
        params = list(model.parameters()) + list(loss_fn.parameters())
        optimizer = optim.Adam(params, lr=0.01)
        
        # Create inputs and targets
        x = torch.randn(10, 5)
        targets = torch.rand(10, 2) * 20  # Random values between 0 and 20
        
        # Store initial parameter values
        initial_weight = model.linear.weight.clone().detach()
        initial_bias = model.linear.bias.clone().detach()
        initial_log_gain = loss_fn.log_gain.clone().detach()
        initial_offset = loss_fn.offset.clone().detach()
        initial_log_read_noise = loss_fn.log_read_noise.clone().detach()
        initial_log_shot_noise = loss_fn.log_shot_noise.clone().detach()
        initial_calib_add = loss_fn.calib_add.clone().detach()
        initial_calib_mult = loss_fn.calib_mult.clone().detach()
        
        # Training loop
        for _ in range(5):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, targets)
            loss.backward()
            optimizer.step()
        
        # Check that parameters have changed
        assert not torch.allclose(model.linear.weight, initial_weight)
        assert not torch.allclose(model.linear.bias, initial_bias)
        assert not torch.allclose(loss_fn.log_gain, initial_log_gain)
        assert not torch.allclose(loss_fn.offset, initial_offset)
        assert not torch.allclose(loss_fn.log_read_noise, initial_log_read_noise)
        assert not torch.allclose(loss_fn.log_shot_noise, initial_log_shot_noise)
        assert not torch.allclose(loss_fn.calib_add, initial_calib_add)
        assert not torch.allclose(loss_fn.calib_mult, initial_calib_mult)
