import pytest
import torch
import numpy as np
from torchregression.categorical import HistogramLoss

class TestHistogramLoss:
    def test_init_valid_params(self):
        # Test initialization with valid parameters
        loss_fn = HistogramLoss(num_bins=10, bin_min=0.0, bin_max=1.0)
        assert loss_fn.num_bins == 10
        assert loss_fn.original_bin_min == 0.0
        assert loss_fn.original_bin_max == 1.0
        assert loss_fn.target_distribution == 'gaussian'
        assert loss_fn.reduction == 'mean'
        
        # Test with one-bin distribution
        loss_fn = HistogramLoss(num_bins=20, bin_min=-1.0, bin_max=1.0, target_distribution='one-bin')
        assert loss_fn.target_distribution == 'one-bin'
        
        # Test with custom sigma
        loss_fn = HistogramLoss(num_bins=30, bin_min=0.0, bin_max=3.0, sigma=0.5)
        assert loss_fn.sigma == 0.5
        
        # Test with different reduction
        loss_fn = HistogramLoss(num_bins=10, bin_min=0.0, bin_max=1.0, reduction='sum')
        assert loss_fn.reduction == 'sum'

    def test_init_invalid_params(self):
        # Test invalid num_bins
        with pytest.raises(ValueError, match="num_bins must be a positive integer"):
            HistogramLoss(num_bins=-5, bin_min=0.0, bin_max=1.0)
        
        # Test invalid bin range
        with pytest.raises(ValueError, match="bin_min must be less than bin_max"):
            HistogramLoss(num_bins=10, bin_min=1.0, bin_max=0.0)
            
        # Test invalid target_distribution
        with pytest.raises(ValueError, match="target_distribution must be"):
            HistogramLoss(num_bins=10, bin_min=0.0, bin_max=1.0, target_distribution='invalid')
            
        # Test invalid sigma
        with pytest.raises(ValueError, match="sigma must be a positive number"):
            HistogramLoss(num_bins=10, bin_min=0.0, bin_max=1.0, sigma=-1.0)
            
        # Test invalid padding
        with pytest.raises(ValueError, match="padding must be a non-negative number"):
            HistogramLoss(num_bins=10, bin_min=0.0, bin_max=1.0, padding=-2.0)
            
        # Test invalid reduction
        with pytest.raises(ValueError, match="reduction must be"):
            HistogramLoss(num_bins=10, bin_min=0.0, bin_max=1.0, reduction='invalid')

    def test_gaussian_distribution(self):
        loss_fn = HistogramLoss(num_bins=10, bin_min=0.0, bin_max=1.0)
        batch_size = 5
        
        # Create dummy predictions and targets
        predictions = torch.randn(batch_size, 10)
        targets = torch.rand(batch_size)
        
        # Ensure forward pass works without errors
        loss = loss_fn(predictions, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1  # Should be scalar with 'mean' reduction
        
        # Test distribution is properly computed
        target_probs = loss_fn.compute_gaussian_target_probs(targets)
        assert target_probs.shape == (batch_size, 10)
        # Check probabilities sum to approximately 1
        assert torch.allclose(target_probs.sum(dim=1), torch.ones(batch_size), atol=1e-5)

    def test_onebin_distribution(self):
        loss_fn = HistogramLoss(num_bins=10, bin_min=0.0, bin_max=1.0, target_distribution='one-bin')
        batch_size = 5
        
        # Create dummy predictions and targets
        predictions = torch.randn(batch_size, 10)
        targets = torch.rand(batch_size)
        
        # Ensure forward pass works without errors
        loss = loss_fn(predictions, targets)
        assert isinstance(loss, torch.Tensor)
        
        # Test one-bin distribution is properly computed
        target_probs = loss_fn.compute_onebin_target_probs(targets)
        assert target_probs.shape == (batch_size, 10)
        # Each row should have exactly one 1.0 value
        assert torch.all(target_probs.sum(dim=1) == 1.0)
        # Each row should be one-hot
        assert torch.all((target_probs == 0) | (target_probs == 1))

    def test_reductions(self):
        batch_size = 3
        predictions = torch.randn(batch_size, 5)
        targets = torch.rand(batch_size)
        
        # Test 'none' reduction
        loss_fn_none = HistogramLoss(num_bins=5, bin_min=0.0, bin_max=1.0, reduction='none')
        loss_none = loss_fn_none(predictions, targets)
        assert loss_none.shape == (batch_size,)
        
        # Test 'mean' reduction
        loss_fn_mean = HistogramLoss(num_bins=5, bin_min=0.0, bin_max=1.0, reduction='mean')
        loss_mean = loss_fn_mean(predictions, targets)
        assert loss_mean.numel() == 1
        assert torch.isclose(loss_mean, loss_none.mean())
        
        # Test 'sum' reduction
        loss_fn_sum = HistogramLoss(num_bins=5, bin_min=0.0, bin_max=1.0, reduction='sum')
        loss_sum = loss_fn_sum(predictions, targets)
        assert loss_sum.numel() == 1
        assert torch.isclose(loss_sum, loss_none.sum())

    def test_input_validation(self):
        loss_fn = HistogramLoss(num_bins=10, bin_min=0.0, bin_max=1.0)
        
        # Test invalid input types
        with pytest.raises(TypeError, match="must be PyTorch tensors"):
            loss_fn(np.zeros((5, 10)), torch.zeros(5))
            
        with pytest.raises(TypeError, match="must be PyTorch tensors"):
            loss_fn(torch.zeros(5, 10), np.zeros(5))
        
        # Test wrong prediction shape
        with pytest.raises(ValueError, match="predictions must have shape"):
            loss_fn(torch.zeros(5, 5), torch.zeros(5))
        
        # Test batch size mismatch
        with pytest.raises(ValueError, match="batch size of predictions"):
            loss_fn(torch.zeros(5, 10), torch.zeros(3))
        
        # Test wrong target dimensions
        with pytest.raises(ValueError, match="targets must have shape"):
            loss_fn(torch.zeros(5, 10), torch.zeros(5, 1))

    def test_numerical_correctness(self):
        # Simple case with known output
        loss_fn = HistogramLoss(num_bins=2, bin_min=0.0, bin_max=2.0, target_distribution='one-bin')
        
        # Perfectly predicted case (first example in bin 0, second in bin 1)
        predictions = torch.tensor([[10.0, -10.0], [-10.0, 10.0]], dtype=torch.float32)
        targets = torch.tensor([0.5, 1.5], dtype=torch.float32)  # Middle of each bin
        
        # Expected behavior: softmax of predictions will be very close to [1,0] and [0,1]
        # One-hot encoded targets will be [1,0] and [0,1], so loss should be very small
        loss = loss_fn(predictions, targets)
        assert loss < 0.01  # Loss should be very small for accurate predictions

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_cuda(self):
        device = torch.device('cuda')
        loss_fn = HistogramLoss(num_bins=10, bin_min=0.0, bin_max=1.0, device=device)
        
        # Check that internal tensors are on the correct device
        assert loss_fn.bin_centers.device.type == 'cuda'
        
        # Test with tensors on CUDA
        predictions = torch.randn(5, 10, device=device)
        targets = torch.rand(5, device=device)
        
        loss = loss_fn(predictions, targets)
        assert loss.device.type == 'cuda'
        
        # Test with tensors on CPU (should be moved to GPU)
        predictions_cpu = torch.randn(5, 10)
        targets_cpu = torch.rand(5)
        
        loss = loss_fn(predictions_cpu, targets_cpu)
        assert loss.device.type == 'cuda'

    def test_edge_cases(self):
        # Test with extreme target values
        loss_fn = HistogramLoss(num_bins=10, bin_min=0.0, bin_max=1.0)
        predictions = torch.randn(3, 10)
        
        # Test with targets at boundaries
        boundary_targets = torch.tensor([0.0, 0.5, 1.0])
        loss = loss_fn(predictions, boundary_targets)
        assert not torch.isnan(loss)
        
        # Test with targets outside boundaries with Gaussian
        outside_targets = torch.tensor([-1.0, 2.0, 0.5])
        loss = loss_fn(predictions, outside_targets)
        assert not torch.isnan(loss)
        
        # Test with targets outside boundaries with one-bin
        loss_fn_onebin = HistogramLoss(num_bins=10, bin_min=0.0, bin_max=1.0, target_distribution='one-bin')
        loss = loss_fn_onebin(predictions, outside_targets)
        assert not torch.isnan(loss)
