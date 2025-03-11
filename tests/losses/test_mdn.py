import pytest
import torch
from torchregression.losses.mdn import MixtureDensityLoss, create_mdn_loss


class TestMixtureDensityLoss:
    """Tests for MixtureDensityLoss implementation."""
    
    def test_initialization(self):
        """Test initialization with different parameters."""
        # Default initialization
        loss_fn = MixtureDensityLoss(n_components=2, n_features=3)
        assert loss_fn.n_components == 2
        assert loss_fn.n_features == 3
        assert loss_fn.covariance_type == 'diagonal'
        assert loss_fn.reduction == 'mean'
        assert loss_fn.expected_output_size == 14  # 2 + 2*2*3 = 14
        
        # Full covariance initialization
        loss_fn = MixtureDensityLoss(n_components=3, n_features=2, covariance_type='full')
        assert loss_fn.covariance_type == 'full'
        assert loss_fn.expected_output_size == 12  # 3 + 3*2 + 3*2*3/2 = 12
        
        # Custom parameters
        loss_fn = MixtureDensityLoss(
            n_components=2, 
            n_features=3, 
            covariance_type='diagonal',
            min_std=0.05, 
            eps=1e-10, 
            reduction='sum'
        )
        assert loss_fn.min_std == 0.05
        assert loss_fn.eps == 1e-10
        assert loss_fn.reduction == 'sum'
        
        # Invalid covariance type
        with pytest.raises(ValueError, match="Unsupported covariance_type"):
            MixtureDensityLoss(n_components=2, n_features=3, covariance_type='invalid')

    def test_extract_distribution_parameters_diagonal(self):
        """Test parameter extraction for diagonal covariance."""
        n_components, n_features = 2, 3
        loss_fn = MixtureDensityLoss(n_components, n_features)
        
        # Create dummy prediction tensor
        batch_size = 5
        y_pred = torch.randn(batch_size, loss_fn.expected_output_size)
        
        weights, means, stds = loss_fn._extract_distribution_parameters(y_pred)
        
        # Check shapes
        assert weights.shape == (batch_size, n_components)
        assert means.shape == (batch_size, n_components, n_features)
        assert stds.shape == (batch_size, n_components, n_features)
        
        # Weights should sum to 1
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size))
        
        # Standard deviations should be positive and at least min_std
        assert (stds >= loss_fn.min_std).all()
        
        # Test with incorrect output size
        with pytest.raises(ValueError, match="Model output size .* doesn't match expected size"):
            loss_fn._extract_distribution_parameters(torch.randn(batch_size, 10))

    def test_extract_distribution_parameters_full(self):
        """Test parameter extraction for full covariance."""
        n_components, n_features = 2, 3
        loss_fn = MixtureDensityLoss(n_components, n_features, covariance_type='full')
        
        # Create dummy prediction tensor
        batch_size = 5
        y_pred = torch.randn(batch_size, loss_fn.expected_output_size)
        
        weights, means, L_matrices = loss_fn._extract_distribution_parameters(y_pred)
        
        # Check shapes
        assert weights.shape == (batch_size, n_components)
        assert means.shape == (batch_size, n_components, n_features) 
        assert L_matrices.shape == (batch_size, n_components, n_features, n_features)
        
        # Weights should sum to 1
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size))
        
        # Check that matrices are lower triangular with positive diagonal
        for b in range(batch_size):
            for c in range(n_components):
                L = L_matrices[b, c]
                # Upper triangular part (excluding diagonal) should be zero
                assert torch.allclose(torch.triu(L, diagonal=1), torch.zeros_like(torch.triu(L, diagonal=1)))
                # Diagonal elements should be positive
                assert (torch.diagonal(L) > 0).all()

    def test_log_prob_diagonal(self):
        """Test log probability calculation with diagonal covariance."""
        n_components, n_features = 2, 3
        loss_fn = MixtureDensityLoss(n_components, n_features)
        
        batch_size = 5
        target = torch.randn(batch_size, n_features)
        means = torch.randn(batch_size, n_components, n_features)
        stds = torch.rand(batch_size, n_components, n_features) + 0.1  # ensure positive
        
        log_probs = loss_fn._log_prob_diagonal(target, means, stds)
        
        # Check shape
        assert log_probs.shape == (batch_size, n_components)
        
        # Verify calculation for one example
        b, c = 0, 0  # batch and component indices for verification
        y = target[b]
        mu = means[b, c]
        sigma = stds[b, c]
        
        # Manual calculation
        norm_dist = (y - mu) / sigma
        exponent = -0.5 * torch.sum(norm_dist**2)
        log_det = -torch.sum(torch.log(sigma + loss_fn.eps))
        const = -0.5 * n_features * loss_fn.log_2pi
        expected_log_prob = exponent + log_det + const
        
        assert torch.isclose(log_probs[b, c], expected_log_prob)

    def test_log_prob_full(self):
        """Test log probability calculation with full covariance."""
        n_components, n_features = 2, 2  # Using n_features=2 for simplicity
        loss_fn = MixtureDensityLoss(n_components, n_features, covariance_type='full')
        
        batch_size = 3
        target = torch.randn(batch_size, n_features)
        means = torch.randn(batch_size, n_components, n_features)
        
        # Create valid lower triangular matrices
        L_matrices = torch.zeros(batch_size, n_components, n_features, n_features)
        for b in range(batch_size):
            for c in range(n_components):
                L = torch.tril(torch.randn(n_features, n_features))
                L.diagonal().copy_(torch.exp(torch.randn(n_features)))  # Positive diagonal
                L_matrices[b, c] = L
        
        log_probs = loss_fn._log_prob_full(target, means, L_matrices)
        
        # Check shape
        assert log_probs.shape == (batch_size, n_components)
        
        # Verify calculation for one example
        b, c = 0, 0  # batch and component indices for verification
        y = target[b]
        mu = means[b, c]
        L = L_matrices[b, c]
        
        # Manual calculation
        residual = y - mu
        z = torch.triangular_solve(residual.unsqueeze(1), L, upper=False)[0].squeeze(1)
        quad_term = torch.sum(z**2)
        log_det = 2 * torch.sum(torch.log(torch.diagonal(L) + loss_fn.eps))
        const_term = n_features * loss_fn.log_2pi
        expected_log_prob = -0.5 * (quad_term + log_det + const_term)
        
        assert torch.isclose(log_probs[b, c], expected_log_prob, rtol=1e-4)

    def test_forward_diagonal(self):
        """Test forward pass with diagonal covariance."""
        n_components, n_features = 2, 3
        batch_size = 5
        
        for reduction in ['none', 'mean', 'sum']:
            loss_fn = MixtureDensityLoss(n_components, n_features, reduction=reduction)
            y_pred = torch.randn(batch_size, loss_fn.expected_output_size)
            target = torch.randn(batch_size, n_features)
            
            loss = loss_fn(y_pred, target)
            
            if reduction == 'none':
                assert loss.shape == (batch_size,)
            else:
                assert loss.ndim == 0  # Scalar

    def test_forward_full(self):
        """Test forward pass with full covariance."""
        n_components, n_features = 2, 2  # Using n_features=2 for simplicity
        batch_size = 5
        
        for reduction in ['none', 'mean', 'sum']:
            loss_fn = MixtureDensityLoss(
                n_components, n_features, 
                covariance_type='full', 
                reduction=reduction
            )
            y_pred = torch.randn(batch_size, loss_fn.expected_output_size)
            target = torch.randn(batch_size, n_features)
            
            loss = loss_fn(y_pred, target)
            
            if reduction == 'none':
                assert loss.shape == (batch_size,)
            else:
                assert loss.ndim == 0  # Scalar

    def test_masked_loss(self):
        """Test loss calculation with masks."""
        n_components, n_features = 2, 3
        loss_fn = MixtureDensityLoss(n_components, n_features)
        
        batch_size = 6
        y_pred = torch.randn(batch_size, loss_fn.expected_output_size)
        target = torch.randn(batch_size, n_features)
        
        # Create a mask where half the samples are masked
        mask = torch.ones(batch_size, n_features, dtype=torch.bool)
        mask[:batch_size//2] = False
        
        # Test with different reductions
        for reduction in ['none', 'mean', 'sum']:
            loss_fn.reduction = reduction
            masked_loss = loss_fn(y_pred, target, mask=mask)
            
            # Calculate reference - only on unmasked samples
            reference_loss = loss_fn(
                y_pred[batch_size//2:], 
                target[batch_size//2:]
            )
            
            if reduction == 'none':
                # First half should be zero, second half should match reference
                assert torch.allclose(masked_loss[:batch_size//2], torch.zeros(batch_size//2))
                assert torch.allclose(masked_loss[batch_size//2:], reference_loss)
            else:
                assert torch.isclose(masked_loss, reference_loss)
        
        # Test with sample-level mask
        sample_mask = torch.ones(batch_size, dtype=torch.bool)
        sample_mask[:batch_size//2] = False
        
        loss_fn.reduction = 'mean'
        sample_masked_loss = loss_fn(y_pred, target, mask=sample_mask)
        feature_masked_loss = loss_fn(y_pred, target, mask=mask)
        
        # Both should give the same result since MDN converts feature masks to sample masks
        assert torch.isclose(sample_masked_loss, feature_masked_loss)

    def test_weighted_loss(self):
        """Test loss calculation with weights."""
        n_components, n_features = 2, 3
        loss_fn = MixtureDensityLoss(n_components, n_features, reduction='sum')
        
        batch_size = 5
        y_pred = torch.randn(batch_size, loss_fn.expected_output_size)
        target = torch.randn(batch_size, n_features)
        
        # Create sample weights
        weights = torch.rand(batch_size)
        
        # Calculate losses
        loss_fn.reduction = 'none'
        unweighted_losses = loss_fn(y_pred, target)
        
        loss_fn.reduction = 'sum'
        weighted_loss = loss_fn(y_pred, target, weights=weights)
        
        # Manual weighting
        expected_weighted_loss = (unweighted_losses * weights).sum()
        assert torch.isclose(weighted_loss, expected_weighted_loss)
        
        # Test with feature-level weights
        feature_weights = torch.rand(batch_size, n_features)
        feature_weighted_loss = loss_fn(y_pred, target, weights=feature_weights)
        
        # Should use average of feature weights as sample weights
        expected_feature_weighted_loss = (unweighted_losses * feature_weights.mean(dim=-1)).sum()
        assert torch.isclose(feature_weighted_loss, expected_feature_weighted_loss)

    def test_input_validation(self):
        """Test input validation."""
        n_components, n_features = 2, 3
        loss_fn = MixtureDensityLoss(n_components, n_features)
        
        # Valid inputs
        y_pred = torch.randn(5, loss_fn.expected_output_size)
        target = torch.randn(5, n_features)
        loss = loss_fn(y_pred, target)  # Should not raise
        
        # Invalid target shape
        invalid_target = torch.randn(5, n_features + 1)
        with pytest.raises(ValueError, match=f"Expected {n_features} features"):
            loss_fn(y_pred, invalid_target)
            
        # Invalid prediction shape
        invalid_pred = torch.randn(5, loss_fn.expected_output_size + 1)
        with pytest.raises(ValueError, match="Model output size .* doesn't match expected size"):
            loss_fn(invalid_pred, target)

    def test_factory_function(self):
        """Test create_mdn_loss factory function."""
        # Basic initialization
        loss_fn = create_mdn_loss(n_components=3, n_features=2)
        assert isinstance(loss_fn, MixtureDensityLoss)
        assert loss_fn.n_components == 3
        assert loss_fn.n_features == 2
        assert loss_fn.covariance_type == 'diagonal'  # default
        
        # Custom parameters
        loss_fn = create_mdn_loss(
            n_components=2,
            n_features=4,
            covariance_type='full',
            min_std=0.1,
            eps=1e-6,
            reduction='sum'
        )
        assert loss_fn.n_components == 2
        assert loss_fn.n_features == 4
        assert loss_fn.covariance_type == 'full'
        assert loss_fn.min_std == 0.1
        assert loss_fn.eps == 1e-6
        assert loss_fn.reduction == 'sum'

    def test_mdn_edge_cases(self):
        """Test MDNLoss with edge cases: zeros, empty tensors, extreme values, NaN/Inf."""
        n_components = 3
        loss_fn = MDNLoss(n_components=n_components)
        
        # Create appropriate shapes for MDN: [batch, 3*n_components]
        # Each component has (weight, mean, log_std)
        
        # Test with zeros
        y_pred_zeros = torch.zeros(10, 3 * n_components)
        # Adjust weights to be valid (sum to 1)
        y_pred_zeros[:, :n_components] = 1.0 / n_components
        y_true_zeros = torch.zeros(10)
        assert torch.isfinite(loss_fn(y_pred_zeros, y_true_zeros))
        
        # Test with empty tensors
        y_pred_empty = torch.tensor([]).reshape(0, 3 * n_components)
        y_true_empty = torch.tensor([])
        assert loss_fn(y_pred_empty, y_true_empty).numel() == 0
        
        # Test with extreme values
        y_pred_large = torch.zeros(1, 3 * n_components)
        y_pred_large[0, :n_components] = 1.0 / n_components  # Equal weights
        y_pred_large[0, n_components] = 1e10                # One large mean
        y_true_small = torch.tensor([0.0])
        assert torch.isfinite(loss_fn(y_pred_large, y_true_small))
        
        # Test with very small/large variances
        y_pred_var = torch.zeros(1, 3 * n_components)
        y_pred_var[0, :n_components] = 1.0 / n_components   # Equal weights
        y_pred_var[0, n_components:2*n_components] = 1.0    # Normal means
        y_pred_var[0, 2*n_components:] = -20.0             # Very small variances
        y_true_var = torch.tensor([1.0])
        assert torch.isfinite(loss_fn(y_pred_var, y_true_var))
        
        # Test with NaN/Inf and masks
        y_pred_nan = torch.zeros(3, 3 * n_components)
        y_pred_nan[:, :n_components] = 1.0 / n_components   # Equal weights
        y_pred_nan[1, n_components] = float('nan')          # NaN in one mean
        y_true_nan = torch.tensor([1.5, 2.5, float('inf')])
        mask = torch.tensor([True, False, False])
        assert torch.isfinite(loss_fn(y_pred_nan, y_true_nan, mask))

import torch
import pytest
from torch.autograd import gradcheck

class TestMDNLossNumericalStability:
    def test_mdn_gradient_flow(self):
        """Test that gradients flow through MDNLoss properly."""
        from torchregression.losses.mdn import MDNLoss
        
        # Create batched input: batch_size=5, n_components=3, n_dims=2
        n_samples = 5
        n_components = 3
        n_dims = 2
        
        # Create params requiring gradients (means, scales, weights)
        means = torch.randn(n_samples, n_components, n_dims, requires_grad=True, dtype=torch.double)
        scales = torch.exp(torch.randn(n_samples, n_components, n_dims, requires_grad=True, dtype=torch.double))
        weights = torch.randn(n_samples, n_components, requires_grad=True, dtype=torch.double)
        
        # Create target
        y_true = torch.randn(n_samples, n_dims, dtype=torch.double)
        
        # Define a function for gradcheck
        def loss_fn(m, s, w, y):
            return MDNLoss(reduction='mean')(m, s, w, y)
        
        # Test with gradcheck
        assert gradcheck(loss_fn, (means, scales, weights, y_true), eps=1e-6, atol=1e-4)
    
    def test_extreme_values(self):
        """Test stability with extreme values."""
        from torchregression.losses.mdn import MDNLoss
        
        n_samples = 3
        n_components = 2
        n_dims = 1
        
        # Normal values
        means = torch.tensor([[[0.0], [1.0]], [[2.0], [3.0]], [[4.0], [5.0]]], requires_grad=True)
        
        # Very small scales - potential numerical issue
        scales_small = torch.tensor([[[1e-8], [1e-8]], [[1e-8], [1e-8]], [[1e-8], [1e-8]]], requires_grad=True)
        
        # Very large scales - potential numerical issue
        scales_large = torch.tensor([[[1e8], [1e8]], [[1e8], [1e8]], [[1e8], [1e8]]], requires_grad=True)
        
        # Equal weight components
        weights = torch.ones(n_samples, n_components, requires_grad=True)
        
        # Target values
        y_true = torch.tensor([[0.1], [2.1], [4.1]])
        
        mdn_loss = MDNLoss(reduction='mean')
        
        # Test with small scales
        loss = mdn_loss(means, scales_small, weights, y_true)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.all(torch.isfinite(means.grad))
        assert torch.all(torch.isfinite(scales_small.grad))
        assert torch.all(torch.isfinite(weights.grad))
        
        # Reset gradients
        means.grad = None
        weights.grad = None
        
        # Test with large scales
        loss = mdn_loss(means, scales_large, weights, y_true)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.all(torch.isfinite(means.grad))
        assert torch.all(torch.isfinite(scales_large.grad))
        assert torch.all(torch.isfinite(weights.grad))
    
    def test_nan_inf_handling(self):
        """Test how MDN loss handles NaN and Inf values with masks."""
        from torchregression.losses.mdn import MDNLoss
        
        n_samples = 3
        n_components = 2
        n_dims = 1
        
        # Create data with some NaNs
        means = torch.tensor([[[0.0], [1.0]], [[float('nan')], [3.0]], [[4.0], [5.0]]], requires_grad=True)
        scales = torch.tensor([[[1.0], [1.0]], [[1.0], [1.0]], [[1.0], [1.0]]], requires_grad=True)
        weights = torch.ones(n_samples, n_components, requires_grad=True)
        y_true = torch.tensor([[0.1], [2.1], [4.1]])
        
        # Create mask to ignore NaN sample
        mask = torch.tensor([True, False, True])
        
        mdn_loss = MDNLoss(reduction='mean')
        
        # This should only use the valid elements
        loss = mdn_loss(means, scales, weights, y_true, mask=mask)
        assert torch.isfinite(loss)
        loss.backward()
        
        # Masked elements should have zero gradient
        assert torch.all(means.grad[1] == 0.0)
        # Other elements should have finite gradients
        assert torch.all(torch.isfinite(means.grad[0]))
        assert torch.all(torch.isfinite(means.grad[2]))
    
    def test_reduction_modes(self):
        """Test different reduction modes for backward pass."""
        from torchregression.losses.mdn import MDNLoss
        
        n_samples = 5
        n_components = 3
        n_dims = 2
        
        means = torch.randn(n_samples, n_components, n_dims, requires_grad=True)
        scales = torch.exp(torch.randn(n_samples, n_components, n_dims, requires_grad=True))
        weights = torch.randn(n_samples, n_components, requires_grad=True)
        y_true = torch.randn(n_samples, n_dims)
        
        # Test mean reduction
        mdn_mean = MDNLoss(reduction='mean')
        loss = mdn_mean(means, scales, weights, y_true)
        loss.backward()
        mean_grad_means = means.grad.clone()
        mean_grad_scales = scales.grad.clone()
        mean_grad_weights = weights.grad.clone()
        
        # Reset gradients
        means.grad = None
        scales.grad = None
        weights.grad = None
        
        # Test sum reduction
        mdn_sum = MDNLoss(reduction='sum')
        loss = mdn_sum(means, scales, weights, y_true)
        loss.backward()
        sum_grad_means = means.grad.clone()
        sum_grad_scales = scales.grad.clone()
        sum_grad_weights = weights.grad.clone()
        
        # Reset gradients
        means.grad = None
        scales.grad = None
        weights.grad = None
        
        # Test none reduction
        mdn_none = MDNLoss(reduction='none')
        loss = mdn_none(means, scales, weights, y_true)
        loss.mean().backward()
        none_grad_means = means.grad.clone()
        none_grad_scales = scales.grad.clone()
        none_grad_weights = weights.grad.clone()
        
        # Mean and sum should give different gradients
        assert not torch.allclose(mean_grad_means, sum_grad_means)
        # Mean and manual mean over none should be similar
        assert torch.allclose(mean_grad_means, none_grad_means)
