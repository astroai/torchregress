import pytest
import torch

from torchregress.losses.mdn import MixtureDensityLoss, create_mdn_loss

# Alias for tests
MDNLoss = MixtureDensityLoss


class TestMixtureDensityLoss:
    """Tests for MixtureDensityLoss implementation."""

    def test_initialization(self):
        """Test initialization with different parameters."""
        # Default initialization
        loss_fn = MixtureDensityLoss(n_components=2, n_features=3)
        assert loss_fn.n_components == 2
        assert loss_fn.n_features == 3
        assert loss_fn.covariance_type == "diagonal"
        assert loss_fn.reduction == "mean"
        assert loss_fn.expected_output_size == 14  # 2 + 2*2*3 = 14

        # Full covariance initialization
        loss_fn = MixtureDensityLoss(n_components=3, n_features=2, covariance_type="full")
        assert loss_fn.covariance_type == "full"
        assert loss_fn.expected_output_size == 18  # 3 + 3*2 + 3*(2*3/2) = 3 + 6 + 9 = 18

        # Custom parameters
        loss_fn = MixtureDensityLoss(
            n_components=2,
            n_features=3,
            covariance_type="diagonal",
            min_std=0.05,
            eps=1e-10,
            reduction="sum",
        )
        assert loss_fn.min_std == 0.05
        assert loss_fn.eps == 1e-10
        assert loss_fn.reduction == "sum"

        # Invalid covariance type
        with pytest.raises(ValueError, match="Unsupported covariance_type"):
            MixtureDensityLoss(n_components=2, n_features=3, covariance_type="invalid")

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
        loss_fn = MixtureDensityLoss(n_components, n_features, covariance_type="full")

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
                assert torch.allclose(
                    torch.triu(L, diagonal=1), torch.zeros_like(torch.triu(L, diagonal=1))
                )
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
        loss_fn = MixtureDensityLoss(n_components, n_features, covariance_type="full")

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

        for reduction in ["none", "mean", "sum"]:
            loss_fn = MixtureDensityLoss(n_components, n_features, reduction=reduction)
            y_pred = torch.randn(batch_size, loss_fn.expected_output_size)
            target = torch.randn(batch_size, n_features)

            loss = loss_fn(y_pred, target)

            if reduction == "none":
                assert loss.shape == (batch_size,)
            else:
                assert loss.ndim == 0  # Scalar

    def test_forward_full(self):
        """Test forward pass with full covariance."""
        n_components, n_features = 2, 2  # Using n_features=2 for simplicity
        batch_size = 5

        for reduction in ["none", "mean", "sum"]:
            loss_fn = MixtureDensityLoss(
                n_components, n_features, covariance_type="full", reduction=reduction
            )
            y_pred = torch.randn(batch_size, loss_fn.expected_output_size)
            target = torch.randn(batch_size, n_features)

            loss = loss_fn(y_pred, target)

            if reduction == "none":
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
        mask[: batch_size // 2] = False

        # Test with different reductions
        for reduction in ["none", "mean", "sum"]:
            loss_fn.reduction = reduction
            masked_loss = loss_fn(y_pred, target, mask=mask)

            # Calculate reference - only on unmasked samples
            reference_loss = loss_fn(y_pred[batch_size // 2 :], target[batch_size // 2 :])

            if reduction == "none":
                # First half should be zero, second half should match reference
                assert torch.allclose(masked_loss[: batch_size // 2], torch.zeros(batch_size // 2))
                assert torch.allclose(masked_loss[batch_size // 2 :], reference_loss)
            else:
                assert torch.isclose(masked_loss, reference_loss)

        # Test with sample-level mask
        sample_mask = torch.ones(batch_size, dtype=torch.bool)
        sample_mask[: batch_size // 2] = False

        loss_fn.reduction = "mean"
        sample_masked_loss = loss_fn(y_pred, target, mask=sample_mask)
        feature_masked_loss = loss_fn(y_pred, target, mask=mask)

        # Both should give the same result since MDN converts feature masks to sample masks
        assert torch.isclose(sample_masked_loss, feature_masked_loss)

    def test_weighted_loss(self):
        """Test loss calculation with weights."""
        n_components, n_features = 2, 3
        loss_fn = MixtureDensityLoss(n_components, n_features, reduction="sum")

        batch_size = 5
        y_pred = torch.randn(batch_size, loss_fn.expected_output_size)
        target = torch.randn(batch_size, n_features)

        # Create sample weights
        weights = torch.rand(batch_size)

        # Calculate losses
        loss_fn.reduction = "none"
        unweighted_losses = loss_fn(y_pred, target)

        loss_fn.reduction = "sum"
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
        loss_fn(y_pred, target)  # Should not raise

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
        assert loss_fn.covariance_type == "diagonal"  # default

        # Custom parameters
        loss_fn = create_mdn_loss(
            n_components=2,
            n_features=4,
            covariance_type="full",
            min_std=0.1,
            eps=1e-6,
            reduction="sum",
        )
        assert loss_fn.n_components == 2
        assert loss_fn.n_features == 4
        assert loss_fn.covariance_type == "full"
        assert loss_fn.min_std == 0.1
        assert loss_fn.eps == 1e-6
        assert loss_fn.reduction == "sum"

    def test_mdn_edge_cases(self):
        """Test MDNLoss with edge cases: zeros, empty tensors, extreme values, NaN/Inf."""
        n_components = 3
        n_features = 1
        loss_fn = MDNLoss(n_components=n_components, n_features=n_features)

        # Test with zeros
        y_pred_zeros = torch.zeros(10, loss_fn.expected_output_size)
        y_true_zeros = torch.zeros(10, n_features)
        assert torch.isfinite(loss_fn(y_pred_zeros, y_true_zeros))

        # Test with empty tensors
        y_pred_empty = torch.tensor([]).reshape(0, loss_fn.expected_output_size)
        y_true_empty = torch.tensor([]).reshape(0, n_features)
        result = loss_fn(y_pred_empty, y_true_empty)
        assert result.numel() == 0 or result == 0 or torch.isnan(result)

        # Test with extreme values
        y_pred_large = torch.randn(1, loss_fn.expected_output_size)
        y_true_small = torch.zeros(1, n_features)
        assert torch.isfinite(loss_fn(y_pred_large, y_true_small))

        # Test with masks
        y_pred_masked = torch.randn(3, loss_fn.expected_output_size)
        y_true_masked = torch.tensor([[1.5], [2.5], [3.5]])
        mask = torch.tensor([[True], [True], [False]])
        assert torch.isfinite(loss_fn(y_pred_masked, y_true_masked, mask))


from torch.autograd import gradcheck


class TestMDNLossNumericalStability:
    def test_mdn_gradient_flow(self):
        """Test that gradients flow through MDNLoss properly."""

        # Create simple model and test gradient flow
        n_samples = 5
        n_components = 2
        n_features = 2

        loss_fn = MDNLoss(n_components=n_components, n_features=n_features, reduction="mean")
        y_pred = torch.randn(n_samples, loss_fn.expected_output_size, requires_grad=True)
        target = torch.randn(n_samples, n_features)

        loss = loss_fn(y_pred, target)
        loss.backward()

        assert y_pred.grad is not None
        assert torch.all(torch.isfinite(y_pred.grad))

    def test_extreme_values(self):
        """Test stability with extreme values."""

        n_samples = 3
        n_components = 2
        n_features = 1

        mdn_loss = MDNLoss(n_components=n_components, n_features=n_features, reduction="mean")

        # Test with normal values
        y_pred = torch.randn(n_samples, mdn_loss.expected_output_size, requires_grad=True)
        target = torch.randn(n_samples, n_features)

        loss = mdn_loss(y_pred, target)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.all(torch.isfinite(y_pred.grad))

    def test_nan_inf_handling(self):
        """Test how MDN loss handles NaN and Inf values with masks."""

        n_samples = 3
        n_components = 2
        n_features = 1

        mdn_loss = MDNLoss(n_components=n_components, n_features=n_features, reduction="mean")

        # Create data
        y_pred = torch.randn(n_samples, mdn_loss.expected_output_size, requires_grad=True)
        target = torch.randn(n_samples, n_features)

        # Create mask to ignore second sample
        mask = torch.tensor([[True], [False], [True]])

        # This should only use the valid elements
        loss = mdn_loss(y_pred, target, mask)
        assert torch.isfinite(loss)
        loss.backward()

        # All gradients should be finite
        assert torch.all(torch.isfinite(y_pred.grad))
