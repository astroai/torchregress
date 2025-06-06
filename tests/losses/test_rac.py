import pytest
import torch
import numpy as np
from torchregress.losses.rag import (
    BinnedRegressionLoss,
    StandardClassificationRegressionLoss,
    OrdinalRegressionLoss,
    HistogramRegressionLoss,
    RegressionAsClassificationLoss,
)


class TestBinnedRegressionLoss:
    """Tests for the base BinnedRegressionLoss class."""

    def test_init_valid_params(self):
        """Test initialization with different valid parameters."""
        # Test initialization with default parameters
        loss_fn = BinnedRegressionLoss(bins=10, min_value=0.0, max_value=1.0)
        assert loss_fn.n_bins == 10
        assert loss_fn.bin_edges[0].item() == 0.0
        assert loss_fn.bin_edges[-1].item() == 1.0
        assert loss_fn.soft_targets is True
        assert loss_fn.sigma == 0.1
        assert loss_fn.reduction == "mean"

        # Test with custom bins tensor
        bins = torch.linspace(-1.0, 1.0, 11)
        loss_fn = BinnedRegressionLoss(bins=bins)
        assert loss_fn.n_bins == 10
        assert torch.allclose(loss_fn.bin_edges, bins)

        # Test with different parameters
        loss_fn = BinnedRegressionLoss(
            bins=20,
            min_value=-1.0,
            max_value=1.0,
            soft_targets=False,
            sigma=0.5,
            reduction="sum",
            extrapolate_beyond_bins=True,
            noise_aware=True,
            adaptive_sigma=True,
            normalize_targets=False,
        )
        assert loss_fn.n_bins == 20
        assert loss_fn.bin_edges[0].item() == -1.0
        assert loss_fn.bin_edges[-1].item() == 1.0
        assert loss_fn.soft_targets is False
        assert loss_fn.sigma == 0.5
        assert loss_fn.reduction == "sum"
        assert loss_fn.extrapolate_beyond_bins is True
        assert loss_fn.noise_aware is True
        assert loss_fn.adaptive_sigma is True
        assert loss_fn.normalize_targets is False

    def test_bin_properties(self):
        """Test that bin centers and widths are correctly calculated."""
        # Create bins with known properties
        loss_fn = BinnedRegressionLoss(bins=5, min_value=0.0, max_value=1.0)

        # Expected bin edges: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        expected_edges = torch.linspace(0.0, 1.0, 6)
        assert torch.allclose(loss_fn.bin_edges, expected_edges)

        # Expected bin centers: [0.1, 0.3, 0.5, 0.7, 0.9]
        expected_centers = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        assert torch.allclose(loss_fn.bin_centers, expected_centers)

        # Expected bin widths: [0.2, 0.2, 0.2, 0.2, 0.2]
        expected_widths = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
        assert torch.allclose(loss_fn.bin_widths, expected_widths)

        # Test with non-uniform bins
        custom_edges = torch.tensor([0.0, 0.1, 0.3, 0.7, 1.0])
        loss_fn = BinnedRegressionLoss(bins=custom_edges)

        # Expected bin centers: [0.05, 0.2, 0.5, 0.85]
        expected_centers = torch.tensor([0.05, 0.2, 0.5, 0.85])
        assert torch.allclose(loss_fn.bin_centers, expected_centers)

        # Expected bin widths: [0.1, 0.2, 0.4, 0.3]
        expected_widths = torch.tensor([0.1, 0.2, 0.4, 0.3])
        assert torch.allclose(loss_fn.bin_widths, expected_widths)

    def test_handle_out_of_range(self):
        """Test handling of values outside the bin range."""
        # Test with extrapolation enabled
        loss_fn = BinnedRegressionLoss(
            bins=5, min_value=0.0, max_value=1.0, extrapolate_beyond_bins=True
        )
        target = torch.tensor([[-0.5], [0.2], [1.5]])
        result = loss_fn._handle_out_of_range(target)
        # Should keep values as is when extrapolating
        assert torch.allclose(result, target)

        # Test with extrapolation disabled (default)
        loss_fn = BinnedRegressionLoss(
            bins=5, min_value=0.0, max_value=1.0, extrapolate_beyond_bins=False
        )
        target = torch.tensor([[-0.5], [0.2], [1.5]])
        result = loss_fn._handle_out_of_range(target)
        # Should clamp values to range
        expected = torch.tensor([[0.0], [0.2], [1.0]])
        assert torch.allclose(result, expected)

    def test_get_target_distribution_soft(self):
        """Test generation of soft target distributions."""
        loss_fn = BinnedRegressionLoss(bins=3, min_value=0.0, max_value=1.0, soft_targets=True)
        target = torch.tensor([[0.0], [0.5], [1.0]])

        # Test soft targets
        distributions = loss_fn._get_target_distribution(target)
        assert distributions.shape == (3, 3)  # (batch_size, n_bins)

        # Highest probability should be at the bin containing the target
        max_probs = torch.argmax(distributions, dim=1)
        # For 3 bins [0-0.33, 0.33-0.66, 0.66-1.0], expect targets in bins [0, 1, 2]
        expected_bins = torch.tensor([0, 1, 2])
        assert torch.allclose(max_probs, expected_bins)

        # Probabilities should sum to 1 for each target
        sums = distributions.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums))

    def test_get_target_distribution_hard(self):
        """Test generation of hard target distributions (bin indices)."""
        loss_fn = BinnedRegressionLoss(bins=3, min_value=0.0, max_value=1.0, soft_targets=False)
        target = torch.tensor([[0.0], [0.5], [1.0]])

        # Test hard targets
        bin_indices = loss_fn._get_target_distribution(target)
        assert bin_indices.shape == (3,)  # batch_size

        # Expected bin indices for targets [0.0, 0.5, 1.0] with 3 bins [0-0.33, 0.33-0.66, 0.66-1.0]
        expected_indices = torch.tensor([0, 1, 2])
        assert torch.allclose(bin_indices, expected_indices)

    def test_get_target_distribution_with_uncertainty(self):
        """Test target distribution generation with uncertainty."""
        loss_fn = BinnedRegressionLoss(
            bins=3, min_value=0.0, max_value=1.0, soft_targets=True, noise_aware=True
        )
        target = torch.tensor([[0.25], [0.5], [0.75]])
        uncertainty = torch.tensor([[0.1], [0.2], [0.3]])

        # Test with uncertainty
        distributions = loss_fn._get_target_distribution(target, uncertainty)
        assert distributions.shape == (3, 3)  # (batch_size, n_bins)

        # Higher uncertainty should lead to more spread-out distributions
        # We can verify this by checking the entropy or the ratio of max prob
        max_probs = torch.max(distributions, dim=1)[0]
        # Lower max_prob indicates more spread out
        # The distribution for higher uncertainty should have lower max_prob
        assert max_probs[0] > max_probs[1] > max_probs[2]

    def test_get_target_distribution_with_adaptive_sigma(self):
        """Test target distribution generation with adaptive sigma."""
        loss_fn = BinnedRegressionLoss(
            bins=3, min_value=0.0, max_value=1.0, soft_targets=True, adaptive_sigma=True
        )
        target = torch.tensor([[0.25], [0.5], [0.75]])

        # Get distributions
        distributions = loss_fn._get_target_distribution(target)
        assert distributions.shape == (3, 3)  # (batch_size, n_bins)

        # Create a loss with the same parameters but non-adaptive sigma
        loss_fn_fixed = BinnedRegressionLoss(
            bins=3, min_value=0.0, max_value=1.0, soft_targets=True, adaptive_sigma=False
        )

        # Get distributions with fixed sigma
        distributions_fixed = loss_fn_fixed._get_target_distribution(target)

        # The distributions should be different due to adaptive sigma
        # We can't easily predict the exact difference, but they shouldn't be identical
        assert not torch.allclose(distributions, distributions_fixed)


class TestStandardClassificationRegressionLoss:
    """Tests for StandardClassificationRegressionLoss."""

    def test_init_and_params(self):
        """Test initialization and parameter storage."""
        loss_fn = StandardClassificationRegressionLoss(
            bins=5,
            min_value=-1.0,
            max_value=1.0,
            loss_type="cross_entropy",
            label_smoothing=0.1,
            focal_gamma=2.5,
        )
        assert loss_fn.n_bins == 5
        assert loss_fn.loss_type == "cross_entropy"
        assert loss_fn.label_smoothing == 0.1
        assert loss_fn.focal_gamma == 2.5

        # Test inheritance from base class
        assert loss_fn.bin_edges[0].item() == -1.0
        assert loss_fn.bin_edges[-1].item() == 1.0
        assert loss_fn.soft_targets is True
        assert loss_fn.sigma == 0.1

    def test_extract_distribution_parameters(self):
        """Test extraction of distribution parameters from model outputs."""
        loss_fn = StandardClassificationRegressionLoss(bins=3)
        logits = torch.tensor([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0]])

        params = loss_fn._extract_distribution_parameters(logits)
        assert "logits" in params
        assert "bin_probs" in params
        assert "bin_centers" in params
        assert "bin_widths" in params

        # Check softmax conversion
        expected_probs = torch.softmax(logits, dim=1)
        assert torch.allclose(params["bin_probs"], expected_probs)

        # Check shapes
        assert params["bin_probs"].shape == (2, 3)
        assert params["logits"].shape == (2, 3)

    def test_calculate_nll_soft_targets(self):
        """Test negative log-likelihood calculation with soft targets."""
        loss_fn = StandardClassificationRegressionLoss(
            bins=3, min_value=0.0, max_value=1.0, soft_targets=True
        )

        # Create mock parameters
        logits = torch.tensor([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0]])
        params = loss_fn._extract_distribution_parameters(logits)

        # Create mock target distributions
        target_dist = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])

        # Test cross-entropy loss
        loss_fn.loss_type = "cross_entropy"
        nll = loss_fn._calculate_nll(target_dist, params)
        assert nll.shape == (2,)
        assert not torch.isnan(nll).any()

        # Test KL divergence loss
        loss_fn.loss_type = "kl_div"
        nll = loss_fn._calculate_nll(target_dist, params)
        assert nll.shape == (2,)
        assert not torch.isnan(nll).any()

        # Test focal loss
        loss_fn.loss_type = "focal"
        nll = loss_fn._calculate_nll(target_dist, params)
        assert nll.shape == (2,)
        assert not torch.isnan(nll).any()

        # Test direct NLL loss
        loss_fn.loss_type = "nll"
        nll = loss_fn._calculate_nll(target_dist, params)
        assert nll.shape == (2,)
        assert not torch.isnan(nll).any()

        # Test unsupported loss type
        loss_fn.loss_type = "invalid"
        with pytest.raises(ValueError, match="Unsupported loss_type"):
            loss_fn._calculate_nll(target_dist, params)

    def test_calculate_nll_hard_targets(self):
        """Test negative log-likelihood calculation with hard targets."""
        loss_fn = StandardClassificationRegressionLoss(
            bins=3, min_value=0.0, max_value=1.0, soft_targets=False, label_smoothing=0.1
        )

        # Create mock parameters
        logits = torch.tensor([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0]])
        params = loss_fn._extract_distribution_parameters(logits)

        # Create mock target indices (bin indices)
        target_indices = torch.tensor([0, 1])

        # Test cross-entropy loss
        loss_fn.loss_type = "cross_entropy"
        nll = loss_fn._calculate_nll(target_indices, params)
        assert nll.shape == (2,)
        assert not torch.isnan(nll).any()

        # Test focal loss
        loss_fn.loss_type = "focal"
        nll = loss_fn._calculate_nll(target_indices, params)
        assert nll.shape == (2,)
        assert not torch.isnan(nll).any()

        # Test other loss types default to cross-entropy for hard targets
        loss_fn.loss_type = "kl_div"
        nll = loss_fn._calculate_nll(target_indices, params)
        assert nll.shape == (2,)
        assert not torch.isnan(nll).any()

    def test_forward_soft_targets(self):
        """Test full forward pass with soft targets."""
        loss_fn = StandardClassificationRegressionLoss(
            bins=3, min_value=0.0, max_value=1.0, soft_targets=True
        )
        logits = torch.tensor(
            [[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0]]  # Predicts first bin  # Predicts second bin
        )
        targets = torch.tensor([[0.1], [0.5]])  # First in bin 0, second in bin 1

        # Test basic forward pass
        loss = loss_fn(logits, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1  # Should be a scalar with 'mean' reduction
        assert not torch.isnan(loss)
        assert loss > 0  # Loss should be positive

        # Test forward with mask
        mask = torch.tensor([True, False])  # Mask out second sample
        loss_masked = loss_fn(logits, targets, mask=mask)
        assert not torch.isnan(loss_masked)

        # Test forward with weights
        weights = torch.tensor([0.5, 1.5])  # Weight second sample more
        loss_weighted = loss_fn(logits, targets, weights=weights)
        assert not torch.isnan(loss_weighted)

        # Test forward with uncertainty
        uncertainty = torch.tensor([[0.1], [0.2]])
        loss_uncertain = loss_fn(logits, targets, uncertainty=uncertainty)
        assert not torch.isnan(loss_uncertain)

        # Test with 'none' reduction
        loss_fn_none = StandardClassificationRegressionLoss(
            bins=3, reduction="none", soft_targets=True
        )
        loss_none = loss_fn_none(logits, targets)
        assert loss_none.shape == (2,)
        assert not torch.isnan(loss_none).any()

    def test_forward_hard_targets(self):
        """Test full forward pass with hard targets."""
        loss_fn = StandardClassificationRegressionLoss(
            bins=3, min_value=0.0, max_value=1.0, soft_targets=False
        )
        logits = torch.tensor(
            [[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0]]  # Predicts first bin  # Predicts second bin
        )
        targets = torch.tensor([[0.1], [0.5]])  # First in bin 0, second in bin 1

        loss = loss_fn(logits, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1
        assert not torch.isnan(loss)
        assert loss > 0

    def test_different_loss_types_integration(self):
        """Integration test with different loss types."""
        targets = torch.tensor([[0.1], [0.5]])
        logits = torch.tensor(
            [
                [1.0, -1.0, 0.0],  # Predicts first bin with high confidence
                [-1.0, 2.0, -1.0],  # Predicts second bin with high confidence
            ]
        )

        # These targets match the predictions well, so losses should be low

        # Cross entropy
        ce_loss = StandardClassificationRegressionLoss(bins=3, loss_type="cross_entropy")
        ce_result = ce_loss(logits, targets)

        # KL divergence
        kl_loss = StandardClassificationRegressionLoss(bins=3, loss_type="kl_div")
        kl_result = kl_loss(logits, targets)

        # Focal loss
        focal_loss = StandardClassificationRegressionLoss(
            bins=3, loss_type="focal", focal_gamma=2.0
        )
        focal_result = focal_loss(logits, targets)

        # All should return valid losses
        assert not torch.isnan(ce_result)
        assert not torch.isnan(kl_result)
        assert not torch.isnan(focal_result)

        # Now create poor predictions to test loss behavior
        bad_logits = torch.tensor(
            [
                [-1.0, 0.0, 1.0],  # Predicts third bin (wrong)
                [2.0, -1.0, -1.0],  # Predicts first bin (wrong)
            ]
        )

        # Loss should be higher for bad predictions
        ce_bad = ce_loss(bad_logits, targets)
        kl_bad = kl_loss(bad_logits, targets)
        focal_bad = focal_loss(bad_logits, targets)

        assert ce_bad > ce_result
        assert kl_bad > kl_result
        assert focal_bad > focal_result

    def test_out_of_range_targets(self):
        """Test handling of targets outside the bin range."""
        loss_fn = StandardClassificationRegressionLoss(bins=3, min_value=0.0, max_value=1.0)
        logits = torch.randn(3, 3)

        # Test with targets outside the range
        out_of_range_targets = torch.tensor([[-0.5], [0.5], [1.5]])

        # By default, these should be clamped to the valid range
        loss = loss_fn(logits, out_of_range_targets)
        assert not torch.isnan(loss)

        # Test with extrapolation enabled
        loss_fn_extrapolate = StandardClassificationRegressionLoss(
            bins=3, min_value=0.0, max_value=1.0, extrapolate_beyond_bins=True
        )
        loss_extrapolate = loss_fn_extrapolate(logits, out_of_range_targets)
        assert not torch.isnan(loss_extrapolate)

        # The losses should be different due to different target handling
        assert loss.item() != loss_extrapolate.item()

    def test_decode_prediction(self):
        """Test decoding predictions back to continuous values."""
        loss_fn = StandardClassificationRegressionLoss(bins=3, min_value=0.0, max_value=1.0)

        # Create logits with clear predictions
        logits = torch.tensor(
            [
                [5.0, -5.0, -5.0],  # Strongly predicts bin 0 (center ~0.167)
                [-5.0, 5.0, -5.0],  # Strongly predicts bin 1 (center ~0.5)
                [-5.0, -5.0, 5.0],  # Strongly predicts bin 2 (center ~0.833)
            ]
        )

        # Decode predictions
        decoded = loss_fn.decode_prediction(logits)
        assert decoded.shape == (3, 1)  # Should return [batch_size, 1]

        # Check if values are close to bin centers
        bin_centers = loss_fn.bin_centers
        expected = torch.tensor([[bin_centers[0]], [bin_centers[1]], [bin_centers[2]]])
        assert torch.allclose(decoded, expected, atol=1e-3)

    def test_get_distribution(self):
        """Test retrieving the full distribution from predictions."""
        loss_fn = StandardClassificationRegressionLoss(bins=3, min_value=0.0, max_value=1.0)

        # Create logits
        logits = torch.tensor([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0]])

        # Get distribution
        dist = loss_fn.get_distribution(logits)
        assert isinstance(dist, dict)
        assert "bin_probs" in dist
        assert "logits" in dist
        assert "bin_centers" in dist
        assert "bin_widths" in dist

        # Check shapes
        assert dist["bin_probs"].shape == (2, 3)
        assert dist["logits"].shape == (2, 3)

        # Probabilities should sum to 1
        assert torch.allclose(dist["bin_probs"].sum(dim=1), torch.ones(2))


class TestOrdinalRegressionLoss:
    """Tests for OrdinalRegressionLoss."""

    def test_init_and_params(self):
        """Test initialization and parameter storage."""
        loss_fn = OrdinalRegressionLoss(
            bins=5, min_value=-1.0, max_value=1.0, loss_type="bce", focal_gamma=3.0
        )
        assert loss_fn.n_bins == 5
        assert loss_fn.loss_type == "bce"
        assert loss_fn.focal_gamma == 3.0

        # Test inheritance from base class
        assert loss_fn.bin_edges[0].item() == -1.0
        assert loss_fn.bin_edges[-1].item() == 1.0
        assert loss_fn.soft_targets is True
        assert loss_fn.sigma == 0.1

        # Test encoding matrix creation
        assert hasattr(loss_fn, "encoding_matrix")
        assert loss_fn.encoding_matrix.shape == (5, 4)  # n_bins x (n_bins-1)

        # Check encoding matrix structure - for ordinal regression:
        # 1st row: [0, 0, 0, 0] - bin 0 has no thresholds crossed
        # 2nd row: [1, 0, 0, 0] - bin 1 crosses 1st threshold
        # 3rd row: [1, 1, 0, 0] - bin 2 crosses 1st and 2nd thresholds
        # etc.
        expected_matrix = torch.zeros(5, 4)
        for i in range(1, 5):  # For bins 1 through 4
            expected_matrix[i, :i] = 1.0

        assert torch.allclose(loss_fn.encoding_matrix, expected_matrix)

    def test_extract_distribution_parameters(self):
        """Test distribution parameter extraction from binary logits."""
        loss_fn = OrdinalRegressionLoss(bins=4)

        # Binary logits for 4 bins (3 thresholds)
        # Logits represent: "Is value > threshold_k?" for k=0,1,2
        binary_logits = torch.tensor(
            [
                [2.0, 0.0, -2.0],  # [0.88, 0.50, 0.12] after sigmoid
                # Predicts: likely > t0, uncertain t1, unlikely > t2
                # Most likely in bin 2 (crossed t0, t1, but not t2)
                [-2.0, -2.0, -2.0],  # [0.12, 0.12, 0.12] after sigmoid
                # Predicts: unlikely > any threshold
                # Most likely in bin 0
            ]
        )

        params = loss_fn._extract_distribution_parameters(binary_logits)

        # Check parameter keys
        assert "binary_logits" in params
        assert "binary_probs" in params
        assert "bin_probs" in params
        assert "bin_centers" in params
        assert "bin_widths" in params

        # Check binary probabilities (sigmoid of logits)
        expected_binary_probs = torch.sigmoid(binary_logits)
        assert torch.allclose(params["binary_probs"], expected_binary_probs)

        # Check bin probability shapes
        assert params["bin_probs"].shape == (2, 4)  # batch_size x n_bins

        # Check probabilities sum to 1 for each sample
        assert torch.allclose(params["bin_probs"].sum(dim=1), torch.ones(2))

        # For first sample: should have highest probability in bin 2
        # For second sample: should have highest probability in bin 0
        max_prob_bins = torch.argmax(params["bin_probs"], dim=1)
        assert max_prob_bins[0] == 2
        assert max_prob_bins[1] == 0

    def test_ordinal_target_conversion(self):
        """Test conversion of targets to ordinal encoding."""
        loss_fn = OrdinalRegressionLoss(bins=4, min_value=0.0, max_value=1.0)

        # Test with soft targets
        target_dist = torch.tensor(
            [
                [0.1, 0.2, 0.6, 0.1],  # Most weight in bin 2
                [0.7, 0.2, 0.1, 0.0],  # Most weight in bin 0
            ]
        )

        # When called via _calculate_nll, the distribution gets converted to ordinal
        params = {"binary_logits": torch.randn(2, 3)}  # batch_size x (n_bins-1)

        # To test conversion, we need access to the actual ordinal targets
        # We're mainly testing that no errors occur and the shape is correct
        loss = loss_fn._calculate_nll(target_dist, params)
        assert not torch.isnan(loss).any()

        # Test with hard targets
        bin_indices = torch.tensor([2, 0])  # First sample in bin 2, second in bin 0

        loss = loss_fn._calculate_nll(bin_indices, params)
        assert not torch.isnan(loss).any()

        # Expected ordinal encoding for bin indices [2, 0]:
        # bin 2 -> [1, 1, 0]  (crosses thresholds 0, 1, but not 2)
        # bin 0 -> [0, 0, 0]  (crosses no thresholds)

    def test_calculate_nll_different_loss_types(self):
        """Test NLL calculation with different loss types."""
        loss_fn = OrdinalRegressionLoss(bins=4, min_value=0.0, max_value=1.0)

        # Create binary logits
        binary_logits = torch.tensor(
            [
                [2.0, 1.0, -2.0],  # Predicts bin 2 (p ~ [0, 0, 0.88, 0.12])
                [-2.0, -2.0, -2.0],  # Predicts bin 0 (p ~ [0.88, 0.12, 0, 0])
            ]
        )

        params = loss_fn._extract_distribution_parameters(binary_logits)

        # Create target that matches predictions well
        bin_indices = torch.tensor([2, 0])  # First in bin 2, second in bin 0

        # Test BCE loss
        loss_fn.loss_type = "bce"
        nll_bce = loss_fn._calculate_nll(bin_indices, params)
        assert nll_bce.shape == (2,)
        assert not torch.isnan(nll_bce).any()

        # Test focal loss
        loss_fn.loss_type = "focal"
        loss_fn.focal_gamma = 2.0
        nll_focal = loss_fn._calculate_nll(bin_indices, params)
        assert nll_focal.shape == (2,)
        assert not torch.isnan(nll_focal).any()

        # Test invalid loss type
        loss_fn.loss_type = "invalid"
        with pytest.raises(ValueError, match="Unsupported loss_type"):
            loss_fn._calculate_nll(bin_indices, params)

    def test_forward_soft_targets(self):
        """Test full forward pass with soft targets."""
        loss_fn = OrdinalRegressionLoss(bins=4, min_value=0.0, max_value=1.0, soft_targets=True)

        # Binary logits for 4 bins (3 thresholds)
        binary_logits = torch.tensor(
            [[2.0, 1.0, -2.0], [-2.0, -1.0, -2.0]]  # Predicts bin 2  # Predicts bin 0
        )

        targets = torch.tensor([[0.7], [0.1]])  # First in bin ~2, second in bin ~0

        # Test basic forward pass
        loss = loss_fn(binary_logits, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1  # Should be scalar with 'mean' reduction
        assert not torch.isnan(loss)
        assert loss > 0

        # Test with mask
        mask = torch.tensor([True, False])
        loss_masked = loss_fn(binary_logits, targets, mask=mask)
        assert not torch.isnan(loss_masked)

        # Test with weights
        weights = torch.tensor([1.5, 0.5])
        loss_weighted = loss_fn(binary_logits, targets, weights=weights)
        assert not torch.isnan(loss_weighted)

        # Test with uncertainty
        uncertainty = torch.tensor([[0.1], [0.2]])
        loss_uncertain = loss_fn(binary_logits, targets, uncertainty=uncertainty)
        assert not torch.isnan(loss_uncertain)

    def test_forward_hard_targets(self):
        """Test full forward pass with hard targets."""
        loss_fn = OrdinalRegressionLoss(bins=4, min_value=0.0, max_value=1.0, soft_targets=False)

        binary_logits = torch.tensor(
            [[2.0, 1.0, -2.0], [-2.0, -1.0, -2.0]]  # Predicts bin 2  # Predicts bin 0
        )

        targets = torch.tensor([[0.7], [0.1]])  # First in bin ~2, second in bin ~0

        loss = loss_fn(binary_logits, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1
        assert not torch.isnan(loss)
        assert loss > 0

    def test_wrong_output_size(self):
        """Test handling of incorrect output size."""
        loss_fn = OrdinalRegressionLoss(bins=4)

        # Wrong output size: should be batch_size x (n_bins-1)
        wrong_logits = torch.randn(2, 5)  # 5 outputs instead of 3 (4-1)
        targets = torch.rand(2, 1)

        with pytest.raises(ValueError, match="Expected y_pred to have"):
            loss_fn(wrong_logits, targets)

    def test_out_of_range_targets(self):
        """Test handling of targets outside the bin range."""
        loss_fn = OrdinalRegressionLoss(bins=4, min_value=0.0, max_value=1.0)

        binary_logits = torch.randn(3, 3)  # batch_size=3, n_bins-1=3
        out_of_range_targets = torch.tensor([[-0.5], [0.5], [1.5]])

        # By default, should clamp to valid range
        loss = loss_fn(binary_logits, out_of_range_targets)
        assert not torch.isnan(loss)

        # Test with extrapolation
        loss_fn_extrapolate = OrdinalRegressionLoss(
            bins=4, min_value=0.0, max_value=1.0, extrapolate_beyond_bins=True
        )
        loss_extrapolate = loss_fn_extrapolate(binary_logits, out_of_range_targets)
        assert not torch.isnan(loss_extrapolate)


class TestHistogramRegressionLoss:
    """Tests for HistogramRegressionLoss."""

    def test_init_and_params(self):
        """Test initialization and parameter storage."""
        loss_fn = HistogramRegressionLoss(
            bins=5,
            min_value=-1.0,
            max_value=1.0,
            loss_type="kl_div",
            normalize_targets=True,
            wasserstein_p=2,
        )
        assert loss_fn.n_bins == 5
        assert loss_fn.loss_type == "kl_div"
        assert loss_fn.normalize_targets is True
        assert loss_fn.wasserstein_p == 2

        # Test inheritance from base class
        assert loss_fn.bin_edges[0].item() == -1.0
        assert loss_fn.bin_edges[-1].item() == 1.0
        assert loss_fn.soft_targets is True
        assert loss_fn.sigma == 0.1

    def test_extract_distribution_params_from_logits(self):
        """Test extracting distribution parameters from logits."""
        loss_fn = HistogramRegressionLoss(bins=3)

        # Test with logits input
        logits = torch.tensor(
            [
                [1.0, -1.0, 0.0],  # After softmax: ~[0.57, 0.08, 0.35]
                [-1.0, 2.0, -1.0],  # After softmax: ~[0.05, 0.90, 0.05]
            ]
        )

        params = loss_fn._extract_distribution_parameters(logits)

        # Check parameter keys
        assert "logits" in params
        assert "bin_probs" in params
        assert "bin_centers" in params
        assert "bin_widths" in params

        # Check bin probabilities (softmax of logits)
        expected_probs = torch.softmax(logits, dim=1)
        assert torch.allclose(params["bin_probs"], expected_probs)

        # Check probabilities sum to 1 for each sample
        assert torch.allclose(params["bin_probs"].sum(dim=1), torch.ones(2))

    def test_extract_distribution_params_from_probs(self):
        """Test extracting distribution parameters from probabilities."""
        loss_fn = HistogramRegressionLoss(bins=3)

        # Test with probability input (values between 0 and 1)
        probs = torch.tensor([[0.6, 0.1, 0.3], [0.2, 0.7, 0.1]])

        params = loss_fn._extract_distribution_parameters(probs)

        # Check probs are returned as is (but normalized to sum to 1)
        assert torch.allclose(params["bin_probs"], probs)

        # Check logits are log of probabilities
        expected_logits = torch.log(probs + 1e-10)
        assert torch.allclose(params["logits"], expected_logits)

        # Check probabilities sum to 1
        assert torch.allclose(params["bin_probs"].sum(dim=1), torch.ones(2))

        # Test with unnormalized positive values
        unnormalized = torch.tensor([[6.0, 1.0, 3.0], [2.0, 7.0, 1.0]])  # Sum = 10  # Sum = 10

        params = loss_fn._extract_distribution_parameters(unnormalized)
        expected_normalized = unnormalized / 10.0  # Divide by sum

        assert torch.allclose(params["bin_probs"], expected_normalized)
        assert torch.allclose(params["bin_probs"].sum(dim=1), torch.ones(2))

    def test_calculate_nll_different_loss_types(self):
        """Test NLL calculation with different loss types."""
        loss_fn = HistogramRegressionLoss(bins=3, min_value=0.0, max_value=1.0)

        # Create predictions
        pred_probs = torch.tensor(
            [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]  # Predicts mostly bin 0  # Predicts mostly bin 1
        )

        params = loss_fn._extract_distribution_parameters(pred_probs)

        # Create target distribution that exactly matches predictions
        target_dist = pred_probs.clone()

        # Test KL divergence loss - should be near zero for identical distributions
        loss_fn.loss_type = "kl_div"
        nll_kl = loss_fn._calculate_nll(target_dist, params)
        assert nll_kl.shape == (2,)
        assert torch.all(nll_kl < 1e-5)  # Should be very close to zero

        # Test cross-entropy loss - should be minimal for identical distributions
        loss_fn.loss_type = "cross_entropy"
        nll_ce = loss_fn._calculate_nll(target_dist, params)
        assert nll_ce.shape == (2,)
        assert torch.all(nll_ce < 1.0)  # Should be relatively small

        # Test Wasserstein distance - should be zero for identical distributions
        loss_fn.loss_type = "wasserstein"
        loss_fn.wasserstein_p = 1
        nll_w1 = loss_fn._calculate_nll(target_dist, params)
        assert nll_w1.shape == (2,)
        assert torch.all(nll_w1 < 1e-5)  # Should be very close to zero

        # Test with Wasserstein-2
        loss_fn.wasserstein_p = 2
        nll_w2 = loss_fn._calculate_nll(target_dist, params)
        assert nll_w2.shape == (2,)
        assert torch.all(nll_w2 < 1e-5)  # Should be very close to zero

        # Test with mismatched distributions
        mismatched_dist = torch.tensor(
            [
                [0.1, 0.2, 0.7],  # Very different from [0.7, 0.2, 0.1]
                [0.7, 0.2, 0.1],  # Very different from [0.1, 0.8, 0.1]
            ]
        )

        # KL divergence for mismatched distributions should be larger
        loss_fn.loss_type = "kl_div"
        nll_kl_mismatched = loss_fn._calculate_nll(mismatched_dist, params)
        assert torch.all(nll_kl_mismatched > nll_kl)  # Should be larger

        # Test invalid loss type
        loss_fn.loss_type = "invalid"
        with pytest.raises(ValueError, match="Unsupported loss_type"):
            loss_fn._calculate_nll(target_dist, params)

    def test_forward_soft_targets(self):
        """Test full forward pass with soft targets."""
        loss_fn = HistogramRegressionLoss(bins=3, min_value=0.0, max_value=1.0, soft_targets=True)

        # Histogram probs for 3 bins
        hist_probs = torch.tensor(
            [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]  # Predicts mostly bin 0  # Predicts mostly bin 1
        )

        targets = torch.tensor([[0.1], [0.5]])  # First in bin 0, second in bin 1

        # Test basic forward pass
        loss = loss_fn(hist_probs, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1  # Should be scalar with 'mean' reduction
        assert not torch.isnan(loss)
        assert loss >= 0  # Loss should be non-negative

        # Test with mask
        mask = torch.tensor([True, False])
        loss_masked = loss_fn(hist_probs, targets, mask=mask)
        assert not torch.isnan(loss_masked)

        # Test with weights
        weights = torch.tensor([1.5, 0.5])
        loss_weighted = loss_fn(hist_probs, targets, weights=weights)
        assert not torch.isnan(loss_weighted)

        # Test with uncertainty
        uncertainty = torch.tensor([[0.1], [0.2]])
        loss_uncertain = loss_fn(hist_probs, targets, uncertainty=uncertainty)
        assert not torch.isnan(loss_uncertain)

        # Test with 'none' reduction
        loss_fn_none = HistogramRegressionLoss(bins=3, reduction="none", soft_targets=True)
        loss_none = loss_fn_none(hist_probs, targets)
        assert loss_none.shape == (2,)
        assert not torch.isnan(loss_none).any()

    def test_forward_hard_targets(self):
        """Test full forward pass with hard targets."""
        loss_fn = HistogramRegressionLoss(bins=3, min_value=0.0, max_value=1.0, soft_targets=False)

        hist_probs = torch.tensor(
            [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]  # Predicts mostly bin 0  # Predicts mostly bin 1
        )

        targets = torch.tensor([[0.1], [0.5]])  # First in bin 0, second in bin 1

        # With hard targets, the targets should be converted to one-hot distributions
        loss = loss_fn(hist_probs, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1
        assert not torch.isnan(loss)
        assert loss >= 0

    def test_wrong_output_size(self):
        """Test handling of incorrect output size."""
        loss_fn = HistogramRegressionLoss(bins=3)

        # Wrong output size: should be batch_size x n_bins
        wrong_hist = torch.randn(2, 5)  # 5 outputs instead of 3
        targets = torch.rand(2, 1)

        with pytest.raises(ValueError, match="Expected y_pred to have"):
            loss_fn(wrong_hist, targets)

    def test_out_of_range_targets(self):
        """Test handling of targets outside the bin range."""
        loss_fn = HistogramRegressionLoss(bins=3, min_value=0.0, max_value=1.0)

        hist_probs = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]])

        out_of_range_targets = torch.tensor([[-0.5], [0.5], [1.5]])

        # By default, should clamp to valid range
        loss = loss_fn(hist_probs, out_of_range_targets)
        assert not torch.isnan(loss)

        # Test with extrapolation
        loss_fn_extrapolate = HistogramRegressionLoss(
            bins=3, min_value=0.0, max_value=1.0, extrapolate_beyond_bins=True
        )
        loss_extrapolate = loss_fn_extrapolate(hist_probs, out_of_range_targets)
        assert not torch.isnan(loss_extrapolate)


class TestRegressionAsClassificationLoss:
    """Tests for RegressionAsClassificationLoss."""

    def test_init_and_params(self):
        """Test initialization and parameter storage."""
        loss_fn = RegressionAsClassificationLoss(
            bins=5,
            min_value=-1.0,
            max_value=1.0,
            order_aware=True,
            smooth_targets=True,
            sigma=0.3,
            loss_type="kl_div",
            adaptive_sigma=True,
            focal_gamma=3.0,
        )
        assert loss_fn.n_bins == 5
        assert loss_fn.order_aware is True
        assert loss_fn.soft_targets is True  # Aliased from smooth_targets
        assert loss_fn.sigma == 0.3
        assert loss_fn.loss_type == "kl_div"
        assert loss_fn.adaptive_sigma is True
        assert loss_fn.focal_gamma == 3.0

        # Test inheritance from base class
        assert loss_fn.bin_edges[0].item() == -1.0
        assert loss_fn.bin_edges[-1].item() == 1.0

        # Test encoding matrix for ordinal mode
        assert hasattr(loss_fn, "encoding_matrix")
        assert loss_fn.encoding_matrix.shape == (5, 4)  # n_bins x (n_bins-1)

        # Test without order_aware
        loss_fn_no_ordinal = RegressionAsClassificationLoss(bins=5, order_aware=False)
        assert not hasattr(loss_fn_no_ordinal, "encoding_matrix")

    def test_extract_distribution_parameters_standard_mode(self):
        """Test extracting distribution parameters in standard mode."""
        loss_fn = RegressionAsClassificationLoss(bins=3, order_aware=True)

        # Standard mode: y_pred has n_bins outputs
        logits = torch.tensor(
            [[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0]]  # Predicts mostly bin 0  # Predicts mostly bin 1
        )

        params = loss_fn._extract_distribution_parameters(logits)

        # Check parameter keys
        assert "bin_probs" in params
        assert "logits" in params
        assert "bin_centers" in params
        assert "bin_widths" in params

        # Check bin probability shapes and values
        assert params["bin_probs"].shape == (2, 3)  # batch_size x n_bins
        expected_probs = torch.softmax(logits, dim=1)
        assert torch.allclose(params["bin_probs"], expected_probs)

        # Check with probability input
        probs = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]])

        params_from_probs = loss_fn._extract_distribution_parameters(probs)
        assert torch.allclose(params_from_probs["bin_probs"], probs)

        # Check logits are created correctly from probs
        expected_logits = torch.log(probs + 1e-10)
        assert torch.allclose(params_from_probs["logits"], expected_logits)

    def test_extract_distribution_parameters_ordinal_mode(self):
        """Test extracting distribution parameters in ordinal mode."""
        loss_fn = RegressionAsClassificationLoss(bins=3, order_aware=True)

        # Ordinal mode: y_pred has (n_bins-1) outputs
        binary_logits = torch.tensor(
            [
                [2.0, -2.0],  # After sigmoid: [0.88, 0.12]
                # Predicts: bin 0 with ~12% prob, bin 1 with ~76% prob, bin 2 with ~12% prob
                [-2.0, 2.0],  # After sigmoid: [0.12, 0.88]
                # Predicts: bin 0 with ~88% prob, bin 1 with ~10% prob, bin 2 with ~2% prob
            ]
        )

        params = loss_fn._extract_distribution_parameters(binary_logits)

        # Check parameter keys
        assert "bin_probs" in params
        assert "binary_probs" in params
        assert "binary_logits" in params
        assert "bin_centers" in params
        assert "bin_widths" in params

        # Check binary probability shapes and values
        assert params["binary_probs"].shape == (2, 2)  # batch_size x (n_bins-1)
        expected_binary_probs = torch.sigmoid(binary_logits)
        assert torch.allclose(params["binary_probs"], expected_binary_probs)

        # Check bin probability shapes
        assert params["bin_probs"].shape == (2, 3)  # batch_size x n_bins

        # Check probabilities sum to 1 for each sample
        assert torch.allclose(params["bin_probs"].sum(dim=1), torch.ones(2))

    def test_forward_standard_mode_soft_targets(self):
        """Test forward pass in standard mode with soft targets."""
        loss_fn = RegressionAsClassificationLoss(
            bins=3,
            min_value=0.0,
            max_value=1.0,
            order_aware=True,  # Should still use standard mode based on input shape
            smooth_targets=True,
        )

        # Standard mode logits (n_bins outputs)
        logits = torch.tensor(
            [[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0]]  # Predicts mostly bin 0  # Predicts mostly bin 1
        )

        targets = torch.tensor([[0.1], [0.5]])  # First in bin 0, second in bin 1

        # Test basic forward pass
        loss = loss_fn(logits, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1  # Should be a scalar with 'mean' reduction
        assert not torch.isnan(loss)
        assert loss > 0

        # Test with different loss types
        for loss_type in ["cross_entropy", "kl_div", "focal", "wasserstein"]:
            loss_fn.loss_type = loss_type
            loss = loss_fn(logits, targets)
            assert not torch.isnan(loss)
            assert loss > 0

    def test_forward_standard_mode_hard_targets(self):
        """Test forward pass in standard mode with hard targets."""
        loss_fn = RegressionAsClassificationLoss(
            bins=3,
            min_value=0.0,
            max_value=1.0,
            order_aware=True,  # Should still use standard mode based on input shape
            smooth_targets=False,
        )

        # Standard mode logits (n_bins outputs)
        logits = torch.tensor(
            [[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0]]  # Predicts mostly bin 0  # Predicts mostly bin 1
        )

        targets = torch.tensor([[0.1], [0.5]])  # First in bin 0, second in bin 1

        # Test basic forward pass
        loss = loss_fn(logits, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1
        assert not torch.isnan(loss)
        assert loss > 0

        # Test with different loss types
        for loss_type in ["cross_entropy", "focal"]:  # Only these make sense for hard targets
            loss_fn.loss_type = loss_type
            loss = loss_fn(logits, targets)
            assert not torch.isnan(loss)
            assert loss > 0

    def test_forward_ordinal_mode_soft_targets(self):
        """Test forward pass in ordinal mode with soft targets."""
        loss_fn = RegressionAsClassificationLoss(
            bins=3, min_value=0.0, max_value=1.0, order_aware=True, smooth_targets=True
        )

        # Ordinal mode binary logits (n_bins-1 outputs)
        binary_logits = torch.tensor(
            [
                [1.0, -1.0],  # After sigmoid: [0.73, 0.27]
                # Predicts: bin 0 with ~27% prob, bin 1 with ~46% prob, bin 2 with ~27% prob
                [-1.0, 1.0],  # After sigmoid: [0.27, 0.73]
                # Predicts: bin 0 with ~73% prob, bin 1 with ~20% prob, bin 2 with ~7% prob
            ]
        )

        targets = torch.tensor([[0.5], [0.1]])  # First in bin 1, second in bin 0

        # Test basic forward pass with ordinal mode
        loss = loss_fn(binary_logits, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1
        assert not torch.isnan(loss)
        assert loss > 0

    def test_forward_ordinal_mode_hard_targets(self):
        """Test forward pass in ordinal mode with hard targets."""
        loss_fn = RegressionAsClassificationLoss(
            bins=3, min_value=0.0, max_value=1.0, order_aware=True, smooth_targets=False
        )

        # Ordinal mode binary logits (n_bins-1 outputs)
        binary_logits = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])  # Predicts bin 1  # Predicts bin 0

        targets = torch.tensor([[0.5], [0.1]])  # First in bin 1, second in bin 0

        # Test basic forward pass with ordinal mode
        loss = loss_fn(binary_logits, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1
        assert not torch.isnan(loss)
        assert loss > 0

    def test_wrong_output_size(self):
        """Test handling of incorrect output sizes."""
        loss_fn = RegressionAsClassificationLoss(bins=3, order_aware=True)

        # Wrong output size in standard mode
        wrong_std_logits = torch.randn(2, 4)  # Should be 2, 3 (n_bins)
        targets = torch.rand(2, 1)

        with pytest.raises(ValueError, match="Expected y_pred with"):
            loss_fn(wrong_std_logits, targets)

        # Wrong output size in ordinal mode
        wrong_ord_logits = torch.randn(2, 3)  # Should be 2, 2 (n_bins-1)

        # Check we can detect this is wrong for ordinal mode
        # when we force ordinal mode (despite output shape matching standard mode)
        with pytest.raises(ValueError, match="For ordinal mode, expected y_pred with"):
            # This is a bit hacky but we need to test the ordinal-specific error path
            # Make sure binary_logits has wrong shape for ordinal mode
            binary_logits = torch.randn(2, 3)  # Should be 2, 2 (n_bins-1)
            # Force execution path to treat this as ordinal mode
            loss_fn._extract_distribution_parameters = lambda x: {"binary_logits": x}
            loss_fn(binary_logits, targets)

    def test_masking(self):
        """Test masking behavior."""
        loss_fn = RegressionAsClassificationLoss(bins=3, min_value=0.0, max_value=1.0)

        # Standard mode logits
        logits = torch.tensor(
            [
                [1.0, -1.0, 0.0],  # Predicts mostly bin 0
                [-1.0, 2.0, -1.0],  # Predicts mostly bin 1
                [0.0, 0.0, 0.0],  # Uniform prediction
            ]
        )

        targets = torch.tensor([[0.1], [0.5], [0.9]])  # In bins 0, 1, 2

        # Create mask that excludes the third sample
        mask = torch.tensor([True, True, False])

        # Loss with mask
        loss_masked = loss_fn(logits, targets, mask=mask)

        # Loss with just first two samples
        loss_first_two = loss_fn(logits[:2], targets[:2])

        # Loss with all samples
        loss_all = loss_fn(logits, targets)

        # Masked loss should be closer to the loss of just the first two samples
        assert abs(loss_masked.item() - loss_first_two.item()) < 1e-5
        # And should be different from the loss of all samples
        assert loss_masked.item() != loss_all.item()

    def test_sample_weights(self):
        """Test sample weights behavior."""
        loss_fn = RegressionAsClassificationLoss(bins=3, min_value=0.0, max_value=1.0)

        logits = torch.tensor(
            [[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0]]  # Predicts mostly bin 0  # Predicts mostly bin 1
        )

        targets = torch.tensor([[0.1], [0.5]])  # In bins 0, 1

        # Equal weights
        weights_equal = torch.tensor([1.0, 1.0])
        loss_equal = loss_fn(logits, targets, weights=weights_equal)

        # Weight first sample more
        weights_first = torch.tensor([2.0, 0.0])
        loss_first = loss_fn(logits, targets, weights=weights_first)

        # Weight second sample more
        weights_second = torch.tensor([0.0, 2.0])
        loss_second = loss_fn(logits, targets, weights=weights_second)

        # Check that weights affect the loss
        assert loss_equal.item() != loss_first.item()
        assert loss_equal.item() != loss_second.item()
        assert loss_first.item() != loss_second.item()

    def test_uncertainty_handling(self):
        """Test uncertainty parameter handling."""
        loss_fn = RegressionAsClassificationLoss(
            bins=3, min_value=0.0, max_value=1.0, noise_aware=True  # Enable uncertainty handling
        )

        logits = torch.tensor(
            [[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0]]  # Predicts mostly bin 0  # Predicts mostly bin 1
        )

        targets = torch.tensor([[0.1], [0.5]])  # In bins 0, 1

        # Test with uncertainty
        uncertainty = torch.tensor([[0.1], [0.3]])  # More uncertainty in second sample
        loss_with_uncertainty = loss_fn(logits, targets, uncertainty=uncertainty)

        # Test without uncertainty
        loss_without_uncertainty = loss_fn(logits, targets)

        # They should be different due to different target distributions
        assert loss_with_uncertainty.item() != loss_without_uncertainty.item()

    def test_out_of_range_targets(self):
        """Test handling of targets outside the bin range."""
        # Test with extrapolation disabled (default)
        loss_fn_clamp = RegressionAsClassificationLoss(
            bins=3, min_value=0.0, max_value=1.0, extrapolate_beyond_bins=False
        )

        # Test with extrapolation enabled
        loss_fn_extrapolate = RegressionAsClassificationLoss(
            bins=3, min_value=0.0, max_value=1.0, extrapolate_beyond_bins=True
        )

        logits = torch.randn(3, 3)
        out_of_range_targets = torch.tensor([[-0.5], [0.5], [1.5]])

        # Both should produce valid losses
        loss_clamp = loss_fn_clamp(logits, out_of_range_targets)
        loss_extrapolate = loss_fn_extrapolate(logits, out_of_range_targets)

        assert not torch.isnan(loss_clamp)
        assert not torch.isnan(loss_extrapolate)
        # The losses should be different due to different target handling
        assert loss_clamp.item() != loss_extrapolate.item()

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        loss_fn = RegressionAsClassificationLoss(bins=3, min_value=0.0, max_value=1.0)

        # Extreme logits
        extreme_logits = torch.tensor(
            [
                [1000.0, -1000.0, 0.0],  # Extremely confident in bin 0
                [-1000.0, 1000.0, -1000.0],  # Extremely confident in bin 1
            ]
        )

        targets = torch.tensor([[0.1], [0.5]])  # In bins 0, 1

        # Should not produce NaN or inf
        loss = loss_fn(extreme_logits, targets)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        # Test with very small/large targets
        extreme_targets = torch.tensor([[1e-10], [1 - 1e-10]])
        loss = loss_fn(extreme_logits, extreme_targets)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_factory_function(self):
        """Test that the regression_as_classification factory function works correctly."""
        from torchregress.losses.rag import regression_as_classification

        # Basic usage
        loss_fn = regression_as_classification(bins=5, min_value=-1.0, max_value=1.0)
        assert isinstance(loss_fn, RegressionAsClassificationLoss)
        assert loss_fn.n_bins == 5
        assert loss_fn.bin_edges[0].item() == -1.0
        assert loss_fn.bin_edges[-1].item() == 1.0

        # Test with robust_to_noise=True
        loss_fn_robust = regression_as_classification(bins=5, robust_to_noise=True)
        assert loss_fn_robust.loss_type == "kl_div"  # KL div handles noise better
        assert loss_fn_robust.adaptive_sigma is True

        # Test with auto_adapt and different bin counts
        loss_fn_small_bins = regression_as_classification(bins=8, auto_adapt=True)
        loss_fn_large_bins = regression_as_classification(bins=25, auto_adapt=True)

        # Sigma should adapt based on bin count
        assert loss_fn_small_bins.sigma > loss_fn_large_bins.sigma

    def test_uncertainty_regression_factory(self):
        """Test the uncertainty_regression factory function."""
        from torchregress.losses.rag import uncertainty_regression

        loss_fn = uncertainty_regression(bins=20, min_value=-1.0, max_value=1.0)

        assert isinstance(loss_fn, RegressionAsClassificationLoss)
        assert loss_fn.n_bins == 20
        assert loss_fn.bin_edges[0].item() == -1.0
        assert loss_fn.bin_edges[-1].item() == 1.0
        assert loss_fn.soft_targets is True

        # Should be configured for robustness
        assert loss_fn.adaptive_sigma is True

    def test_rag_edge_cases(self):
        """Test RAGLoss with edge cases: zeros, empty tensors, extreme values, NaN/Inf."""
        loss_fn = RAGLoss()

        # Test with zeros
        y_pred_zeros = torch.zeros(10, 2)  # [mean, log_std]
        y_true_zeros = torch.zeros(10)
        assert torch.isfinite(loss_fn(y_pred_zeros, y_true_zeros))

        # Test with empty tensors
        y_pred_empty = torch.tensor([]).reshape(0, 2)
        y_true_empty = torch.tensor([])
        assert loss_fn(y_pred_empty, y_true_empty).numel() == 0

        # Test with extreme values
        y_pred_large = torch.tensor([[1e10, 0.0]])  # Large mean, normal std
        y_true_small = torch.tensor([0.0])
        assert torch.isfinite(loss_fn(y_pred_large, y_true_small))

        # Test with very small/large variance
        y_pred_var = torch.tensor([[1.0, -20.0]])  # Mean 1, very small variance
        y_true_var = torch.tensor([1.0])
        assert torch.isfinite(loss_fn(y_pred_var, y_true_var))

        y_pred_var2 = torch.tensor([[1.0, 20.0]])  # Mean 1, very large variance
        assert torch.isfinite(loss_fn(y_pred_var2, y_true_var))

        # Test with NaN/Inf and masks
        y_pred_nan = torch.tensor([[1.0, 0.0], [float("nan"), 0.0], [3.0, 1.0]])
        y_true_nan = torch.tensor([1.5, 2.5, float("inf")])
        mask = torch.tensor([True, False, False])
        assert torch.isfinite(loss_fn(y_pred_nan, y_true_nan, mask))


import torch
import pytest
from torch.autograd import gradcheck


class TestRAGLossNumericalStability:
    def test_rag_gradient_flow(self):
        """Test that gradients flow through RAGLoss properly."""
        from torchregress.losses.rag import RAGLoss

        # Create inputs that require gradients
        y_pred = torch.randn(10, 1, requires_grad=True, dtype=torch.double)
        y_var = torch.exp(torch.randn(10, 1, requires_grad=True, dtype=torch.double))
        y_true = torch.randn(10, 1, dtype=torch.double)

        # Define a function for gradcheck
        def loss_fn(pred, var, target):
            return RAGLoss(reduction="mean")(pred, var, target)

        # Test with gradcheck
        assert gradcheck(loss_fn, (y_pred, y_var, y_true), eps=1e-6, atol=1e-4)

    def test_extreme_values(self):
        """Test stability with extreme values."""
        from torchregress.losses.rag import RAGLoss

        # Very large and small values
        y_pred_large = torch.tensor([1e5, 1e10, 1e15], requires_grad=True)
        y_var_large = torch.tensor([1e4, 1e8, 1e12], requires_grad=True)
        y_true_large = torch.tensor([1e5 + 1, 1e10 + 10, 1e15 + 100])

        y_pred_small = torch.tensor([1e-5, 1e-10, 1e-15], requires_grad=True)
        y_var_small = torch.tensor([1e-4, 1e-8, 1e-12], requires_grad=True)
        y_true_small = torch.tensor([1e-5 + 1e-7, 1e-10 + 1e-12, 1e-15 + 1e-17])

        # Test with large values
        rag_loss = RAGLoss(reduction="mean")

        # Large values test
        loss = rag_loss(y_pred_large, y_var_large, y_true_large)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.all(torch.isfinite(y_pred_large.grad))
        assert torch.all(torch.isfinite(y_var_large.grad))

        # Small values test
        y_pred_small.grad = None
        y_var_small.grad = None
        loss = rag_loss(y_pred_small, y_var_small, y_true_small)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.all(torch.isfinite(y_pred_small.grad))
        assert torch.all(torch.isfinite(y_var_small.grad))

    def test_nan_inf_handling(self):
        """Test how RAG loss handles NaN and Inf values with masks."""
        from torchregress.losses.rag import RAGLoss

        # Create data with some NaNs and Infs
        y_pred = torch.tensor([1.0, float("nan"), 3.0, float("inf")], requires_grad=True)
        y_var = torch.tensor([1.0, 1.0, float("nan"), 1.0], requires_grad=True)
        y_true = torch.tensor([1.1, 2.0, 3.0, 4.0])
        mask = torch.tensor([True, False, False, False])  # Mask out NaNs and Infs

        rag_loss = RAGLoss(reduction="mean")

        # This should only use the valid elements
        loss = rag_loss(y_pred, y_var, y_true, mask=mask)
        assert torch.isfinite(loss)
        loss.backward()
        # Only the unmasked elements should have gradients
        assert torch.isfinite(y_pred.grad[0])
        assert torch.isfinite(y_var.grad[0])
        # Masked elements should have zero gradient
        assert y_pred.grad[1] == 0.0
        assert y_pred.grad[2] == 0.0
        assert y_pred.grad[3] == 0.0
        assert y_var.grad[1] == 0.0
        assert y_var.grad[2] == 0.0
        assert y_var.grad[3] == 0.0

    def test_reduction_modes(self):
        """Test different reduction modes for backward pass."""
        from torchregress.losses.rag import RAGLoss

        y_pred = torch.randn(10, 1, requires_grad=True)
        y_var = torch.exp(torch.randn(10, 1, requires_grad=True))
        y_true = torch.randn(10, 1)

        # Test mean reduction
        rag_mean = RAGLoss(reduction="mean")
        loss = rag_mean(y_pred, y_var, y_true)
        loss.backward()
        mean_grad_pred = y_pred.grad.clone()
        mean_grad_var = y_var.grad.clone()

        # Reset gradients
        y_pred.grad = None
        y_var.grad = None

        # Test sum reduction
        rag_sum = RAGLoss(reduction="sum")
        loss = rag_sum(y_pred, y_var, y_true)
        loss.backward()
        sum_grad_pred = y_pred.grad.clone()
        sum_grad_var = y_var.grad.clone()

        # Reset gradients
        y_pred.grad = None
        y_var.grad = None

        # Test none reduction
        rag_none = RAGLoss(reduction="none")
        loss = rag_none(y_pred, y_var, y_true)
        loss.mean().backward()
        none_grad_pred = y_pred.grad.clone()
        none_grad_var = y_var.grad.clone()

        # Mean and sum should give different gradients
        assert not torch.allclose(mean_grad_pred, sum_grad_pred)
        # Mean and manual mean over none should be similar
        assert torch.allclose(mean_grad_pred, none_grad_pred)
        assert torch.allclose(mean_grad_var, none_grad_var)
