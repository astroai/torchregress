import torch
import unittest
from torch.autograd import gradcheck
from torchregress.losses.poisson import (
    PoissonDevianceLoss,
    ZeroInflatedPoissonNLLLoss,
    NegativeBinomialNLLLoss,
    PoissonLikelihoodRatioLoss,
)


class TestPoissonLosses(unittest.TestCase):
    def setUp(self):
        # Use fixed seed for reproducibility
        torch.manual_seed(42)
        self.batch_size = 4
        self.n_features = 5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.y_true = torch.randint(
            0, 10, (self.batch_size, self.n_features), device=self.device
        ).float()  # Integer counts
        self.y_pred = (
            torch.rand(self.batch_size, self.n_features, device=self.device) * 5
        )  # Positive predictions
        self.mask = torch.randint(
            0, 2, (self.batch_size, self.n_features), device=self.device
        ).bool()
        self.weights = torch.rand(
            self.batch_size, self.n_features, device=self.device
        )  # test weights

        # Add deterministic test case for exact comparison
        self.det_y_true = torch.tensor([[2.0, 0.0], [3.0, 1.0]], device=self.device)
        self.det_y_pred = torch.tensor([[2.0, 0.5], [3.0, 1.0]], device=self.device)
        self.det_mask = torch.tensor([[True, True], [True, False]], device=self.device)
        self.det_weights = torch.tensor([[1.0, 1.0], [1.0, 1.0]], device=self.device)
        self.det_pi_logits = torch.tensor([[0.0, 2.0], [-1.0, 0.0]], device=self.device)

        # Edge cases
        self.zero_true = torch.zeros_like(self.y_true)
        self.small_pred = torch.ones_like(self.y_pred) * 1e-6
        self.large_pred = torch.ones_like(self.y_pred) * 1e6

        # Additional extreme test cases
        self.neg_true = torch.ones_like(self.y_true) * -1  # Invalid negative counts
        self.extreme_pi_logits = torch.tensor([[10.0, -10.0], [10.0, -10.0]], device=self.device)  # Extreme ZI probabilities
        self.nan_pred = torch.full_like(self.y_pred, float('nan'))  # NaN predictions

    def test_poisson_deviance(self):
        loss_fn = PoissonDevianceLoss().to(self.device)
        loss = loss_fn(self.y_true, self.y_pred, self.mask, self.weights)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

        loss_no_mask = loss_fn(self.y_true, self.y_pred, weights=self.weights)
        self.assertTrue(torch.is_tensor(loss_no_mask))
        self.assertFalse(torch.isnan(loss_no_mask).any())

        # Test with deterministic values
        det_loss = loss_fn(self.det_y_true, self.det_y_pred, self.det_mask)
        # Poisson deviance has term: (pred - true + true * (log(true) - log(pred)))
        expected_value = (
            (2 - 2 + 2 * (torch.log(2) - torch.log(2)))
            + (0.5 - 0 + 0)
            + (3 - 3 + 3 * (torch.log(3) - torch.log(3)))
        ) / 3  # Should be 0.5/3
        self.assertTrue(torch.allclose(det_loss, expected_value, rtol=1e-4))

        # Test with edge cases
        loss_zeros = loss_fn(self.zero_true, self.y_pred)
        self.assertFalse(torch.isnan(loss_zeros).any())

        loss_small = loss_fn(self.y_true, self.small_pred)
        self.assertFalse(torch.isnan(loss_small).any())

        # Test with large predictions (edge case)
        loss_large = loss_fn(self.y_true, self.large_pred)
        self.assertFalse(torch.isnan(loss_large).any())
        self.assertFalse(torch.isinf(loss_large).any())
        
        # Test with negative targets (should raise error)
        with self.assertRaises(ValueError):
            loss_fn(self.neg_true, self.y_pred)

        # Test learn variance
        loss_fn_var = PoissonDevianceLoss(learn_variance=True).to(self.device)
        loss_fn_var.zero_grad()  # Clear previous gradients
        loss_var = loss_fn_var(self.y_true, self.y_pred, self.mask, self.weights)
        self.assertTrue(torch.is_tensor(loss_var))
        self.assertFalse(torch.isnan(loss_var).any())
        loss_var.backward()
        self.assertIsNotNone(loss_fn_var.log_variance.grad)

    def test_zero_inflated_poisson_nll(self):
        loss_fn = ZeroInflatedPoissonNLLLoss().to(self.device)
        pi_logits = torch.randn(self.batch_size, self.n_features, device=self.device)  # logits
        loss = loss_fn(self.y_true, self.y_pred, pi_logits, self.mask, self.weights)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

        loss_no_mask = loss_fn(self.y_true, self.y_pred, pi_logits, weights=self.weights)
        self.assertTrue(torch.is_tensor(loss_no_mask))
        self.assertFalse(torch.isnan(loss_no_mask).any())

        # Test with deterministic values
        det_loss = loss_fn(self.det_y_true, self.det_y_pred, self.det_pi_logits, self.det_mask)
        # Calculate expected value for ZIP loss
        poisson_loss = (2 * torch.log(2) - 2 + 0.5 + 3 * torch.log(3) - 3) / 3
        # For first value (y=2): -log(sigmoid(-0.0)) = -log(0.5) ≈ 0.693
        # For second value (y=0): -log(sigmoid(2.0)) ≈ -log(0.881) ≈ 0.126
        # For third value (y=3): -log(sigmoid(-1.0)) = -log(0.269) ≈ 1.313
        zi_component = (0.693 + 0.126 + 1.313) / 3
        expected_zip_loss = -poisson_loss + zi_component
        self.assertTrue(torch.allclose(det_loss, expected_zip_loss, rtol=1e-3))

        # Test with edge cases
        loss_zeros = loss_fn(self.zero_true, self.y_pred, pi_logits)
        self.assertFalse(torch.isnan(loss_zeros).any())

        # Test with extreme zero-inflation probabilities
        extreme_loss = loss_fn(self.det_y_true, self.det_y_pred, self.extreme_pi_logits)
        self.assertFalse(torch.isnan(extreme_loss).any())
        self.assertFalse(torch.isinf(extreme_loss).any())
        
        # Test with large predictions
        large_loss = loss_fn(self.y_true, self.large_pred, pi_logits)
        self.assertFalse(torch.isnan(large_loss).any())
        self.assertFalse(torch.isinf(large_loss).any())

        # Test learn variance
        loss_fn_var = ZeroInflatedPoissonNLLLoss(learn_variance=True).to(self.device)
        loss_fn_var.zero_grad()  # Clear previous gradients
        loss_var = loss_fn_var(self.y_true, self.y_pred, pi_logits, self.mask, self.weights)
        self.assertTrue(torch.is_tensor(loss_var))
        self.assertFalse(torch.isnan(loss_var).any())
        loss_var.backward()
        self.assertIsNotNone(loss_fn_var.log_variance.grad)

    def test_negative_binomial_nll(self):
        loss_fn = NegativeBinomialNLLLoss().to(self.device)
        # Fixed theta parameter since learn_theta is False by default
        theta = 2.0

        loss = loss_fn(self.y_true, self.y_pred, self.mask, self.weights, theta)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

        loss_no_mask = loss_fn(self.y_true, self.y_pred, weights=self.weights, theta=theta)
        self.assertTrue(torch.is_tensor(loss_no_mask))
        self.assertFalse(torch.isnan(loss_no_mask).any())

        # Test with deterministic values
        det_loss = loss_fn(self.det_y_true, self.det_y_pred, self.det_mask, theta=theta)
        self.assertFalse(torch.isnan(det_loss).any())

        # Test with theta as tensor
        theta_tensor = torch.tensor(2.0, device=self.device)
        loss_tensor_theta = loss_fn(self.y_true, self.y_pred, self.mask, theta=theta_tensor)
        self.assertFalse(torch.isnan(loss_tensor_theta).any())

        # Test with very small theta (approaching Poisson distribution)
        small_theta = 1e-5
        loss_small_theta = loss_fn(self.y_true, self.y_pred, theta=small_theta)
        self.assertFalse(torch.isnan(loss_small_theta).any())
        self.assertFalse(torch.isinf(loss_small_theta).any())
        
        # Test with very large theta (approaching normal distribution)
        large_theta = 1e5
        loss_large_theta = loss_fn(self.y_true, self.y_pred, theta=large_theta)
        self.assertFalse(torch.isnan(loss_large_theta).any())
        self.assertFalse(torch.isinf(loss_large_theta).any())

        # Test learn_theta
        loss_fn_learn = NegativeBinomialNLLLoss(learn_theta=True).to(self.device)
        loss_fn_learn.zero_grad()
        loss_learn = loss_fn_learn(self.y_true, self.y_pred, self.mask)
        self.assertTrue(torch.is_tensor(loss_learn))
        self.assertFalse(torch.isnan(loss_learn).any())
        loss_learn.backward()
        self.assertIsNotNone(loss_fn_learn.log_theta.grad)

        # Test with edge cases
        loss_zeros = loss_fn(self.zero_true, self.y_pred, theta=theta)
        self.assertFalse(torch.isnan(loss_zeros).any())

        loss_small = loss_fn(self.y_true, self.small_pred, theta=theta)
        self.assertFalse(torch.isnan(loss_small).any())

    def test_poisson_likelihood_ratio_loss(self):
        """Test PoissonLikelihoodRatioLoss functionality."""
        loss_fn = PoissonLikelihoodRatioLoss().to(self.device)
        loss = loss_fn(self.y_true, self.y_pred, self.mask, self.weights)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

        loss_no_mask = loss_fn(self.y_true, self.y_pred, weights=self.weights)
        self.assertTrue(torch.is_tensor(loss_no_mask))
        self.assertFalse(torch.isnan(loss_no_mask).any())

        # Test with deterministic values
        det_loss = loss_fn(self.det_y_true, self.det_y_pred, self.det_mask)
        # Manual calculation for likelihood ratio loss
        # For first element (2.0, 2.0): 2*(2-2) + 2*2*ln(2/2) = 0
        # For second element (0.0, 0.5): 2*(0.5-0) = 1
        # For third element (3.0, 3.0): 2*(3-3) + 2*3*ln(3/3) = 0
        # Total: (0 + 1 + 0) / 3 = 1/3 = 0.333...
        expected_value = 1.0 / 3.0
        self.assertTrue(torch.allclose(det_loss, expected_value, rtol=1e-4))

        # Test with edge cases
        loss_zeros = loss_fn(self.zero_true, self.y_pred)
        self.assertFalse(torch.isnan(loss_zeros).any())

        loss_small = loss_fn(self.y_true, self.small_pred)
        self.assertFalse(torch.isnan(loss_small).any())

        # Test with large predictions
        loss_large = loss_fn(self.y_true, self.large_pred)
        self.assertFalse(torch.isnan(loss_large).any())
        self.assertFalse(torch.isinf(loss_large).any())

        # Test log_input
        loss_fn_log = PoissonLikelihoodRatioLoss(log_input=True).to(self.device)
        # Use a safer epsilon value for numerical stability
        eps = 1e-6
        log_pred = torch.log(self.y_pred + eps)
        loss_log = loss_fn_log(self.y_true, log_pred, self.mask, self.weights)
        self.assertTrue(torch.is_tensor(loss_log))
        self.assertFalse(torch.isnan(loss_log).any())

    def test_reduction_modes(self):
        """Test all loss functions with different reduction modes."""

        # Test PoissonDevianceLoss
        dev_loss_none = PoissonDevianceLoss(reduction="none").to(self.device)
        dev_loss_none_out = dev_loss_none(self.y_true, self.y_pred)
        self.assertEqual(dev_loss_none_out.shape, self.y_true.shape)

        dev_loss_mean = PoissonDevianceLoss(reduction="mean").to(self.device)
        dev_loss_sum = PoissonDevianceLoss(reduction="sum").to(self.device)
        dev_loss_mean_out = dev_loss_mean(self.y_true, self.y_pred)
        dev_loss_sum_out = dev_loss_sum(self.y_true, self.y_pred)
        self.assertTrue(torch.is_tensor(dev_loss_mean_out) and dev_loss_mean_out.numel() == 1)
        self.assertTrue(torch.is_tensor(dev_loss_sum_out) and dev_loss_sum_out.numel() == 1)

        # Test ZeroInflatedPoissonNLLLoss
        pi_logits = torch.randn(self.batch_size, self.n_features, device=self.device)
        zip_loss_none = ZeroInflatedPoissonNLLLoss(reduction="none").to(self.device)
        zip_loss_none_out = zip_loss_none(self.y_true, self.y_pred, pi_logits)
        self.assertEqual(zip_loss_none_out.shape, self.y_true.shape)

        zip_loss_mean = ZeroInflatedPoissonNLLLoss(reduction="mean").to(self.device)
        zip_loss_sum = ZeroInflatedPoissonNLLLoss(reduction="sum").to(self.device)
        zip_loss_mean_out = zip_loss_mean(self.y_true, self.y_pred, pi_logits)
        zip_loss_sum_out = zip_loss_sum(self.y_true, self.y_pred, pi_logits)
        self.assertTrue(torch.is_tensor(zip_loss_mean_out) and zip_loss_mean_out.numel() == 1)
        self.assertTrue(torch.is_tensor(zip_loss_sum_out) and zip_loss_sum_out.numel() == 1)

        # Test NegativeBinomialNLLLoss
        theta = torch.tensor(2.0, device=self.device)
        nb_loss_none = NegativeBinomialNLLLoss(reduction="none").to(self.device)
        nb_loss_none_out = nb_loss_none(self.y_true, self.y_pred, theta=theta)
        self.assertEqual(nb_loss_none_out.shape, self.y_true.shape)

        nb_loss_mean = NegativeBinomialNLLLoss(reduction="mean").to(self.device)
        nb_loss_sum = NegativeBinomialNLLLoss(reduction="sum").to(self.device)
        nb_loss_mean_out = nb_loss_mean(self.y_true, self.y_pred, theta=theta)
        nb_loss_sum_out = nb_loss_sum(self.y_true, self.y_pred, theta=theta)
        self.assertTrue(torch.is_tensor(nb_loss_mean_out) and nb_loss_mean_out.numel() == 1)
        self.assertTrue(torch.is_tensor(nb_loss_sum_out) and nb_loss_sum_out.numel() == 1)

        # Test PoissonLikelihoodRatioLoss
        lr_loss_none = PoissonLikelihoodRatioLoss(reduction="none").to(self.device)
        lr_loss_none_out = lr_loss_none(self.y_true, self.y_pred)
        self.assertEqual(lr_loss_none_out.shape, self.y_true.shape)

        lr_loss_mean = PoissonLikelihoodRatioLoss(reduction="mean").to(self.device)
        lr_loss_sum = PoissonLikelihoodRatioLoss(reduction="sum").to(self.device)
        lr_loss_mean_out = lr_loss_mean(self.y_true, self.y_pred)
        lr_loss_sum_out = lr_loss_sum(self.y_true, self.y_pred)
        self.assertTrue(torch.is_tensor(lr_loss_mean_out) and lr_loss_mean_out.numel() == 1)
        self.assertTrue(torch.is_tensor(lr_loss_sum_out) and lr_loss_sum_out.numel() == 1)

    def test_gradient_flow(self):
        """Test gradient flow through all loss functions."""
        # Create trainable parameters
        y_pred_param = (
            torch.rand(self.batch_size, self.n_features, device=self.device, requires_grad=True) * 5
        )
        pi_logits_param = torch.randn(
            self.batch_size, self.n_features, device=self.device, requires_grad=True
        )

        # Reset gradients
        y_pred_param.grad = None

        # Test PoissonDevianceLoss
        dev_loss_fn = PoissonDevianceLoss().to(self.device)
        dev_loss = dev_loss_fn(self.y_true, y_pred_param)
        dev_loss.backward()
        self.assertIsNotNone(y_pred_param.grad)
        self.assertFalse(torch.isnan(y_pred_param.grad).any())

        # Reset gradients
        y_pred_param.grad = None

        # Test ZeroInflatedPoissonNLLLoss
        zip_loss_fn = ZeroInflatedPoissonNLLLoss().to(self.device)
        zip_loss = zip_loss_fn(self.y_true, y_pred_param, pi_logits_param)
        zip_loss.backward()
        self.assertIsNotNone(y_pred_param.grad)
        self.assertIsNotNone(pi_logits_param.grad)
        self.assertFalse(torch.isnan(y_pred_param.grad).any())
        self.assertFalse(torch.isnan(pi_logits_param.grad).any())

        # Reset gradients
        y_pred_param.grad = None
        pi_logits_param.grad = None

        # Test NegativeBinomialNLLLoss with learned theta
        nb_loss_fn = NegativeBinomialNLLLoss(learn_theta=True).to(self.device)
        theta_before = nb_loss_fn.log_theta.exp().clone()
        nb_loss = nb_loss_fn(self.y_true, y_pred_param)
        nb_loss.backward()
        self.assertIsNotNone(y_pred_param.grad)
        self.assertIsNotNone(nb_loss_fn.log_theta.grad)
        self.assertFalse(torch.isnan(y_pred_param.grad).any())

        # Test if theta updates after optimization step
        optimizer = torch.optim.SGD([nb_loss_fn.log_theta], lr=0.01)
        optimizer.step()
        theta_after = nb_loss_fn.log_theta.exp()
        self.assertFalse(torch.allclose(theta_before, theta_after))

        # Reset gradients
        y_pred_param.grad = None

        # Test PoissonLikelihoodRatioLoss
        lr_loss_fn = PoissonLikelihoodRatioLoss().to(self.device)
        lr_loss = lr_loss_fn(self.y_true, y_pred_param)
        lr_loss.backward()
        self.assertIsNotNone(y_pred_param.grad)
        self.assertFalse(torch.isnan(y_pred_param.grad).any())

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        # Test negative theta for NegativeBinomialNLLLoss
        nb_loss_fn = NegativeBinomialNLLLoss().to(self.device)
        try:
            loss = nb_loss_fn(self.y_true, self.y_pred, theta=-1.0)
            # If no error, check the loss is finite
            self.assertFalse(torch.isnan(loss).any())
            self.assertFalse(torch.isinf(loss).any())
        except ValueError as e:
            # If error raised, it should mention theta
            self.assertIn("theta", str(e).lower())

        # Test invalid reduction mode
        with self.assertRaises(ValueError):
            invalid_loss_fn = PoissonDevianceLoss(reduction="invalid").to(self.device)

    def test_stateful_properties(self):
        """Test stateful properties of loss functions."""
        # Test NegativeBinomialNLLLoss with learn_theta
        nb_loss_fn = NegativeBinomialNLLLoss(learn_theta=True).to(self.device)
        initial_theta = nb_loss_fn.log_theta.exp().clone()

        # Compute loss with gradients
        y_pred_param = torch.rand(self.batch_size, self.n_features, device=self.device, requires_grad=True) * 5
        optimizer = torch.optim.SGD([nb_loss_fn.log_theta], lr=0.1)
        loss = nb_loss_fn(self.y_true, y_pred_param)
        loss.backward()
        optimizer.step()

        # Verify log_theta has changed
        updated_theta = nb_loss_fn.log_theta.exp()
        self.assertFalse(torch.allclose(initial_theta, updated_theta))

        # Test learnable variance in PoissonDevianceLoss
        loss_fn_var = PoissonDevianceLoss(learn_variance=True).to(self.device)
        initial_variance = loss_fn_var.log_variance.exp().clone()
        
        # Compute loss and update variance
        var_optimizer = torch.optim.SGD([loss_fn_var.log_variance], lr=0.1)
        var_loss = loss_fn_var(self.y_true, self.y_pred)
        var_loss.backward()
        var_optimizer.step()
        
        # Verify log_variance has changed
        updated_variance = loss_fn_var.log_variance.exp()
        self.assertFalse(torch.allclose(initial_variance, updated_variance))
        
        # Test that the variance parameter persists
        loss_fn_var2 = loss_fn_var  # Reference the same object
        self.assertTrue(torch.allclose(loss_fn_var2.log_variance, loss_fn_var.log_variance))

        # Compute another loss, should use updated variance
        new_loss = loss_fn_var2(self.y_true, self.y_pred)
        self.assertTrue(torch.is_tensor(new_loss))
        self.assertFalse(torch.isnan(new_loss).any())

    def test_nan_handling(self):
        """Test that losses handle NaN inputs properly."""
        # For PoissonDevianceLoss
        dev_loss_fn = PoissonDevianceLoss().to(self.device)
        
        # NaN targets should be masked by default
        mask = torch.ones_like(self.y_true, dtype=torch.bool)
        nan_target = self.y_true.clone()
        nan_target[0, 0] = float('nan')
        mask[0, 0] = False
        
        loss = dev_loss_fn(nan_target, self.y_pred, mask)
        self.assertFalse(torch.isnan(loss).any())
        
        # NegativeBinomialNLLLoss should handle edge cases
        nb_loss_fn = NegativeBinomialNLLLoss().to(self.device)
        theta = 1.0
        
        # Test masking NaN values
        nb_loss = nb_loss_fn(nan_target, self.y_pred, mask, theta=theta)
        self.assertFalse(torch.isnan(nb_loss).any())
        
        # ZeroInflatedPoissonNLLLoss should handle extreme probabilities
        zip_loss_fn = ZeroInflatedPoissonNLLLoss().to(self.device)
        pi_logits = torch.randn_like(self.y_pred)
        
        # Set some extreme values in pi_logits
        pi_logits[0, 0] = 100.0  # Probability ≈ 1
        pi_logits[0, 1] = -100.0  # Probability ≈ 0
        
        zip_loss = zip_loss_fn(self.y_true, self.y_pred, pi_logits)
        self.assertFalse(torch.isnan(zip_loss).any())
        
        # NaN in mask should be treated as False
        nan_mask = self.mask.clone().float()
        nan_mask[0, 0] = float('nan')
        nan_mask = nan_mask.bool()
        
        # This should not raise error and not produce NaN loss
        try:
            loss_with_nan_mask = dev_loss_fn(self.y_true, self.y_pred, nan_mask)
            self.assertFalse(torch.isnan(loss_with_nan_mask).any())
        except:
            self.fail("Loss function failed with NaN in mask")
    
    def test_numerical_gradients(self):
        """Test numerical gradient correctness using gradcheck."""
        # Prepare small tensors for gradcheck (which is computationally intensive)
        small_batch = 2
        small_features = 3
        
        y_true_small = torch.randint(1, 5, (small_batch, small_features), device=self.device).float()
        y_pred_small = torch.rand(small_batch, small_features, device=self.device).requires_grad_(True) * 3 + 0.5
        
        # PoissonDevianceLoss gradcheck
        def fn_deviance(pred):
            return PoissonDevianceLoss(reduction='sum')(pred, y_true_small)
        
        gradcheck_result = gradcheck(
            fn_deviance, 
            y_pred_small, 
            eps=1e-3, 
            atol=1e-3, 
            check_grad_dtypes=True
        )
        self.assertTrue(gradcheck_result)
        
        # PoissonLikelihoodRatioLoss gradcheck
        def fn_likelihood_ratio(pred):
            return PoissonLikelihoodRatioLoss(reduction='sum')(pred, y_true_small)
        
        gradcheck_result = gradcheck(
            fn_likelihood_ratio, 
            y_pred_small, 
            eps=1e-3, 
            atol=1e-3, 
            check_grad_dtypes=True
        )
        self.assertTrue(gradcheck_result)
        
        # For more complex losses, we'll do a simpler check
        # Testing gradient consistency for NegativeBinomialNLLLoss
        def check_grad_consistency(loss_fn, inputs, target):
            """Check if gradients are consistent when computed multiple times."""
            loss_fn.zero_grad()
            loss1 = loss_fn(target, inputs)
            loss1.backward()
            grad1 = inputs.grad.clone()
            
            inputs.grad.zero_()
            loss2 = loss_fn(target, inputs)
            loss2.backward()
            grad2 = inputs.grad.clone()
            
            # Gradients should be identical for deterministic functions
            return torch.allclose(grad1, grad2)
        
        # Test NegativeBinomialNLLLoss gradient consistency
        loss_fn = NegativeBinomialNLLLoss().to(self.device)
        inputs = torch.rand(small_batch, small_features, device=self.device, requires_grad=True)
        self.assertTrue(check_grad_consistency(loss_fn, inputs, y_true_small))
    
    def test_double_backward(self):
        """Test second-order gradients (important for some optimization algorithms)."""
        small_batch = 2
        small_features = 3
        
        y_true = torch.randint(1, 5, (small_batch, small_features), device=self.device).float()
        y_pred = torch.rand(small_batch, small_features, device=self.device).requires_grad_(True) * 3 + 0.5
        
        # Test double backward for PoissonDevianceLoss
        loss_fn = PoissonDevianceLoss().to(self.device)
        
        # First backward pass
        loss = loss_fn(y_true, y_pred)
        grad_outputs = torch.ones_like(loss)
        gradients = torch.autograd.grad(
            loss, y_pred, grad_outputs=grad_outputs, create_graph=True, retain_graph=True
        )[0]
        
        # Second backward pass
        grad_sum = gradients.sum()
        second_grads = torch.autograd.grad(grad_sum, y_pred)
        
        self.assertIsNotNone(second_grads[0])
        self.assertFalse(torch.isnan(second_grads[0]).any())
        self.assertFalse(torch.isinf(second_grads[0]).any())
        
        # Also test NegativeBinomialNLLLoss with learn_theta
        nb_loss_fn = NegativeBinomialNLLLoss(learn_theta=True).to(self.device)
        
        # First backward pass
        loss = nb_loss_fn(y_true, y_pred)
        grad_outputs = torch.ones_like(loss)
        gradients_pred = torch.autograd.grad(
            loss, y_pred, grad_outputs=grad_outputs, create_graph=True, retain_graph=True
        )[0]
        
        # Second backward pass on prediction
        grad_sum = gradients_pred.sum()
        second_grads_pred = torch.autograd.grad(
            grad_sum, y_pred, retain_graph=True
        )[0]
        
        # Second backward pass on log_theta
        second_grads_theta = torch.autograd.grad(
            grad_sum, nb_loss_fn.log_theta, retain_graph=True
        )[0]
        
        self.assertIsNotNone(second_grads_pred)
        self.assertIsNotNone(second_grads_theta)
        self.assertFalse(torch.isnan(second_grads_pred).any())
        self.assertFalse(torch.isnan(second_grads_theta).any())

    def test_shape_mismatch(self):
        """Test error handling for shape mismatches."""
        wrong_shape = torch.rand(self.batch_size + 1, self.n_features, device=self.device)
        
        # Test PoissonDevianceLoss
        dev_loss_fn = PoissonDevianceLoss().to(self.device)
        with self.assertRaises(ValueError):
            dev_loss_fn(self.y_true, wrong_shape)
        
        # Test ZeroInflatedPoissonNLLLoss
        zip_loss_fn = ZeroInflatedPoissonNLLLoss().to(self.device)
        pi_logits = torch.randn_like(self.y_true)
        with self.assertRaises(ValueError):
            zip_loss_fn(self.y_true, wrong_shape, pi_logits)
        
        # Test when pi_logits has wrong shape
        wrong_pi_shape = torch.randn(self.batch_size + 1, self.n_features, device=self.device)
        with self.assertRaises(ValueError):
            zip_loss_fn(self.y_true, self.y_pred, wrong_pi_shape)
        
        # Test NegativeBinomialNLLLoss
        nb_loss_fn = NegativeBinomialNLLLoss().to(self.device)
        with self.assertRaises(ValueError):
            nb_loss_fn(self.y_true, wrong_shape)
        
        # Test PoissonLikelihoodRatioLoss
        lr_loss_fn = PoissonLikelihoodRatioLoss().to(self.device)
        with self.assertRaises(ValueError):
            lr_loss_fn(self.y_true, wrong_shape)

if __name__ == "__main__":
    unittest.main()
