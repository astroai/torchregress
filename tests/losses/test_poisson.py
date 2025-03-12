import torch
import unittest
from torch.autograd import gradcheck
from torchregression.losses.poisson import (
    PoissonNLL,
    PoissonDeviance,
    ZeroInflatedPoissonNLL,
    NegativeBinomialNLL,
    PoissonNLLLoss,
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

    def test_poisson_nll(self):
        loss_fn = PoissonNLL().to(self.device)
        loss = loss_fn(self.y_true, self.y_pred, self.mask, self.weights)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

        loss_no_mask = loss_fn(self.y_true, self.y_pred, weights=self.weights)
        self.assertTrue(torch.is_tensor(loss_no_mask))
        self.assertFalse(torch.isnan(loss_no_mask).any())

        # Test with deterministic values
        det_loss = loss_fn(self.det_y_true, self.det_y_pred, self.det_mask)
        # Using approx equal for floating point comparison
        expected_value = (
            2 * torch.log(2) - 2 + 0.5 + 3 * torch.log(3) - 3
        ) / 3  # Manual calculation of Poisson NLL
        self.assertTrue(torch.allclose(det_loss, -expected_value, rtol=1e-4))

        # Test log_input
        loss_fn_log = PoissonNLL(log_input=True).to(self.device)
        # Use a safer epsilon value for numerical stability
        eps = 1e-6
        log_pred = torch.log(self.y_pred + eps)
        loss_log = loss_fn_log(self.y_true, log_pred, self.mask, self.weights)
        self.assertTrue(torch.is_tensor(loss_log))
        self.assertFalse(torch.isnan(loss_log).any())

        # Test with edge cases
        loss_zeros = loss_fn(self.zero_true, self.y_pred)
        self.assertFalse(torch.isnan(loss_zeros).any())

        loss_small = loss_fn(self.y_true, self.small_pred)
        self.assertFalse(torch.isnan(loss_small).any())

        # Test learn variance
        loss_fn_var = PoissonNLL(learn_variance=True).to(self.device)
        loss_fn_var.zero_grad()  # Clear previous gradients if any
        loss_var = loss_fn_var(self.y_true, self.y_pred, self.mask, self.weights)
        self.assertTrue(torch.is_tensor(loss_var))
        self.assertFalse(torch.isnan(loss_var).any())
        loss_var.backward()
        self.assertIsNotNone(loss_fn_var.log_variance.grad)  # Check gradient

    def test_poisson_deviance(self):
        loss_fn = PoissonDeviance().to(self.device)
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

        # Test learn variance
        loss_fn_var = PoissonDeviance(learn_variance=True).to(self.device)
        loss_fn_var.zero_grad()  # Clear previous gradients
        loss_var = loss_fn_var(self.y_true, self.y_pred, self.mask, self.weights)
        self.assertTrue(torch.is_tensor(loss_var))
        self.assertFalse(torch.isnan(loss_var).any())
        loss_var.backward()
        self.assertIsNotNone(loss_fn_var.log_variance.grad)

    def test_zero_inflated_poisson_nll(self):
        loss_fn = ZeroInflatedPoissonNLL().to(self.device)
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

        # Test learn variance
        loss_fn_var = ZeroInflatedPoissonNLL(learn_variance=True).to(self.device)
        loss_fn_var.zero_grad()  # Clear previous gradients
        loss_var = loss_fn_var(self.y_true, self.y_pred, pi_logits, self.mask, self.weights)
        self.assertTrue(torch.is_tensor(loss_var))
        self.assertFalse(torch.isnan(loss_var).any())
        loss_var.backward()
        self.assertIsNotNone(loss_fn_var.log_variance.grad)

    def test_negative_binomial_nll(self):
        loss_fn = NegativeBinomialNLL().to(self.device)
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

        # Test learn_theta
        loss_fn_learn = NegativeBinomialNLL(learn_theta=True).to(self.device)
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

    def test_nans(self):
        loss_fn = PoissonNLL().to(self.device)

        y_true_nan = self.y_true.clone()
        y_true_nan[0, 0] = float("nan")
        loss = loss_fn(y_true_nan, self.y_pred, self.mask)
        self.assertFalse(torch.isnan(loss).any(), "Loss should handle NaN in y_true")

        y_pred_nan = self.y_pred.clone()
        y_pred_nan[0, 0] = float("nan")
        loss = loss_fn(self.y_true, y_pred_nan, self.mask)
        self.assertFalse(torch.isnan(loss).any(), "Loss should handle NaN in y_pred due to masking")

        mask_nan = self.mask.clone().float()  # Convert to float for NaN assignment
        mask_nan[0, 0] = float("nan")
        loss = loss_fn(self.y_true, self.y_pred, mask_nan)
        self.assertFalse(torch.isnan(loss).any(), "Loss should handle NaN in mask")

        weights_nan = self.weights.clone()
        weights_nan[0, 0] = float("nan")
        loss = loss_fn(self.y_true, self.y_pred, self.mask, weights_nan)
        self.assertFalse(torch.isnan(loss).any(), "Loss should handle NaN in weights")

    def test_inf(self):
        loss_fn = PoissonNLL().to(self.device)

        y_true_inf = self.y_true.clone()
        y_true_inf[0, 0] = float("inf")
        # Create a mask to avoid the inf value
        mask = self.mask.clone()
        mask[0, 0] = False
        loss = loss_fn(y_true_inf, self.y_pred, mask)
        self.assertFalse(
            torch.isinf(loss).any(), "Loss should not be inf when inf values are masked out"
        )

        # Test with inf that should be included in calculation
        loss_with_inf = loss_fn(y_true_inf, self.y_pred, self.mask)
        self.assertTrue(
            torch.isinf(loss_with_inf).any(), "Loss should be inf when inf values are included"
        )

        y_pred_inf = self.y_pred.clone()
        y_pred_inf[0, 0] = float("inf")
        mask[0, 0] = False
        loss = loss_fn(self.y_true, y_pred_inf, mask)
        self.assertFalse(
            torch.isinf(loss).any(), "Loss should not be inf when inf values are masked out"
        )

        # Test proper handling of infinite values in mask (should be treated as boolean True)
        mask_inf = self.mask.clone().float()
        mask_inf[0, 0] = float("inf")  # Should be cast to True
        loss = loss_fn(self.y_true, self.y_pred, mask_inf)
        self.assertFalse(
            torch.isnan(loss).any(), "Loss should handle inf in mask by treating as True"
        )

        # Test with finite weights to control the inf contribution
        weights_with_zero = self.weights.clone()
        weights_with_zero[0, 0] = 0.0  # Zero out the weight where we have inf
        loss = loss_fn(y_true_inf, self.y_pred, self.mask, weights_with_zero)
        self.assertFalse(
            torch.isinf(loss).any(), "Loss should not be inf when inf values have zero weight"
        )

        weights_inf = self.weights.clone()
        weights_inf[0, 0] = float("inf")
        loss = loss_fn(self.y_true, self.y_pred, self.mask, weights_inf)
        self.assertTrue(torch.isinf(loss).any(), "Loss should be inf when weights are inf")

    def test_reduction_modes(self):
        """Test all loss functions with different reduction modes."""
        # Test PoissonNLL
        loss_none = PoissonNLL(reduction="none").to(self.device)
        loss_none_out = loss_none(self.y_true, self.y_pred)
        self.assertEqual(loss_none_out.shape, self.y_true.shape)

        loss_mean = PoissonNLL(reduction="mean").to(self.device)
        loss_sum = PoissonNLL(reduction="sum").to(self.device)
        loss_mean_out = loss_mean(self.y_true, self.y_pred)
        loss_sum_out = loss_sum(self.y_true, self.y_pred)
        self.assertTrue(torch.is_tensor(loss_mean_out) and loss_mean_out.numel() == 1)
        self.assertTrue(torch.is_tensor(loss_sum_out) and loss_sum_out.numel() == 1)
        self.assertAlmostEqual(
            loss_mean_out.item(),
            loss_sum_out.item() / (self.batch_size * self.n_features),
            places=5,
        )

        # Test PoissonDeviance
        dev_loss_none = PoissonDeviance(reduction="none").to(self.device)
        dev_loss_none_out = dev_loss_none(self.y_true, self.y_pred)
        self.assertEqual(dev_loss_none_out.shape, self.y_true.shape)

        dev_loss_mean = PoissonDeviance(reduction="mean").to(self.device)
        dev_loss_sum = PoissonDeviance(reduction="sum").to(self.device)
        dev_loss_mean_out = dev_loss_mean(self.y_true, self.y_pred)
        dev_loss_sum_out = dev_loss_sum(self.y_true, self.y_pred)
        self.assertTrue(torch.is_tensor(dev_loss_mean_out) and dev_loss_mean_out.numel() == 1)
        self.assertTrue(torch.is_tensor(dev_loss_sum_out) and dev_loss_sum_out.numel() == 1)

        # Test ZeroInflatedPoissonNLL
        pi_logits = torch.randn(self.batch_size, self.n_features, device=self.device)
        zip_loss_none = ZeroInflatedPoissonNLL(reduction="none").to(self.device)
        zip_loss_none_out = zip_loss_none(self.y_true, self.y_pred, pi_logits)
        self.assertEqual(zip_loss_none_out.shape, self.y_true.shape)

        zip_loss_mean = ZeroInflatedPoissonNLL(reduction="mean").to(self.device)
        zip_loss_sum = ZeroInflatedPoissonNLL(reduction="sum").to(self.device)
        zip_loss_mean_out = zip_loss_mean(self.y_true, self.y_pred, pi_logits)
        zip_loss_sum_out = zip_loss_sum(self.y_true, self.y_pred, pi_logits)
        self.assertTrue(torch.is_tensor(zip_loss_mean_out) and zip_loss_mean_out.numel() == 1)
        self.assertTrue(torch.is_tensor(zip_loss_sum_out) and zip_loss_sum_out.numel() == 1)

        # Test NegativeBinomialNLL
        theta = torch.tensor(2.0, device=self.device)
        nb_loss_none = NegativeBinomialNLL(reduction="none").to(self.device)
        nb_loss_none_out = nb_loss_none(self.y_true, self.y_pred, theta=theta)
        self.assertEqual(nb_loss_none_out.shape, self.y_true.shape)

        nb_loss_mean = NegativeBinomialNLL(reduction="mean").to(self.device)
        nb_loss_sum = NegativeBinomialNLL(reduction="sum").to(self.device)
        nb_loss_mean_out = nb_loss_mean(self.y_true, self.y_pred, theta=theta)
        nb_loss_sum_out = nb_loss_sum(self.y_true, self.y_pred, theta=theta)
        self.assertTrue(torch.is_tensor(nb_loss_mean_out) and nb_loss_mean_out.numel() == 1)
        self.assertTrue(torch.is_tensor(nb_loss_sum_out) and nb_loss_sum_out.numel() == 1)

    def test_gradient_flow(self):
        """Test gradient flow through all loss functions."""
        # Create trainable parameters
        y_pred_param = (
            torch.rand(self.batch_size, self.n_features, device=self.device, requires_grad=True) * 5
        )
        pi_logits_param = torch.randn(
            self.batch_size, self.n_features, device=self.device, requires_grad=True
        )

        # Test PoissonNLL
        loss_fn = PoissonNLL().to(self.device)
        loss = loss_fn(self.y_true, y_pred_param)
        loss.backward()
        self.assertIsNotNone(y_pred_param.grad)
        self.assertFalse(torch.isnan(y_pred_param.grad).any())

        # Reset gradients
        y_pred_param.grad = None

        # Test PoissonDeviance
        dev_loss_fn = PoissonDeviance().to(self.device)
        dev_loss = dev_loss_fn(self.y_true, y_pred_param)
        dev_loss.backward()
        self.assertIsNotNone(y_pred_param.grad)
        self.assertFalse(torch.isnan(y_pred_param.grad).any())

        # Reset gradients
        y_pred_param.grad = None

        # Test ZeroInflatedPoissonNLL
        zip_loss_fn = ZeroInflatedPoissonNLL().to(self.device)
        zip_loss = zip_loss_fn(self.y_true, y_pred_param, pi_logits_param)
        zip_loss.backward()
        self.assertIsNotNone(y_pred_param.grad)
        self.assertIsNotNone(pi_logits_param.grad)
        self.assertFalse(torch.isnan(y_pred_param.grad).any())
        self.assertFalse(torch.isnan(pi_logits_param.grad).any())

        # Reset gradients
        y_pred_param.grad = None
        pi_logits_param.grad = None

        # Test NegativeBinomialNLL with learned theta
        nb_loss_fn = NegativeBinomialNLL(learn_theta=True).to(self.device)
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

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        # Test negative predictions
        loss_fn = PoissonNLL().to(self.device)
        negative_preds = -torch.rand(self.batch_size, self.n_features, device=self.device)

        # Should handle negative predictions correctly (e.g., by clamping or error)
        try:
            loss = loss_fn(self.y_true, negative_preds)
            # If no error, check the loss is finite
            self.assertFalse(torch.isnan(loss).any())
            self.assertFalse(torch.isinf(loss).any())
        except ValueError as e:
            # If error raised, it should mention negative values
            self.assertIn("negative", str(e).lower())

        # Test negative theta for NegativeBinomialNLL
        nb_loss_fn = NegativeBinomialNLL().to(self.device)
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
            invalid_loss_fn = PoissonNLL(reduction="invalid").to(self.device)

    def test_stateful_properties(self):
        """Test stateful properties of loss functions."""
        # Test PoissonNLL with learn_variance
        loss_fn_var = PoissonNLL(learn_variance=True).to(self.device)
        initial_variance = loss_fn_var.log_variance.exp().clone()

        # Create trainable parameter
        y_pred_param = (
            torch.rand(self.batch_size, self.n_features, device=self.device, requires_grad=True) * 5
        )

        # Compute loss with gradients
        optimizer = torch.optim.SGD([loss_fn_var.log_variance], lr=0.1)
        loss = loss_fn_var(self.y_true, y_pred_param)
        loss.backward()
        optimizer.step()

        # Verify log_variance has changed
        updated_variance = loss_fn_var.log_variance.exp()
        self.assertFalse(torch.allclose(initial_variance, updated_variance))

        # Test NegativeBinomialNLL with learn_theta
        nb_loss_fn = NegativeBinomialNLL(learn_theta=True).to(self.device)
        initial_theta = nb_loss_fn.log_theta.exp().clone()

        # Compute loss with gradients
        optimizer = torch.optim.SGD([nb_loss_fn.log_theta], lr=0.1)
        loss = nb_loss_fn(self.y_true, y_pred_param)
        loss.backward()
        optimizer.step()

        # Verify log_theta has changed
        updated_theta = nb_loss_fn.log_theta.exp()
        self.assertFalse(torch.allclose(initial_theta, updated_theta))

        # Test that the variance parameter persists
        loss_fn_var2 = loss_fn_var  # Reference the same object
        self.assertTrue(torch.allclose(loss_fn_var2.log_variance, loss_fn_var.log_variance))

        # Compute another loss, should use updated variance
        new_loss = loss_fn_var2(self.y_true, self.y_pred)
        self.assertTrue(torch.is_tensor(new_loss))
        self.assertFalse(torch.isnan(new_loss).any())

    def test_poisson_edge_cases(self):
        """Test PoissonNLLLoss with edge cases: zeros, empty tensors, extreme values, NaN/Inf."""
        loss_fn = PoissonNLLLoss(log_input=False)

        # Test with zeros
        y_pred_zeros = torch.zeros(10)
        y_true_zeros = torch.zeros(10)
        assert torch.isfinite(loss_fn(y_pred_zeros, y_true_zeros))

        # Test with empty tensors
        y_pred_empty = torch.tensor([])
        y_true_empty = torch.tensor([])
        assert loss_fn(y_pred_empty, y_true_empty).numel() == 0

        # Test with extreme values - note: Poisson requires non-negative values
        y_pred_large = torch.tensor([1e10])
        y_true_large = torch.tensor([1e5])
        assert torch.isfinite(loss_fn(y_pred_large, y_true_large))

        # Test with small positive values
        y_pred_small = torch.tensor([1e-10])
        y_true_small = torch.tensor([1e-10])
        assert torch.isfinite(loss_fn(y_pred_small, y_true_small))

        # Test with NaN/Inf and masks
        y_pred = torch.tensor([1.0, float("nan"), 3.0])
        y_true = torch.tensor([1.5, 2.5, float("inf")])
        mask = torch.tensor([True, False, False])
        assert torch.isfinite(loss_fn(y_pred, y_true, mask))


class TestPoissonLossNumericalStability:
    def test_poisson_loss_gradient_flow(self):
        """Test that gradients flow through PoissonNLLLoss properly."""
        # Create positive inputs that require gradients
        y_pred = torch.exp(torch.randn(10, 1, requires_grad=True, dtype=torch.double))
        y_true = torch.abs(torch.randn(10, 1, dtype=torch.double)) + 1.0  # positive values

        # Test with gradcheck
        loss_fn = PoissonNLLLoss(reduction="mean", log_input=False)
        assert gradcheck(loss_fn, (y_pred, y_true), eps=1e-6, atol=1e-4)

        # Test with log input
        y_pred_log = torch.log(y_pred.detach()).requires_grad_(True)
        loss_fn_log = PoissonNLLLoss(reduction="mean", log_input=True)
        assert gradcheck(loss_fn_log, (y_pred_log, y_true), eps=1e-6, atol=1e-4)

    def test_extreme_values(self):
        """Test stability with extreme values."""
        # Small values (close to zero)
        y_pred_small = torch.tensor([1e-5, 1e-7, 1e-9], requires_grad=True)
        # Large values
        y_pred_large = torch.tensor([1e3, 1e4, 1e5], requires_grad=True)

        y_true = torch.tensor([1.0, 10.0, 100.0])

        # Test with non-log input
        poisson_loss = PoissonNLLLoss(reduction="mean", log_input=False)

        # Small values test
        loss = poisson_loss(y_pred_small, y_true)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.all(torch.isfinite(y_pred_small.grad))

        # Large values test
        y_pred_large.grad = None
        loss = poisson_loss(y_pred_large, y_true)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.all(torch.isfinite(y_pred_large.grad))

        # Test with log input
        log_poisson_loss = PoissonNLLLoss(reduction="mean", log_input=True)

        # Log of small/large values
        y_pred_log_small = torch.log(y_pred_small.detach()).requires_grad_(True)
        y_pred_log_large = torch.log(y_pred_large.detach()).requires_grad_(True)

        # Small values test (log input)
        loss = log_poisson_loss(y_pred_log_small, y_true)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.all(torch.isfinite(y_pred_log_small.grad))

        # Large values test (log input)
        y_pred_log_large.grad = None
        loss = log_poisson_loss(y_pred_log_large, y_true)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.all(torch.isfinite(y_pred_log_large.grad))

    def test_zeros_in_target(self):
        """Test handling of zeros in target."""
        # Predictions with gradient
        y_pred = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        # Targets with zeros
        y_true = torch.tensor([0.0, 1.0, 0.0])

        # Non-log input
        poisson_loss = PoissonNLLLoss(reduction="mean", log_input=False)
        loss = poisson_loss(y_pred, y_true)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.all(torch.isfinite(y_pred.grad))

        # Log input
        y_pred.grad = None
        log_pred = torch.log(y_pred).requires_grad_(True)
        log_poisson_loss = PoissonNLLLoss(reduction="mean", log_input=True)
        loss = log_poisson_loss(log_pred, y_true)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.all(torch.isfinite(log_pred.grad))

    def test_reduction_modes(self):
        """Test different reduction modes for backward pass."""
        y_pred = torch.exp(torch.randn(10, 1, requires_grad=True))
        y_true = torch.abs(torch.randn(10, 1)) + 0.1  # positive values

        # Test mean reduction
        poisson_mean = PoissonNLLLoss(reduction="mean", log_input=False)
        loss = poisson_mean(y_pred, y_true)
        loss.backward()
        mean_grad = y_pred.grad.clone()

        # Test sum reduction
        y_pred.grad = None
        poisson_sum = PoissonNLLLoss(reduction="sum", log_input=False)
        loss = poisson_sum(y_pred, y_true)
        loss.backward()
        sum_grad = y_pred.grad.clone()

        # Test none reduction
        y_pred.grad = None
        poisson_none = PoissonNLLLoss(reduction="none", log_input=False)
        loss = poisson_none(y_pred, y_true)
        loss.mean().backward()
        none_grad = y_pred.grad.clone()

        # Mean and sum should give different gradients
        assert not torch.allclose(mean_grad, sum_grad)
        # Mean and manual mean over none should be similar
        assert torch.allclose(mean_grad, none_grad, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
