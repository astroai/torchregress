import torch
import unittest
from torchregression.poisson import PoissonNLL, ModifiedPoissonNLL, ZeroInflatedPoissonNLL, NegativeBinomialNLL

class TestPoissonLosses(unittest.TestCase):
    def setUp(self):
        # Use fixed seed for reproducibility
        torch.manual_seed(42)
        self.batch_size = 4
        self.n_features = 5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.y_true = torch.randint(0, 10, (self.batch_size, self.n_features), device=self.device).float() # Integer counts
        self.y_pred = torch.rand(self.batch_size, self.n_features, device=self.device) * 5  # Positive predictions
        self.mask = torch.randint(0, 2, (self.batch_size, self.n_features), device=self.device).bool()
        self.weights = torch.rand(self.batch_size, self.n_features, device=self.device) #test weights
        
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
        expected_value = (2*torch.log(2) - 2 + 0.5 + 3*torch.log(3) - 3) / 3  # Manual calculation of Poisson NLL
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

    def test_modified_poisson_nll(self):
        loss_fn = ModifiedPoissonNLL().to(self.device)
        loss = loss_fn(self.y_true, self.y_pred, self.mask, self.weights)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

        loss_no_mask = loss_fn(self.y_true, self.y_pred, weights=self.weights)
        self.assertTrue(torch.is_tensor(loss_no_mask))
        self.assertFalse(torch.isnan(loss_no_mask).any())
        
        # Test with deterministic values
        det_loss = loss_fn(self.det_y_true, self.det_y_pred, self.det_mask)
        # Modified Poisson loss has term: (pred - true + true * (log(true) - log(pred)))
        expected_value = ((2 - 2 + 2*(torch.log(2) - torch.log(2))) + 
                         (0.5 - 0 + 0) + 
                         (3 - 3 + 3*(torch.log(3) - torch.log(3)))) / 3  # Should be 0.5/3
        self.assertTrue(torch.allclose(det_loss, expected_value, rtol=1e-4))
        
        # Test with edge cases
        loss_zeros = loss_fn(self.zero_true, self.y_pred)
        self.assertFalse(torch.isnan(loss_zeros).any())
        
        loss_small = loss_fn(self.y_true, self.small_pred)
        self.assertFalse(torch.isnan(loss_small).any())

        # Test learn variance
        loss_fn_var = ModifiedPoissonNLL(learn_variance=True).to(self.device)
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
        poisson_loss = (2*torch.log(2) - 2 + 0.5 + 3*torch.log(3) - 3) / 3
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
        y_true_nan[0,0] = float('nan')
        loss = loss_fn(y_true_nan, self.y_pred, self.mask)
        self.assertFalse(torch.isnan(loss).any(), "Loss should handle NaN in y_true")

        y_pred_nan = self.y_pred.clone()
        y_pred_nan[0,0] = float('nan')
        loss = loss_fn(self.y_true, y_pred_nan, self.mask)
        self.assertFalse(torch.isnan(loss).any(), "Loss should handle NaN in y_pred due to masking")

        mask_nan = self.mask.clone().float()  # Convert to float for NaN assignment
        mask_nan[0, 0] = float('nan')
        loss = loss_fn(self.y_true, self.y_pred, mask_nan)
        self.assertFalse(torch.isnan(loss).any(), "Loss should handle NaN in mask")

        weights_nan = self.weights.clone()
        weights_nan[0,0] = float('nan')
        loss = loss_fn(self.y_true, self.y_pred, self.mask, weights_nan)
        self.assertFalse(torch.isnan(loss).any(), "Loss should handle NaN in weights")

    def test_inf(self):
        loss_fn = PoissonNLL().to(self.device)

        y_true_inf = self.y_true.clone()
        y_true_inf[0,0] = float('inf')
        # Create a mask to avoid the inf value
        mask = self.mask.clone()
        mask[0,0] = False
        loss = loss_fn(y_true_inf, self.y_pred, mask)
        self.assertFalse(torch.isinf(loss).any(), "Loss should not be inf when inf values are masked out")

        # Test with inf that should be included in calculation
        loss_with_inf = loss_fn(y_true_inf, self.y_pred, self.mask)
        self.assertTrue(torch.isinf(loss_with_inf).any(), "Loss should be inf when inf values are included")

        y_pred_inf = self.y_pred.clone()
        y_pred_inf[0,0] = float('inf')
        mask[0,0] = False
        loss = loss_fn(self.y_true, y_pred_inf, mask)
        self.assertFalse(torch.isinf(loss).any(), "Loss should not be inf when inf values are masked out")

        # Test proper handling of infinite values in mask (should be treated as boolean True)
        mask_inf = self.mask.clone().float()
        mask_inf[0,0] = float('inf')  # Should be cast to True
        loss = loss_fn(self.y_true, self.y_pred, mask_inf)
        self.assertFalse(torch.isnan(loss).any(), "Loss should handle inf in mask by treating as True")
        
        # Test with finite weights to control the inf contribution
        weights_with_zero = self.weights.clone()
        weights_with_zero[0,0] = 0.0  # Zero out the weight where we have inf
        loss = loss_fn(y_true_inf, self.y_pred, self.mask, weights_with_zero)
        self.assertFalse(torch.isinf(loss).any(), "Loss should not be inf when inf values have zero weight")

        weights_inf = self.weights.clone()
        weights_inf[0,0] = float('inf')
        loss = loss_fn(self.y_true, self.y_pred, self.mask, weights_inf)
        self.assertTrue(torch.isinf(loss).any(), "Loss should be inf when weights are inf")

if __name__ == '__main__':
    unittest.main()