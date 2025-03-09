import torch
import unittest
from torchregression.tweedie import TweedieLoss

class TestTweedieLoss(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.n_features = 5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.y_true = torch.randn(self.batch_size, self.n_features, device=self.device).float()
        # For Gamma and compound tests, we need positive y_true values
        self.y_true_pos = torch.rand(self.batch_size, self.n_features, device=self.device) * 5 + 0.1  # Positive true values
        self.y_pred = torch.rand(self.batch_size, self.n_features, device=self.device) * 5 + 0.1  # Positive predictions
        self.mask = torch.randint(0, 2, (self.batch_size, self.n_features), device=self.device).bool()
        self.weights = torch.rand(self.batch_size, self.n_features, device=self.device) # test weights

    def test_tweedie_loss_gaussian(self):
        """Test Tweedie loss with p=0 (Gaussian distribution)"""
        loss_fn = TweedieLoss(p=0.0).to(self.device) #Gaussian
        loss = loss_fn(self.y_true, self.y_pred, self.mask)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

    def test_tweedie_loss_poisson(self):
        """Test Tweedie loss with p=1 (Poisson distribution)"""
        loss_fn = TweedieLoss(p=1.0).to(self.device) #Poisson
        loss = loss_fn(self.y_true_pos, self.y_pred, self.mask)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

    def test_tweedie_loss_gamma(self):
        """Test Tweedie loss with p=2 (Gamma distribution)"""
        loss_fn = TweedieLoss(p=2.0).to(self.device) #Gamma
        loss = loss_fn(self.y_true_pos, self.y_pred, self.mask)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

    def test_tweedie_loss_compound(self):
        """Test Tweedie loss with p=1.5 (Compound Poisson-Gamma distribution)"""
        loss_fn = TweedieLoss(p=1.5).to(self.device) #Compound Poisson-Gamma
        loss = loss_fn(self.y_true_pos, self.y_pred, self.mask)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

    def test_tweedie_loss_weighted(self):
        """Test Tweedie loss with sample weights"""
        loss_fn = TweedieLoss(p=1.5).to(self.device)
        loss_unweighted = loss_fn(self.y_true_pos, self.y_pred, self.mask)
        loss_weighted = loss_fn(self.y_true_pos, self.y_pred, self.mask, self.weights)
        self.assertTrue(torch.is_tensor(loss_weighted))
        self.assertFalse(torch.isnan(loss_weighted).any())
        # Weighted and unweighted should be different
        self.assertNotEqual(loss_unweighted.item(), loss_weighted.item())

    def test_tweedie_loss_reduction(self):
        """Test different reduction methods"""
        # Test mean reduction (default)
        loss_fn_mean = TweedieLoss(p=1.0, reduction='mean').to(self.device)
        loss_mean = loss_fn_mean(self.y_true_pos, self.y_pred, self.mask)
        self.assertTrue(torch.is_tensor(loss_mean))
        self.assertEqual(loss_mean.dim(), 0)  # Scalar output
        
        # Test sum reduction
        loss_fn_sum = TweedieLoss(p=1.0, reduction='sum').to(self.device)
        loss_sum = loss_fn_sum(self.y_true_pos, self.y_pred, self.mask)
        self.assertTrue(torch.is_tensor(loss_sum))
        self.assertEqual(loss_sum.dim(), 0)  # Scalar output
        
        # Test none reduction
        loss_fn_none = TweedieLoss(p=1.0, reduction='none').to(self.device)
        loss_none = loss_fn_none(self.y_true_pos, self.y_pred, self.mask)
        self.assertTrue(torch.is_tensor(loss_none))
        self.assertEqual(loss_none.shape, self.y_true_pos.shape)  # Per-element loss

    def test_tweedie_loss_nans(self):
        """Test that NaNs are handled correctly in inputs"""
        loss_fn = TweedieLoss(p=1.5).to(self.device)

        # Test NaN in true values
        y_true_nan = self.y_true_pos.clone()
        y_true_nan[0, 0] = float('nan')
        loss = loss_fn(y_true_nan, self.y_pred, self.mask)
        self.assertFalse(torch.isnan(loss).any())

        # Test NaN in predictions
        y_pred_nan = self.y_pred.clone()
        y_pred_nan[0, 0] = float('nan')
        loss = loss_fn(self.y_true_pos, y_pred_nan, self.mask)
        self.assertFalse(torch.isnan(loss).any())

        # Test NaN in mask - using boolean mask
        mask_nan = self.mask.clone()
        # Set an entry to False that would have a NaN in the calculation
        y_true_nan[0, 1] = float('nan')
        mask_nan[0, 1] = False  # Mask out the NaN value
        loss = loss_fn(y_true_nan, self.y_pred, mask_nan)
        self.assertFalse(torch.isnan(loss).any())

    def test_tweedie_loss_edge_cases(self):
        """Test edge cases like zero predictions and extreme values"""
        loss_fn = TweedieLoss(p=1.5).to(self.device)
        
        # Test with zero predictions (should handle numerical stability)
        y_pred_zeros = torch.zeros_like(self.y_pred)
        # Make a mask that avoids mathematical impossibilities
        safe_mask = self.mask.clone()
        safe_mask[self.y_true_pos == 0] = False
        loss = loss_fn(self.y_true_pos, y_pred_zeros, safe_mask)
        self.assertFalse(torch.isnan(loss).any())
        self.assertFalse(torch.isinf(loss).any())
        
        # Test with very small predictions
        y_pred_small = torch.ones_like(self.y_pred) * 1e-10
        loss = loss_fn(self.y_true_pos, y_pred_small, safe_mask)
        self.assertFalse(torch.isnan(loss).any())
        self.assertFalse(torch.isinf(loss).any())
        
        # Test with very large predictions
        y_pred_large = torch.ones_like(self.y_pred) * 1e10
        loss = loss_fn(self.y_true_pos, y_pred_large, safe_mask)
        self.assertFalse(torch.isnan(loss).any())
        self.assertFalse(torch.isinf(loss).any())

    def test_tweedie_loss_invalid_p(self):
        """Test that invalid p values raise appropriate errors"""
        with self.assertRaises(ValueError):
            TweedieLoss(p=0.5)  # Invalid p
        with self.assertRaises(ValueError):
            TweedieLoss(p=-1)  # Invalid p

if __name__ == '__main__':
    unittest.main()