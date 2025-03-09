import torch
import unittest
from torchregression.quantile import QuantileLoss, PinballLoss

class TestQuantileLosses(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.n_features = 5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.y_true = torch.randn(self.batch_size, self.n_features, device=self.device)
        self.y_pred = torch.randn(self.batch_size, self.n_features, device=self.device)
        self.mask = torch.randint(0, 2, (self.batch_size, self.n_features), device=self.device).bool()

    def test_quantile_loss(self):
        loss_fn = QuantileLoss(tau=0.5).to(self.device) #test median
        loss = loss_fn(self.y_true, self.y_pred, self.mask)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

        loss_no_mask = loss_fn(self.y_true, self.y_pred)
        self.assertTrue(torch.is_tensor(loss_no_mask))
        self.assertFalse(torch.isnan(loss_no_mask).any())

    def test_pinball_loss(self):
        loss_fn = PinballLoss(quantile=0.25).to(self.device) #test different quantile
        loss = loss_fn(self.y_true, self.y_pred, self.mask)
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss).any())

        loss_no_mask = loss_fn(self.y_true, self.y_pred)
        self.assertTrue(torch.is_tensor(loss_no_mask))
        self.assertFalse(torch.isnan(loss_no_mask).any())

    def test_nans(self):
      loss_fn = QuantileLoss(tau=0.5).to(self.device)

      y_true_nan = self.y_true.clone()
      y_true_nan[0,0] = float('nan')
      loss = loss_fn(y_true_nan, self.y_pred, self.mask)
      self.assertFalse(torch.isnan(loss).any()) #Should still gives a value

      y_pred_nan = self.y_pred.clone()
      y_pred_nan[0,0] = float('nan')
      loss = loss_fn(self.y_true, y_pred_nan, self.mask)
      self.assertFalse(torch.isnan(loss).any()) #Should still gives a value


      mask_nan = self.mask.clone().float()
      mask_nan[0,0] = float('nan')
      loss = loss_fn(self.y_true, self.y_pred, mask_nan)
      self.assertFalse(torch.isnan(loss).any()) #Should still gives a value


    def test_inf(self):
      loss_fn = QuantileLoss(tau=0.5).to(self.device)
      y_true_inf = self.y_true.clone()
      y_true_inf[0,0] = float('inf')
      loss = loss_fn(y_true_inf, self.y_pred, self.mask)
      self.assertTrue(torch.isinf(loss).any())

      y_pred_inf = self.y_pred.clone()
      y_pred_inf[0,0] = float('inf')
      loss = loss_fn(self.y_true, y_pred_inf, self.mask)
      self.assertTrue(torch.isinf(loss).any())


      mask_inf = self.mask.clone().float()
      mask_inf[0,0] = float('inf') #Should be casted to True
      loss = loss_fn(self.y_true, self.y_pred, mask_inf)
      self.assertFalse(torch.isnan(loss).any()) #Should still gives a value

if __name__ == '__main__':
    unittest.main()