import torch
from .base import MaskedLoss

class QuantileLoss(MaskedLoss):
    """
    Quantile Loss (Pinball Loss) for Quantile Regression.

    Args:
        tau (float or torch.Tensor): The quantile(s) to estimate (0 < tau < 1).
        reduction (str): Specifies the reduction to apply to the output:
             'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(self, tau=0.5, reduction='mean'):
        super().__init__(reduction=reduction)
        tau = self._validate_quantile(tau)
        # Register quantile as a buffer so it will be properly moved to device with model
        self.register_buffer('quantile', tau)

    def _validate_quantile(self, tau):
        if isinstance(tau, float):
            if not 0 < tau < 1:
                raise ValueError("tau must be between 0 and 1")
            return torch.tensor(tau, dtype=torch.float32) 
        elif isinstance(tau, torch.Tensor):
            if not torch.all((0 < tau) & (tau < 1)):
                raise ValueError("All tau values must be between 0 and 1")
            return tau.to(torch.float32)
        else:
            raise TypeError("tau must be a float or a torch.Tensor")

    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculates the Quantile Loss.

        Args:
            y_true (torch.Tensor): Original values (batch_size, n_features)
            y_pred (torch.Tensor): Predicted values (batch_size, n_features)
            mask (torch.Tensor, optional): Mask (batch_size, n_features)
            weights (torch.Tensor, optional): Sample weights (batch_size, n_features)

        Returns:
            torch.Tensor: The Quantile Loss (scalar)
        """
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)

        diff = y_true - y_pred
        
        # Handle broadcasting for tensor quantiles
        if self.quantile.numel() > 1:
            # Add broadcasting dimensions if required
            if self.quantile.ndim < diff.ndim:
                # Create proper shape for broadcasting
                shape = [1] * (diff.ndim - 1) + [-1]
                q = self.quantile.view(*shape)
            else:
                q = self.quantile
        else:
            q = self.quantile
            
        # Calculate quantile loss
        loss = torch.max(q * diff, (q - 1) * diff)
        return self._reduce(loss, mask, weights)


class PinballLoss(QuantileLoss):
    """
    Pinball Loss (same as QuantileLoss).
    
    Args:
        quantile (float): The quantile to estimate (0 < quantile < 1).
        reduction (str): Specifies the reduction to apply to the output:
             'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(self, quantile=0.5, reduction='mean'):
        super().__init__(tau=quantile, reduction=reduction)

class MultiQuantileLoss(MaskedLoss):
    """
    Loss for simultaneously predicting multiple quantiles.

    Args:
        quantiles (list or torch.Tensor): List of quantiles to estimate (0 < q < 1).
        reduction (str): Specifies the reduction to apply to the output:
             'none' | 'mean' | 'sum'. Default: 'mean'
    
    Input shape:
        y_true: (batch_size, n_features)
        y_pred: (batch_size * num_quantiles, n_features) where predictions for each 
                quantile are stacked in the batch dimension.
    """
    def __init__(self, quantiles, reduction='mean'):
        super().__init__(reduction=reduction)
        if isinstance(quantiles, list):
            quantiles = torch.tensor(quantiles, dtype=torch.float32)
        if not isinstance(quantiles, torch.Tensor):
            raise TypeError("quantiles must be a list or a torch.Tensor")
        if not torch.all((0 < quantiles) & (quantiles < 1)):
            raise ValueError("All quantile values must be between 0 and 1")
        # Register quantiles as a buffer so it moves to the correct device with the model
        self.register_buffer('quantiles', quantiles)
        self.num_quantiles = self.quantiles.numel()
        # Use our own quantile loss implementation directly
        self.quantile_loss = QuantileLoss(tau=self.quantiles, reduction='none')
        
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculates the Multi-Quantile Loss.

        Args:
            y_true (torch.Tensor): Original values (batch_size, n_features)
            y_pred (torch.Tensor): Predicted values (batch_size * num_quantiles, n_features).
                   The predictions should be organized with batch_size blocks for each quantile:
                   [q1_batch1, q1_batch2, ..., q2_batch1, q2_batch2, ...]
            mask (torch.Tensor, optional): Mask (batch_size, n_features)
            weights (torch.Tensor, optional): Optional sample weights

        Returns:
            torch.Tensor: The Multi-Quantile Loss
        """
        batch_size, n_features = y_true.shape
        
        # Validate input dimensions
        expected_pred_size = batch_size * self.num_quantiles
        if y_pred.shape[0] != expected_pred_size:
            raise ValueError(f"Expected y_pred to have first dimension size {expected_pred_size} "
                           f"(batch_size {batch_size} * num_quantiles {self.num_quantiles}), "
                           f"but got {y_pred.shape[0]}")
                           
        # Reshape y_true and mask for broadcasting with multiple quantiles
        y_true = y_true.repeat_interleave(self.num_quantiles, dim=0)
        if mask is not None:
            mask = mask.repeat_interleave(self.num_quantiles, dim=0)
        if weights is not None:
            weights = weights.repeat_interleave(self.num_quantiles, dim=0)
            
        # Use the quantile loss with the expanded inputs
        loss = self.quantile_loss(y_true, y_pred, mask, weights)
        return self._reduce(loss, mask, weights)

class LogLinQuantileLoss(MaskedLoss):
    """
    Log-linearized quantile loss from CatBoost.
    See https://catboost.ai/docs/en/concepts/loss-functions-regression#LogLinQuantile

    Args:
        quantile (float): The quantile, between 0 and 1.
        a (float): Smoothing parameter. Default from CatBoost.
        reduction (str): Specifies the reduction to apply to the output:
             'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(self, quantile=0.5, a=1.0):
        super().__init__()
        if not 0 < quantile < 1:
            raise ValueError("quantile must be between 0 and 1")
        # Register parameters as buffers for proper device management
        self.register_buffer('quantile', torch.tensor(quantile, dtype=torch.float32))
        self.register_buffer('a', torch.tensor(a, dtype=torch.float32))
        
    def forward(self, y_true, y_pred, mask=None):
        """
        Calculates the Log-Linearized Quantile Loss.

        Args:
            y_true (torch.Tensor): Original values (batch_size, n_features)
            y_pred (torch.Tensor): Predicted values (batch_size, n_features)
            mask (torch.Tensor, optional): Mask (batch_size, n_features)

        Returns:
            torch.Tensor: The Log-Linearized Quantile Loss (scalar)
        """
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Note: CatBoost uses pred - true (opposite of standard quantile loss)
        diff = y_pred - y_true
        
        # Use log-sum-exp for numerical stability
        alpha = torch.where(diff > 0, self.quantile, 1 - self.quantile)
        
        # Avoid division by zero for stability
        eps = torch.finfo(y_true.dtype).eps
        a_safe = torch.maximum(self.a, torch.tensor(eps, device=self.a.device))
        
        loss = a_safe * torch.log(torch.exp(-diff/a_safe) + torch.exp((alpha-1)*diff/a_safe))
        return self._reduce(loss, mask)