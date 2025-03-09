import torch
from .base import MaskedLoss

class ExpectileLoss(MaskedLoss):
    """Expectile loss function.
    
    Expectile loss generalizes the squared error (L2 loss) by weighting positive and negative
    residuals differently. It's defined as:
    
    L(y_true, y_pred) = |τ - I(y_true < y_pred)| * (y_true - y_pred)²
    
    where I() is the indicator function: 1 when condition is true, 0 otherwise.
    
    When τ=0.5, this is equivalent to the Mean Squared Error (MSE) loss.
    When τ<0.5, the loss penalizes overestimation more than underestimation.
    When τ>0.5, the loss penalizes underestimation more than overestimation.
    
    Args:
        tau (float): Value between [0,1]. Asymmetry parameter.
             Default: 0.5 (equivalent to MSE)
        reduction (str): Specifies the reduction to apply to the output:
             'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(self, tau=0.5, reduction='mean'):
        super().__init__(reduction=reduction)
        if not 0 <= tau <= 1:
            raise ValueError("tau must be between 0 and 1 (inclusive)")
        self.register_buffer('tau', torch.tensor(tau, dtype=torch.float32))

    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Args:
            y_true (torch.Tensor): Ground truth values
            y_pred (torch.Tensor): Predicted values
            mask (torch.Tensor, optional): Optional mask for ignoring certain values
            weights (torch.Tensor, optional): Optional weights for each element
            
        Returns:
            torch.Tensor: Expectile loss value according to reduction method
        """
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)

        diff = y_true - y_pred
        # Use where condition to apply tau or (1-tau) depending on sign of difference
        alpha = torch.where(diff > 0, self.tau, 1-self.tau)
        # Using torch.square for better numerical stability
        loss = alpha * torch.square(diff)
        
        return self._reduce(loss, mask, weights)