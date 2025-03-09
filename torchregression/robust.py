import torch
import torch.nn.functional as F
from .base import MaskedLoss

class L1Loss(MaskedLoss):
    """
    L1 loss with masked input support.
    
    Computes the mean absolute error between the elements.
    Formula: loss(x, y) = |x - y|
    
    Args:
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
    """
    def __init__(self, reduction='mean'):
        super().__init__(reduction=reduction)
        
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate L1 Loss.
        
        Args:
            y_true (torch.Tensor): Target values
            y_pred (torch.Tensor): Predicted values
            mask (torch.Tensor, optional): Optional mask
            weights (torch.Tensor, optional): Optional weights
            
        Returns:
            torch.Tensor: L1 loss
        """
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        loss = torch.abs(y_true - y_pred)
        return self._reduce(loss, mask, weights)

class HuberLoss(MaskedLoss):
    """
    Huber loss with masked input support.
    
    Creates a criterion that uses a squared term if the absolute
    element-wise error falls below delta and a delta-scaled L1 term otherwise.
    
    Formula:
    - If |y_true - y_pred| < delta:
      loss(x, y) = 0.5 * (y_true - y_pred)^2
    - Otherwise:
      loss(x, y) = delta * (|y_true - y_pred| - 0.5 * delta)
      
    Args:
        delta (float): Controls the point where the loss transitions from quadratic to linear
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
    """
    def __init__(self, delta=1.0, reduction='mean'):
        super().__init__(reduction=reduction)
        self.delta = delta

    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate Huber Loss.
        
        Args:
            y_true (torch.Tensor): Target values
            y_pred (torch.Tensor): Predicted values
            mask (torch.Tensor, optional): Optional mask
            weights (torch.Tensor, optional): Optional weights
            
        Returns:
            torch.Tensor: Huber loss
        """
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        diff = torch.abs(y_true - y_pred)
        loss = torch.where(diff < self.delta,
                          0.5 * diff**2,
                          self.delta * (diff - 0.5 * self.delta))
        
        return self._reduce(loss, mask, weights)

class PseudoHuberLoss(MaskedLoss):
    """
    Pseudo-Huber loss with masked input support.
    
    A smooth approximation to the Huber loss that is differentiable
    at the point of transition between quadratic and linear.
    
    Formula: loss(x, y) = delta^2 * (sqrt(1 + ((x - y)/delta)^2) - 1)
    
    Args:
        delta: Controls the transition point between L2 and L1 behavior
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
    """
    def __init__(self, delta=1.0, reduction='mean'):
        super().__init__(reduction=reduction)
        self.delta = delta

    def forward(self, y_true, y_pred, mask=None, weights=None):
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        diff = y_true - y_pred
        # Clip extremely large values for numerical stability
        scaled_diff = torch.clamp(diff / self.delta, min=-1e6, max=1e6)
        loss = self.delta**2 * (torch.sqrt(1 + scaled_diff**2) - 1)
        return self._reduce(loss, mask, weights)


class LogCoshLoss(MaskedLoss):
    """
    Log-cosh loss with masked input support.
    
    A smoothed version of L1 loss that is twice differentiable everywhere.
    
    Formula: loss(x, y) = log(cosh(x - y))
    
    For large values, this loss approaches |x - y| - log(2), 
    and for small values, it approaches (x - y)^2/2.
    
    Args:
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
    """
    def __init__(self, reduction='mean'):
        super().__init__(reduction=reduction)

    def forward(self, y_true, y_pred, mask=None, weights=None):
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        diff = y_true - y_pred
        # Using softplus for numerical stability with large inputs
        loss = diff + F.softplus(-2 * diff) - torch.log(torch.tensor(2.0, device=diff.device))
        return self._reduce(loss, mask, weights)

class CharbonnierLoss(MaskedLoss):
    """
    Charbonnier loss with masked input support.
    
    A smooth approximation of L1 loss, used in computer vision tasks.
    
    Formula: loss(x, y) = sqrt((x - y)^2 + epsilon^2)^alpha
    
    Args:
        alpha: Power parameter, typically 0.5 for L1-like behavior
        epsilon: Small constant for numerical stability
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
    """
    def __init__(self, alpha=0.5, epsilon=1e-6, reduction='mean'):
        super().__init__(reduction=reduction)
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, y_true, y_pred, mask=None, weights=None):
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        diff = y_true - y_pred
        loss = torch.pow(diff**2 + self.epsilon**2, self.alpha)
        return self._reduce(loss, mask, weights)
    
class LqLoss(MaskedLoss):
    """
    Lq loss function. Generalization of L1 and L2 loss.

    Formula: loss(x, y) = |x - y|^q

    Args:
        q: (float) the exponent. q >= 1. q=1 is L1 loss, q=2 is L2 loss.
           Values of q < 1 are generally *not* convex and may be difficult to optimize.
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
    """
    def __init__(self, q=2.0, reduction='mean'):
        super().__init__(reduction=reduction)
        if q < 1.0:
            print('Warning: q < 1, Lq loss is not convex')
        self.register_buffer('q', torch.tensor(q, dtype=torch.float32))

    def forward(self, y_true, y_pred, mask=None, weights=None):
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        diff = torch.abs(y_true - y_pred)
        # Clip very small values to prevent numerical issues when q < 1
        diff = torch.clamp(diff, min=1e-10)
        loss = torch.pow(diff, self.q)
        return self._reduce(loss, mask, weights)