"""
Robust loss functions for regression.

This module provides implementations of loss functions that are
robust against outliers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base import RegressionLoss
from ..utils.validation import validate_positive

class HuberLoss(RegressionLoss):
    """
    Huber Loss: less sensitive to outliers than MSE.
    
    L(y, f(x)) = 0.5 * (y - f(x))^2 if |y - f(x)| <= delta
                 delta * (|y - f(x)| - 0.5 * delta) otherwise
    
    Args:
        delta (float): Threshold where the loss changes from quadratic to linear.
                      Default: 1.0
        reduction (str): 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.delta = validate_positive(delta, 'delta')
        
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate Huber loss.
        
        Args:
            y_true: Ground truth values [batch_size, ...]
            y_pred: Predicted values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]
            
        Returns:
            Huber loss value
        """
        self._validate_inputs(y_true, y_pred, mask=mask)
        
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Calculate absolute error
        abs_diff = torch.abs(y_true - y_pred)
        
        # Apply Huber formula
        loss = torch.where(
            abs_diff <= self.delta,
            0.5 * abs_diff * abs_diff,
            self.delta * (abs_diff - 0.5 * self.delta)
        )
        
        # Apply weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            loss = loss * weights
            
        return self._reduce(loss, mask)


class L1Loss(RegressionLoss):
    """
    Mean Absolute Error (L1) Loss.
    L(y, f(x)) = |y - f(x)|
    
    Args:
        reduction (str): 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def forward(self, y_true, y_pred, mask=None, weights=None):
        self._validate_inputs(y_true, y_pred, mask=mask)
        
        # Calculate residuals and absolute value
        residuals = self._calculate_residuals(y_true, y_pred, mask)
        abs_error = torch.abs(residuals)
        
        # Apply weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            abs_error = abs_error * weights
        
        return self._reduce(abs_error, mask)


class PseudoHuberLoss(RegressionLoss):
    """
    Pseudo-Huber Loss: a smooth approximation to the Huber loss.
    
    L(y, f(x)) = delta^2 * (sqrt(1 + ((y - f(x))/delta)^2) - 1)
    
    Args:
        delta (float): Controls the smoothness. Default: 1.0
        reduction (str): 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.delta = validate_positive(delta, 'delta')
        
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate Pseudo-Huber loss.
        
        Args:
            y_true: Ground truth values [batch_size, ...]
            y_pred: Predicted values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]
            
        Returns:
            Pseudo-Huber loss value
        """
        self._validate_inputs(y_true, y_pred, mask=mask)
        
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Calculate scaled difference
        scaled_diff = (y_true - y_pred) / self.delta
        
        # Calculate pseudo-huber
        loss = self.delta**2 * (torch.sqrt(1.0 + scaled_diff**2) - 1.0)
        
        # Apply weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            loss = loss * weights
            
        return self._reduce(loss, mask)


class LogCoshLoss(RegressionLoss):
    """
    Log-Cosh Loss: A smooth approximation of Huber loss.
    
    L(y, f(x)) = log(cosh(y - f(x)))
    
    Args:
        scale (float): Scaling factor to control smoothness. Default: 1.0
        reduction (str): 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(self, scale: float = 1.0, reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.scale = validate_positive(scale, 'scale')
        
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate Log-Cosh loss.
        
        Args:
            y_true: Ground truth values [batch_size, ...]
            y_pred: Predicted values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]
            
        Returns:
            Log-Cosh loss value
        """
        self._validate_inputs(y_true, y_pred, mask=mask)
        
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Calculate scaled difference and apply log-cosh
        diff = self.scale * (y_true - y_pred)
        
        # Use a stable formula for log-cosh
        abs_diff = torch.abs(diff)
        loss = abs_diff + torch.log1p(torch.exp(-2.0 * abs_diff)) - torch.log(torch.tensor(2.0))
        
        # Apply weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            loss = loss * weights
            
        return self._reduce(loss, mask)


class CharbonnierLoss(RegressionLoss):
    """
    Charbonnier Loss: A smooth alternative to L1 loss.
    
    L(y, f(x)) = sqrt((y - f(x))^2 + eps^2)
    
    Args:
        eps (float): Small constant for numerical stability. Default: 1e-3
        reduction (str): 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(self, eps: float = 1e-3, reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.eps = validate_positive(eps, 'eps')
        
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate Charbonnier loss.
        
        Args:
            y_true: Ground truth values [batch_size, ...]
            y_pred: Predicted values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]
            
        Returns:
            Charbonnier loss value
        """
        self._validate_inputs(y_true, y_pred, mask=mask)
        
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Calculate squared difference and apply charbonnier formula
        squared_diff = (y_true - y_pred)**2
        loss = torch.sqrt(squared_diff + self.eps**2)
        
        # Apply weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            loss = loss * weights
            
        return self._reduce(loss, mask)


class LqLoss(RegressionLoss):
    """
    Lq Loss: A generalization of L1 (q=1) and L2 (q=2) losses.
    
    L(y, f(x)) = |y - f(x)|^q
    
    Args:
        q (float): Order of the norm. Default: 1.5
        eps (float): Small constant for numerical stability. Default: 1e-8
        reduction (str): 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(self, q: float = 1.5, eps: float = 1e-8, reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.q = validate_positive(q, 'q')
        self.eps = eps
        
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate Lq loss.
        
        Args:
            y_true: Ground truth values [batch_size, ...]
            y_pred: Predicted values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]
            
        Returns:
            Lq loss value
        """
        self._validate_inputs(y_true, y_pred, mask=mask)
        
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Calculate absolute difference and raise to power q
        abs_diff = torch.abs(y_true - y_pred) + self.eps
        loss = abs_diff ** self.q
        
        # Apply weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            loss = loss * weights
            
        return self._reduce(loss, mask)


class TukeyBiweightLoss(RegressionLoss):
    """
    Tukey's biweight loss function, extremely robust against outliers.
    
    L(y, f(x)) = c^2/6 * (1 - (1 - ((y - f(x))/c)^2)^3) if |y - f(x)| <= c
                 c^2/6 otherwise
    
    Args:
        c (float): Cutoff parameter. Default: 4.685
        reduction (str): 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(self, c: float = 4.685, reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.c = validate_positive(c, 'c')
        self.c_squared_div_6 = (c**2) / 6.0
        
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate Tukey's biweight loss.
        
        Args:
            y_true: Ground truth values [batch_size, ...]
            y_pred: Predicted values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]
            
        Returns:
            Tukey's biweight loss value
        """
        self._validate_inputs(y_true, y_pred, mask=mask)
        
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Calculate error and scaled value
        diff = y_true - y_pred
        scaled_diff = diff / self.c
        abs_scaled_diff = torch.abs(scaled_diff)
        
        # Apply Tukey's formula with cutoff
        loss = torch.where(
            abs_scaled_diff <= 1.0,
            self.c_squared_div_6 * (1.0 - (1.0 - scaled_diff**2)**3),
            torch.tensor(self.c_squared_div_6, device=y_true.device)
        )
        
        # Apply weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            loss = loss * weights
            
        return self._reduce(loss, mask)


class WinsorizedLoss(RegressionLoss):
    """
    Winsorized loss that clips the error at quantile thresholds.
    
    Args:
        q (float): Quantile for clipping (0.0 to 0.5). Default: 0.1
        base_loss (str): Base loss after clipping ('l1', 'l2'). Default: 'l2'
        reduction (str): 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(self, q: float = 0.1, base_loss: str = 'l2', reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        if not 0.0 <= q < 0.5:
            raise ValueError("q must be between 0.0 and 0.5")
        self.q = q
        
        if base_loss not in ['l1', 'l2']:
            raise ValueError("base_loss must be 'l1' or 'l2'")
        self.base_loss = base_loss
        
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate Winsorized loss.
        
        Args:
            y_true: Ground truth values [batch_size, ...]
            y_pred: Predicted values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]
            
        Returns:
            Winsorized loss value
        """
        self._validate_inputs(y_true, y_pred, mask=mask)
        
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Calculate quantiles for clipping if needed
        if self.q > 0:
            if mask is not None:
                # Calculate only on non-masked values
                valid_residuals = residuals[mask]
                lower_q = torch.quantile(valid_residuals, self.q)
                upper_q = torch.quantile(valid_residuals, 1.0 - self.q)
            else:
                lower_q = torch.quantile(residuals, self.q)
                upper_q = torch.quantile(residuals, 1.0 - self.q)
                
            # Clip residuals
            residuals = torch.clamp(residuals, lower_q, upper_q)
        
        # Apply base loss
        if self.base_loss == 'l1':
            loss = torch.abs(residuals)
        else:  # l2
            loss = residuals**2
            
        # Apply weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            loss = loss * weights
            
        return self._reduce(loss, mask)


class LogBarrierLoss(RegressionLoss):
    """
    Log Barrier loss for robust regression.
    
    This loss implements a logarithmic barrier function that gracefully limits
    the influence of large errors.
    
    L(y, f(x)) = -log(1 - min(|y - f(x)|/rho, 1-eps)^2)
    
    Args:
        rho: Scale parameter defining the error threshold
        eps: Small constant to ensure loss remains finite
        reduction: 'none' | 'mean' | 'sum'
    """
    def __init__(self, rho: float = 1.0, eps: float = 1e-8, reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.rho = rho
        self.eps = eps
    
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate Log Barrier loss.
        
        Args:
            y_true: Ground truth values [batch_size, ...]
            y_pred: Predicted values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]
            
        Returns:
            Log Barrier loss
        """
        self._validate_inputs(y_true, y_pred, mask=mask)
        
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Calculate error ratio
        abs_error = torch.abs(y_true - y_pred)
        ratio = abs_error / self.rho
        
        # Clip ratio to avoid barrier approaching infinity
        ratio = torch.clamp(ratio, max=1.0-self.eps)
        
        # Calculate log barrier loss
        loss = -torch.log(1.0 - ratio**2 + self.eps)
        
        # Apply weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            loss = loss * weights
            
        return self._reduce(loss, mask)


class TukeyBiweightLoss(RegressionLoss):
    """
    Tukey's biweight (bisquare) loss for robust regression.
    
    This loss completely ignores errors beyond a certain threshold, making it
    highly robust against outliers, but non-convex.
    
    L(y, f(x)) = c^2/6 * (1 - (1 - (r/c)^2)^3)  if |r| <= c
                 c^2/6                          otherwise
    
    where r = y - f(x) and c is the tuning constant.
    
    Args:
        c: Tuning constant (typical value 4.685)
        reduction: 'none' | 'mean' | 'sum'
    """
    def __init__(self, c: float = 4.685, reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.c = c
        self.c_squared_over_6 = c**2 / 6.0
    
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate Tukey's biweight loss.
        
        Args:
            y_true: Ground truth values [batch_size, ...]
            y_pred: Predicted values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]
            
        Returns:
            Tukey's biweight loss
        """
        self._validate_inputs(y_true, y_pred, mask=mask)
        
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Calculate residuals
        residuals = y_true - y_pred
        abs_residuals = torch.abs(residuals)
        
        # Calculate loss based on whether residuals exceed threshold
        scaled_residuals = abs_residuals / self.c
        squared_scaled_residuals = scaled_residuals**2
        
        # For |r| <= c: c^2/6 * (1 - (1 - (r/c)^2)^3)
        # For |r| >  c: c^2/6
        mask_within = abs_residuals <= self.c
        loss = torch.ones_like(residuals) * self.c_squared_over_6
        if torch.any(mask_within):
            loss[mask_within] = self.c_squared_over_6 * (
                1.0 - (1.0 - squared_scaled_residuals[mask_within])**3
            )
        
        # Apply weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            loss = loss * weights
            
        return self._reduce(loss, mask)


class AdaptiveHuberLoss(RegressionLoss):
    """
    Adaptive Huber loss with automatic delta estimation.
    
    This loss dynamically computes the delta parameter of the Huber loss
    based on the data, making it adaptive to different scales.
    
    Args:
        quantile: Quantile of absolute errors to use for delta estimation
        scale_factor: Additional scaling factor for delta
        reduction: 'none' | 'mean' | 'sum'
    """
    def __init__(self, quantile: float = 0.8, scale_factor: float = 1.0, reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.quantile = quantile
        self.scale_factor = scale_factor
        
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate Adaptive Huber loss.
        
        Args:
            y_true: Ground truth values [batch_size, ...]
            y_pred: Predicted values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]
            
        Returns:
            Adaptive Huber loss
        """
        self._validate_inputs(y_true, y_pred, mask=mask)
        
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Calculate residuals
        residuals = y_true - y_pred
        abs_residuals = torch.abs(residuals)
        
        # Determine delta dynamically based on the quantile of absolute residuals
        with torch.no_grad():
            if mask is not None:
                # Get only valid residuals for quantile calculation
                valid_residuals = abs_residuals[mask]
                if valid_residuals.numel() == 0:
                    # Default delta if no valid residuals
                    delta = torch.tensor(1.0, device=y_true.device)
                else:
                    delta = torch.quantile(valid_residuals, self.quantile) * self.scale_factor
            else:
                delta = torch.quantile(abs_residuals, self.quantile) * self.scale_factor
        
        # Calculate Huber loss with the adaptive delta
        quadratic_mask = abs_residuals <= delta
        loss = torch.zeros_like(abs_residuals)
        
        # Quadratic region
        loss[quadratic_mask] = 0.5 * abs_residuals[quadratic_mask]**2
        
        # Linear region
        linear_mask = ~quadratic_mask
        loss[linear_mask] = delta * (abs_residuals[linear_mask] - 0.5 * delta)
        
        # Apply weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            loss = loss * weights
            
        return self._reduce(loss, mask)


class WinsorizedLoss(RegressionLoss):
    """
    Winsorized loss for robust regression.
    
    This loss replaces extreme residuals with more moderate values,
    truncating the influence of outliers.
    
    Args:
        quantile_low: Lower quantile for winsorization (0-1)
        quantile_high: Upper quantile for winsorization (0-1)
        reduction: 'none' | 'mean' | 'sum'
    """
    def __init__(self, quantile_low: float = 0.05, quantile_high: float = 0.95, reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.quantile_low = quantile_low
        self.quantile_high = quantile_high
        
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate Winsorized loss.
        
        Args:
            y_true: Ground truth values [batch_size, ...]
            y_pred: Predicted values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]
            
        Returns:
            Winsorized loss
        """
        self._validate_inputs(y_true, y_pred, mask=mask)
        
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Determine quantile thresholds
        with torch.no_grad():
            if mask is not None:
                valid_residuals = residuals[mask]
                if valid_residuals.numel() == 0:
                    # Default thresholds if no valid residuals
                    lower_threshold = torch.tensor(-1.0, device=residuals.device)
                    upper_threshold = torch.tensor(1.0, device=residuals.device)
                else:
                    lower_threshold = torch.quantile(valid_residuals, self.quantile_low)
                    upper_threshold = torch.quantile(valid_residuals, self.quantile_high)
            else:
                lower_threshold = torch.quantile(residuals, self.quantile_low)
                upper_threshold = torch.quantile(residuals, self.quantile_high)
        
        # Winsorize residuals
        winsorized_residuals = torch.clamp(residuals, min=lower_threshold, max=upper_threshold)
        
        # Calculate squared error on winsorized residuals
        loss = winsorized_residuals**2
        
        # Apply weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            loss = loss * weights
            
        return self._reduce(loss, mask)


class ClippedLoss(RegressionLoss):
    """
    Clipped loss for robust regression.
    
    This loss clips errors above a threshold, limiting the influence of outliers.
    
    Args:
        threshold: Error threshold beyond which the loss is clipped
        base_loss: Base loss function ('l1', 'mse')
        reduction: 'none' | 'mean' | 'sum'
    """
    def __init__(self, threshold: float = 1.0, base_loss: str = 'mse', reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.threshold = threshold
        self.base_loss = base_loss
        
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate Clipped loss.
        
        Args:
            y_true: Ground truth values [batch_size, ...]
            y_pred: Predicted values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]
            
        Returns:
            Clipped loss
        """
        self._validate_inputs(y_true, y_pred, mask=mask)
        
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        if self.base_loss == 'l1':
            # Calculate absolute error
            error = torch.abs(residuals)
        else:  # 'mse'
            # Calculate squared error
            error = residuals**2
        
        # Clip error at threshold
        loss = torch.clamp(error, max=self.threshold)
        
        # Apply weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            loss = loss * weights
            
        return self._reduce(loss, mask)


class FairLoss(RegressionLoss):
    """
    Fair loss for robust regression.
    
    This loss grows less than linearly with the absolute error, making it
    less sensitive to outliers than MSE or MAE.
    
    L(y, f(x)) = c^2 * (|r|/c - log(1 + |r|/c))
    
    where r = y - f(x) and c is the scale parameter.
    
    Args:
        c: Scale parameter
        reduction: 'none' | 'mean' | 'sum'
        eps: Small constant for numerical stability
    """
    def __init__(self, c: float = 1.0, reduction: str = 'mean', eps: float = 1e-8):
        super().__init__(reduction=reduction)
        self.c = c
        self.eps = eps
        
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate Fair loss.
        
        Args:
            y_true: Ground truth values [batch_size, ...]
            y_pred: Predicted values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]
            
        Returns:
            Fair loss
        """
        self._validate_inputs(y_true, y_pred, mask=mask)
        
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Calculate residuals
        residuals = y_true - y_pred
        abs_residuals = torch.abs(residuals)
        
        # Calculate fair loss
        scaled_residuals = abs_residuals / self.c
        loss = self.c**2 * (scaled_residuals - torch.log(1.0 + scaled_residuals + self.eps))
        
        # Apply weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            loss = loss * weights
            
        return self._reduce(loss, mask)


class CauchyLoss(RegressionLoss):
    """
    Cauchy loss for robust regression.
    
    This loss uses the negative log of the Cauchy distribution density.
    It is very robust to outliers but non-convex.
    
    L(y, f(x)) = log(1 + (r/c)^2)
    
    where r = y - f(x) and c is the scale parameter.
    
    Args:
        c: Scale parameter
        reduction: 'none' | 'mean' | 'sum'
    """
    def __init__(self, c: float = 1.0, reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.c = c
        
    def forward(self, y_true, y_pred, mask=None, weights=None):
        """
        Calculate Cauchy loss.
        
        Args:
            y_true: Ground truth values [batch_size, ...]
            y_pred: Predicted values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]
            
        Returns:
            Cauchy loss
        """
        self._validate_inputs(y_true, y_pred, mask=mask)
        
        # Apply mask if provided
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Calculate Cauchy loss
        scaled_residuals = residuals / self.c
        loss = torch.log(1.0 + scaled_residuals**2)
        
        # Apply weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            loss = loss * weights
            
        return self._reduce(loss, mask)