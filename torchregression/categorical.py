import torch
import torch.nn.functional as F

from .base import MaskedLoss
from typing import Optional

class MaskedSoftmaxCELoss(MaskedLoss):
    """
    Cross-entropy loss with softmax, supporting masked values and probabilistic targets.

    This loss is suitable for both standard classification (with one-hot encoded targets)
    and for regression-as-classification with probabilistic targets (as in Vega et al., 2021).

    Args:
        reduction (str): 'none', 'mean', or 'sum'. Default: 'mean'.
        eps (float): Small constant for numerical stability.  Default: 1e-8.
    """
    def __init__(self, reduction: str = 'mean', eps: float = 1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f"Invalid reduction: {reduction}")


    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculates the masked softmax cross-entropy loss.

        Args:
            y_true: Target values.  Can be either:
                - One-hot encoded: (batch_size, num_classes) with 0s and 1s.
                - Probabilistic:   (batch_size, num_classes) with probabilities summing to 1.
            y_pred: Predicted values (logits) (batch_size, num_classes).
            mask: Optional mask (batch_size, num_classes).  If None, all values are used.

        Returns:
            loss: The cross-entropy loss.
        """
        # Apply the mask *before* softmax (important for correctness)
        y_true = self._apply_mask(y_true, mask)
        y_pred = self._apply_mask(y_pred, mask)

        # Softmax over the predicted logits
        y_pred_softmax = F.softmax(y_pred, dim=-1)

        # Calculate the cross-entropy. Add eps for numerical stability
        loss = -torch.sum(y_true * torch.log(y_pred_softmax + self.eps), dim=-1)

        return self._reduce(loss, mask)

class SoftmaxCELoss(MaskedSoftmaxCELoss):
    """
    Cross-entropy loss with softmax (no mask).
    Just calls MaskedSoftmaxCELoss, for ease of use.

    """
    def __init__(self, reduction: str = 'mean', eps: float = 1e-8):
        super().__init__(reduction=reduction, eps=eps)

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
          raise ValueError('This class does not expect a mask, use MaskedSoftmaxCELoss')
        return super().forward(y_true, y_pred, mask=None)

class HistogramLoss(MaskedLoss):
    """
    Implements the Histogram Loss for regression.

    This class provides a numerically stable and robust implementation of the
    Histogram Loss, as described in "Investigating the Histogram Loss in Regression".
    It supports different target distribution types (Gaussian and one-bin).

    Args:
        num_bins (int): The number of bins in the histogram.
        bin_min (float): The minimum value of the support of the histogram.
        bin_max (float): The maximum value of the support of the histogram.
        target_distribution (str, optional):  Type of the target distribution.
            'gaussian' (default) or 'one-bin'.
        sigma (float, optional): Standard deviation for the Gaussian target
            distribution.  Defaults to (bin_max - bin_min) / num_bins * 2  (twice the bin width).  
            Ignored if target_distribution is 'one-bin'.
        padding (float, optional) Added padding to each side of the support, as multiple of sigma. 
            Defaults to 3.0.
        reduction (str): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'
        device (str or torch.device, optional) the device in which to run the computations.
            Default: None (uses current device)
            
    Examples:
        >>> # Basic usage with default Gaussian distribution
        >>> loss_fn = HistogramLoss(num_bins=100, bin_min=0.0, bin_max=1.0)
        >>> predictions = model(inputs)  # shape: [batch_size, 100]
        >>> loss = loss_fn(predictions, targets)  # targets shape: [batch_size]
    """

    def __init__(self, num_bins, bin_min, bin_max,
                 target_distribution='gaussian', sigma=None, padding=3.0,
                 reduction='mean', device=None):
        super().__init__(reduction=reduction)

        if not isinstance(num_bins, int) or num_bins <= 0:
            raise ValueError("num_bins must be a positive integer")
        if not isinstance(bin_min, (int, float)) or not isinstance(bin_max, (int, float)):
            raise ValueError("bin_min and bin_max must be numeric")
        if bin_min >= bin_max:
            raise ValueError("bin_min must be less than bin_max")
        if target_distribution not in ['gaussian', 'one-bin']:
            raise ValueError("target_distribution must be 'gaussian' or 'one-bin'")
        if sigma is not None and (not isinstance(sigma, (int, float)) or sigma <= 0):
            raise ValueError("sigma must be a positive number")
        if not isinstance(padding, (int,float)) or padding < 0:
            raise ValueError("padding must be a non-negative number")

        self.num_bins = num_bins
        self.original_bin_min = bin_min  # Store original values for reference
        self.original_bin_max = bin_max
        self.target_distribution = target_distribution
        self.padding = padding
        self.device = device

        # Calculate bin width and sigma
        original_range = bin_max - bin_min
        base_bin_width = original_range / num_bins
        
        # Set sigma (if not provided) and adjust bins based on padding
        if sigma is None:
            self.sigma = 2 * base_bin_width  # Default as described in the paper
        else:
            self.sigma = sigma
        
        # If using padding, expand the histogram range
        if padding > 0 and target_distribution == 'gaussian':
            padding_amount = self.padding * self.sigma
            self.bin_min = bin_min - padding_amount
            self.bin_max = bin_max + padding_amount
            total_range = self.bin_max - self.bin_min
            self.bin_width = total_range / num_bins
        else:
            self.bin_min = bin_min
            self.bin_max = bin_max
            self.bin_width = base_bin_width

        # Pre-compute bin centers - move to device during forward pass
        self.register_buffer('bin_centers', torch.linspace(
            self.bin_min + self.bin_width / 2,
            self.bin_max - self.bin_width / 2,
            self.num_bins))
        
        # Pre-compute sqrt(2) for efficiency
        self.register_buffer('sqrt_2', torch.sqrt(torch.tensor(2.0)))

    def compute_gaussian_target_probs(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute target probabilities for Gaussian distribution.
        
        Args:
            targets (torch.Tensor): Ground truth values [batch_size]
            
        Returns:
            torch.Tensor: Target probabilities [batch_size, num_bins]
        """
        # Expand dimensions for broadcasting
        targets = targets.unsqueeze(-1)  # [batch_size, 1]
        
        # Compute bin boundaries
        bin_left = self.bin_centers - self.bin_width / 2   # [num_bins]
        bin_right = self.bin_centers + self.bin_width / 2  # [num_bins]
        
        # Compute CDFs at bin boundaries
        normalized_left = (bin_left - targets) / (self.sigma * self.sqrt_2)
        normalized_right = (bin_right - targets) / (self.sigma * self.sqrt_2)
        
        # Use torch.erf directly with clamping for numerical stability
        cdf_left = 0.5 * (1 + torch.erf(normalized_left).clamp(-1, 1))
        cdf_right = 0.5 * (1 + torch.erf(normalized_right).clamp(-1, 1))
        
        # Probability mass in each bin
        return cdf_right - cdf_left

    def compute_onebin_target_probs(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute target probabilities for one-bin distribution.
        
        Args:
            targets (torch.Tensor): Ground truth values [batch_size]
            
        Returns:
            torch.Tensor: Target probabilities [batch_size, num_bins]
        """
        # Find the bin index for each target
        bin_indices = torch.floor((targets - self.bin_min) / self.bin_width).long()
        
        # Clip bin indices to valid range [0, num_bins-1]
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
        
        # Create one-hot encoded target probabilities
        batch_size = targets.size(0)
        target_probs = torch.zeros(batch_size, self.num_bins, device=targets.device)
        target_probs.scatter_(1, bin_indices.unsqueeze(-1), 1.0)
        
        return target_probs

    def forward(self, predictions, targets, mask=None, weights=None):
        """
        Calculates the Histogram Loss.

        Args:
            predictions (torch.Tensor): Predicted histogram logits (before softmax).
                Shape: (batch_size, num_bins).
            targets (torch.Tensor): Ground truth target values. Shape: (batch_size,).
            mask (torch.Tensor, optional): Optional mask (not used in this loss).
            weights (torch.Tensor, optional): Optional weights (not used in this loss).

        Returns:
            torch.Tensor: The computed Histogram Loss based on the reduction method.
        """
        # Validate inputs
        if not torch.is_tensor(predictions) or not torch.is_tensor(targets):
            raise TypeError("predictions and targets must be PyTorch tensors")
            
        if predictions.shape[1] != self.num_bins:
            raise ValueError(f"predictions must have shape (batch_size, {self.num_bins}), "
                             f"but got {predictions.shape}")
        if targets.shape[0] != predictions.shape[0]:
            raise ValueError(f"The batch size of predictions ({predictions.shape[0]}) "
                             f"must match the batch size of targets ({targets.shape[0]})")
        if targets.ndim != 1:  # Ensure targets are 1D
            raise ValueError(f"targets must have shape (batch_size,), but got {targets.shape}")

        # Ensure tensors are on the correct device
        if predictions.device != self.bin_centers.device:
            predictions = predictions.to(self.bin_centers.device)
        if targets.device != self.bin_centers.device:
            targets = targets.to(self.bin_centers.device)

        # Compute target probabilities based on distribution type
        if self.target_distribution == 'gaussian':
            target_probs = self.compute_gaussian_target_probs(targets)
        else:  # 'one-bin'
            target_probs = self.compute_onebin_target_probs(targets)
        
        # Compute cross-entropy loss using log_softmax for numerical stability
        log_probs = F.log_softmax(predictions, dim=1)
        loss = -(target_probs * log_probs).sum(dim=1)  # Sum across bins

        # Convert the existing multi-reduction approach to use our standard _reduce method
        return self._reduce(loss, None, weights)
class RegressionAsClassificationLoss(MaskedLoss):
    """
    Treats a regression problem as a multi-class classification problem by discretizing
    the target variable into bins.

    Args:
        method (str): Method for binning: 'uniform', 'quantile', or 'fixed'.
        num_bins (int): The number of bins to use.  Ignored if method is 'fixed'.
        min_val (float): Minimum value of the target range (for 'uniform' and auto binning).
        max_val (float): Maximum value of the target range (for 'uniform' and auto binning).
        bin_edges (torch.Tensor, optional):  Pre-defined bin edges (for 'fixed' method).
                                            If provided, overrides num_bins, min_val, and max_val.
        classification_loss (str):  Which classification loss to use: 'cross_entropy' (default) or 'focal'.
        gamma (float): Focusing parameter for focal loss. Only used if classification_loss='focal'.
        alpha (float or torch.Tensor):  Class weights for cross-entropy or focal loss.  If float, it's used
                                       as a global weight.  If a Tensor, it should have shape (num_bins,).
    """

    def __init__(self, method='uniform', num_bins=10, min_val=0.0, max_val=1.0,
                 bin_edges=None, classification_loss='cross_entropy', gamma=2.0, alpha=None):
        super().__init__()

        self.method = method.lower()
        if self.method not in ['uniform', 'quantile', 'fixed']:
            raise ValueError("method must be 'uniform', 'quantile', or 'fixed'")

        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val
        self.bin_edges = bin_edges

        if self.method == 'fixed':
            if self.bin_edges is None:
                raise ValueError("bin_edges must be provided when method='fixed'")
            if not isinstance(self.bin_edges, torch.Tensor):
                self.bin_edges = torch.tensor(self.bin_edges, dtype=torch.float32)
            self.num_bins = self.bin_edges.numel() - 1  # Number of bins is one less than the number of edges


        self.classification_loss = classification_loss.lower()
        if self.classification_loss not in ['cross_entropy', 'focal']:
            raise ValueError("classification_loss must be 'cross_entropy' or 'focal'")
        self.gamma = gamma

        self.alpha = alpha
        if isinstance(self.alpha, float):
          self.alpha = torch.tensor(self.alpha, dtype=torch.float32) #convert to tensor
        if self.alpha is not None and isinstance(self.alpha, torch.Tensor):
            if self.alpha.ndim !=1 or self.alpha.shape[0] != self.num_bins:
                raise ValueError(f'Alpha must be a float, or 1D tensor of size the number of bins:{self.num_bins}')

    def _compute_bins(self, y_true):
        """Computes the bin edges based on the chosen method."""
        device = y_true.device

        if self.method == 'uniform':
            self.bin_edges = torch.linspace(self.min_val, self.max_val, steps=self.num_bins + 1, device=device)
        elif self.method == 'quantile':
            # Ensure enough unique values for quantiles
            unique_values = torch.unique(y_true)
            if unique_values.numel() <= self.num_bins:
                # Fallback to uniform if not enough unique values
                print("Warning: Not enough unique values for quantile binning. Falling back to uniform.")
                self.bin_edges = torch.linspace(self.min_val, self.max_val, steps=self.num_bins + 1, device=device)
            else:
                quantiles = torch.linspace(0, 1, steps=self.num_bins + 1, device=device)
                self.bin_edges = torch.quantile(y_true, quantiles, dim=0) # Compute quantiles along the batch dimension
                # Ensure strict monotonicity of bin edges:
                self.bin_edges = torch.unique(self.bin_edges, sorted=True)
                if self.bin_edges.numel() < self.num_bins + 1: #if not enough bins
                    print("Warning: Not enough unique values for all quantiles. Some bins will be empty")
                    self.bin_edges = torch.linspace(self.min_val, self.max_val, steps=self.num_bins + 1, device=device)
        # 'fixed' case is handled in __init__

        return self.bin_edges


    def _digitize(self, y_true):
        """Converts the continuous target values to bin indices."""
        # Find the bin index for each value
        bin_indices = torch.bucketize(y_true, self.bin_edges[1:-1]) #exclude first and last
        return bin_indices

    def forward(self, y_true, y_pred, mask=None):
        """
        Calculates the Regression-as-Classification loss.

        Args:
            y_true: Ground truth values (batch_size, n_features).
            y_pred: Predicted logits (batch_size, n_features, num_bins).
            mask: (Optional) Mask (batch_size, n_features).

        Returns:
            loss: The loss (scalar).
        """
        if self.method != "fixed":
            bin_edges = self._compute_bins(y_true) #compute if not using fixed bins.
        else:
             bin_edges = self.bin_edges.to(y_true.device)
        bin_indices = self._digitize(y_true) #get the indices
        bin_indices = self._apply_mask(bin_indices, mask) #apply mask
        y_pred = self._apply_mask(y_pred, mask)
        if len(y_pred.shape) == 3: #we have multiple features
          # Flatten the features and batch dimensions to be (N, num_classes).
          y_pred = y_pred.transpose(1,2).reshape(-1, self.num_bins)  # (batch_size * n_features, num_bins)
          bin_indices = bin_indices.reshape(-1) # (batch_size * n_features)
        #else we suppose it is (N, num_classes)

        # Handle alpha (class weights)
        if self.alpha is not None:
            alpha = self.alpha.to(y_true.device)
            if isinstance(self.alpha, torch.Tensor):
                alpha = alpha.view(1, -1)  # Reshape to (1, num_bins) for broadcasting
        else:
            alpha = None

        if self.classification_loss == 'cross_entropy':
            loss = F.cross_entropy(y_pred, bin_indices.long(), reduction='none', weight=alpha)
        elif self.classification_loss == 'focal':
            log_probs = F.log_softmax(y_pred, dim=-1) #for numerical stability
            probs = torch.exp(log_probs)
            pt = probs.gather(1, bin_indices.unsqueeze(-1)).squeeze(-1)
            if alpha is not None:
               at = alpha.gather(0, bin_indices)
               loss = -at * (1 - pt)**self.gamma * log_probs.gather(1, bin_indices.unsqueeze(-1)).squeeze(-1)
            else:
                loss = -(1 - pt)**self.gamma * log_probs.gather(1, bin_indices.unsqueeze(-1)).squeeze(-1)
        else:
            raise ValueError(f'Invalid classification loss {self.classification_loss}')

        return self._reduce(loss, mask) #the mask was already used, but just in case
    