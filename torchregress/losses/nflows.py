"""
Normalizing Flow loss functions for regression tasks.

This module provides loss functions for regression models that use
normalizing flows to model complex output distributions.
Uses the zuko package for efficient implementation of various flows.
"""

from typing import Optional, Dict
import torch

try:
    from zuko.flows import Flow, MAF, RealNVP, NSF, IAF

    ZUKO_AVAILABLE = True
except ImportError:
    ZUKO_AVAILABLE = False

from .base import DistributionLoss
from ..utils.tensor_ops import apply_mask, masked_reduction


class NormalizingFlowLoss(DistributionLoss):
    """
    Negative Log-Likelihood loss for normalizing flow models using zuko.

    This loss allows modeling complex multi-dimensional target distributions
    for regression tasks using various normalizing flow architectures.

    Args:
        n_features (int): Number of output features (dimensions)
        flow_type (str): Type of flow to use ('realnvp', 'maf', 'nsf', 'iaf')
            Default: 'realnvp'
        n_blocks (int): Number of transformation blocks in the flow
            Default: 3
        hidden_features (int): Size of hidden layers in coupling/autoregressive networks
            Default: 64
        n_hidden_layers (int): Number of hidden layers in coupling/autoregressive networks
            Default: 2
        base_distribution (str): Base distribution ('normal' or 'uniform')
            Default: 'normal'
        activation (str): Activation function for hidden layers
            Default: 'relu'
        dropout (float): Dropout rate for hidden layers
            Default: 0.0
        batch_norm (bool): Whether to use batch normalization
            Default: False
        reduction (str): Specifies the reduction to apply: 'none' | 'mean' | 'sum'
            Default: 'mean'

    Mathematical Formulation:
        Normalizing flows transform a simple base distribution into a complex target
        distribution through a series of invertible transformations. The negative log
        likelihood is given by:

        NLL = -log(p_X(x)) = -log(p_Z(f(x))) - log|det(df/dx)|

        where p_Z is the density of the base distribution, f is the invertible
        transformation, and |det(df/dx)| is the absolute determinant of the Jacobian.

    Notes:
        - Requires the 'zuko' package: `pip install zuko`
        - The model should output parameters for the flow, which are handled
          by the zuko package internally
        - Different flow types (RealNVP, MAF, NSF, IAF) have different modeling
          capacities and computational characteristics

    Examples:
        >>> import torch
        >>> # First, create a flow loss function
        >>> loss_fn = NormalizingFlowLoss(n_features=2, flow_type='realnvp', n_blocks=3)
        >>> # Model predictions are flow parameters
        >>> y_pred = torch.randn(10, 100)  # Example flow parameters
        >>> target = torch.randn(10, 2)    # Target values
        >>> loss = loss_fn(y_pred, target)
    """

    def __init__(
        self,
        n_features: int,
        flow_type: str = "realnvp",
        n_blocks: int = 3,
        hidden_features: int = 64,
        n_hidden_layers: int = 2,
        base_distribution: str = "normal",
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
        reduction: str = "mean",
    ):
        if not ZUKO_AVAILABLE:
            raise ImportError(
                "The zuko package is required for NormalizingFlowLoss. "
                "Install it with: pip install zuko"
            )

        super().__init__(reduction=reduction)
        self.n_features = n_features
        self.flow_type = flow_type.lower()
        self.n_blocks = n_blocks
        self.hidden_features = hidden_features
        self.n_hidden_layers = n_hidden_layers
        self.base_distribution = base_distribution
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm

        # Validate flow type
        self._validate_flow_type()

    def _validate_flow_type(self) -> None:
        """
        Validate the flow type selection.

        Raises:
            ValueError: If an invalid flow type is provided
        """
        valid_types = ["realnvp", "maf", "nsf", "iaf"]
        if self.flow_type not in valid_types:
            raise ValueError(
                f"Invalid flow_type: {self.flow_type}. " f"Must be one of: {', '.join(valid_types)}"
            )

    def _create_flow(self, params_dict: Dict[str, torch.Tensor]) -> "Flow":
        """
        Create a flow model from parameters dictionary.

        Args:
            params_dict: Dictionary of flow parameters from model output

        Returns:
            Flow model instance

        Raises:
            RuntimeError: If there's an issue creating the flow
        """
        # Common arguments for all flows
        common_args = {
            "features": self.n_features,
            "transforms": self.n_blocks,
            "hidden_features": self.hidden_features,
            "hidden_layers": self.n_hidden_layers,
            "activation": self.activation,
            "dropout": self.dropout,
            "batch_norm": self.batch_norm,
            "base_dist": self.base_distribution,
        }

        try:
            # Create flow based on type
            if self.flow_type == "realnvp":
                flow = RealNVP(**common_args)
            elif self.flow_type == "maf":
                flow = MAF(**common_args)
            elif self.flow_type == "nsf":
                flow = NSF(**common_args)
            elif self.flow_type == "iaf":
                flow = IAF(**common_args)

            # Set the flow parameters
            flow.set_parameters(params_dict)
            return flow
        except Exception as e:
            raise RuntimeError(f"Failed to create flow model: {str(e)}")

    def _extract_distribution_parameters(self, y_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract flow parameters from model predictions.

        Args:
            y_pred: Model predictions containing flow parameters

        Returns:
            Dictionary of flow parameters
        """
        # If y_pred is already a dictionary of parameters, return it directly
        if isinstance(y_pred, dict):
            return y_pred

        # Otherwise, we assume the model outputs the serialized parameters directly
        # We'll create a valid parameter dictionary expected by zuko
        return {"params": y_pred}

    def _calculate_nll(
        self,
        target: torch.Tensor,
        params: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate negative log-likelihood for the flow model.

        Args:
            target: Target values [batch_size, n_features]
            params: Flow parameters
            mask: Optional mask [batch_size, n_features]

        Returns:
            Negative log-likelihood [batch_size]
        """
        # Create flow
        flow = self._create_flow(params)

        # Calculate log probability
        log_prob = flow.log_prob(target)

        # Return negative log-likelihood
        return -log_prob

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate normalizing flow negative log-likelihood loss.

        Args:
            y_pred: Flow parameters from model
            target: Ground truth values [batch_size, n_features]
            mask: Optional boolean mask [batch_size, n_features]
            weights: Optional sample weights [batch_size]

        Returns:
            Negative log-likelihood loss

        Raises:
            ValueError: If target shape doesn't match expected features
        """
        # Validate inputs
        if target.shape[-1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features in target, got {target.shape[-1]}"
            )

        # Apply mask if provided
        if mask is not None:
            target_masked = apply_mask(target, mask)
        else:
            target_masked = target

        # Extract parameters
        params = self._extract_distribution_parameters(y_pred)

        # Calculate negative log-likelihood
        nll = self._calculate_nll(target_masked, params, mask)

        # Apply weights if provided
        if weights is not None:
            # Handle different weight shapes
            if weights.dim() > 1 and weights.shape[1] > 1:
                # Average across features if weights are per-feature
                weights = weights.mean(dim=1)
            nll = nll * weights

        # Apply reduction
        return masked_reduction(nll, mask, self.reduction)

    def sample(self, y_pred: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Generate samples from the flow distribution.

        Args:
            y_pred: Flow parameters from model
            n_samples: Number of samples to generate per input
                Default: 1

        Returns:
            Samples [batch_size, n_samples, n_features]
        """
        # Extract parameters
        params = self._extract_distribution_parameters(y_pred)

        # Create flow
        flow = self._create_flow(params)

        # Generate samples - shape depends on the flow implementation
        # Typically [batch_size, n_samples, n_features] or [batch_size * n_samples, n_features]
        samples = flow.sample(n_samples)

        # Reshape if needed to ensure [batch_size, n_samples, n_features]
        if samples.dim() == 2:
            batch_size = params["params"].shape[0] if "params" in params else y_pred.shape[0]
            samples = samples.reshape(batch_size, n_samples, self.n_features)

        return samples


def create_flow_loss(
    n_features: int,
    flow_type: str = "realnvp",
    n_blocks: int = 3,
    hidden_features: int = 64,
    n_hidden_layers: int = 2,
    base_distribution: str = "normal",
    activation: str = "relu",
    dropout: float = 0.0,
    batch_norm: bool = False,
    reduction: str = "mean",
) -> DistributionLoss:
    """
    Factory function to create a normalizing flow loss.

    Args:
        n_features (int): Number of features in the target
        flow_type (str): Type of flow ('realnvp', 'maf', 'nsf', 'iaf')
            Default: 'realnvp'
        n_blocks (int): Number of transformation blocks
            Default: 3
        hidden_features (int): Size of hidden layers
            Default: 64
        n_hidden_layers (int): Number of hidden layers
            Default: 2
        base_distribution (str): Base distribution ('normal', 'uniform')
            Default: 'normal'
        activation (str): Activation function
            Default: 'relu'
        dropout (float): Dropout rate
            Default: 0.0
        batch_norm (bool): Whether to use batch normalization
            Default: False
        reduction (str): Loss reduction method
            Default: 'mean'

    Returns:
        NormalizingFlowLoss instance

    Examples:
        >>> # Create a normalizing flow loss for 3D targets
        >>> flow_loss = create_flow_loss(
        ...     n_features=3,
        ...     flow_type='nsf',
        ...     n_blocks=5,
        ...     hidden_features=128
        ... )
        >>> # Use it with model predictions and targets
        >>> loss = flow_loss(model_output, targets)
    """
    return NormalizingFlowLoss(
        n_features=n_features,
        flow_type=flow_type,
        n_blocks=n_blocks,
        hidden_features=hidden_features,
        n_hidden_layers=n_hidden_layers,
        base_distribution=base_distribution,
        activation=activation,
        dropout=dropout,
        batch_norm=batch_norm,
        reduction=reduction,
    )
