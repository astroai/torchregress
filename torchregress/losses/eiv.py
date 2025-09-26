"""
Error-in-Variables (EIV) regression losses.

This module implements various approaches to Error-in-Variables regression,
where both inputs (x) and outputs (y) contain measurement errors.
"""

from typing import Callable, Optional, Tuple, Union

import torch

from ..ensemble.utils import run_ensemble_model
from ..utils.augment import EnsemblePerturbationAugmenter
from ..utils.tensor_ops import (
    apply_mask,
    calculate_gaussian_nll,
    calculate_propagated_variance,
    compute_model_gradients,
    prepare_covariance,
    prepare_cross_covariance,
    prepare_model_input_for_gradients,
)
from .base import RegressionLoss


class BaseEIVLoss(RegressionLoss):
    """
    Base class for Errors-In-Variables regression loss functions.

    This provides common functionality for all EIV loss variants.

    Args:
        model: Model function f(x) that predicts y
        sigma_x: Standard deviation or covariance of feature noise (scalar, vector or matrix)
        sigma_y: Standard deviation or covariance of target noise (scalar, vector or matrix)
        reduction: One of 'none', 'mean', 'sum'
        eps: Small value for numerical stability

    Shape:
        - y_pred: :math:`(N, D_{in})` where N is batch size and D_{in} is input dimension
          (interpreted as observed x with noise)
        - target: :math:`(N, D_{out})` where D_{out} is output dimension
          (interpreted as observed y with noise)
        - Output: scalar if reduction is 'mean' or 'sum', :math:`(N)` if reduction is 'none'
    """

    def __init__(
        self,
        model: Callable,
        sigma_x: Union[float, torch.Tensor],
        sigma_y: Optional[Union[float, torch.Tensor]] = None,
        reduction: str = "mean",
        eps: float = 1e-8,
    ):
        super().__init__(reduction=reduction)
        self.model = model
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.eps = eps

    def _prepare_covariances(
        self, n_features_x: int, n_features_y: int, device: torch.device
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepare covariance matrices for features and targets.

        Args:
            n_features_x: Number of features in input
            n_features_y: Number of features in output
            device: Device to create tensors on

        Returns:
            Tuple of (sigma_x_tensor, sigma_y_tensor)
        """
        sigma_x_tensor = prepare_covariance(self.sigma_x, n_features_x, device)
        sigma_y_tensor = (
            None if self.sigma_y is None else prepare_covariance(self.sigma_y, n_features_y, device)
        )
        return sigma_x_tensor, sigma_y_tensor

    def _prepare_inverse_covariances(
        self,
        sigma_x_tensor: torch.Tensor,
        sigma_y_tensor: torch.Tensor,
        n_features_x: int,
        n_features_y: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate inverse covariance matrices for Mahalanobis distances.

        Args:
            sigma_x_tensor: Covariance matrix for features
            sigma_y_tensor: Covariance matrix for targets
            n_features_x: Number of features in input
            n_features_y: Number of features in output
            device: Device to create tensors on

        Returns:
            Tuple of (sigma_x_inv, sigma_y_inv)
        """
        if sigma_x_tensor.ndim <= 1:
            sigma_x_inv = 1.0 / (sigma_x_tensor + self.eps)
        else:
            try:
                sigma_x_inv = torch.inverse(
                    sigma_x_tensor + torch.eye(n_features_x, device=device) * self.eps
                )
            except RuntimeError:
                # Fallback to SVD-based pseudoinverse for numerical stability
                U, S, Vh = torch.linalg.svd(
                    sigma_x_tensor + torch.eye(n_features_x, device=device) * self.eps
                )
                S_inv = torch.where(S > self.eps, 1.0 / S, torch.zeros_like(S))
                sigma_x_inv = torch.matmul(Vh.t(), torch.matmul(torch.diag(S_inv), U.t()))

        if sigma_y_tensor is None:
            sigma_y_inv = None
        else:
            if sigma_y_tensor.ndim <= 1:
                sigma_y_inv = 1.0 / (sigma_y_tensor + self.eps)
            else:
                try:
                    sigma_y_inv = torch.inverse(
                        sigma_y_tensor + torch.eye(n_features_y, device=device) * self.eps
                    )
                except RuntimeError:
                    # Fallback to SVD-based pseudoinverse for numerical stability
                    U, S, Vh = torch.linalg.svd(
                        sigma_y_tensor + torch.eye(n_features_y, device=device) * self.eps
                    )
                    S_inv = torch.where(S > self.eps, 1.0 / S, torch.zeros_like(S))
                    sigma_y_inv = torch.matmul(Vh.t(), torch.matmul(torch.diag(S_inv), U.t()))

        return sigma_x_inv, sigma_y_inv

    def _calculate_mahalanobis_distance(
        self, diff: torch.Tensor, sigma_inv: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate weighted Mahalanobis distance.

        Args:
            diff: Difference vector or matrix [batch_size, n_features]
            sigma_inv: Inverse covariance matrix or diagonal vector

        Returns:
            Distance per sample [batch_size]
        """
        if sigma_inv.ndim <= 1:
            return torch.sum(diff**2 * sigma_inv, dim=1)
        else:
            return torch.sum(diff * torch.bmm(diff.unsqueeze(1), sigma_inv).squeeze(1), dim=1)


class FunctionalEIVLoss(BaseEIVLoss):
    """
    Functional Errors-In-Variables Loss.

    This loss implements the functional approach to errors-in-variables modeling,
    where the true values are treated as fixed but unknown parameters.
    It propagates uncertainty from the inputs to the outputs using a
    first-order Taylor approximation through model gradients.

    Args:
        model: Model function f(x) that predicts y
        sigma_x: Standard deviation or covariance of feature noise
        sigma_y: Standard deviation or covariance of target noise (optional)
        monte_carlo: Whether to use Monte Carlo sampling for gradient estimation
        n_samples: Number of MC samples if monte_carlo=True
        reduction: One of 'none', 'mean', 'sum'
        eps: Small value for numerical stability

    Shape:
        - y_pred: :math:`(N, D_{in})` where N is batch size and D_{in} is input dimension
          (interpreted as observed x with noise)
        - target: :math:`(N, D_{out})` where D_{out} is output dimension
          (interpreted as observed y with noise)
        - mask: :math:`(N, D_{out})` or None
        - Output: scalar if reduction is 'mean' or 'sum', :math:`(N)` if reduction is 'none'

    Examples:
        >>> import torch
        >>> from torchregress.losses import FunctionalEIVLoss
        >>>
        >>> # Define a simple model
        >>> model = lambda x: x[:, 0:1] * 2 + x[:, 1:2]
        >>>
        >>> # Create loss with diagonal covariance
        >>> loss_fn = FunctionalEIVLoss(model, sigma_x=torch.tensor([0.2, 0.1]), sigma_y=0.1)
        >>>
        >>> # Generate some data
        >>> y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # x_obs in EIV terminology
        >>> target = torch.tensor([[4.0], [10.0]])           # y_true in EIV terminology
        >>>
        >>> # Compute loss
        >>> loss_value = loss_fn(y_pred, target)
        >>> print(f"Loss: {loss_value.item():.4f}")
    """

    def __init__(
        self,
        model: Callable,
        sigma_x: Union[float, torch.Tensor],
        sigma_y: Optional[Union[float, torch.Tensor]] = None,
        monte_carlo: bool = False,
        n_samples: int = 20,
        reduction: str = "mean",
        eps: float = 1e-8,
    ):
        super().__init__(model, sigma_x, sigma_y, reduction, eps)
        self.monte_carlo = monte_carlo
        self.n_samples = n_samples

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate Functional EIV loss.

        Args:
            y_pred: Input features (x_obs) with noise [batch_size, n_features_x]
            target: Target values (y_true) with noise [batch_size, n_features_y]
            mask: Optional boolean mask [batch_size, n_features_y]
            weights: Optional sample weights [batch_size]

        Returns:
            Loss tensor (scalar if reduction is applied)
        """
        # Validate inputs
        self._validate_inputs(y_pred, target, mask)

        # For EIV losses, we interpret parameters differently:
        # y_pred is actually the observed x values (with noise)
        # target is the observed y values (with noise)
        x_obs = y_pred
        y_true = apply_mask(target, mask)

        batch_size, n_features_x = x_obs.shape
        n_features_y = y_true.shape[1]
        device = x_obs.device

        # Prepare noise parameters
        sigma_x_tensor, sigma_y_tensor = self._prepare_covariances(
            n_features_x, n_features_y, device
        )

        if not self.monte_carlo:
            # Analytical approach: use gradients to propagate uncertainty
            x_grad = prepare_model_input_for_gradients(x_obs)
            model_output = self.model(x_grad)

            # Apply mask if needed
            if mask is not None:
                model_output = apply_mask(model_output, mask)

            residuals = y_true - model_output

            # Calculate gradients and propagate variance
            grad = compute_model_gradients(model_output, x_grad, n_features_y)

            # Propagate variance from inputs to outputs
            propagated_var = calculate_propagated_variance(
                grad, sigma_x_tensor, sigma_y=sigma_y_tensor
            )

            # Calculate negative log-likelihood
            loss = calculate_gaussian_nll(residuals, propagated_var, eps=self.eps)
        else:
            # Monte Carlo approach
            loss = self._monte_carlo_forward(
                x_obs,
                y_true,
                sigma_x_tensor,
                sigma_y_tensor,
                batch_size,
                n_features_x,
                n_features_y,
                device,
                mask,
            )

        # Apply weights and reduction
        return self._reduce_with_mask(loss, mask, weights)

    def _monte_carlo_forward(
        self,
        x_obs: torch.Tensor,
        y_true: torch.Tensor,
        sigma_x_tensor: torch.Tensor,
        sigma_y_tensor: Optional[torch.Tensor],
        batch_size: int,
        n_features_x: int,
        n_features_y: int,
        device: torch.device,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Monte Carlo implementation of variance propagation"""
        # Vectorized sampling around observed values
        if sigma_x_tensor.ndim <= 1:
            noise = torch.randn(
                self.n_samples, batch_size, n_features_x, device=device
            ) * sigma_x_tensor.view(1, 1, n_features_x)
        else:
            chol = torch.linalg.cholesky(
                sigma_x_tensor + torch.eye(n_features_x, device=device) * self.eps
            )
            base_noise = torch.randn(self.n_samples, batch_size, n_features_x, device=device)
            noise = base_noise @ chol.T
        x_samples = x_obs.unsqueeze(0) + noise
        x_flat = x_samples.reshape(-1, n_features_x)

        # Forward pass for all samples
        with torch.no_grad():
            y_preds_flat = self.model(x_flat)

            # Reshape predictions [n_samples, batch_size, n_features_y]
            if y_preds_flat.shape[-1] != n_features_y and n_features_y == 1:
                y_preds = y_preds_flat.reshape(self.n_samples, batch_size, 1)
            else:
                y_preds = y_preds_flat.reshape(self.n_samples, batch_size, n_features_y)

            # Apply mask if provided
            if mask is not None:
                mask_expanded = mask.unsqueeze(0).expand(self.n_samples, -1, -1)
                y_preds = torch.where(mask_expanded, y_preds, torch.zeros_like(y_preds))

        # Calculate mean prediction across samples
        mean_pred = torch.mean(y_preds, dim=0)  # [batch_size, n_features_y]

        # Calculate covariance
        y_centered = y_preds - mean_pred.unsqueeze(0)  # [n_samples, batch_size, n_features_y]

        # Efficient vectorized batch covariance: [n_samples, batch_size, n_features_y]
        batch_cov = torch.einsum("sbi,sbj->bij", y_centered, y_centered) / (self.n_samples - 1)

        # Add intrinsic output noise if provided
        if sigma_y_tensor is not None:
            if sigma_y_tensor.ndim <= 1:
                # Diagonal case
                batch_cov = batch_cov + torch.diag_embed(sigma_y_tensor)
            else:
                # Full covariance case
                batch_cov = batch_cov + sigma_y_tensor

        # Calculate residuals from mean prediction
        residuals = y_true - mean_pred

        # Calculate negative log-likelihood
        return calculate_gaussian_nll(residuals, batch_cov, eps=self.eps)


class StructuralEIVLoss(BaseEIVLoss):
    """
    Structural Errors-In-Variables Loss.

    This implements the structural approach to errors-in-variables modeling,
    which accounts for correlations between errors in x and y through
    a cross-covariance matrix.

    Args:
        model: Model function f(x) that predicts y
        sigma_x: Covariance of feature noise
        sigma_y: Covariance of target noise
        sigma_xy: Cross-covariance between feature and target noise
        reduction: One of 'none', 'mean', 'sum'
        eps: Small value for numerical stability

    Shape:
        - y_pred: :math:`(N, D_{in})` where N is batch size and D_{in} is input dimension
          (interpreted as observed x with noise)
        - target: :math:`(N, D_{out})` where D_{out} is output dimension
          (interpreted as observed y with noise)
        - mask: :math:`(N, D_{out})` or None
        - Output: scalar if reduction is 'mean' or 'sum', :math:`(N)` if reduction is 'none'

    Examples:
        >>> import torch
        >>> from torchregress.losses import StructuralEIVLoss
        >>>
        >>> # Define a simple model
        >>> model = lambda x: x[:, 0:1] * 2 + x[:, 1:2]
        >>>
        >>> # Create loss with cross-covariance
        >>> sigma_x = torch.tensor([[0.04, 0.01], [0.01, 0.01]])  # 2x2 covariance
        >>> sigma_y = torch.tensor([0.01])  # 1x1 covariance
        >>> sigma_xy = torch.tensor([[0.005, 0.002]])  # 1x2 cross-covariance
        >>> loss_fn = StructuralEIVLoss(model, sigma_x, sigma_y, sigma_xy)
        >>>
        >>> # Generate some data
        >>> y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # x_obs in EIV terminology
        >>> target = torch.tensor([[4.0], [10.0]])           # y_true in EIV terminology
        >>>
        >>> # Compute loss
        >>> loss_value = loss_fn(y_pred, target)
        >>> print(f"Loss: {loss_value.item():.4f}")
    """

    def __init__(
        self,
        model: Callable,
        sigma_x: Union[float, torch.Tensor],
        sigma_y: Union[float, torch.Tensor],
        sigma_xy: torch.Tensor,
        reduction: str = "mean",
        eps: float = 1e-8,
    ):
        super().__init__(model, sigma_x, sigma_y, reduction, eps)
        self.sigma_xy = sigma_xy

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate Structural EIV loss.

        Args:
            y_pred: Input features (x_obs) with noise [batch_size, n_features_x]
            target: Target values (y_true) with noise [batch_size, n_features_y]
            mask: Optional boolean mask [batch_size, n_features_y]
            weights: Optional sample weights [batch_size]

        Returns:
            Loss tensor (scalar if reduction is applied)
        """
        # Validate inputs
        self._validate_inputs(y_pred, target, mask)

        # For EIV losses, we interpret parameters differently:
        # y_pred is actually the observed x values (with noise)
        # target is the observed y values (with noise)
        x_obs = y_pred
        y_true = apply_mask(target, mask)

        batch_size, n_features_x = x_obs.shape
        n_features_y = y_true.shape[1]
        device = x_obs.device

        # Prepare input for gradient computation
        x_grad = prepare_model_input_for_gradients(x_obs)

        # Forward pass through the model
        model_output = self.model(x_grad)

        # Apply mask if needed
        if mask is not None:
            model_output = apply_mask(model_output, mask)

        # Calculate residuals
        residuals = y_true - model_output

        # Prepare covariance matrices
        sigma_x_tensor, sigma_y_tensor = self._prepare_covariances(
            n_features_x, n_features_y, device
        )
        sigma_xy_tensor = prepare_cross_covariance(
            self.sigma_xy, n_features_x, n_features_y, device
        )

        # Calculate gradients of predictions with respect to inputs
        grad = compute_model_gradients(model_output, x_grad, n_features_y)

        # Propagate input variance to output variance with cross-covariance
        propagated_var = calculate_propagated_variance(
            grad, sigma_x_tensor, sigma_xy=sigma_xy_tensor, sigma_y=sigma_y_tensor
        )

        # Calculate negative log-likelihood
        loss = calculate_gaussian_nll(residuals, propagated_var, eps=self.eps)

        # Apply weights and reduction
        return self._reduce_with_mask(loss, mask, weights)


class OrthogonalDistanceRegressionLoss(BaseEIVLoss):
    """
    Orthogonal Distance Regression (ODR) loss.

    This loss minimizes the orthogonal (perpendicular) distances from data points
    to the model curve by optimizing latent true x values during the forward pass.

    Args:
        model: Model function f(x) that predicts y
        sigma_x: Standard deviation or covariance of feature noise
        sigma_y: Standard deviation or covariance of target noise
        learning_rate: Learning rate for the latent x optimization
        max_iterations: Maximum iterations for latent x optimization
        tolerance: Convergence criterion for optimization
        reduction: One of 'none', 'mean', 'sum'
        eps: Small value for numerical stability

    Shape:
        - y_pred: :math:`(N, D_{in})` where N is batch size and D_{in} is input dimension
          (interpreted as observed x with noise)
        - target: :math:`(N, D_{out})` where D_{out} is output dimension
          (interpreted as observed y with noise)
        - mask: :math:`(N, D_{out})` or None
        - Output: scalar if reduction is 'mean' or 'sum', :math:`(N)` if reduction is 'none'

    Examples:
        >>> import torch
        >>> from torchregress.losses import OrthogonalDistanceRegressionLoss
        >>>
        >>> # Define a simple model
        >>> model = lambda x: x[:, 0:1] * 2 + x[:, 1:2]
        >>>
        >>> # Create loss with equal weighting of x and y errors
        >>> sigma_x = torch.tensor([1.0, 1.0])  # Equal uncertainty in both inputs
        >>> sigma_y = torch.tensor([1.0])       # Unit uncertainty in output
        >>> loss_fn = OrthogonalDistanceRegressionLoss(model, sigma_x, sigma_y)
        >>>
        >>> # Generate some data
        >>> y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # x_obs in EIV terminology
        >>> target = torch.tensor([[4.0], [10.0]])           # y_true in EIV terminology
        >>>
        >>> # Compute loss
        >>> loss_value = loss_fn(y_pred, target)
        >>> print(f"Loss: {loss_value.item():.4f}")
    """

    def __init__(
        self,
        model: Callable,
        sigma_x: Union[float, torch.Tensor],
        sigma_y: Union[float, torch.Tensor],
        learning_rate: float = 0.01,
        max_iterations: int = 10,
        tolerance: float = 1e-6,
        reduction: str = "mean",
        eps: float = 1e-8,
    ):
        super().__init__(model, sigma_x, sigma_y, reduction, eps)
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate the ODR loss by optimizing latent true x values.

        Args:
            y_pred: Input features (x_obs) with noise [batch_size, n_features_x]
            target: Target values (y_true) with noise [batch_size, n_features_y]
            mask: Optional boolean mask [batch_size, n_features_y]
            weights: Optional sample weights [batch_size]

        Returns:
            Loss tensor (scalar if reduction is applied)
        """
        # Validate inputs
        self._validate_inputs(y_pred, target, mask)

        # For EIV losses, we interpret parameters differently:
        # y_pred is actually the observed x values (with noise)
        # target is the observed y values (with noise)
        x_obs = y_pred
        y_true = apply_mask(target, mask)

        batch_size, n_features_x = x_obs.shape
        n_features_y = y_true.shape[1]
        device = x_obs.device

        # Prepare covariance matrices
        sigma_x_tensor, sigma_y_tensor = self._prepare_covariances(
            n_features_x, n_features_y, device
        )

        # Prepare inverse covariance matrices for Mahalanobis distance
        sigma_x_inv, sigma_y_inv = self._prepare_inverse_covariances(
            sigma_x_tensor, sigma_y_tensor, n_features_x, n_features_y, device
        )

        # Initialize latent true x as observed x with gradient tracking enabled
        x_latent = x_obs.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([x_latent], lr=self.learning_rate)

        # Optimize latent true x values
        prev_loss = float("inf")
        for iteration in range(self.max_iterations):
            optimizer.zero_grad()

            # Forward pass with current latent x
            model_output = self.model(x_latent)

            # Apply mask if needed
            if mask is not None:
                model_output = apply_mask(model_output, mask)

            # Calculate x distance (between observed and latent x)
            x_diff = x_obs - x_latent
            x_dist = self._calculate_mahalanobis_distance(x_diff, sigma_x_inv)

            # Calculate y distance (between observed y and predicted y)
            y_diff = y_true - model_output
            y_dist = self._calculate_mahalanobis_distance(y_diff, sigma_y_inv)

            # Total ODR objective: minimize weighted sum of distances
            total_dist = x_dist + y_dist
            odr_objective = torch.mean(total_dist)

            # Backward pass and update
            odr_objective.backward()
            optimizer.step()

            # Check for convergence
            if abs(prev_loss - odr_objective.item()) < self.tolerance:
                break

            prev_loss = odr_objective.item()

        # Final forward pass with optimized latent x (detached to avoid gradient tracking)
        x_latent_final = x_latent.detach()
        model_output_final = self.model(x_latent_final)

        # Apply mask if needed
        if mask is not None:
            model_output_final = apply_mask(model_output_final, mask)

        # Calculate final orthogonal distances
        x_diff_final = x_obs - x_latent_final
        final_x_dist = self._calculate_mahalanobis_distance(x_diff_final, sigma_x_inv)

        y_diff_final = y_true - model_output_final
        final_y_dist = self._calculate_mahalanobis_distance(y_diff_final, sigma_y_inv)

        # Total loss is the weighted sum of squared orthogonal distances
        loss = final_x_dist + final_y_dist

        # Apply weights
        if weights is not None:
            weights = validate_weights(weights, batch_size)
            loss = loss * weights

        # Apply reduction
        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:  # 'none'
            return loss


class EnsembleEIVLoss(BaseEIVLoss):
    """
    Simple Ensemble Errors-in-Variables Loss.

    This loss implements a straightforward approach to handling uncertainty in inputs
    by generating multiple perturbed versions, running the model on each, and
    averaging the predictions before calculating the loss.

    Args:
        model: Model function f(x) that predicts y
        sigma_x: Standard deviation or covariance of feature noise (scalar, vector, or matrix)
        n_samples: Number of perturbed samples to generate
        perturb_method: Method for perturbing inputs ('gaussian', 'uniform')
        reduction: One of 'none', 'mean', 'sum'
        eps: Small value for numerical stability

    Shape:
        - y_pred: :math:`(N, D_{in})` where N is batch size and D_{in} is input dimension
          (interpreted as observed x with noise)
        - target: :math:`(N, D_{out})` where D_{out} is output dimension
          (interpreted as observed y with noise)
        - mask: :math:`(N, D_{out})` or None
        - Output: scalar if reduction is 'mean' or 'sum', :math:`(N)` if reduction is 'none'
    """

    def __init__(
        self,
        model: Callable,
        sigma_x: Union[float, torch.Tensor],
        n_samples: int = 20,
        perturb_method: str = "gaussian",
        reduction: str = "mean",
        eps: float = 1e-8,
    ):
        super().__init__(model, sigma_x, None, reduction, eps)
        self.n_samples = n_samples
        self.perturb_method = perturb_method
        self.perturbation_augmenter = EnsemblePerturbationAugmenter(
            n_samples=n_samples, perturb_method=perturb_method, sigma=sigma_x
        )

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate Ensemble EIV loss.

        Args:
            y_pred: Input features (x_obs) with noise [batch_size, n_features_x]
            target: Target values (y_true) with noise [batch_size, n_features_y]
            mask: Optional boolean mask [batch_size, n_features_y]
            weights: Optional sample weights [batch_size]

        Returns:
            Loss tensor (scalar if reduction is applied)
        """
        # Validate inputs
        self._validate_inputs(y_pred, target, mask)

        # For EIV losses, we interpret parameters differently:
        # y_pred is actually the observed x values (with noise)
        # target is the observed y values (with noise)
        x_obs = y_pred
        y_true = apply_mask(target, mask)

        # Generate perturbed samples using the augmenter
        perturbed_samples = self.perturbation_augmenter(x_obs)
        perturbed_stacked = torch.stack(perturbed_samples)

        # Run ensemble model prediction using util function
        result = run_ensemble_model(self.model, perturbed_stacked)
        mean_pred = result["mean"]

        # Apply mask if needed
        if mask is not None:
            mean_pred = apply_mask(mean_pred, mask)

        # Calculate squared error loss between averaged prediction and target
        squared_error = (mean_pred - y_true) ** 2
        loss = torch.sum(squared_error, dim=1)

        # Apply weights and reduction
        return self._reduce_with_mask(loss, mask, weights)
