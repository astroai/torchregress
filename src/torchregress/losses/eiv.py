"""
Error-in-Variables (EIV) regression losses.

This module implements various approaches to Error-in-Variables regression,
where both inputs (x) and outputs (y) contain measurement errors.
"""

from typing import Any, Callable, Optional, Tuple, Union, cast

import torch

from ..utils.augment import EnsemblePerturbationAugmenter
from ..utils.tensor_ops import (
    apply_mask,
    calculate_gaussian_nll,
    calculate_propagated_variance,
    compute_model_gradients,
    prepare_cross_covariance,
    prepare_model_input_for_gradients,
)
from ..utils.validation import validate_weights
from .base import RegressionLoss
from .loss_registry import register_regression_loss
from .mdn import MDNLoss
from .ordinal import OrdinalCrossEntropyLoss


class BaseEIVLoss(RegressionLoss):
    """
    Base class for Errors-In-Variables regression loss functions.

    This provides common functionality for all EIV loss variants.

    Args:
        model: Model function f(x) that predicts y
        sigma_x: Standard deviation (scalar/vector) or covariance matrix of feature noise.
            Per-sample noise is supported with shapes [batch, n_features] (diagonal stddev)
            or [batch, n_features, n_features] (full covariance).
        sigma_y: Standard deviation (scalar/vector) or covariance matrix of target noise.
            Per-sample noise is supported with shapes [batch, n_features] (diagonal stddev)
            or [batch, n_features, n_features] (full covariance).
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
        eps: float = 1e-3,
    ):
        super().__init__(reduction=reduction)
        self.model = model
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.eps = eps

    def _validate_inputs(
        self, y_pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> None:
        """
        Validate EIV inputs where y_pred represents noisy inputs (x_obs).
        """
        if y_pred.shape[0] != target.shape[0]:
            raise ValueError(
                f"Batch size mismatch: y_pred has {y_pred.shape[0]} rows, "
                f"target has {target.shape[0]} rows."
            )
        if mask is not None and mask.shape != target.shape:
            raise ValueError(f"Mask shape {mask.shape} must match target shape {target.shape}")

    def _prepare_covariances(
        self,
        n_features_x: int,
        n_features_y: int,
        device: torch.device,
        batch_size: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        sigma_x: Optional[Union[float, torch.Tensor]] = None,
        sigma_y: Optional[Union[float, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepare covariance matrices for features and targets.

        Args:
            n_features_x: Number of features in input
            n_features_y: Number of features in output
            device: Device to create tensors on
            batch_size: Optional batch size for per-sample noise
            dtype: Optional dtype for the tensors

        Returns:
            Tuple of (sigma_x_tensor, sigma_y_tensor)
        """
        sigma_x_value = self.sigma_x if sigma_x is None else sigma_x
        sigma_y_value = self.sigma_y if sigma_y is None else sigma_y
        sigma_x_tensor = self._prepare_covariance_from_sigma(
            sigma_x_value, n_features_x, device, batch_size, dtype
        )
        sigma_y_tensor = None
        if sigma_y_value is not None:
            sigma_y_tensor = self._prepare_covariance_from_sigma(
                sigma_y_value, n_features_y, device, batch_size, dtype
            )
        return sigma_x_tensor, sigma_y_tensor

    def explicit(self) -> "ExplicitEIVAdapter":
        """Return an explicit-call adapter with ``x_obs`` as the first argument."""
        return ExplicitEIVAdapter(self)

    def _prepare_covariance_from_sigma(
        self,
        sigma: Union[float, torch.Tensor],
        n_features: int,
        device: torch.device,
        batch_size: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Interpret sigma as a standard deviation or covariance and return covariance.

        Supports per-sample noise with shapes [batch, n_features] (diagonal stddev)
        or [batch, n_features, n_features] (full covariance).
        """
        if isinstance(sigma, (int, float)):
            return torch.eye(n_features, device=device, dtype=dtype) * float(sigma) ** 2

        if isinstance(sigma, torch.Tensor):
            sigma = sigma.to(device)
            if sigma.numel() == 1:
                return torch.eye(n_features, device=device, dtype=dtype) * float(sigma.item()) ** 2
            if sigma.ndim == 1:
                if sigma.shape[0] != n_features:
                    raise ValueError(
                        f"sigma shape {tuple(sigma.shape)} doesn't match required size {n_features}"
                    )
                return torch.diag(sigma**2)
            if sigma.ndim == 2:
                if sigma.shape != (n_features, n_features):
                    if batch_size is not None and sigma.shape == (batch_size, n_features):
                        return torch.diag_embed(sigma**2)
                    raise ValueError(
                        f"sigma matrix shape {tuple(sigma.shape)} doesn't match "
                        f"expected shape ({n_features}, {n_features})"
                    )
                if not torch.allclose(sigma, sigma.t()):
                    sigma = (sigma + sigma.t()) / 2
                return sigma
            if sigma.ndim == 3:
                if sigma.shape[-2:] != (n_features, n_features):
                    raise ValueError(
                        f"sigma matrix shape {tuple(sigma.shape)} doesn't match "
                        f"expected shape (*, {n_features}, {n_features})"
                    )
                if batch_size is not None and sigma.shape[0] != batch_size:
                    raise ValueError(
                        f"sigma shape {tuple(sigma.shape)} doesn't match batch size {batch_size}"
                    )
                sigma_t = sigma.transpose(-1, -2)
                if not torch.allclose(sigma, sigma_t):
                    sigma = (sigma + sigma_t) / 2
                return sigma
            raise ValueError(f"sigma must be scalar, vector, or matrix, got {sigma.ndim}D tensor")

        raise TypeError(f"sigma must be float or tensor, got {type(sigma).__name__}")

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
                jitter = (
                    torch.eye(n_features_x, device=device, dtype=sigma_x_tensor.dtype) * self.eps
                )
                sigma_x_stable = sigma_x_tensor + jitter
                chol = torch.linalg.cholesky(sigma_x_stable)
                sigma_x_inv = torch.cholesky_inverse(chol)
            except RuntimeError:
                sigma_x_inv = torch.linalg.pinv(sigma_x_tensor)

        if sigma_y_tensor is None:
            sigma_y_inv = None
        else:
            if sigma_y_tensor.ndim <= 1:
                sigma_y_inv = 1.0 / (sigma_y_tensor + self.eps)
            else:
                try:
                    jitter = (
                        torch.eye(n_features_y, device=device, dtype=sigma_y_tensor.dtype)
                        * self.eps
                    )
                    sigma_y_stable = sigma_y_tensor + jitter
                    chol = torch.linalg.cholesky(sigma_y_stable)
                    sigma_y_inv = torch.cholesky_inverse(chol)
                except RuntimeError:
                    sigma_y_inv = torch.linalg.pinv(sigma_y_tensor)

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
        if sigma_inv.ndim == 2:
            return torch.sum(diff * (diff @ sigma_inv), dim=1)
        return torch.sum(diff * torch.bmm(diff.unsqueeze(1), sigma_inv).squeeze(1), dim=1)


class ExplicitEIVAdapter:
    """Adapter exposing EIV losses with an explicit ``(x_obs, target)`` call surface."""

    def __init__(self, loss: BaseEIVLoss):
        self.loss = loss

    def __call__(
        self,
        x_obs: torch.Tensor,
        target: torch.Tensor,
        *,
        sigma_x: Optional[Union[float, torch.Tensor]] = None,
        sigma_y: Optional[Union[float, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:

        return cast(
            torch.Tensor,
            self.loss(
                x_obs,
                target,
                mask=mask,
                weights=weights,
                sigma_x=sigma_x,
                sigma_y=sigma_y,
                **kwargs,
            ),
        )


@register_regression_loss("input_noise_marginalization")
class InputNoiseMarginalizationLoss(RegressionLoss):
    """Monte-Carlo input-noise marginalization with an explicit ``x_obs`` entry surface.

    This is a simpler EIV-style baseline for tabular measurement-error problems:
    perturb observed inputs according to their reported uncertainties, run the model on
    those perturbations, and average the chosen downstream loss.
    """

    def __init__(
        self,
        model: Callable,
        base_loss: Callable[..., torch.Tensor],
        sigma_x: Optional[Union[float, torch.Tensor]] = None,
        *,
        n_samples: int = 4,
        min_sigma: float = 1.0e-4,
        antithetic: bool = False,
        pass_sigma_x_to_model: bool = False,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.model = model
        self.base_loss = base_loss
        self.sigma_x = sigma_x
        self.n_samples = int(n_samples)
        self.min_sigma = float(min_sigma)
        self.antithetic = bool(antithetic)
        self.pass_sigma_x_to_model = bool(pass_sigma_x_to_model)

    def _sigma_spec(
        self,
        x_obs: torch.Tensor,
        sigma_x: Optional[Union[float, torch.Tensor]],
    ) -> tuple[str, torch.Tensor]:
        sigma_value = self.sigma_x if sigma_x is None else sigma_x
        if sigma_value is None:
            raise ValueError("sigma_x must be provided either at init time or in forward()")
        if isinstance(sigma_value, (int, float)):
            return "diag", torch.full_like(x_obs, float(sigma_value)).clamp_min(self.min_sigma)
        sigma = sigma_value.to(device=x_obs.device, dtype=x_obs.dtype)
        if sigma.numel() == 1:
            return "diag", torch.full_like(x_obs, float(sigma.item())).clamp_min(self.min_sigma)
        if sigma.ndim == 1:
            if sigma.shape[0] != x_obs.shape[-1]:
                raise ValueError(
                    f"sigma_x shape {tuple(sigma.shape)} must match feature dim {x_obs.shape[-1]}"
                )
            return "diag", sigma.view(1, -1).expand_as(x_obs).clamp_min(self.min_sigma)
        if sigma.ndim == 2:
            if sigma.shape == x_obs.shape:
                return "diag", sigma.clamp_min(self.min_sigma)
            if sigma.shape == (x_obs.shape[-1], x_obs.shape[-1]):
                return "full_shared", sigma
        if sigma.ndim == 3:
            if sigma.shape[0] != x_obs.shape[0] or sigma.shape[-2:] != (
                x_obs.shape[-1],
                x_obs.shape[-1],
            ):
                raise ValueError(
                    "sigma_x with 3 dimensions must have shape "
                    f"({x_obs.shape[0]}, {x_obs.shape[-1]}, {x_obs.shape[-1]})"
                )
            return "full_batched", sigma
        raise ValueError(
            "sigma_x must be scalar, [features], [batch, features], [features, features], "
            "or [batch, features, features]"
        )

    def _draw_perturbations(
        self,
        observed: torch.Tensor,
        sigma: torch.Tensor,
        *,
        mode: str = "diag",
        n_samples: Optional[int] = None,
        antithetic: Optional[bool] = None,
    ) -> torch.Tensor:
        """Draw Gaussian perturbations for explicit input-noise marginalization."""
        sample_count = int(self.n_samples if n_samples is None else n_samples)
        use_antithetic = self.antithetic if antithetic is None else bool(antithetic)
        if sample_count <= 0:
            raise ValueError("n_samples must be >= 1")
        draw_count = sample_count if not use_antithetic else (sample_count + 1) // 2
        base_noise = torch.randn(
            draw_count,
            *observed.shape,
            device=observed.device,
            dtype=observed.dtype,
        )
        if use_antithetic:
            base_noise = torch.cat([base_noise, -base_noise], dim=0)[:sample_count]

        if mode == "diag":
            return observed.unsqueeze(0) + base_noise * sigma.unsqueeze(0)

        if mode in ("full_shared", "full_batched"):
            dim = observed.shape[-1]
            eye = torch.eye(dim, device=observed.device, dtype=observed.dtype)
            chol = torch.linalg.cholesky(sigma + eye * (self.min_sigma**2))
            noise = (base_noise.unsqueeze(-2) @ chol.transpose(-1, -2)).squeeze(-2)
            return observed.unsqueeze(0) + noise

        raise ValueError(f"Unknown sigma_x sampling mode: {mode}")

    def sample_predictions(
        self,
        x_obs: torch.Tensor,
        *,
        sigma_x: Optional[Union[float, torch.Tensor]] = None,
        n_samples: Optional[int] = None,
        antithetic: Optional[bool] = None,
    ) -> Union[torch.Tensor, list[Any]]:
        """Return model predictions under input-noise perturbations.

        Tensor outputs are stacked to ``[n_samples, batch, ...]``. Structured outputs are
        returned as a list aligned with the perturbation draws.
        """
        if isinstance(sigma_x, torch.nn.Module):
            raise TypeError("sigma_x cannot be a Module in this context")
        mode, sigma_x_t = self._sigma_spec(x_obs, sigma_x)
        if mode == "diag":
            sigma_x_t = sigma_x_t.clamp_min(self.min_sigma)
        perturbed = self._draw_perturbations(
            x_obs,
            sigma_x_t,
            mode=mode,
            n_samples=n_samples,
            antithetic=antithetic,
        )
        outputs = [self._model_forward(sample, sigma_x_t) for sample in perturbed]
        if outputs and isinstance(outputs[0], torch.Tensor):
            return torch.stack(cast(list[torch.Tensor], outputs))
        return outputs

    def _model_forward(
        self,
        sample: torch.Tensor,
        sigma_x_t: torch.Tensor,
    ) -> Any:
        if self.pass_sigma_x_to_model:
            return self.model(sample, sigma_x_t)
        return self.model(sample)

    def predictive_average(
        self,
        x_obs: torch.Tensor,
        *,
        sigma_x: Optional[Union[float, torch.Tensor]] = None,
        n_samples: Optional[int] = None,
        antithetic: Optional[bool] = None,
        transform: Optional[Callable[[Union[torch.Tensor, list[Any]]], Any]] = None,
    ) -> Any:
        """Average predictions over perturbed inputs for test-time marginalization.

        If ``transform`` is provided, it is applied to the stacked predictions and its
        result is returned. Otherwise tensor outputs are averaged across the sample axis.
        """
        outputs = self.sample_predictions(
            x_obs,
            sigma_x=sigma_x,
            n_samples=n_samples,
            antithetic=antithetic,
        )
        if transform is not None:
            return transform(outputs)
        if isinstance(outputs, torch.Tensor):
            return outputs.mean(dim=0)
        raise ValueError(
            "predictive_average requires a transform when model outputs are not tensors."
        )

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        x_obs: Optional[torch.Tensor] = None,
        sigma_x: Optional[Union[float, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        observed = y_pred if x_obs is None else x_obs
        predictions = self.sample_predictions(observed, sigma_x=sigma_x)

        if not isinstance(predictions, torch.Tensor):
            # Fallback to loop for structured outputs (e.g. lists of tuples/dicts)
            losses = []
            kwargs_no_red = {k: v for k, v in kwargs.items() if k != "reduction"}
            for prediction in predictions:
                losses.append(
                    cast(
                        torch.Tensor,
                        self.base_loss(
                            prediction,
                            target,
                            mask=mask,
                            weights=weights,
                            reduction="none",
                            **kwargs_no_red,
                        ),
                    )
                )
            return torch.stack(losses).mean(dim=0)

        # Vectorized path for standard Tensor outputs (much faster)
        n_samples, batch_size = predictions.shape[:2]
        # Flatten: [n_samples * batch_size, ...]
        flat_preds = predictions.reshape(-1, *predictions.shape[2:])

        # Tile target/mask/weights: [batch_size, ...] -> [n_samples * batch_size, ...]
        def tile(t: torch.Tensor) -> torch.Tensor:
            repeat_dims = [n_samples] + [1] * (t.dim() - 1)
            return t.repeat(*repeat_dims)

        flat_target = tile(target)
        flat_mask = tile(mask) if mask is not None else None
        flat_weights = tile(weights) if weights is not None else None

        # Force reduction='none' to average over samples properly before final reduction
        kwargs_no_red = {k: v for k, v in kwargs.items() if k != "reduction"}
        flat_losses = self.base_loss(
            flat_preds,
            flat_target,
            mask=flat_mask,
            weights=flat_weights,
            reduction="none",
            **kwargs_no_red,
        )

        # If base_loss already reduced (e.g. mean), we just return it.
        # But for EIV we expect base_loss usually to be 'none' when called internally
        # so we can average over samples ourselves.
        # Check if reduction is applied inside base_loss
        if flat_losses.dim() == 0:
            return flat_losses

        # Reshape losses back to [n_samples, batch_size, ...] and average over samples
        sample_losses = flat_losses.reshape(n_samples, batch_size, *flat_losses.shape[1:])
        return self._reduce(sample_losses.mean(dim=0), mask, weights)


class NoisyInputPredictor(torch.nn.Module):
    """High-level wrapper for models generating predictions under noisy inputs.

    This simplifies the 'input-noise marginalization' pattern for test-time inference.

    Args:
        model: Underlying model receiving (possibly perturbed) inputs.
        sigma_x: Standard deviation or covariance of feature noise.
        n_samples: Number of MC perturbations to draw.
        antithetic: Use antithetic sampling if supported.
    """

    def __init__(
        self,
        model: Callable,
        sigma_x: Optional[Union[float, torch.Tensor]] = None,
        n_samples: int = 16,
        antithetic: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.marginalizer = InputNoiseMarginalizationLoss(
            model=model,
            base_loss=lambda p, t: p,  # Dummy base loss; unused by predictive methods
            sigma_x=sigma_x,
            n_samples=n_samples,
            antithetic=antithetic,
        )

    def forward(
        self,
        x_obs: torch.Tensor,
        *,
        sigma_x: Optional[Union[float, torch.Tensor]] = None,
        n_samples: Optional[int] = None,
        antithetic: Optional[bool] = None,
        transform: Optional[Callable[[Union[torch.Tensor, list[Any]]], Any]] = None,
    ) -> Any:
        """Return the marginalized predictive output across noise perturbations.

        Defaults to the mean over perturbations. If ``transform`` is provided, it is
        applied to the stacked predictions (either a tensor [n_samples, batch, features]
        or a list of structured outputs).
        """
        return self.marginalizer.predictive_average(
            x_obs,
            sigma_x=sigma_x,
            n_samples=n_samples,
            antithetic=antithetic,
            transform=transform,
        )

    def sample_predictions(
        self,
        x_obs: torch.Tensor,
        *,
        sigma_x: Optional[Union[float, torch.Tensor]] = None,
        n_samples: Optional[int] = None,
        antithetic: Optional[bool] = None,
    ) -> torch.Tensor:
        """Return the raw stacked predictions [n_samples, batch_size, ...]."""
        res = self.marginalizer.sample_predictions(
            x_obs,
            sigma_x=sigma_x,
            n_samples=n_samples,
            antithetic=antithetic,
        )
        if not isinstance(res, torch.Tensor):
            raise TypeError(
                "NoisyInputPredictor expects the model to return a Tensor for raw sampling."
            )
        return res


@register_regression_loss("input_noise_mdn")
class InputNoiseMDNLoss(InputNoiseMarginalizationLoss):
    """Input-noise marginalization specifically for MDN heads.

    Args:
        model: Underlying model predicting MDN parameters.
        n_components: Number of mixture components.
        n_features: Number of output features.
        sigma_x: Standard deviation or covariance of feature noise.
        **kwargs: Passed to InputNoiseMarginalizationLoss and MixtureDensityLoss.
    """

    def __init__(
        self,
        model: Callable,
        n_components: int,
        n_features: int = 1,
        sigma_x: Optional[Union[float, torch.Tensor]] = None,
        reduction: str = "mean",
        **kwargs: Any,
    ) -> None:
        mdn_kwargs = {
            "n_components": n_components,
            "n_features": n_features,
            "covariance_type": kwargs.pop("covariance_type", "diagonal"),
            "min_std": kwargs.pop("min_std", 1e-3),
            "eps": kwargs.pop("eps", 1e-8),
        }
        base_loss = MDNLoss(**mdn_kwargs)
        super().__init__(
            model=model,
            base_loss=base_loss,
            sigma_x=sigma_x,
            reduction=reduction,
            **kwargs,
        )


@register_regression_loss("input_noise_binned_pdf")
class InputNoiseBinnedPDFLoss(InputNoiseMarginalizationLoss):
    """Input-noise marginalization specifically for binned-PDF / ordinal heads.

    Args:
        model: Underlying model predicting class logits.
        sigma_x: Standard deviation or covariance of feature noise.
        **kwargs: Passed to InputNoiseMarginalizationLoss and OrdinalCrossEntropyLoss.
    """

    def __init__(
        self,
        model: Callable,
        sigma_x: Optional[Union[float, torch.Tensor]] = None,
        reduction: str = "mean",
        **kwargs: Any,
    ) -> None:
        ordinal_kwargs = {
            "label_smoothing": kwargs.pop("label_smoothing", 0.0),
        }
        base_loss = OrdinalCrossEntropyLoss(**ordinal_kwargs)
        super().__init__(
            model=model,
            base_loss=base_loss,
            sigma_x=sigma_x,
            reduction=reduction,
            **kwargs,
        )


@register_regression_loss("functional_eiv")
class FunctionalEIVLoss(BaseEIVLoss):
    """
    Functional Errors-In-Variables Loss.

    This loss implements the functional approach to errors-in-variables modeling,
    where the true values are treated as fixed but unknown parameters.
    It propagates uncertainty from the inputs to the outputs using a
    first-order Taylor approximation through model gradients.

    Args:
        model: Model function f(x) that predicts y
        sigma_x: Standard deviation (scalar/vector) or covariance matrix of feature noise
        sigma_y: Standard deviation (scalar/vector) or covariance matrix of target noise (optional)
        mode: Variance propagation mode — ``"analytical"`` (Jacobian variance + analytical mean),
            ``"mc"`` (MC empirical variance + MC mean), ``"hybrid"`` (Jacobian variance +
            MC mean — stable on nonlinear, degrades MC advantage on linear).
            Default: ``"analytical"``.
        n_samples: Number of MC samples if mode is ``"mc"`` or ``"hybrid"``
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
        >>> # Analytical mode (default)
        >>> loss_fn = FunctionalEIVLoss(model, sigma_x=torch.tensor([0.2, 0.1]), sigma_y=0.1)
        >>>
        >>> # Monte Carlo mode
        >>> loss_fn = FunctionalEIVLoss(model, sigma_x=0.2, mode="mc", n_samples=50)
        >>>
        >>> # Hybrid mode: Jacobian variance + MC mean
        >>> loss_fn = FunctionalEIVLoss(model, sigma_x=0.2, mode="hybrid", n_samples=20)
        >>>
        >>> # Generate some data
        >>> y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # x_obs in EIV terminology
        >>> target = torch.tensor([[4.0], [10.0]])           # y_true in EIV terminology
        >>>
        >>> # Compute loss
        >>> loss_value = loss_fn(y_pred, target)
        >>> print(f"Loss: {loss_value.item():.4f}")
    """

    _VALID_MODES = ("analytical", "mc", "hybrid")

    def __init__(
        self,
        model: Callable,
        sigma_x: Union[float, torch.Tensor],
        sigma_y: Optional[Union[float, torch.Tensor]] = None,
        mode: str = "analytical",
        monte_carlo: bool = False,
        n_samples: int = 20,
        reduction: str = "mean",
        eps: float = 1e-3,
    ):
        super().__init__(model, sigma_x, sigma_y, reduction, eps)
        # monte_carlo kwarg is deprecated; use mode instead
        if monte_carlo:
            if mode != "analytical":
                raise ValueError(
                    "Cannot set both mode= and monte_carlo=True; use mode='mc' instead"
                )
            mode = "mc"
        if mode not in self._VALID_MODES:
            raise ValueError(f"mode must be one of {self._VALID_MODES}, got {mode!r}")
        self.mode = mode
        self.monte_carlo = self.mode == "mc"  # backward compat: True only for pure MC
        self.n_samples = n_samples

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
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
        sigma_x_override = kwargs.pop("sigma_x", None)
        sigma_y_override = kwargs.pop("sigma_y", None)

        batch_size, n_features_x = x_obs.shape
        n_features_y = y_true.shape[1]
        device = x_obs.device

        # Prepare noise parameters
        sigma_x_tensor, sigma_y_tensor = self._prepare_covariances(
            n_features_x,
            n_features_y,
            device,
            batch_size,
            dtype=x_obs.dtype,
            sigma_x=sigma_x_override,
            sigma_y=sigma_y_override,
        )

        if self.mode == "analytical":
            # Analytical approach: use gradients to propagate uncertainty
            with torch.enable_grad():
                x_grad = prepare_model_input_for_gradients(x_obs)
                model_output = self.model(x_grad)

                # Apply mask if needed
                if mask is not None:
                    model_output = apply_mask(model_output, mask)

                residuals = y_true - model_output

                # Calculate gradients and propagate variance
                grad = compute_model_gradients(model_output, x_grad, n_features_y)

                # Propagate variance from inputs to outputs (gradients allowed —
                # the log(var) NLL term naturally balances Jacobian shrinkage
                # against residual accuracy, enabling attenuation-bias correction)
                propagated_var = calculate_propagated_variance(
                    grad, sigma_x_tensor, sigma_y=sigma_y_tensor
                )

                # Calculate negative log-likelihood (var fixed, residuals trainable)
                loss = calculate_gaussian_nll(residuals, propagated_var, eps=self.eps)

        elif self.mode == "mc":
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

        else:  # mode == "hybrid"
            loss = self._hybrid_forward(
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

    def _generate_monte_carlo_noise(
        self,
        sigma_x_tensor: torch.Tensor,
        batch_size: int,
        n_features_x: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Generate Monte Carlo noise based on sigma_x."""
        if sigma_x_tensor.ndim <= 1:
            return torch.randn(
                self.n_samples, batch_size, n_features_x, device=device, dtype=dtype
            ) * sigma_x_tensor.view(1, 1, n_features_x)

        chol = torch.linalg.cholesky(
            sigma_x_tensor + torch.eye(n_features_x, device=device, dtype=dtype) * self.eps
        )
        base_noise = torch.randn(
            self.n_samples, batch_size, n_features_x, device=device, dtype=dtype
        )

        if sigma_x_tensor.ndim == 2:
            return base_noise @ chol.T
        return torch.einsum("sbn,bnm->sbm", base_noise, chol)

    def _get_monte_carlo_predictions(
        self,
        x_obs: torch.Tensor,
        noise: torch.Tensor,
        batch_size: int,
        n_features_x: int,
        n_features_y: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass and reshape for Monte Carlo samples."""
        x_samples = x_obs.unsqueeze(0) + noise
        x_flat = x_samples.reshape(-1, n_features_x)

        # Forward pass for all samples
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

        return y_preds

    def _calculate_monte_carlo_covariance(
        self,
        y_preds: torch.Tensor,
        mean_pred: torch.Tensor,
        sigma_y_tensor: Optional[torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        """Calculate covariance from Monte Carlo samples and intrinsic noise."""
        y_centered = y_preds - mean_pred.unsqueeze(0)  # [n_samples, batch_size, n_features_y]

        # Efficient vectorized batch covariance: [n_samples, batch_size, n_features_y]
        batch_cov = torch.einsum("sbi,sbj->bij", y_centered, y_centered) / (self.n_samples - 1)

        # Add intrinsic output noise if provided
        if sigma_y_tensor is not None:
            if sigma_y_tensor.ndim <= 1:
                # Diagonal case
                batch_cov = batch_cov + torch.diag_embed(sigma_y_tensor)
            elif sigma_y_tensor.ndim == 2 and sigma_y_tensor.shape[0] == batch_size:
                batch_cov = batch_cov + torch.diag_embed(sigma_y_tensor)
            else:
                # Full covariance case
                batch_cov = batch_cov + sigma_y_tensor

        return batch_cov

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
        noise = self._generate_monte_carlo_noise(
            sigma_x_tensor, batch_size, n_features_x, device, x_obs.dtype
        )

        # Get predictions for all samples
        y_preds = self._get_monte_carlo_predictions(
            x_obs, noise, batch_size, n_features_x, n_features_y, mask
        )

        # Calculate mean prediction across samples
        mean_pred = torch.mean(y_preds, dim=0)  # [batch_size, n_features_y]

        # Calculate covariance
        batch_cov = self._calculate_monte_carlo_covariance(
            y_preds, mean_pred, sigma_y_tensor, batch_size
        )

        # Calculate residuals from mean prediction
        residuals = y_true - mean_pred

        # Calculate negative log-likelihood (gradients flow through variance
        # for attenuation-bias correction)
        return calculate_gaussian_nll(residuals, batch_cov, eps=self.eps)

    def _hybrid_forward(
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
        """Hybrid: analytical Jacobian variance + MC perturbation mean.

        On nonlinear data, the MC empirical variance estimator creates a
        pathological loss landscape because noisy variance estimates produce
        enormous gradient noise (35x louder than clean gradient, nearly
        orthogonal to truth).  The analytical Jacobian variance is stable
        but uses the unperturbed mean.  The hybrid uses:
        - MC perturbations for the mean (bias-corrected)
        - Analytical Jacobian for the variance (stable, smooth gradients)
        - Gaussian NLL to combine them
        """
        # ── 1. Analytical Jacobian variance (stable) ──────────
        with torch.enable_grad():
            x_grad = prepare_model_input_for_gradients(x_obs)
            model_output = self.model(x_grad)
            if mask is not None:
                model_output = apply_mask(model_output, mask)
            grad = compute_model_gradients(model_output, x_grad, n_features_y)
            propagated_var = calculate_propagated_variance(
                grad, sigma_x_tensor, sigma_y=sigma_y_tensor
            )

        # ── 2. MC perturbation mean (bias-corrected) ─────────
        noise = self._generate_monte_carlo_noise(
            sigma_x_tensor, batch_size, n_features_x, device, x_obs.dtype
        )
        y_preds = self._get_monte_carlo_predictions(
            x_obs, noise, batch_size, n_features_x, n_features_y, mask
        )
        mean_pred = torch.mean(y_preds, dim=0)  # [batch_size, n_features_y]

        # ── 3. Gaussian NLL: Jacobian variance + MC mean ──
        residuals = y_true - mean_pred
        return calculate_gaussian_nll(residuals, propagated_var, eps=self.eps)


@register_regression_loss("structural_eiv")
class StructuralEIVLoss(BaseEIVLoss):
    """
    Structural Errors-In-Variables Loss.

    This implements the structural approach to errors-in-variables modeling,
    which accounts for correlations between errors in x and y through
    a cross-covariance matrix.

    Args:
        model: Model function f(x) that predicts y
        sigma_x: Standard deviation (scalar/vector) or covariance matrix of feature noise
        sigma_y: Standard deviation (scalar/vector) or covariance matrix of target noise
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
        >>> # Per-sample feature measurement errors (diagonal stddev per sample)
        >>> sigma_x = torch.tensor([[0.2, 0.1], [0.3, 0.05]])
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
        eps: float = 1e-3,
    ):
        super().__init__(model, sigma_x, sigma_y, reduction, eps)
        self.sigma_xy = sigma_xy

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
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
        sigma_x_override = kwargs.pop("sigma_x", None)
        sigma_y_override = kwargs.pop("sigma_y", None)

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
            n_features_x,
            n_features_y,
            device,
            batch_size,
            dtype=x_obs.dtype,
            sigma_x=sigma_x_override,
            sigma_y=sigma_y_override,
        )
        sigma_xy_tensor = prepare_cross_covariance(
            self.sigma_xy, n_features_x, n_features_y, device, dtype=x_obs.dtype
        )

        # Calculate gradients of predictions with respect to inputs
        grad = compute_model_gradients(model_output, x_grad, n_features_y)

        # Propagate input variance to output variance with cross-covariance
        # (gradients allowed — log(var) term balances Jacobian shrinkage)
        propagated_var = calculate_propagated_variance(
            grad, sigma_x_tensor, sigma_xy=sigma_xy_tensor, sigma_y=sigma_y_tensor
        )

        # Calculate negative log-likelihood (var fixed, residuals trainable)
        loss = calculate_gaussian_nll(residuals, propagated_var, eps=self.eps)

        # Apply weights and reduction
        return self._reduce_with_mask(loss, mask, weights)


@register_regression_loss("odr")
class OrthogonalDistanceRegressionLoss(BaseEIVLoss):
    """
    Orthogonal Distance Regression (ODR) loss.

    This loss minimizes the orthogonal (perpendicular) distances from data points
    to the model curve by optimizing latent true x values during the forward pass.

    Args:
        model: Model function f(x) that predicts y
        sigma_x: Standard deviation (scalar/vector) or covariance matrix of feature noise
        sigma_y: Standard deviation (scalar/vector) or covariance matrix of target noise
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
        eps: float = 1e-3,
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
        **kwargs: Any,
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
        sigma_x_override = kwargs.pop("sigma_x", None)
        sigma_y_override = kwargs.pop("sigma_y", None)

        batch_size, n_features_x = x_obs.shape
        n_features_y = y_true.shape[1]
        device = x_obs.device

        # Prepare covariance matrices
        sigma_x_tensor, sigma_y_tensor = self._prepare_covariances(
            n_features_x,
            n_features_y,
            device,
            batch_size,
            dtype=x_obs.dtype,
            sigma_x=sigma_x_override,
            sigma_y=sigma_y_override,
        )

        # Prepare inverse covariance matrices for Mahalanobis distance
        sigma_x_inv, sigma_y_inv = self._prepare_inverse_covariances(
            sigma_x_tensor,
            cast(torch.Tensor, sigma_y_tensor),
            n_features_x,
            n_features_y,
            device,
        )

        # Initialize latent true x as observed x with gradient tracking enabled
        x_latent = x_obs.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([x_latent], lr=self.learning_rate)

        # Optimize latent true x values
        prev_loss = float("inf")
        # Check convergence less frequently to avoid GPU-CPU sync overhead
        check_interval = 5

        with torch.enable_grad():
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

                # Check for convergence periodically
                if (iteration + 1) % check_interval == 0:
                    current_loss = odr_objective.item()
                    if abs(prev_loss - current_loss) < self.tolerance:
                        break
                    prev_loss = current_loss

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
            weights = cast(torch.Tensor, weights)
            loss = loss * weights

        # Apply reduction
        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:  # 'none'
            return loss


@register_regression_loss("ensemble_eiv")
class EnsembleEIVLoss(BaseEIVLoss):
    """
    Simple Ensemble Errors-in-Variables Loss.

    This loss implements a straightforward approach to handling uncertainty in inputs
    by generating multiple perturbed versions, running the model on each, and
    averaging the predictions before calculating the loss.

    Args:
        model: Model function f(x) that predicts y
        sigma_x: Standard deviation (scalar/vector) or covariance matrix of feature noise
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
        eps: float = 1e-3,
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
        **kwargs: Any,
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

        # Flatten for a single forward pass to keep gradients
        flat_inputs = perturbed_stacked.reshape(-1, perturbed_stacked.shape[-1])
        preds_flat = self.model(flat_inputs)
        if preds_flat.dim() == 1:
            preds_flat = preds_flat.unsqueeze(1)
        preds = preds_flat.reshape(self.n_samples, x_obs.shape[0], -1)
        mean_pred = preds.mean(dim=0)

        # Apply mask if needed
        if mask is not None:
            mean_pred = apply_mask(mean_pred, mask)

        # Calculate squared error loss between averaged prediction and target
        squared_error = (mean_pred - y_true) ** 2
        loss = torch.sum(squared_error, dim=1)

        # Apply weights and reduction
        return self._reduce_with_mask(loss, mask, weights)


def create_eiv_loss(
    model: Callable[..., torch.Tensor],
    loss_type: str = "functional",
    **kwargs: Any,
) -> torch.nn.Module:
    """
    Convenience factory for EIV loss variants.

    Args:
        model: Predictive model/function used by the EIV loss.
        loss_type: One of ``functional``, ``structural``, ``odr``/``orthogonal``,
            or ``ensemble``.
        **kwargs: Constructor kwargs for the selected loss class.
    """
    key = loss_type.lower()
    if key == "functional":
        return FunctionalEIVLoss(model=model, **kwargs)
    if key in {"functional_hybrid", "hybrid_eiv"}:
        return FunctionalEIVLoss(model=model, mode="hybrid", **kwargs)
    if key == "structural":
        return StructuralEIVLoss(model=model, **kwargs)
    if key in {"odr", "orthogonal", "orthogonal_distance_regression"}:
        return OrthogonalDistanceRegressionLoss(model=model, **kwargs)
    if key == "ensemble":
        return EnsembleEIVLoss(model=model, **kwargs)
    if key in {"input_noise_marginalization", "mc_input_noise"}:
        return InputNoiseMarginalizationLoss(model=model, **kwargs)
    if key in {"input_noise_mdn", "eiv_mdn"}:
        return InputNoiseMDNLoss(model=model, **kwargs)
    if key in {"input_noise_binned_pdf", "eiv_binned_pdf"}:
        return InputNoiseBinnedPDFLoss(model=model, **kwargs)
    raise ValueError(
        "loss_type must be one of {'functional', 'functional_hybrid', "
        "'structural', 'odr', 'ensemble', 'input_noise_marginalization', "
        "'input_noise_mdn', 'input_noise_binned_pdf'}, "
        f"got {loss_type!r}"
    )
