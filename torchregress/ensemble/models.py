"""
Ensemble model implementations.

This module provides concrete implementations of ensemble models
for regression tasks with uncertainty estimation.
"""

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from torchregress.losses.mdn import MixtureDensityLoss
from torchregress.utils.ordinal import cumulative_logits_to_pmf

from .base import BaseEnsembleModel
from .layers import BatchEnsembleLinear
from .utils import parse_heteroscedastic_output


def _variance_from_logvar(
    log_var: torch.Tensor,
    *,
    min_logvar: float = -8.0,
    max_logvar: float = 6.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Convert log-variance to variance with the same stabilization used in training."""
    return torch.exp(log_var.clamp(min=min_logvar, max=max_logvar)).clamp_min(eps)


def _stack_member_tensors(outputs: Union[torch.Tensor, list[Any]]) -> torch.Tensor:
    """Stack per-member tensor outputs, validating structured ensembles early."""
    if isinstance(outputs, torch.Tensor):
        return outputs
    tensors = [pred for pred in outputs if isinstance(pred, torch.Tensor)]
    if len(tensors) != len(outputs):
        raise ValueError("Expected tensor outputs from all ensemble members.")
    return torch.stack(tensors)


def _variance_across_members(stacked: torch.Tensor, *, dim: int) -> torch.Tensor:
    """Sample variance with a stable fallback for singleton ensembles."""
    unbiased = stacked.shape[dim] > 1
    return torch.var(stacked, dim=dim, unbiased=unbiased)


def _support_moments(
    probs: torch.Tensor,
    support_values: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return predictive mean/variance for discrete support values."""
    support = (
        torch.arange(probs.shape[-1], device=probs.device, dtype=probs.dtype)
        if support_values is None
        else support_values.to(device=probs.device, dtype=probs.dtype)
    )
    view_shape = (1,) * (probs.ndim - 1) + (support.numel(),)
    support = support.view(*view_shape)
    mean = torch.sum(probs * support, dim=-1)
    second_moment = torch.sum(probs * support.square(), dim=-1)
    variance = (second_moment - mean.square()).clamp_min(1.0e-8)
    return mean, variance


class HeteroscedasticEnsembleModel(BaseEnsembleModel):
    """
    Ensemble model with heteroscedastic uncertainty estimation.

    This model assumes each ensemble member predicts both mean and variance.

    Args:
        base_model: Base model class or instance to ensemble
        ensemble_size: Number of ensemble members
        device: Device to use
    """

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make prediction with uncertainty estimates.

        Args:
            x: Input tensor [batch_size, ...]

        Returns:
            Dictionary with mean, epistemic and aleatoric variance
        """
        with torch.no_grad():
            # Get predictions from all ensemble members
            predictions = self.forward(x)

            # Separate means and log_vars using utility function
            means = []
            log_vars = []

            for pred in predictions:
                mean, log_var = parse_heteroscedastic_output(pred)
                means.append(mean)
                log_vars.append(log_var)

            # Stack means and calculate ensemble mean [batch_size, output_dim]
            stacked_means = torch.stack(means)
            ensemble_mean = torch.mean(stacked_means, dim=0)

            # Convert log_vars to variances with the same stabilization used in training.
            variances = [_variance_from_logvar(log_var) for log_var in log_vars]

            # Stack variances and calculate mean aleatoric uncertainty
            stacked_vars = torch.stack(variances)
            aleatoric_var = torch.mean(stacked_vars, dim=0)

            # Calculate epistemic uncertainty as variance of means
            epistemic_var = _variance_across_members(stacked_means, dim=0)

            # Total predictive variance is sum of epistemic and aleatoric
            total_var = epistemic_var + aleatoric_var

            return {
                "mean": ensemble_mean,
                "variance": total_var,
                "epistemic_variance": epistemic_var,
                "aleatoric_variance": aleatoric_var,
            }

    def predict_full_covariance(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make prediction with full-output covariance estimation, splitting aleatoric and epistemic.
        """
        with torch.no_grad():
            preds = self.forward(x)
            means, vars_ = [], []
            for pred in preds:
                # Expect (mean, log_var) or concatenated [mean|log_var]
                if isinstance(pred, tuple) and len(pred) == 2:
                    mean, log_var = pred
                elif pred.shape[1] % 2 == 0:
                    d = pred.shape[1] // 2
                    mean, log_var = pred[:, :d], pred[:, d:]
                else:
                    raise ValueError("Unexpected output format for heteroscedastic ensemble.")
                means.append(mean)
                vars_.append(_variance_from_logvar(log_var))

            # Stack [M, B, D]
            stacked_means = torch.stack(means)
            mean = torch.mean(stacked_means, dim=0)
            # Epistemic covariance: variance of means
            p = stacked_means.permute(1, 0, 2)  # [B, M, D]
            p_centered = p - mean.unsqueeze(1)
            epi_cov = torch.einsum("bmd,bme->bde", p_centered, p_centered) / (
                self.ensemble_size - 1
            )
            # Aleatoric covariance: mean of member variances as diagonal
            stacked_vars = torch.stack(vars_)  # [M, B, D]
            avg_vars = torch.mean(stacked_vars, dim=0)  # [B, D]
            ale_cov = torch.diag_embed(avg_vars)
            total_cov = epi_cov + ale_cov
            return {
                "mean": mean,
                "epistemic_covariance": epi_cov,
                "aleatoric_covariance": ale_cov,
                "total_covariance": total_cov,
            }


class DeepEnsemble(BaseEnsembleModel):
    """
    Implementation of deep ensembles for uncertainty estimation.

    Reference: "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"
    by Lakshminarayanan, Pritzel, Blundell.

    This model creates an ensemble of independently trained models and combines
    their predictions for improved uncertainty estimation.

    Args:
        base_model: Base model class or instance to ensemble
        ensemble_size: Number of ensemble members
        device: Device to use
    """

    def __init__(
        self,
        base_model: Union[nn.Module, type],
        ensemble_size: int = 5,
        device: str = "cpu",
        **base_model_kwargs: Any,
    ) -> None:
        super().__init__(
            base_model=base_model,
            ensemble_size=ensemble_size,
            device=device,
            **base_model_kwargs,
        )


class BinnedPDFEnsembleModel(BaseEnsembleModel):
    """Deep-ensemble style averaging for discrete PDF / classification heads.

    Each member predicts bin logits. The ensemble predictive distribution is the
    arithmetic mean of the member probabilities, which is the natural model average
    in distribution space.
    """

    def __init__(
        self,
        base_model: Union[nn.Module, type],
        ensemble_size: int = 5,
        device: str = "cpu",
        *,
        support_values: Optional[torch.Tensor] = None,
        **base_model_kwargs: Any,
    ) -> None:
        super().__init__(
            base_model=base_model,
            ensemble_size=ensemble_size,
            device=device,
            **base_model_kwargs,
        )
        self.support_values = support_values

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            logits = _stack_member_tensors(self.forward(x))
            probs = F.softmax(logits, dim=-1).mean(dim=0)
            mean, variance = _support_moments(probs, self.support_values)
            return {
                "probabilities": probs,
                "log_probabilities": torch.log(probs.clamp_min(1.0e-8)),
                "mean": mean,
                "variance": variance,
            }

    def sample(self, x: torch.Tensor, n_samples: int = 100) -> torch.Tensor:
        prediction = self.predict(x)
        probs = prediction["probabilities"]
        support = (
            torch.arange(probs.shape[-1], device=probs.device, dtype=probs.dtype)
            if self.support_values is None
            else self.support_values.to(device=probs.device, dtype=probs.dtype)
        )
        indices = torch.multinomial(probs, n_samples, replacement=True)
        samples = support[indices]
        return samples.transpose(0, 1)


class CumulativeLinkEnsembleModel(BaseEnsembleModel):
    """Deep-ensemble averaging for cumulative-link / ordinal heads."""

    def __init__(
        self,
        base_model: Union[nn.Module, type],
        ensemble_size: int = 5,
        device: str = "cpu",
        *,
        support_values: Optional[torch.Tensor] = None,
        **base_model_kwargs: Any,
    ) -> None:
        super().__init__(
            base_model=base_model,
            ensemble_size=ensemble_size,
            device=device,
            **base_model_kwargs,
        )
        self.support_values = support_values

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            logits = _stack_member_tensors(self.forward(x))
            probs = cumulative_logits_to_pmf(logits).mean(dim=0)
            mean, variance = _support_moments(probs, self.support_values)
            return {
                "probabilities": probs,
                "log_probabilities": torch.log(probs.clamp_min(1.0e-8)),
                "mean": mean,
                "variance": variance,
            }

    def sample(self, x: torch.Tensor, n_samples: int = 100) -> torch.Tensor:
        prediction = self.predict(x)
        probs = prediction["probabilities"]
        support = (
            torch.arange(probs.shape[-1], device=probs.device, dtype=probs.dtype)
            if self.support_values is None
            else self.support_values.to(device=probs.device, dtype=probs.dtype)
        )
        indices = torch.multinomial(probs, n_samples, replacement=True)
        samples = support[indices]
        return samples.transpose(0, 1)


class MDNEnsembleModel(BaseEnsembleModel):
    """Deep-ensemble averaging for MDN heads via mixture-of-mixtures aggregation.

    Member-specific components are *not* aligned before averaging. Instead the
    ensemble predictive density is represented as a mixture whose components are the
    concatenation of each member's mixture components with a ``1 / ensemble_size``
    weight factor. This avoids label-switching issues that make naive parameter
    averaging unstable for MDNs.
    """

    def __init__(
        self,
        base_model: Union[nn.Module, type],
        ensemble_size: int = 5,
        device: str = "cpu",
        *,
        n_components: int,
        n_features: int = 1,
        covariance_type: str = "diagonal",
        **base_model_kwargs: Any,
    ) -> None:
        super().__init__(
            base_model=base_model,
            ensemble_size=ensemble_size,
            device=device,
            **base_model_kwargs,
        )
        self.n_components = int(n_components)
        self.n_features = int(n_features)
        self.covariance_type = str(covariance_type)
        self.loss = MixtureDensityLoss(
            n_components=self.n_components,
            n_features=self.n_features,
            covariance_type=self.covariance_type,
        )
        if self.covariance_type != "diagonal":
            raise NotImplementedError("MDNEnsembleModel currently supports diagonal covariance only.")

    def _predict_components(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_outputs = _stack_member_tensors(self.forward(x))
        weight_list = []
        mean_list = []
        scale_list = []
        for member_output in raw_outputs:
            weights, means, scales = self.loss._extract_distribution_parameters(member_output)
            weight_list.append(weights / float(raw_outputs.shape[0]))
            mean_list.append(means)
            scale_list.append(scales)
        return (
            torch.cat(weight_list, dim=-1),
            torch.cat(mean_list, dim=-2),
            torch.cat(scale_list, dim=-2),
        )

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            weights, means, stds = self._predict_components(x)
            mixture_mean = torch.sum(weights.unsqueeze(-1) * means, dim=-2)
            second_moment = torch.sum(weights.unsqueeze(-1) * (stds.square() + means.square()), dim=-2)
            mixture_var = (second_moment - mixture_mean.square()).clamp_min(1.0e-8)
            return {
                "mixture_weights": weights,
                "component_means": means,
                "component_stds": stds,
                "mean": mixture_mean,
                "variance": mixture_var,
            }

    def sample(self, x: torch.Tensor, n_samples: int = 100) -> torch.Tensor:
        prediction = self.predict(x)
        weights = prediction["mixture_weights"]
        means = prediction["component_means"]
        stds = prediction["component_stds"]
        batch_size = weights.shape[0]
        component_idx = torch.multinomial(weights, n_samples, replacement=True)
        expanded_idx = component_idx.unsqueeze(-1).expand(-1, -1, self.n_features)
        selected_means = torch.gather(
            means.unsqueeze(1).expand(-1, n_samples, -1, -1),
            dim=2,
            index=expanded_idx.unsqueeze(2),
        ).squeeze(2)
        selected_stds = torch.gather(
            stds.unsqueeze(1).expand(-1, n_samples, -1, -1),
            dim=2,
            index=expanded_idx.unsqueeze(2),
        ).squeeze(2)
        samples = torch.randn(batch_size, n_samples, self.n_features, device=weights.device)
        samples = samples * selected_stds + selected_means
        return samples.transpose(0, 1)


class HeteroscedasticBatchEnsembleModel(nn.Module):
    """
    Batch ensemble model with heteroscedastic uncertainty estimation.

    This combines batch ensembling (efficient parameter sharing) with
    heteroscedastic uncertainty estimation.

    Args:
        backbone: Base model architecture (without output head)
        input_size: Size of input features
        output_size: Size of output features (half will be used for variance)
        ensemble_size: Number of ensemble members
        device: Device to use
    """

    def __init__(
        self,
        backbone: nn.Module,
        input_size: int,
        output_size: int,
        ensemble_size: int = 4,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.ensemble_size = ensemble_size
        self.input_size = input_size
        self.output_size = output_size
        self.device = device

        # Final layer is a BatchEnsemble layer with 2*output_size outputs
        # (mean and log_var for each output dimension)
        self.output_layer = BatchEnsembleLinear(
            in_features=input_size,
            out_features=2 * output_size,
            ensemble_size=ensemble_size,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x: Input tensor [batch_size, ...]

        Returns:
            Dictionary with mean and log_var predictions for each ensemble member
        """
        # Extract features using the backbone
        features = self.backbone(x)

        # Pass through batch ensemble layer to get ensemble outputs
        # Output shape: [batch_size, ensemble_size, 2*output_size]
        outputs = self.output_layer(features)

        # Split outputs into mean and log variance
        means = outputs[:, :, : self.output_size]  # [batch_size, ensemble_size, output_size]
        log_vars = outputs[:, :, self.output_size :]  # [batch_size, ensemble_size, output_size]

        return {"means": means, "log_vars": log_vars}

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make prediction with uncertainty estimates.

        Args:
            x: Input tensor [batch_size, ...]

        Returns:
            Dictionary with mean, epistemic and aleatoric variance
        """
        with torch.no_grad():
            outputs = self.forward(x)

            means = outputs["means"]  # [batch_size, ensemble_size, output_size]
            log_vars = outputs["log_vars"]  # [batch_size, ensemble_size, output_size]

            # Convert log_vars to variances with the same stabilization used in training.
            variances = _variance_from_logvar(log_vars)

            # Calculate ensemble mean across members
            ensemble_mean = torch.mean(means, dim=1)  # [batch_size, output_size]

            # Calculate aleatoric uncertainty (mean of predicted variances)
            aleatoric_var = torch.mean(variances, dim=1)  # [batch_size, output_size]

            # Calculate epistemic uncertainty (variance of means)
            epistemic_var = _variance_across_members(means, dim=1)  # [batch_size, output_size]

            # Total predictive variance is sum of epistemic and aleatoric
            total_var = epistemic_var + aleatoric_var  # [batch_size, output_size]

            return {
                "mean": ensemble_mean,
                "variance": total_var,
                "epistemic_variance": epistemic_var,
                "aleatoric_variance": aleatoric_var,
            }
