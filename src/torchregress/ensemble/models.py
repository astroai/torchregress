"""
Ensemble model implementations.

This module provides concrete implementations of ensemble models
for regression tasks with uncertainty estimation.
"""

from typing import Any, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from torchregress.losses.mdn import MixtureDensityLoss
from torchregress.utils.gaussian_output import variance_from_logvar
from torchregress.utils.ordinal import cumulative_logits_to_pmf

from .base import BaseEnsembleModel
from .layers import BatchEnsembleLinear
from .utils import parse_heteroscedastic_output


def _stack_member_tensors(outputs: Union[torch.Tensor, list[Any]]) -> torch.Tensor:
    """Stack per-member tensor outputs, validating structured ensembles early."""
    if isinstance(outputs, torch.Tensor):
        return outputs
    tensors = [pred for pred in outputs if isinstance(pred, torch.Tensor)]
    if len(tensors) != len(outputs):
        raise ValueError("Expected tensor outputs from all ensemble members.")
    return torch.stack(tensors)


def _variance_across_members(
    stacked: torch.Tensor, *, dim: int, correction: int = 0, eps: float = 1.0e-8
) -> torch.Tensor:
    """Variance with a stable fallback for singleton ensembles."""
    return torch.var(stacked, dim=dim, correction=correction).clamp_min(eps)


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


def _ensemble_sample(
    model: BaseEnsembleModel,
    x: torch.Tensor,
    n_samples: int,
    support_values: Optional[torch.Tensor],
) -> torch.Tensor:
    prediction = model.predict(x)
    probs = prediction["probabilities"]
    support = (
        torch.arange(probs.shape[-1], device=probs.device, dtype=probs.dtype)
        if support_values is None
        else support_values.to(device=probs.device, dtype=probs.dtype)
    )
    indices = torch.multinomial(probs, n_samples, replacement=True)
    samples = support[indices]
    return samples.transpose(0, 1)


def _piecewise_uniform_cdf_at_edges(
    probs: torch.Tensor,
    source_edges: torch.Tensor,
    query_edges: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the CDF of a piecewise-uniform discrete PDF at arbitrary edges."""
    src_edges = source_edges.to(device=probs.device, dtype=probs.dtype)
    queries = query_edges.to(device=probs.device, dtype=probs.dtype)
    cdf = torch.cumsum(probs, dim=-1)
    out = torch.empty(
        probs.shape[:-1] + (queries.numel(),),
        device=probs.device,
        dtype=probs.dtype,
    )
    out[..., 0] = 0.0
    out[..., -1] = 1.0
    if queries.numel() <= 2:
        return out

    body = queries[1:-1]
    bin_idx = torch.searchsorted(src_edges[1:-1], body, right=False)
    bin_idx = bin_idx.clamp(min=0, max=probs.shape[-1] - 1)
    lower_edge = src_edges[bin_idx]
    upper_edge = src_edges[bin_idx + 1]
    widths = (upper_edge - lower_edge).clamp_min(1.0e-8)
    frac = ((body - lower_edge) / widths).clamp(0.0, 1.0)
    if cdf.ndim == 1:
        prev_cdf = torch.where(
            bin_idx > 0,
            cdf.index_select(0, (bin_idx - 1).clamp_min(0)),
            torch.zeros_like(frac),
        )
        out[..., 1:-1] = prev_cdf + probs.index_select(0, bin_idx) * frac
        return out

    expanded_idx = bin_idx.view(*((1,) * (probs.ndim - 1)), -1).expand(*probs.shape[:-1], -1)
    gathered_probs = probs.gather(-1, expanded_idx)
    prev_idx = (bin_idx - 1).clamp_min(0)
    gathered_prev = cdf.gather(
        -1,
        prev_idx.view(*((1,) * (probs.ndim - 1)), -1).expand(*probs.shape[:-1], -1),
    )
    zero_prev = torch.zeros_like(gathered_prev)
    prev_cdf = torch.where(expanded_idx > 0, gathered_prev, zero_prev)
    out[..., 1:-1] = prev_cdf + gathered_probs * frac.view(
        *((1,) * (probs.ndim - 1)),
        -1,
    )
    return out


def _random_partition_union(member_bin_edges: Sequence[torch.Tensor]) -> torch.Tensor:
    """Create a shared evaluation grid from the union of member edges."""
    if not member_bin_edges:
        raise ValueError("member_bin_edges must not be empty")
    merged = torch.cat([edges.detach().cpu().reshape(-1) for edges in member_bin_edges])
    unique = torch.unique(merged, sorted=True)
    if unique.numel() < 2:
        raise ValueError("member_bin_edges must define at least one interval")
    return unique


class HeteroscedasticEnsembleModel(BaseEnsembleModel):
    """
    Ensemble model with heteroscedastic uncertainty estimation.

    This model assumes each ensemble member predicts both mean and variance.

    Args:
        base_model: Base model class or instance to ensemble
        ensemble_size: Number of ensemble members
        device: Device to use
    """

    def predict(self, x: torch.Tensor, correction: int = 0) -> Dict[str, torch.Tensor]:
        """
        Make prediction with uncertainty estimates.

        Args:
            x: Input tensor [batch_size, ...]
            correction: Bessel's correction setting. Default: 0

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
            variances = [variance_from_logvar(log_var) for log_var in log_vars]

            # Stack variances and calculate mean aleatoric uncertainty
            stacked_vars = torch.stack(variances)
            aleatoric_var = torch.mean(stacked_vars, dim=0)

            # Calculate epistemic uncertainty as variance of means
            epistemic_var = _variance_across_members(stacked_means, dim=0, correction=correction)

            # Total predictive variance is sum of epistemic and aleatoric
            total_var = (epistemic_var + aleatoric_var).clamp_min(1.0e-8)

            return {
                "mean": ensemble_mean,
                "variance": total_var,
                "epistemic_variance": epistemic_var,
                "aleatoric_variance": aleatoric_var,
            }

    def predict_full_covariance(
        self, x: torch.Tensor, correction: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Make prediction with full-output covariance estimation,
        splitting aleatoric and epistemic.
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
                vars_.append(variance_from_logvar(log_var))

            # Stack [M, B, D]
            stacked_means = torch.stack(means)
            mean = torch.mean(stacked_means, dim=0)
            # Epistemic covariance: variance of means
            p = stacked_means.permute(1, 0, 2)  # [B, M, D]
            p_centered = p - mean.unsqueeze(1)
            # Correct contraction to output dimensions: [B, D, D] instead of [B, M, M]
            epi_cov = torch.einsum("bmd,bme->bde", p_centered, p_centered)
            denom = max(self.ensemble_size - correction, 1)
            epi_cov = epi_cov / denom
            # Aleatoric covariance: mean of member variances as diagonal
            stacked_vars = torch.stack(vars_)  # [M, B, D]
            avg_vars = torch.mean(stacked_vars, dim=0)  # [B, D]
            # Coverage invariants (TOR003): chain .to() on torch.diag_embed because
            # torch.diag_embed does not accept device=/dtype= kwargs natively.
            # Anchor on ``avg_vars`` (= logvar dtype along the mean-across-members
            # path) rather than on a per-member mean: in mixed-precision runs the
            # backbone is fp64 while the logvar head is fp32, so a "members mean"
            # anchor would silently mismatch dtype.
            ale_cov = torch.diag_embed(avg_vars).to(device=avg_vars.device, dtype=avg_vars.dtype)
            total_cov = epi_cov + ale_cov
            # Ensure diagonal is strictly positive for numerical stability
            d_idx = torch.arange(total_cov.shape[-1], device=total_cov.device)
            total_cov[..., d_idx, d_idx] = total_cov[..., d_idx, d_idx].clamp_min(1.0e-8)
            return {
                "mean": mean,
                "epistemic_covariance": epi_cov,
                "aleatoric_covariance": ale_cov,
                "total_covariance": total_cov,
            }


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

    def predict(self, x: torch.Tensor, correction: int = 0) -> Dict[str, torch.Tensor]:
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
        return _ensemble_sample(self, x, n_samples, self.support_values)


class RandomPartitionEnsembleModel(BaseEnsembleModel):
    """Deep ensemble over randomized irregular target partitions.

    Each member predicts probabilities on its own target partition. Predictions
    are aggregated by projecting each member's piecewise-uniform CDF onto a
    common evaluation grid and averaging those CDFs. This avoids the incoherent
    averaging that would come from directly combining logits defined on different
    bin edges.
    """

    _evaluation_bin_edges: torch.Tensor

    def __init__(
        self,
        base_model: Union[nn.Module, type],
        ensemble_size: int = 5,
        device: str = "cpu",
        *,
        member_bin_edges: Sequence[torch.Tensor],
        evaluation_bin_edges: Optional[torch.Tensor] = None,
        **base_model_kwargs: Any,
    ) -> None:
        if len(member_bin_edges) != ensemble_size:
            raise ValueError(
                "member_bin_edges length must match ensemble_size for RandomPartitionEnsembleModel"
            )
        super().__init__(
            base_model=base_model,
            ensemble_size=ensemble_size,
            device=device,
            **base_model_kwargs,
        )
        for idx, edges in enumerate(member_bin_edges):
            self.register_buffer(
                f"_member_bin_edges_{idx}",
                edges.detach().clone().float(),
            )
        resolved_eval_edges = (
            _random_partition_union(member_bin_edges)
            if evaluation_bin_edges is None
            else evaluation_bin_edges.detach().clone().float()
        )
        self.register_buffer("_evaluation_bin_edges", resolved_eval_edges)

    @property
    def member_bin_edges(self) -> list[torch.Tensor]:
        return [getattr(self, f"_member_bin_edges_{idx}") for idx in range(self.ensemble_size)]

    @property
    def evaluation_bin_edges(self) -> torch.Tensor:
        return self._evaluation_bin_edges

    def predict(
        self,
        x: torch.Tensor,
        correction: int = 0,
        *,
        evaluation_bin_edges: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            logits = self.forward(x)
            if not isinstance(logits, torch.Tensor):
                logits = _stack_member_tensors(logits)
            member_probs = torch.softmax(logits, dim=-1)
            eval_edges = (
                self.evaluation_bin_edges
                if evaluation_bin_edges is None
                else evaluation_bin_edges.to(device=x.device, dtype=member_probs.dtype)
            )
            member_cdfs = []
            for idx in range(self.ensemble_size):
                member_cdfs.append(
                    _piecewise_uniform_cdf_at_edges(
                        member_probs[idx],
                        self.member_bin_edges[idx].to(device=x.device, dtype=member_probs.dtype),
                        eval_edges,
                    )
                )
            mean_cdf = torch.stack(member_cdfs, dim=0).mean(dim=0)
            probs = torch.diff(mean_cdf, dim=-1).clamp_min(0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
            centers = 0.5 * (eval_edges[:-1] + eval_edges[1:])
            mean, variance = _support_moments(probs, centers)
            return {
                "probabilities": probs,
                "log_probabilities": torch.log(probs.clamp_min(1.0e-8)),
                "cdf_at_edges": mean_cdf,
                "bin_edges": eval_edges,
                "mean": mean,
                "variance": variance,
            }

    def sample(
        self,
        x: torch.Tensor,
        n_samples: int = 100,
        *,
        evaluation_bin_edges: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prediction = self.predict(x, evaluation_bin_edges=evaluation_bin_edges)
        probs = prediction["probabilities"]
        edges = prediction["bin_edges"]
        centers = 0.5 * (edges[:-1] + edges[1:])
        indices = torch.multinomial(probs, n_samples, replacement=True)
        samples = centers[indices]
        return samples.transpose(0, 1)


class BatchEnsembleMLPBackbone(nn.Module):
    """Shared-backbone BatchEnsemble MLP.

    This is a parameter-efficient ensemble backbone that applies BatchEnsemble
    rank-1 perturbations throughout the hidden stack, rather than only in the
    output layer. It is a practical shared-backbone ensemble path that maps
    cleanly to tabular architectures such as TabM-like efficient ensembles.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        ensemble_size: int = 4,
        hidden_dims: Optional[Sequence[int]] = None,
        activation: str = "ReLU",
        layer_norm: bool = True,
        dropout: float = 0.0,
        residual: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        dims = [int(dim) for dim in (hidden_dims or [hidden_size, hidden_size])]
        if not dims:
            raise ValueError("BatchEnsembleMLPBackbone requires at least one hidden dimension")
        act_cls = getattr(nn, activation)
        self.feature_dim = dims[-1]
        self.activation = act_cls()
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.layer_norm = bool(layer_norm)
        self.residual = bool(residual)
        self.input_layer = BatchEnsembleLinear(
            input_size,
            dims[0],
            ensemble_size=ensemble_size,
            device=device,
        )
        self.input_norm = nn.LayerNorm(dims[0]) if self.layer_norm else nn.Identity()
        self.hidden_layers = nn.ModuleList()
        self.hidden_norms = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        prev_dim = dims[0]
        for dim in dims[1:]:
            self.hidden_layers.append(
                BatchEnsembleLinear(
                    prev_dim,
                    dim,
                    ensemble_size=ensemble_size,
                    device=device,
                )
            )
            self.hidden_norms.append(nn.LayerNorm(dim) if self.layer_norm else nn.Identity())
            if self.residual:
                if prev_dim == dim:
                    self.shortcuts.append(nn.Identity())
                else:
                    self.shortcuts.append(
                        BatchEnsembleLinear(
                            prev_dim,
                            dim,
                            ensemble_size=ensemble_size,
                            device=device,
                        )
                    )
            else:
                self.shortcuts.append(nn.Identity())
            prev_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        out = self.input_norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        for layer, norm, shortcut in zip(self.hidden_layers, self.hidden_norms, self.shortcuts):
            residual = out
            out = layer(out)
            out = norm(out)
            if self.residual:
                residual = shortcut(residual)
                out = out + residual
            out = self.activation(out)
            out = self.dropout(out)
        return out


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

    def predict(self, x: torch.Tensor, correction: int = 0) -> Dict[str, torch.Tensor]:
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
        return _ensemble_sample(self, x, n_samples, self.support_values)


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
            raise NotImplementedError(
                "MDNEnsembleModel currently supports diagonal covariance only."
            )

    def _predict_components(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def predict(self, x: torch.Tensor, correction: int = 0) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            weights, means, stds = self._predict_components(x)
            mixture_mean = torch.sum(weights.unsqueeze(-1) * means, dim=-2)
            second_moment = torch.sum(
                weights.unsqueeze(-1) * (stds.square() + means.square()), dim=-2
            )
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
        backbone: Base model architecture (without output head). Must expose
            ``feature_dim`` (the output dimensionality of its forward pass).
        output_size: Size of output features (half will be used for variance)
        ensemble_size: Number of ensemble members
        device: Device to use
    """

    def __init__(
        self,
        backbone: nn.Module,
        output_size: int,
        ensemble_size: int = 4,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.ensemble_size = ensemble_size
        self.output_size = output_size
        self.device = device

        # Final layer is a BatchEnsemble layer with 2*output_size outputs
        # (mean and log_var for each output dimension)
        self.output_layer = BatchEnsembleLinear(
            in_features=backbone.feature_dim,  # type: ignore[arg-type]
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

    def predict(self, x: torch.Tensor, correction: int = 0) -> Dict[str, torch.Tensor]:
        """
        Make prediction with uncertainty estimates.

        Args:
            x: Input tensor [batch_size, ...]
            correction: Bessel's correction setting. Default: 0

        Returns:
            Dictionary with mean, epistemic and aleatoric variance
        """
        with torch.no_grad():
            outputs = self.forward(x)

            means = outputs["means"]  # [batch_size, ensemble_size, output_size]
            log_vars = outputs["log_vars"]  # [batch_size, ensemble_size, output_size]

            # Convert log_vars to variances with the same stabilization used in training.
            variances = variance_from_logvar(log_vars)

            # Calculate ensemble mean across members
            ensemble_mean = torch.mean(means, dim=1)  # [batch_size, output_size]

            # Calculate aleatoric uncertainty (mean of predicted variances)
            aleatoric_var = torch.mean(variances, dim=1)  # [batch_size, output_size]

            # Calculate epistemic uncertainty (variance of means)
            epistemic_var = _variance_across_members(
                means, dim=1, correction=correction
            )  # [batch_size, output_size]

            # Total predictive variance is sum of epistemic and aleatoric
            total_var = (epistemic_var + aleatoric_var).clamp_min(1.0e-8)

            return {
                "mean": ensemble_mean,
                "variance": total_var,
                "epistemic_variance": epistemic_var,
                "aleatoric_variance": aleatoric_var,
            }
