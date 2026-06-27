"""
Advanced ensemble methods for regression with uncertainty quantification.
"""

from typing import List, Tuple, cast

import torch
import torch.nn as nn
from torch import Tensor


def _batched_ensemble_forward(models: nn.ModuleList, x: Tensor, method: str = "stack") -> Tensor:
    """
    Fast batched forward pass for homogeneous ensembles.
    Falls back to sequential loops if models have different architectures or gradients are required.
    """
    if len(models) == 0:
        return torch.empty(0, device=x.device)

    # vmap execution loses the backward graph to original models' parameters.
    # We only use the fast path if we don't need gradients for the base models.
    requires_grad = torch.is_grad_enabled() and any(
        p.requires_grad for m in models for p in m.parameters()
    )

    if not requires_grad:
        try:
            from torch.func import functional_call, stack_module_state, vmap

            params, buffers = stack_module_state(models)
            base_model = models[0]

            def fmodel(p: dict, b: dict, x_val: Tensor) -> Tensor:
                return functional_call(base_model, (p, b), (x_val,))

            preds = vmap(fmodel, in_dims=(0, 0, None))(params, buffers, x)
            preds = preds.transpose(0, 1)

            if method == "cat":
                return preds.reshape(x.shape[0], -1)
            return preds
        except Exception:
            pass

    if method == "cat":
        return torch.cat([model(x) for model in models], dim=1)
    return torch.stack([model(x) for model in models], dim=1)


class SoftmaxModelCombiner(nn.Module):
    """
    Softmax-weighted model combiner for ensemble regression.

    Combines predictions from multiple models using a learned softmax weighting.
    """

    def __init__(
        self,
        models: List[nn.Module],
    ) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_models = len(models)
        # Initialize model weights (logits) uniformly
        self.model_weights = nn.Parameter(torch.zeros(self.n_models))

    def forward(self, x: Tensor) -> Tensor:
        """
        Calculate combined predictions using weighted average.
        """
        # Get model weights (probabilities)
        model_probs = torch.softmax(self.model_weights, dim=0)

        # Get predictions from all models
        preds = _batched_ensemble_forward(self.models, x, method="stack")

        # Weighted average of predictions
        weighted_pred = torch.sum(preds * model_probs.view(1, -1, 1), dim=1)

        return weighted_pred

    def get_model_weights(self) -> Tensor:
        """Get the current model weights (probabilities)."""
        return torch.softmax(self.model_weights, dim=0)

    def predict_with_uncertainty(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Get predictions with uncertainty estimates.
        """
        model_probs = torch.softmax(self.model_weights, dim=0)

        preds = _batched_ensemble_forward(self.models, x, method="stack")

        # Weighted mean
        mean_pred = torch.sum(preds * model_probs.view(1, -1, 1), dim=1)

        # Variance of means (law of total variance — epistemic uncertainty)
        mean_diffs = preds - mean_pred.unsqueeze(1)
        var_of_means = torch.sum((mean_diffs**2) * model_probs.view(1, -1, 1), dim=1)

        return mean_pred, var_of_means


class StackingEnsemble(nn.Module):
    """
    Stacking ensemble with meta-learner for regression.

    Uses a meta-learner to combine base model predictions.
    """

    def __init__(
        self,
        models: List[nn.Module],
        meta_learner: nn.Module,
    ) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)
        self.meta_learner = meta_learner

    def forward(self, x: Tensor) -> Tensor:
        """
        Calculate stacking ensemble loss.
        """
        preds = _batched_ensemble_forward(self.models, x, method="cat")
        return cast(Tensor, self.meta_learner(preds))
