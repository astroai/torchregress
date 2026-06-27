"""
Latent-input neural regression for errors-in-variables.

This module provides a lightweight LatentNN-style algorithm for tabular
regression with noisy inputs. The method jointly optimizes network parameters
and latent clean inputs, using a Gaussian quadratic penalty to keep the latent
inputs close to the observed values.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..utils.validation import check_tensor


class LatentNN:
    """
    Latent-input neural regressor for noisy-feature problems.

    This is a compact neural analogue of latent-variable errors-in-variables
    regression. Training jointly updates:
    - the network parameters `theta`
    - per-sample latent clean inputs `x_latent`

    The objective is:

    .. math::

        L = L_model(f_theta(x_latent), y) +
            lambda_x * mean(((x_latent - x_obs) / sigma_x)^2)

    where `L_model` can be a point or probabilistic regression loss.

    Args:
        model_factory: Callable returning a freshly initialized model.
        loss_fn: Loss used on model predictions and targets. Defaults to MSE.
        sigma_x: Known input standard deviation. May be scalar, feature vector,
            or per-sample matrix with shape `(N, D)`.
        sigma_y: Optional target standard deviation. When provided, it rescales
            the model loss by `mean(1 / sigma_y^2)` within each batch.
        epochs: Number of optimization epochs.
        lr: Learning rate for model parameters.
        latent_lr: Learning rate for latent inputs. Defaults to 0.1.
        batch_size: Mini-batch size. Defaults to full-batch training.
        weight_decay: Weight decay on model parameters.
        latent_weight_decay: Optional weight decay on latent inputs.
        latent_penalty_weight: Additional multiplier on the latent-input penalty.
        max_grad_norm: Optional gradient clipping threshold.
    """

    def __init__(
        self,
        *,
        model_factory: Callable[[], nn.Module],
        loss_fn: nn.Module | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        sigma_x: float | torch.Tensor = 1.0,
        sigma_y: float | torch.Tensor | None = None,
        epochs: int = 500,
        lr: float = 1.0e-3,
        latent_lr: float | None = 0.1,
        batch_size: int | None = None,
        weight_decay: float = 0.0,
        latent_weight_decay: float = 0.0,
        latent_penalty_weight: float = 1.0,
        max_grad_norm: float | None = None,
    ) -> None:
        self.model_factory = model_factory
        self.loss_fn = loss_fn or nn.MSELoss()
        self.sigma_x_input = sigma_x
        self.sigma_y_input = sigma_y
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.latent_lr = float(latent_lr) if latent_lr is not None else 0.1
        self.batch_size = batch_size
        self.weight_decay = float(weight_decay)
        self.latent_weight_decay = float(latent_weight_decay)
        self.latent_penalty_weight = float(latent_penalty_weight)
        self.max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None

        self.model: nn.Module | None = None
        self.x_latent_: torch.Tensor | None = None

    @staticmethod
    def _per_sample_quadratic(values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 1:
            return values.pow(2)
        return values.reshape(values.shape[0], -1).pow(2).sum(dim=1)

    def _expand_sigma(
        self,
        sigma: float | torch.Tensor | None,
        reference: torch.Tensor,
        *,
        name: str,
    ) -> torch.Tensor:
        if sigma is None:
            return torch.ones_like(reference)
        if isinstance(sigma, (int, float)):
            return torch.full_like(reference, float(sigma))
        sigma_t = sigma.to(device=reference.device, dtype=reference.dtype)
        if sigma_t.ndim == 0:
            return torch.full_like(reference, float(sigma_t.item()))
        if sigma_t.shape == reference.shape:
            return sigma_t
        if sigma_t.ndim == 1 and sigma_t.shape[0] == reference.shape[-1]:
            return sigma_t.reshape(1, -1).expand_as(reference)
        if sigma_t.ndim == 2 and sigma_t.shape[0] == 1 and sigma_t.shape[1] == reference.shape[-1]:
            return sigma_t.expand_as(reference)
        if sigma_t.ndim == 2 and sigma_t.shape[1] == 1 and sigma_t.shape[0] == reference.shape[0]:
            return sigma_t.expand_as(reference)
        raise ValueError(
            f"{name} must be scalar, feature vector, or match reference shape; "
            f"got {tuple(sigma_t.shape)} for reference {tuple(reference.shape)}"
        )

    def fit(
        self,
        X_observed: torch.Tensor,
        y_observed: torch.Tensor,
        *,
        X_val: torch.Tensor | None = None,
        y_val: torch.Tensor | None = None,
    ) -> "LatentNN":
        """Fit the model and latent inputs jointly."""
        check_tensor(X_observed, "X_observed")
        check_tensor(y_observed, "y_observed")
        if X_observed.ndim != 2:
            raise ValueError("X_observed must be a 2D tensor (N, D)")
        if y_observed.ndim != 2:
            raise ValueError("y_observed must be a 2D tensor (N, K)")
        if X_observed.shape[0] != y_observed.shape[0]:
            raise ValueError("X_observed and y_observed must have matching batch dimension")
        if X_val is not None:
            check_tensor(X_val, "X_val")
        if y_val is not None:
            check_tensor(y_val, "y_val")

        model = self.model_factory().to(X_observed.device)
        x_latent = nn.Parameter(X_observed.clone())

        sigma_x = self._expand_sigma(self.sigma_x_input, X_observed, name="sigma_x").clamp_min(
            1.0e-6
        )
        sigma_y = self._expand_sigma(self.sigma_y_input, y_observed, name="sigma_y").clamp_min(
            1.0e-6
        )

        indices = torch.arange(X_observed.shape[0], device=X_observed.device)
        batch_size = min(int(self.batch_size or X_observed.shape[0]), X_observed.shape[0])
        loader = DataLoader(
            TensorDataset(indices, X_observed, y_observed, sigma_x, sigma_y),
            batch_size=batch_size,
            shuffle=True,
        )

        optimizer = torch.optim.Adam(
            [
                {
                    "params": model.parameters(),
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": [x_latent],
                    "lr": self.latent_lr,
                    "weight_decay": self.latent_weight_decay,
                },
            ]
        )

        for _ in range(self.epochs):
            model.train()
            for batch_idx, x_obs_b, y_b, sigma_x_b, sigma_y_b in loader:
                optimizer.zero_grad()
                x_latent_b = x_latent[batch_idx]
                pred = model(x_latent_b)
                if isinstance(self.loss_fn, nn.MSELoss):
                    model_residual = (pred - y_b) / sigma_y_b
                    model_loss = self._per_sample_quadratic(model_residual).mean()
                else:
                    model_loss = self.loss_fn(pred, y_b)
                    if isinstance(model_loss, torch.Tensor) and model_loss.ndim > 0:
                        model_loss = model_loss.mean()
                latent_residual = (x_latent_b - x_obs_b) / sigma_x_b
                latent_penalty = self._per_sample_quadratic(latent_residual).mean()
                loss = model_loss + self.latent_penalty_weight * latent_penalty
                loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_([x_latent], self.max_grad_norm)
                optimizer.step()

        self.model = model.eval()
        self.x_latent_ = x_latent.detach()
        return self

    def predict(
        self,
        X: torch.Tensor,
        *,
        sigma_x: float | torch.Tensor | None = None,
        n_samples: int = 20,
    ) -> torch.Tensor:
        """Predict with the fitted network.

        Args:
            X: Input tensor (noisy observations).
            sigma_x: Optional known input noise stddev. When provided, performs
                test-time MC input-noise marginalization by perturbing inputs
                and averaging predictions, bridging the distribution shift
                between clean latent training and noisy test data.
            n_samples: Number of MC perturbations when sigma_x is provided.

        Returns:
            Predictions tensor.
        """
        check_tensor(X, "X")
        if self.model is None:
            raise RuntimeError("LatentNN must be fit before predicting")

        device = next(self.model.parameters()).device

        if sigma_x is not None:
            X = X.to(device)
            sigma_x_t = self._expand_sigma(sigma_x, X, name="sigma_x").clamp_min(1.0e-6)
            with torch.no_grad():
                noise = torch.randn(n_samples, *X.shape, device=device, dtype=X.dtype)
                perturbed = X.unsqueeze(0) + noise * sigma_x_t.unsqueeze(0)
                flat = perturbed.reshape(-1, X.shape[-1])
                preds = self.model(flat)
                preds = preds.reshape(n_samples, X.shape[0], -1)
                return preds.mean(dim=0)

        with torch.no_grad():
            return self.model(X.to(device))
