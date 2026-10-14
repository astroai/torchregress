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

        device = X_observed.device
        n_rows = X_observed.shape[0]
        model = self.model_factory().to(device)
        # Latent clean inputs are optimised with a *lazy* (row-sparse) Adam: a
        # mini-batch step touches only its own rows' values and moments, with
        # per-row step counts for bias correction.  A dense ``torch.optim.Adam``
        # on an (N, D) parameter costs O(N * D) per step -- O(N^2 D / B) per
        # epoch, days at N ~ 1e6 -- and keeps moving rows outside the batch
        # through momentum on a zero gradient.  Full-batch training is identical
        # to dense Adam.
        x_latent = X_observed.detach().clone()
        m_latent = torch.zeros_like(x_latent)
        v_latent = torch.zeros_like(x_latent)
        steps = torch.zeros(n_rows, 1, device=device, dtype=x_latent.dtype)
        beta1, beta2, adam_eps = 0.9, 0.999, 1.0e-8

        sigma_x = self._expand_sigma(self.sigma_x_input, X_observed, name="sigma_x").clamp_min(
            1.0e-6
        )
        sigma_y = self._expand_sigma(self.sigma_y_input, y_observed, name="sigma_y").clamp_min(
            1.0e-6
        )

        batch_size = min(int(self.batch_size or n_rows), n_rows)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for _ in range(self.epochs):
            model.train()
            # Device-side shuffling: no per-row Python indexing (DataLoader over a
            # TensorDataset costs one __getitem__ per row per epoch).
            order = torch.randperm(n_rows, device=device)
            for start in range(0, n_rows, batch_size):
                batch_idx = order[start : start + batch_size]
                x_obs_b = X_observed[batch_idx]
                y_b = y_observed[batch_idx]
                sigma_x_b = sigma_x[batch_idx]
                sigma_y_b = sigma_y[batch_idx]
                x_latent_b = x_latent[batch_idx].requires_grad_(True)

                optimizer.zero_grad()
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
                    torch.nn.utils.clip_grad_norm_([x_latent_b], self.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    grad = x_latent_b.grad
                    assert grad is not None
                    rows = x_latent_b.detach()
                    if self.latent_weight_decay:
                        grad = grad + self.latent_weight_decay * rows
                    m_b = beta1 * m_latent[batch_idx] + (1 - beta1) * grad
                    v_b = beta2 * v_latent[batch_idx] + (1 - beta2) * grad.pow(2)
                    t_b = steps[batch_idx] + 1
                    m_hat = m_b / (1 - beta1**t_b)
                    v_hat = v_b / (1 - beta2**t_b)
                    m_latent[batch_idx] = m_b
                    v_latent[batch_idx] = v_b
                    steps[batch_idx] = t_b
                    x_latent[batch_idx] = rows - self.latent_lr * m_hat / (v_hat.sqrt() + adam_eps)

        self.model = model.eval()
        self.x_latent_ = x_latent
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
            # Row chunks bound the (n_samples, rows, D) perturbation tensor.
            chunk = max(1, (1 << 22) // max(1, n_samples * X.shape[-1]))
            outs = []
            with torch.no_grad():
                for start in range(0, X.shape[0], chunk):
                    xb, sb = X[start : start + chunk], sigma_x_t[start : start + chunk]
                    noise = torch.randn(n_samples, *xb.shape, device=device, dtype=X.dtype)
                    flat = (xb.unsqueeze(0) + noise * sb.unsqueeze(0)).reshape(-1, X.shape[-1])
                    outs.append(self.model(flat).reshape(n_samples, xb.shape[0], -1).mean(dim=0))
            return torch.cat(outs, dim=0)

        with torch.no_grad():
            return self.model(X.to(device))
