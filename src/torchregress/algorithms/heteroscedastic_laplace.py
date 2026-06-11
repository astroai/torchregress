"""
Effective Bayesian Heteroscedastic Regression using natural parameterization heads
and last-layer Laplace posterior approximation.

Reference: Immer et al., "Effective Bayesian Heteroscedastic Regression
with Deep Neural Networks" (NeurIPS 2023).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..prediction import PredictiveBatch


class NaturalHeteroscedasticHead(nn.Module):
    """
    Gaussian natural parameterization head.

    Predicts natural parameters:
    eta_1 = mean / var
    eta_2 = -1 / (2 * var)

    Transforms them to standard parameters mean and log_var.
    To ensure positivity of precision (variance > 0), a link function
    is used on the outputs representing eta_2.
    """

    def __init__(self, in_features: int, out_features: int, link_fn: str = "exp") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.link_fn = link_fn
        self.linear = nn.Linear(in_features, 2 * out_features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.linear(x)
        f1, f2 = torch.chunk(out, 2, dim=-1)

        if self.link_fn == "exp":
            # precision = exp(f2) -> var = exp(-f2)
            # mean = f1 / precision = f1 * var
            var = torch.exp(-f2)
            mean = f1 * var
            log_var = -f2
        elif self.link_fn == "softplus":
            precision = nn.functional.softplus(f2)
            var = 1.0 / (precision + 1e-8)
            mean = f1 * var
            log_var = torch.log(var + 1e-8)
        else:
            raise ValueError(f"Unknown link_fn: {self.link_fn}")

        return mean, log_var


class NaturalReparamHead(nn.Module):
    """
    Reparameterization helper that maps natural parameter vectors f1, f2 to mean and log_var.
    """

    def __init__(self, link_fn: str = "exp") -> None:
        super().__init__()
        self.link_fn = link_fn

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.link_fn == "exp":
            var = torch.exp(-f2)
            mean = f1 * var
            log_var = -f2
        elif self.link_fn == "softplus":
            precision = nn.functional.softplus(f2)
            var = 1.0 / (precision + 1e-8)
            mean = f1 * var
            log_var = torch.log(var + 1e-8)
        else:
            raise ValueError(f"Unknown link_fn: {self.link_fn}")
        return mean, log_var


class HeteroscedasticLaplaceRegressor(nn.Module):
    """
    Heteroscedastic Laplace Regressor.

    Performs last-layer Laplace posterior approximation over the weight and bias parameters
    of the heteroscedastic head. Supports natural parameterization heads.
    """

    def __init__(
        self,
        base_model: nn.Module,
        head: nn.Module,
        prior_precision: float = 1.0,
        n_samples: int = 30,
        jitter: float = 1e-6,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.head = head
        self.prior_precision = prior_precision
        self.n_samples = n_samples
        self.jitter = jitter
        self.is_fitted = False

        self.register_buffer("post_var_weight", None)
        self.register_buffer("post_var_bias", None)

    def _get_head_linear(self) -> nn.Linear:
        if hasattr(self.head, "linear") and isinstance(self.head.linear, nn.Linear):
            return self.head.linear
        if isinstance(self.head, nn.Linear):
            return self.head
        raise TypeError(
            "head must be an nn.Linear or contain a .linear attribute of type nn.Linear"
        )

    def fit(
        self,
        train_loader: DataLoader,
        lr: float = 1e-3,
        epochs: int = 10,
        device: Union[str, torch.device] = "cpu",
    ) -> HeteroscedasticLaplaceRegressor:
        """
        Train the model parameters and compute the last-layer Laplace posterior.
        """
        device = torch.device(device)
        self.to(device)

        # 1. Train model parameters (if epochs > 0)
        if epochs > 0:
            self.train()
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            for epoch in range(epochs):
                for x_batch, y_batch in train_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()

                    # Forward
                    feats = self.base_model(x_batch)
                    mean, log_var = self.head(feats)
                    var = torch.exp(log_var)

                    # NLL loss
                    nll = 0.5 * (
                        math.log(2 * math.pi) + log_var + (y_batch - mean) ** 2 / (var + 1e-8)
                    )
                    loss = nll.mean()

                    # Add L2 prior regularization on head parameters
                    head_linear = self._get_head_linear()
                    l2_reg = (
                        0.5
                        * self.prior_precision
                        * (head_linear.weight.square().sum() + head_linear.bias.square().sum())
                    )
                    total_loss = loss + l2_reg / len(train_loader.dataset)

                    total_loss.backward()
                    optimizer.step()

        # 2. Compute Laplace diagonal empirical Fisher w.r.t head linear parameters
        self.eval()
        head_linear = self._get_head_linear()
        head_params = {
            "weight": head_linear.weight,
            "bias": head_linear.bias,
        }

        # Accumulators
        fisher_weight = torch.zeros_like(head_linear.weight)
        fisher_bias = torch.zeros_like(head_linear.bias)

        # Extract all features and compute gradients per sample
        def single_sample_loss(params, feat, y):
            w = params["weight"]
            b = params["bias"]
            out = torch.nn.functional.linear(feat, w, b)
            if hasattr(self.head, "link_fn"):
                f1, f2 = torch.chunk(out, 2, dim=-1)
                if self.head.link_fn == "exp":
                    var = torch.exp(-f2)
                    mean = f1 * var
                    log_var = -f2
                elif self.head.link_fn == "softplus":
                    precision = nn.functional.softplus(f2)
                    var = 1.0 / (precision + 1e-8)
                    mean = f1 * var
                    log_var = torch.log(var + 1e-8)
            else:
                mean, log_var = torch.chunk(out, 2, dim=-1)

            var = torch.exp(log_var)
            nll = 0.5 * (math.log(2 * math.pi) + log_var + (y - mean) ** 2 / (var + 1e-8))
            return nll.sum()

        grad_fn = torch.func.grad(single_sample_loss, argnums=0)
        grad_batch_fn = torch.vmap(grad_fn, in_dims=(None, 0, 0))

        with torch.no_grad():
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                feats = self.base_model(x_batch)

                # Compute per-sample gradients in the batch
                # grads is a dict containing tensors of shape [B, ...]
                grads = grad_batch_fn(head_params, feats, y_batch)

                # Accumulate empirical Fisher diagonal
                fisher_weight += torch.sum(grads["weight"] ** 2, dim=0)
                fisher_bias += torch.sum(grads["bias"] ** 2, dim=0)

        # 3. Compute posterior variances: var = 1 / (Fisher + prior_precision)
        self.post_var_weight = 1.0 / (fisher_weight + self.prior_precision + self.jitter)
        self.post_var_bias = 1.0 / (fisher_bias + self.prior_precision + self.jitter)
        self.is_fitted = True

        return self

    def predict_distribution(
        self,
        x: torch.Tensor,
        n_samples: Optional[int] = None,
    ) -> PredictiveBatch:
        """
        Predictive distribution using MC sampling from the last-layer Laplace posterior.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict_distribution")

        n = n_samples or self.n_samples
        self.eval()

        head_linear = self._get_head_linear()

        # Sample head weights and biases
        # post_var shape matching weight: [2*out_features, hidden_dim]
        # post_var shape matching bias: [2*out_features]
        w_mean = head_linear.weight
        b_mean = head_linear.bias

        # 1. Forward backbone once to get features
        with torch.no_grad():
            feats = self.base_model(x)  # [B, hidden_dim]

        # 2. Sample and evaluate
        sampled_means = []
        sampled_vars = []
        sampled_points = []

        with torch.no_grad():
            for _ in range(n):
                w_sample = w_mean + torch.randn_like(w_mean) * torch.sqrt(self.post_var_weight)
                b_sample = b_mean + torch.randn_like(b_mean) * torch.sqrt(self.post_var_bias)

                out = torch.nn.functional.linear(feats, w_sample, b_sample)
                if hasattr(self.head, "link_fn"):
                    f1, f2 = torch.chunk(out, 2, dim=-1)
                    if self.head.link_fn == "exp":
                        var = torch.exp(-f2)
                        mean = f1 * var
                    elif self.head.link_fn == "softplus":
                        precision = nn.functional.softplus(f2)
                        var = 1.0 / (precision + 1e-8)
                        mean = f1 * var
                else:
                    mean, log_var = torch.chunk(out, 2, dim=-1)
                    var = torch.exp(log_var)

                # Sample target prediction (mean + noise)
                noise = torch.randn_like(mean) * torch.sqrt(var + 1e-8)
                point = mean + noise

                sampled_means.append(mean)
                sampled_vars.append(var)
                sampled_points.append(point)

        # [n_samples, B, out_features]
        means_tensor = torch.stack(sampled_means)
        vars_tensor = torch.stack(sampled_vars)
        points_tensor = torch.stack(sampled_points)

        # Epistemic: variance of means across samples
        epistemic = means_tensor.var(dim=0)
        # Aleatoric: mean of predicted variances
        aleatoric = vars_tensor.mean(dim=0)
        total_var = epistemic + aleatoric
        total_std = torch.sqrt(total_var + 1e-8)

        # Final predictions: mean of sampled means
        final_mean = means_tensor.mean(dim=0)

        # Reshape samples to [B, n_samples] for PredictiveBatch compatibility (if 1D output)
        if final_mean.shape[-1] == 1:
            samples_flat = points_tensor.squeeze(-1).transpose(0, 1)  # [B, n_samples]
        else:
            samples_flat = points_tensor.transpose(0, 1)  # [B, n_samples, out_features]

        return PredictiveBatch(
            point=final_mean,
            mean=final_mean,
            std=total_std,
            samples=samples_flat,
            extra={
                "epistemic_variance": epistemic,
                "aleatoric_variance": aleatoric,
            },
        )
