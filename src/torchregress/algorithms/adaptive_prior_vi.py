"""
Adaptive-Prior Bayesian Uncertainty under Distribution Shift (VIDS-style).

Reference: Slavutsky & Blei, "Quantifying Uncertainty in the Presence of
Distribution Shifts" (arXiv:2506.03942 / NeurIPS 2025).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..prediction import PredictiveBatch


class SyntheticEnvironmentSampler:
    """
    Bootstrap-based sampler for generating synthetic shifted environments.
    """

    def __init__(self, bootstrap_fraction: float = 0.3, n_environments: int = 32) -> None:
        self.bootstrap_fraction = bootstrap_fraction
        self.n_environments = n_environments

    def sample(self, X: torch.Tensor, Y: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        N = X.shape[0]
        k = max(1, int(self.bootstrap_fraction * N))
        environments = []
        for _ in range(self.n_environments):
            indices = torch.randint(0, N, (k,), device=X.device)
            environments.append((X[indices], Y[indices]))
        return environments


class AdaptivePriorGuide(nn.Module):
    """
    Posterior guide network that amortizes variational posterior parameters.

    References
    ----------
    .. [1] Slavutsky, et al. (2025). VIDS: Variational Inference under Covariate Shift.
       In *NeurIPS 2025*. https://arxiv.org/abs/2506.18283
    """

    def __init__(
        self,
        in_features: int,
        target_dim: int,
        param_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        # Input size: context_X (2 * in_features) + context_Y (2 * target_dim)
        input_dim = 2 * in_features + 2 * target_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * param_dim),
        )

    def forward(
        self, context_X: torch.Tensor, context_Y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # context_X: [2 * in_features], context_Y: [2 * target_dim]
        inp = torch.cat([context_X, context_Y], dim=-1)
        out = self.net(inp)
        mean, log_var = torch.chunk(out, 2, dim=-1)
        # Clamp log_var for numerical stability
        log_var = torch.clamp(log_var, min=-10.0, max=5.0)
        return mean, log_var


class AdaptivePriorNetwork(nn.Module):
    """
    Prior network mapping training context and query test features to parameter prior parameters.
    """

    def __init__(
        self,
        in_features: int,
        param_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        # Input size: context_X (2 * in_features) + query feature (in_features)
        input_dim = 3 * in_features
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * param_dim),
        )

    def forward(
        self, context_X: torch.Tensor, x_query: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # context_X: [2 * in_features], x_query: [B, in_features]
        # Expand context_X to match batch size
        B = x_query.shape[0]
        context_expanded = context_X.unsqueeze(0).expand(B, -1)
        inp = torch.cat([context_expanded, x_query], dim=-1)
        out = self.net(inp)
        mean, log_var = torch.chunk(out, 2, dim=-1)
        log_var = torch.clamp(log_var, min=-10.0, max=5.0)
        return mean, log_var


class VIDSRegressor(nn.Module):
    """
    VIDS Variational Regressor under covariate shifts.

    References
    ----------
    .. [1] Slavutsky, et al. (2025). VIDS: Variational Inference under Covariate Shift.
       In *NeurIPS 2025*. https://arxiv.org/abs/2506.18283
    """

    def __init__(
        self,
        in_features: int,
        target_dim: int = 1,
        hidden_dim: int = 64,
        prior_variance_init: float = 1.0,
        noise_variance_init: float = 0.1,
        jitter: float = 1e-6,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.target_dim = target_dim
        self.jitter = jitter

        # Weight and bias parameter dimension: P = target_dim * in_features + target_dim
        self.param_dim = target_dim * in_features + target_dim

        # Guide and Prior networks
        self.guide = AdaptivePriorGuide(in_features, target_dim, self.param_dim, hidden_dim)
        self.prior_net = AdaptivePriorNetwork(in_features, self.param_dim, hidden_dim)

        # Log observation noise variance
        self.log_noise_var = nn.Parameter(torch.tensor(math.log(noise_variance_init)))

        self.register_buffer("x_train_mean", None)
        self.register_buffer("x_train_std", None)
        self.is_fitted = False

    def _get_context(
        self, X: torch.Tensor, Y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        mean_X = X.mean(dim=0)
        std_X = X.std(dim=0) + self.jitter
        context_X = torch.cat([mean_X, std_X], dim=-1)

        context_Y = None
        if Y is not None:
            mean_Y = Y.mean(dim=0)
            std_Y = Y.std(dim=0) + self.jitter
            context_Y = torch.cat([mean_Y, std_Y], dim=-1)

        return context_X, context_Y

    def _split_weights_bias(
        self, params_sampled: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # params_sampled shape: [S, P]
        S = params_sampled.shape[0]
        w_size = self.target_dim * self.in_features
        w_flat = params_sampled[:, :w_size]
        b = params_sampled[:, w_size:]
        w = w_flat.reshape(S, self.target_dim, self.in_features)
        return w, b

    def fit(
        self,
        x_train_features: torch.Tensor,
        y_train: torch.Tensor,
        n_environments: int = 32,
        bootstrap_fraction: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 50,
        beta: float = 1.0,
        n_samples: int = 10,
    ) -> VIDSRegressor:
        """
        Fits the variational posterior and adaptive prior using bootstrapped environments.
        """
        device = x_train_features.device
        self.to(device)

        # Store training context summaries for test-time prior conditioning
        self.x_train_mean = x_train_features.mean(dim=0)
        self.x_train_std = x_train_features.std(dim=0) + self.jitter

        sampler = SyntheticEnvironmentSampler(bootstrap_fraction, n_environments)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.train()
        for epoch in range(epochs):
            # Re-sample synthetic environments each epoch to prevent overfitting
            envs = sampler.sample(x_train_features, y_train)
            epoch_loss = 0.0

            for X_env, Y_env in envs:
                optimizer.zero_grad()

                context_X, context_Y = self._get_context(X_env, Y_env)

                # 1. Predict posterior guide parameters
                post_mean, post_log_var = self.guide(context_X, context_Y)
                post_std = torch.exp(0.5 * post_log_var)

                # 2. Sample parameters from variational posterior
                eps = torch.randn(n_samples, self.param_dim, device=device)
                params_sampled = post_mean.unsqueeze(0) + eps * post_std.unsqueeze(
                    0
                )  # [n_samples, P]

                # Split parameters into weight and bias
                w_samples, b_samples = self._split_weights_bias(
                    params_sampled
                )  # [S, target_dim, in_features], [S, target_dim]

                # 3. Compute expected NLL over the environment
                # Compute predictions: [S, B, target_dim]
                # X_env: [B, in_features], w_samples: [S, target_dim, in_features]
                y_pred = torch.einsum("bi,sji->sbj", X_env, w_samples) + b_samples.unsqueeze(1)

                noise_var = torch.exp(self.log_noise_var)
                # Sum NLL over target dimensions and data points
                se = (Y_env.unsqueeze(0) - y_pred) ** 2  # [S, B, target_dim]
                nll = 0.5 * (se / (noise_var + 1e-8) + self.log_noise_var + math.log(2 * math.pi))
                nll_loss = nll.mean(dim=0).sum()

                # 4. Compute KL divergence against adaptive prior
                # Prior parameters are predicted for each point x_n in the environment
                prior_mean, prior_log_var = self.prior_net(context_X, X_env)  # [B, P], [B, P]
                prior_var = torch.exp(prior_log_var)
                post_var = post_std**2

                # Analytical KL for diagonal Gaussians: post w.r.t prior
                # KL = 0.5 * sum( log(prior_var/post_var) +
                #      (post_var + (post_mean - prior_mean)^2)/prior_var - 1 )
                kl = 0.5 * (
                    prior_log_var
                    - post_log_var.unsqueeze(0)
                    + (post_var.unsqueeze(0) + (post_mean.unsqueeze(0) - prior_mean) ** 2)
                    / (prior_var + 1e-8)
                    - 1.0
                )  # [B, P]
                kl_loss = kl.mean(dim=0).sum()

                # ELBO loss
                loss = nll_loss + beta * kl_loss
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

        self.is_fitted = True
        return self

    def predict_distribution(
        self, x_test: torch.Tensor, n_samples: Optional[int] = None
    ) -> PredictiveBatch:
        """
        Evaluates the adaptive prior on the test points to output predictions and uncertainty.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict_distribution")

        n = n_samples or 30
        self.eval()

        device = x_test.device
        context_X = torch.cat([self.x_train_mean, self.x_train_std], dim=-1)

        with torch.no_grad():
            # Predict prior parameters for each test point
            # prior_mean, prior_log_var shapes: [B, P]
            prior_mean, prior_log_var = self.prior_net(context_X, x_test)
            prior_std = torch.exp(0.5 * prior_log_var)

            B = x_test.shape[0]
            sampled_predictions = []

            # Sample S weights and biases for each test point separately
            for s in range(n):
                eps = torch.randn(B, self.param_dim, device=device)
                params_sampled = prior_mean + eps * prior_std  # [B, P]

                # Map parameters to weight and bias for each sample
                # weight: [B, target_dim, in_features], bias: [B, target_dim]
                w_size = self.target_dim * self.in_features
                w_flat = params_sampled[:, :w_size]
                b = params_sampled[:, w_size:]
                w = w_flat.reshape(B, self.target_dim, self.in_features)

                # Predict: mean = sum_j w_ij * x_j + b_i
                pred_mean = torch.sum(w * x_test.unsqueeze(1), dim=-1) + b  # [B, target_dim]

                # Sample point = mean + observation noise
                noise_var = torch.exp(self.log_noise_var)
                noise = torch.randn_like(pred_mean) * torch.sqrt(noise_var + 1e-8)
                point = pred_mean + noise

                sampled_predictions.append((pred_mean, point))

        means_list, points_list = zip(*sampled_predictions)
        means_tensor = torch.stack(means_list)  # [n_samples, B, target_dim]
        points_tensor = torch.stack(points_list)  # [n_samples, B, target_dim]

        # Epistemic: variance of means across samples
        epistemic = means_tensor.var(dim=0)
        # Aleatoric: observation noise variance
        aleatoric = torch.exp(self.log_noise_var) + torch.zeros_like(epistemic)
        total_var = epistemic + aleatoric
        total_std = torch.sqrt(total_var + 1e-8)

        final_mean = means_tensor.mean(dim=0)

        # Reshape samples to [B, n_samples] for PredictiveBatch compatibility (if 1D output)
        if self.target_dim == 1:
            samples_flat = points_tensor.squeeze(-1).transpose(0, 1)  # [B, n_samples]
        else:
            samples_flat = points_tensor.transpose(0, 1)  # [B, n_samples, target_dim]

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
