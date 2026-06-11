"""
Bayesian Neural Networks for uncertainty estimation.

This module provides variational Bayesian neural network layers and models
for principled uncertainty quantification in regression tasks.

Reference: Blundell et al., "Weight Uncertainty in Neural Networks" (ICML 2015)
"""

import math
from typing import Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VariationalLinear(nn.Module):
    """
    Variational Bayesian linear layer.

    Uses the local reparameterization trick for efficient gradient estimation.
    Weights are sampled from a Gaussian posterior: w ~ N(mu, sigma^2).

    Args:
        in_features: Size of input features
        out_features: Size of output features
        prior_sigma: Prior standard deviation (default: 1.0)
        bias: Whether to include bias (default: True)

    Example:
        >>> layer = VariationalLinear(64, 32)
        >>> output = layer(x)  # Different output each call due to sampling
        >>> kl_div = layer.kl_divergence()  # For ELBO loss
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_sigma: float = 1.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma
        self.use_bias = bias

        # Weight parameters: mean and log(sigma)
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(torch.zeros(out_features, in_features))

        # Bias parameters
        if bias:
            self.bias_mu = nn.Parameter(torch.zeros(out_features))
            self.bias_log_sigma = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_log_sigma", None)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters using Xavier/Glorot initialization."""
        # Initialize mean with Xavier
        nn.init.xavier_normal_(self.weight_mu)

        # Initialize log_sigma to give small initial variance
        nn.init.constant_(self.weight_log_sigma, -3.0)

        if self.use_bias:
            nn.init.zeros_(self.bias_mu)
            nn.init.constant_(self.bias_log_sigma, -3.0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with weight sampling.

        Uses local reparameterization: instead of sampling weights,
        we sample the output directly for efficiency.
        """
        # Compute mean of output
        output_mu = F.linear(x, self.weight_mu, self.bias_mu)

        # Compute variance of output (local reparameterization)
        weight_sigma = torch.exp(self.weight_log_sigma)
        output_sigma_sq = F.linear(x.pow(2), weight_sigma.pow(2))

        if self.use_bias:
            bias_sigma = torch.exp(self.bias_log_sigma)
            output_sigma_sq = output_sigma_sq + bias_sigma.pow(2)

        output_sigma = torch.sqrt(output_sigma_sq + 1e-8)

        # Sample output using reparameterization trick
        eps = torch.randn_like(output_mu)
        return output_mu + output_sigma * eps

    def kl_divergence(self) -> Tensor:
        """
        Compute KL divergence between posterior and prior.

        KL(q(w) || p(w)) where q(w) = N(mu, sigma^2) and p(w) = N(0, prior_sigma^2)
        """
        weight_sigma = torch.exp(self.weight_log_sigma)
        prior_sigma = self.prior_sigma

        # KL divergence for Gaussian: log(sigma_p/sigma_q) + (sigma_q^2 + mu^2)/(2*sigma_p^2) - 0.5
        kl = (
            math.log(prior_sigma)
            - self.weight_log_sigma
            + (weight_sigma.pow(2) + self.weight_mu.pow(2)) / (2 * prior_sigma**2)
            - 0.5
        )
        kl_sum = kl.sum()

        if self.use_bias:
            bias_sigma = torch.exp(self.bias_log_sigma)
            kl_bias = (
                math.log(prior_sigma)
                - self.bias_log_sigma
                + (bias_sigma.pow(2) + self.bias_mu.pow(2)) / (2 * prior_sigma**2)
                - 0.5
            )
            kl_sum = kl_sum + kl_bias.sum()

        return cast(Tensor, kl_sum)


class BayesianNeuralNetwork(nn.Module):
    """
    Bayesian Neural Network for regression with uncertainty estimation.

    Uses variational inference with the ELBO (Evidence Lower Bound) objective.

    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension
        prior_sigma: Prior standard deviation for weights
        n_samples: Number of MC samples for inference (default: 30)

    Example:
        >>> model = BayesianNeuralNetwork(input_dim=10, hidden_dims=[64, 32], output_dim=1)
        >>> # Training
        >>> pred = model(x)
        >>> nll_loss = F.mse_loss(pred, y)
        >>> kl_loss = model.kl_divergence()
        >>> total_loss = nll_loss + kl_loss / n_train  # ELBO
        >>> # Inference
        >>> mean, std = model.predict_with_uncertainty(x_test)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        prior_sigma: float = 1.0,
        n_samples: int = 30,
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.prior_sigma = prior_sigma

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(VariationalLinear(prev_dim, hidden_dim, prior_sigma))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(VariationalLinear(prev_dim, output_dim, prior_sigma))
        self.network = nn.Sequential(*layers)

        # Store variational layers for KL computation
        self.variational_layers: list[VariationalLinear] = [
            layer for layer in self.network if isinstance(layer, VariationalLinear)
        ]

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with weight sampling."""
        return cast(Tensor, self.network(x))

    def kl_divergence(self) -> Tensor:
        """Compute total KL divergence across all variational layers."""
        kl = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.variational_layers:
            kl = kl + layer.kl_divergence()
        return kl

    def elbo_loss(
        self,
        pred: Tensor,
        target: Tensor,
        n_data: int,
        beta: float = 1.0,
    ) -> Tensor:
        """
        Compute ELBO loss for training.

        ELBO = E[log p(y|x,w)] - beta * KL(q(w) || p(w)) / n_data

        Args:
            pred: Model predictions
            target: Ground truth targets
            n_data: Total number of training samples (for KL scaling)
            beta: KL weighting factor (default: 1.0)

        Returns:
            Negative ELBO loss (to minimize)
        """
        # Reconstruction loss (negative log likelihood)
        nll = F.mse_loss(pred, target, reduction="mean")

        # KL divergence scaled by dataset size
        kl = self.kl_divergence() / n_data

        return nll + beta * kl

    def mc_forward(self, x: Tensor, n_samples: Optional[int] = None) -> Tensor:
        """
        Multiple forward passes for uncertainty estimation.

        Args:
            x: Input tensor [batch_size, ...]
            n_samples: Number of samples (uses default if None)

        Returns:
            Stacked predictions [n_samples, batch_size, output_dim]
        """
        n = n_samples or self.n_samples

        with torch.no_grad():
            repeat_dims = [n] + [1] * (x.dim() - 1)
            x_expanded = x.repeat(*repeat_dims)
            preds = self.forward(x_expanded)
            return preds.view(n, x.shape[0], *preds.shape[1:])

    def predict_with_uncertainty(
        self,
        x: Tensor,
        n_samples: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Make prediction with uncertainty estimate.

        Args:
            x: Input tensor [batch_size, ...]
            n_samples: Number of MC samples (uses default if None)

        Returns:
            Tuple of (mean, std) predictions
        """
        samples = self.mc_forward(x, n_samples)
        return samples.mean(dim=0), samples.std(dim=0)

    def predict_interval(
        self,
        x: Tensor,
        confidence: float = 0.95,
        n_samples: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute prediction intervals using MC samples.

        Args:
            x: Input tensor [batch_size, ...]
            confidence: Confidence level (default 0.95)
            n_samples: Number of MC samples (uses default if None)

        Returns:
            Tuple of (lower, upper) bounds
        """
        samples = self.mc_forward(x, n_samples)
        alpha = 1 - confidence
        lower = torch.quantile(samples, alpha / 2, dim=0)
        upper = torch.quantile(samples, 1 - alpha / 2, dim=0)
        return lower, upper


class HeteroscedasticBNN(nn.Module):
    """
    Heteroscedastic BNN that predicts both mean and variance.

    Combines epistemic uncertainty (from weight uncertainty) with
    aleatoric uncertainty (predicted variance).

    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension
        prior_sigma: Prior standard deviation for weights
        n_samples: Number of MC samples for inference

    Example:
        >>> model = HeteroscedasticBNN(input_dim=10, hidden_dims=[64, 32], output_dim=1)
        >>> mean, aleatoric, epistemic = model.predict_with_decomposition(x_test)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        prior_sigma: float = 1.0,
        n_samples: int = 30,
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.output_dim = output_dim

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(VariationalLinear(prev_dim, hidden_dim, prior_sigma))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer predicts mean and log_variance
        layers.append(VariationalLinear(prev_dim, 2 * output_dim, prior_sigma))
        self.network = nn.Sequential(*layers)

        self.variational_layers: list[VariationalLinear] = [
            layer for layer in self.network if isinstance(layer, VariationalLinear)
        ]

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass returning mean and log_variance.

        Returns:
            Tuple of (mean, log_var)
        """
        out = self.network(x)
        mean = out[..., : self.output_dim]
        log_var = out[..., self.output_dim :]
        return mean, log_var

    def kl_divergence(self) -> Tensor:
        """Compute total KL divergence."""
        kl = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.variational_layers:
            kl = kl + layer.kl_divergence()
        return kl

    def mc_forward(self, x: Tensor, n_samples: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        """
        Multiple forward passes for uncertainty estimation.

        Returns:
            Tuple of (means, log_vars) each [n_samples, batch_size, output_dim]
        """
        n = n_samples or self.n_samples

        with torch.no_grad():
            repeat_dims = [n] + [1] * (x.dim() - 1)
            x_expanded = x.repeat(*repeat_dims)
            means, log_vars = self.forward(x_expanded)

            means = means.view(n, x.shape[0], *means.shape[1:])
            log_vars = log_vars.view(n, x.shape[0], *log_vars.shape[1:])

        return means, log_vars

    def predict_with_decomposition(
        self,
        x: Tensor,
        n_samples: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Predict with decomposed uncertainty.

        Returns:
            Tuple of (mean, aleatoric_variance, epistemic_variance)
        """
        means, log_vars = self.mc_forward(x, n_samples)

        # Epistemic: variance of means across samples
        epistemic = means.var(dim=0)

        # Aleatoric: mean of predicted variances
        aleatoric = torch.exp(log_vars).mean(dim=0)

        # Mean prediction
        mean = means.mean(dim=0)

        return mean, aleatoric, epistemic

    def predict_with_uncertainty(
        self,
        x: Tensor,
        n_samples: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Predict with total uncertainty.

        Returns:
            Tuple of (mean, total_std)
        """
        mean, aleatoric, epistemic = self.predict_with_decomposition(x, n_samples)
        total_std = torch.sqrt(aleatoric + epistemic)
        return mean, total_std

    def predict_interval(
        self,
        x: Tensor,
        confidence: float = 0.95,
        n_samples: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute prediction intervals.

        Args:
            x: Input tensor
            confidence: Confidence level
            n_samples: Number of MC samples

        Returns:
            Tuple of (lower, upper) bounds
        """
        means, log_vars = self.mc_forward(x, n_samples)

        # Sample from predicted distributions
        stds = torch.exp(0.5 * log_vars)
        eps = torch.randn_like(means)
        samples = means + stds * eps  # [n_samples, batch, output_dim]

        alpha = 1 - confidence
        lower = torch.quantile(samples, alpha / 2, dim=0)
        upper = torch.quantile(samples, 1 - alpha / 2, dim=0)

        return lower, upper
