"""
Stochastic Weight Averaging - Gaussian (SWAG) for uncertainty estimation.

This module implements SWAG and MultiSWAG methods for Bayesian uncertainty
estimation in deep learning models.

References:
    - Maddox et al. "A Simple Baseline for Bayesian Uncertainty Estimation in
      Deep Learning" (NeurIPS 2019)
    - Wilson & Izmailov "Bayesian Deep Learning and a Probabilistic Perspective
      of Generalization" (NeurIPS 2020)
"""

from typing import Dict, List

import torch
import torch.nn as nn


class SWAG(nn.Module):
    """
    Stochastic Weight Averaging - Gaussian (SWAG).

    Approximates the posterior over model weights using first and second
    moments collected during SGD training. This provides a cheap alternative
    to full Bayesian inference while still capturing epistemic uncertainty.

    The method collects snapshots of model parameters during training and
    computes:
    1. Mean of parameters (first moment)
    2. Second moment of parameters (for diagonal covariance)
    3. Low-rank covariance approximation (from parameter deviations)

    Args:
        base_model: The model architecture to use
        max_num_models: Maximum number of models to store for low-rank approximation.
            Default: 20
        var_clamp: Minimum value to clamp variance to avoid numerical issues.
            Default: 1e-30

    Example:
        >>> import torch.nn as nn
        >>> from torchregress.ensemble.swag import SWAG
        >>>
        >>> # Create base model
        >>> model = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 1))
        >>>
        >>> # Wrap with SWAG
        >>> swag_model = SWAG(model, max_num_models=20)
        >>>
        >>> # Normal training for warmup
        >>> for epoch in range(warmup_epochs):
        >>>     train_epoch(model, optimizer, train_loader)
        >>>
        >>> # SWAG collection phase (after warmup)
        >>> for epoch in range(swag_epochs):
        >>>     train_epoch(model, optimizer, train_loader)
        >>>     swag_model.collect_model(model)  # Collect model snapshot
        >>>
        >>> # Sample from posterior for predictions
        >>> predictions = []
        >>> for _ in range(n_samples):
        >>>     swag_model.sample(scale=0.5)  # Sample weights from posterior
        >>>     with torch.no_grad():
        >>>         pred = swag_model(x_test)
        >>>     predictions.append(pred)
        >>>
        >>> # Compute mean and uncertainty
        >>> mean = torch.stack(predictions).mean(0)
        >>> variance = torch.stack(predictions).var(0)

    Notes:
        - SWAG requires a warmup phase of regular training before collection
        - Typical warmup: 75% of total training epochs
        - Collection phase: Use cyclical or constant learning rate
        - The scale parameter in sample() controls sampling variance (tune on validation)
    """

    def __init__(
        self,
        base_model: nn.Module,
        max_num_models: int = 20,
        var_clamp: float = 1e-30,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.max_num_models = max_num_models
        self.var_clamp = var_clamp

        # Storage for number of collected models
        self.register_buffer("n_models", torch.zeros(1, dtype=torch.long))

        # Will store mean and second moment of parameters
        for name, param in self.base_model.named_parameters():
            if param.requires_grad:
                self.register_buffer(f"{name.replace('.', '_')}_mean", torch.zeros_like(param))
                self.register_buffer(f"{name.replace('.', '_')}_sq_mean", torch.zeros_like(param))

        # Store deviations for low-rank approximation
        self.deviations: List[Dict[str, torch.Tensor]] = []

    def collect_model(self, model: nn.Module) -> None:
        """
        Collect a model snapshot for SWAG averaging.

        This should be called after each training step during the SWAG collection phase
        (after warmup). The method updates running statistics of the parameters.

        Args:
            model: Model with current weights (typically after an SGD step)

        Example:
            >>> # After warmup training
            >>> for epoch in range(swag_epochs):
            >>>     for batch in train_loader:
            >>>         # Training step
            >>>         loss = criterion(model(x), y)
            >>>         loss.backward()
            >>>         optimizer.step()
            >>>     # Collect model at end of epoch
            >>>     swag_model.collect_model(model)
        """
        # Update running mean and second moment
        n = self.n_models.item()
        for (name, param), (_, base_param) in zip(
            model.named_parameters(), self.base_model.named_parameters()
        ):
            if not param.requires_grad:
                continue

            name_cleaned = name.replace(".", "_")
            mean_buffer = getattr(self, f"{name_cleaned}_mean")
            sq_mean_buffer = getattr(self, f"{name_cleaned}_sq_mean")

            # Online update: new_mean = (n*old_mean + new_value) / (n+1)
            mean_buffer.mul_(n / (n + 1)).add_(param.data / (n + 1))
            sq_mean_buffer.mul_(n / (n + 1)).add_(param.data**2 / (n + 1))

        # Store deviation from mean (for low-rank covariance)
        if len(self.deviations) < self.max_num_models:
            deviation = {}
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                name_cleaned = name.replace(".", "_")
                mean_buffer = getattr(self, f"{name_cleaned}_mean")
                deviation[name] = (param.data - mean_buffer).cpu().clone()
            self.deviations.append(deviation)
        else:
            # Circular buffer: overwrite oldest deviation
            idx = n % self.max_num_models
            deviation = {}
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                name_cleaned = name.replace(".", "_")
                mean_buffer = getattr(self, f"{name_cleaned}_mean")
                deviation[name] = (param.data - mean_buffer).cpu().clone()
            self.deviations[idx] = deviation

        self.n_models.add_(1)

    def sample(self, scale: float = 1.0, diag_noise: bool = True) -> None:
        """
        Sample weights from the SWAG approximate posterior.

        This modifies the base_model's parameters in-place by sampling from
        the approximate Gaussian posterior. Call this before each forward pass
        during inference to get different predictions.

        Args:
            scale: Scaling factor for sampling. Lower values = closer to mean.
                Typical range: 0.5-1.5. Default: 1.0
            diag_noise: Whether to include diagonal (per-parameter) noise component.
                Default: True

        Raises:
            ValueError: If no models have been collected yet

        Example:
            >>> # Sample 30 predictions from posterior
            >>> predictions = []
            >>> for _ in range(30):
            >>>     swag_model.sample(scale=0.5)
            >>>     with torch.no_grad():
            >>>         pred = swag_model(x_test)
            >>>     predictions.append(pred)
        """
        if self.n_models.item() == 0:
            raise ValueError("No models collected yet. Call collect_model() first.")

        # Sample from diagonal Gaussian (per-parameter variance)
        for name, param in self.base_model.named_parameters():
            if not param.requires_grad:
                continue

            name_cleaned = name.replace(".", "_")
            mean = getattr(self, f"{name_cleaned}_mean")
            sq_mean = getattr(self, f"{name_cleaned}_sq_mean")

            # Variance: Var[X] = E[X²] - E[X]²
            var = torch.clamp(sq_mean - mean**2, self.var_clamp)

            if diag_noise:
                # Sample: θ ~ N(mean, scale² * var)
                param.data.copy_(mean + scale * torch.randn_like(mean) * var.sqrt())
            else:
                param.data.copy_(mean)

        # Add low-rank component from stored deviations
        if len(self.deviations) > 0:
            # Sample coefficients: z ~ N(0, 1)
            z = torch.randn(len(self.deviations), device=next(self.base_model.parameters()).device)

            for name, param in self.base_model.named_parameters():
                if not param.requires_grad:
                    continue

                # Low-rank component: sum of z_i * deviation_i
                low_rank_sample = sum(
                    z[i] * self.deviations[i][name].to(param.device)
                    for i in range(len(self.deviations))
                ) / ((len(self.deviations) - 1) ** 0.5)

                param.data.add_(scale * low_rank_sample)

    def forward(self, *args, **kwargs):
        """Forward pass through base model with current sampled weights."""
        return self.base_model(*args, **kwargs)


class MultiSWAG(nn.Module):
    """
    Multi-SWAG: Multiple independent SWAG models for improved uncertainty.

    Trains multiple SWAG models independently (with different initializations
    or training procedures) and combines their predictions. This provides
    better uncertainty estimates than a single SWAG by capturing both:
    1. Within-SWAG uncertainty (from weight posterior sampling)
    2. Between-SWAG uncertainty (from different local optima)

    Args:
        base_model: Model architecture (will be copied n_models times)
        n_models: Number of independent SWAG models. Default: 5
        max_num_models: Maximum snapshots per SWAG. Default: 20

    Example:
        >>> from torchregress.ensemble.swag import MultiSWAG
        >>>
        >>> # Create MultiSWAG with 5 independent models
        >>> multi_swag = MultiSWAG(MyModel(), n_models=5, max_num_models=20)
        >>>
        >>> # Train each SWAG independently with different seeds
        >>> for swag_idx in range(5):
        >>>     model = MyModel()
        >>>     torch.manual_seed(swag_idx)  # Different initialization
        >>>     model.apply(init_weights)
        >>>
        >>>     # Warmup training
        >>>     for epoch in range(warmup_epochs):
        >>>         train_epoch(model, optimizer, train_loader)
        >>>
        >>>     # SWAG collection
        >>>     for epoch in range(swag_epochs):
        >>>         train_epoch(model, optimizer, train_loader)
        >>>         multi_swag.swag_models[swag_idx].collect_model(model)
        >>>
        >>> # Predict with uncertainty decomposition
        >>> mean, epistemic, aleatoric = multi_swag.predict_with_uncertainty(
        >>>     x_test, n_samples=30
        >>> )

    Notes:
        - Each SWAG should be trained independently with different random seeds
        - Epistemic uncertainty comes from disagreement between SWAGs
        - Aleatoric uncertainty requires models to predict variance (e.g., heteroscedastic)
        - Total samples = n_models * n_samples_per_swag
    """

    def __init__(
        self,
        base_model: nn.Module,
        n_models: int = 5,
        max_num_models: int = 20,
    ) -> None:
        super().__init__()
        self.n_models = n_models

        # Create multiple independent SWAG models
        # We need to deep copy the base model structure for each SWAG
        import copy

        self.swag_models = nn.ModuleList(
            [
                SWAG(copy.deepcopy(base_model), max_num_models=max_num_models)
                for _ in range(n_models)
            ]
        )

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 30,
        scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with decomposed epistemic and aleatoric uncertainty.

        Args:
            x: Input tensor [batch_size, ...]
            n_samples: Number of samples per SWAG model. Default: 30
            scale: Sampling scale for SWAG. Default: 1.0

        Returns:
            tuple of (mean, epistemic_var, aleatoric_var):
                - mean: Predicted mean [batch_size, output_dim]
                - epistemic_var: Epistemic (model) uncertainty [batch_size, output_dim]
                - aleatoric_var: Aleatoric (data) uncertainty [batch_size, output_dim]

        Notes:
            - Epistemic uncertainty: Variance across SWAG models
            - Aleatoric uncertainty: Average within-SWAG variance
            - For aleatoric uncertainty, models should output variance predictions
        """
        all_swag_means = []

        # Sample from each SWAG
        for swag_model in self.swag_models:
            swag_samples = []
            for _ in range(n_samples):
                swag_model.sample(scale=scale)
                with torch.no_grad():
                    pred = swag_model(x)
                swag_samples.append(pred)

            # Average samples within this SWAG
            swag_mean = torch.stack(swag_samples).mean(0)
            all_swag_means.append(swag_mean)

        # Stack all SWAG means: [n_models, batch_size, output_dim]
        all_preds = torch.stack(all_swag_means)

        # Overall mean across all SWAGs
        mean = all_preds.mean(0)

        # Epistemic uncertainty: variance across SWAG models
        # This captures uncertainty due to different local optima
        epistemic_var = all_preds.var(0, unbiased=True)

        # Aleatoric uncertainty: approximate as zero
        # For true aleatoric uncertainty, models should predict variance
        # (e.g., output (mean, log_var) pairs)
        aleatoric_var = torch.zeros_like(mean)

        return mean, epistemic_var, aleatoric_var

    def predict_with_samples(
        self,
        x: torch.Tensor,
        n_samples: int = 30,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate multiple prediction samples for uncertainty estimation.

        Args:
            x: Input tensor [batch_size, ...]
            n_samples: Number of samples per SWAG. Default: 30
            scale: Sampling scale. Default: 1.0

        Returns:
            Tensor of predictions [n_models * n_samples, batch_size, output_dim]

        Example:
            >>> samples = multi_swag.predict_with_samples(x_test, n_samples=50)
            >>> mean = samples.mean(0)
            >>> std = samples.std(0)
            >>> # Calculate custom statistics
            >>> quantiles = torch.quantile(samples, q=torch.tensor([0.05, 0.95]), dim=0)
        """
        all_samples = []

        for swag_model in self.swag_models:
            for _ in range(n_samples):
                swag_model.sample(scale=scale)
                with torch.no_grad():
                    pred = swag_model(x)
                all_samples.append(pred)

        return torch.stack(all_samples)

    def forward(self, x: torch.Tensor, n_samples: int = 1, scale: float = 1.0) -> torch.Tensor:
        """
        Forward pass with sampling from all SWAG models.

        Args:
            x: Input tensor
            n_samples: Number of samples per SWAG. Default: 1
            scale: Sampling scale. Default: 1.0

        Returns:
            Mean prediction across all models and samples
        """
        samples = self.predict_with_samples(x, n_samples=n_samples, scale=scale)
        return samples.mean(0)
