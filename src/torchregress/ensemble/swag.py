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

from collections.abc import Mapping
from typing import Any, cast

import torch
import torch.nn as nn

from torchregress.utils.gaussian_output import variance_from_logvar


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

    References
    ----------
    .. [1] Maddox, W. J., Izmailov, P., Garipov, T., Vetrov, D. P., & Wilson, A. G. (2019).
       A Simple Baseline for Bayesian Uncertainty Estimation in Deep Learning.
       In *NeurIPS 2019*. https://arxiv.org/abs/1902.02476
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

        # Cache parameter names replaced with underscore to avoid overhead
        self._name_map = {}

        # Will store mean and second moment of parameters
        for name, param in self.base_model.named_parameters():
            if param.requires_grad:
                name_cleaned = name.replace(".", "_")
                self._name_map[name] = name_cleaned
                self.register_buffer(f"{name_cleaned}_mean", torch.zeros_like(param))
                self.register_buffer(f"{name_cleaned}_sq_mean", torch.zeros_like(param))

        # TR-ENS-01: Do NOT place posterior over BatchNorm running stats. They are
        # deterministic population estimates, not weights; sampling them breaks
        # equivariance (Maddox Alg.1 keeps BN in train+recalibrate, not Gaussian).
        # Only floating buffers >1 element that are NOT BN running buffers are tracked
        # (e.g., LayerNorm already has no running buffers). Scalars/int are skipped.
        self._buffer_name_map: dict[str, str] = {}
        for name, buf in self.base_model.named_buffers():
            if not buf.is_floating_point() or buf.numel() <= 1:
                continue
            # skip BatchNorm running stats
            if "running_mean" in name or "running_var" in name or "num_batches_tracked" in name:
                continue
            # heuristic: skip any buffer attached to a BatchNorm module
            module_name = name.rsplit(".", 1)[0] if "." in name else ""
            try:
                mod = self.base_model.get_submodule(module_name) if module_name else None
                if isinstance(
                    mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
                ):
                    continue
            except AttributeError:
                pass
            name_cleaned = name.replace(".", "_")
            self._buffer_name_map[name] = name_cleaned
            self.register_buffer(f"{name_cleaned}_mean", torch.zeros_like(buf))
            self.register_buffer(f"{name_cleaned}_sq_mean", torch.zeros_like(buf))
        # TR-ENS-02: low-rank deviations are persistent registered buffers
        # (created lazily on first collect on the model's device) so they are
        # included in state_dict and never pay per-sample H2D copies.

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
        n = int(cast(torch.Tensor, self.n_models).item())
        for (name, param), (_, base_param) in zip(
            model.named_parameters(), self.base_model.named_parameters()
        ):
            if not param.requires_grad:
                continue

            name_cleaned = self._name_map[name]
            mean_buffer = getattr(self, f"{name_cleaned}_mean")
            sq_mean_buffer = getattr(self, f"{name_cleaned}_sq_mean")

            # Online update: new_mean = (n*old_mean + new_value) / (n+1)
            mean_buffer.mul_(n / (n + 1)).add_(param.data / (n + 1))
            sq_mean_buffer.mul_(n / (n + 1)).add_(param.data**2 / (n + 1))

        # Update tracked buffers (e.g. BN running stats) the same way.
        for (name, buf), (_, base_buf) in zip(
            model.named_buffers(), self.base_model.named_buffers()
        ):
            if name not in self._buffer_name_map:
                continue
            name_cleaned = self._buffer_name_map[name]
            mean_buffer = getattr(self, f"{name_cleaned}_mean")
            sq_mean_buffer = getattr(self, f"{name_cleaned}_sq_mean")
            mean_buffer.mul_(n / (n + 1)).add_(buf.detach() / (n + 1))
            sq_mean_buffer.mul_(n / (n + 1)).add_(buf.detach().square() / (n + 1))

        # Store deviation from mean (for low-rank covariance)
        idx = n % self.max_num_models
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            name_cleaned = self._name_map[name]
            mean_buffer = getattr(self, f"{name_cleaned}_mean")
            devs_name = f"{name_cleaned}_devs"
            if not hasattr(self, devs_name):
                self.register_buffer(
                    devs_name,
                    torch.zeros((self.max_num_models,) + param.shape, device=param.device),
                )
            getattr(self, devs_name)[idx] = param.data - mean_buffer

        cast(torch.Tensor, self.n_models).add_(1)

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
        if int(cast(torch.Tensor, self.n_models).item()) == 0:
            raise ValueError("No models collected yet. Call collect_model() first.")

        # Sample from diagonal Gaussian (per-parameter variance)
        for name, param in self.base_model.named_parameters():
            if not param.requires_grad:
                continue

            name_cleaned = self._name_map[name]
            mean = getattr(self, f"{name_cleaned}_mean")
            sq_mean = getattr(self, f"{name_cleaned}_sq_mean")

            # Variance: Var[X] = E[X²] - E[X]²
            var = torch.clamp(sq_mean - mean**2, self.var_clamp)

            if diag_noise:
                # Sample: θ ~ N(mean, scale² * var)
                param.data.copy_(mean + scale * torch.randn_like(mean) * var.sqrt())
            else:
                param.data.copy_(mean)

        # TR-ENS-01: sampled buffers (e.g. BN running stats) are written back
        # into the base model so forward passes use posterior buffer draws.
        for name, buf in self.base_model.named_buffers():
            if name not in self._buffer_name_map:
                continue
            name_cleaned = self._buffer_name_map[name]
            mean = getattr(self, f"{name_cleaned}_mean")
            sq_mean = getattr(self, f"{name_cleaned}_sq_mean")
            var = torch.clamp(sq_mean - mean**2, self.var_clamp)
            buf.data.copy_(mean + scale * torch.randn_like(mean) * var.sqrt())

        # Add low-rank component from stored deviations
        n_models_val = int(cast(torch.Tensor, self.n_models).item())
        num_deviations = min(n_models_val, self.max_num_models)

        if num_deviations > 0:
            # Sample coefficients: z ~ N(0, 1)
            device = next(self.base_model.parameters()).device
            z = torch.randn(num_deviations, device=device)

            # Denominator for scaling. Avoid division by zero if only one model.
            k = num_deviations
            denom = (k - 1) ** 0.5 if k > 1 else 1.0

            for name, param in self.base_model.named_parameters():
                if not param.requires_grad:
                    continue

                # TR-ENS-02: deviations live in registered buffers on-device.
                dev_stack = getattr(self, f"{self._name_map[name]}_devs")[:num_deviations]

                # Calculate low-rank sample with efficient tensor dot product
                low_rank_sample = torch.tensordot(z, dev_stack, dims=([0], [0])) / denom

                param.data.add_(scale * low_rank_sample)

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        """Passthrough that lazily registers ``*_devs`` buffers before loading.

        Deviation buffers are created on first collect; a freshly constructed
        SWAG does not have them yet, so register any missing ones from the
        incoming checkpoint first (TR-ENS-02 round-trip guarantee).
        """
        for name in list(state_dict):
            if name.endswith("_devs") and not hasattr(self, name):
                buf = state_dict[name]
                self.register_buffer(name, torch.zeros_like(buf))
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
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

    References
    ----------
    .. [1] Maddox, W. J., Izmailov, P., Garipov, T., Vetrov, D. P., & Wilson, A. G. (2019).
       A Simple Baseline for Bayesian Uncertainty Estimation in Deep Learning.
       In *NeurIPS 2019*. https://arxiv.org/abs/1902.02476
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
        correction: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with decomposed epistemic and aleatoric uncertainty.

        Args:
            x: Input tensor [batch_size, ...]
            n_samples: Number of samples per SWAG model. Default: 30
            scale: Sampling scale for SWAG. Default: 1.0
            correction: Bessel's correction setting. Default: 0

        Returns:
            tuple of (mean, epistemic_var, aleatoric_var):
                - mean: Predicted mean [batch_size, output_dim]
                - epistemic_var: Total epistemic (model) uncertainty (between-mode
                  + within-mode) [batch_size, output_dim]
                - aleatoric_var: Aleatoric (data) uncertainty [batch_size, output_dim]
        """
        all_swag_means = []
        all_swag_within_vars = []
        all_swag_aleatorics = []
        is_heteroscedastic = False

        # Sample from each SWAG
        for swag_model_module in self.swag_models:
            swag_model = cast(SWAG, swag_model_module)
            swag_means_s = []
            swag_vars_s = []

            for _ in range(n_samples):
                swag_model.sample(scale=scale)
                with torch.no_grad():
                    pred = swag_model(x)

                # Unpack heteroscedastic predictions if available
                if isinstance(pred, tuple) and len(pred) == 2:
                    is_heteroscedastic = True
                    mean_s, log_var_s = pred
                    var_s = variance_from_logvar(log_var_s)
                elif isinstance(pred, torch.Tensor) and pred.ndim > 1 and pred.shape[-1] % 2 == 0:
                    # Check if output dimension is even (potential concatenated mean/logvar)
                    # We compare shape[-1] with even to split, but let's be careful.
                    # BaseEnsembleModel has parse_heteroscedastic_output or similar utility.
                    # Let's check format. For standard torchregress heteroscedastic models,
                    # the last dim is split in half.
                    is_heteroscedastic = True
                    d = pred.shape[-1] // 2
                    mean_s, log_var_s = pred[..., :d], pred[..., d:]
                    var_s = variance_from_logvar(log_var_s)
                else:
                    mean_s = pred
                    var_s = torch.zeros_like(pred)

                swag_means_s.append(mean_s)
                swag_vars_s.append(var_s)

            # Expected predictions within this SWAG mode
            stacked_means_s = torch.stack(swag_means_s)
            swag_mean = torch.mean(stacked_means_s, dim=0)

            # Within-mode epistemic uncertainty (variance of weight samples)
            swag_within_var = torch.var(stacked_means_s, dim=0, correction=correction)

            # Expected aleatoric uncertainty for this mode
            swag_aleatoric = torch.mean(torch.stack(swag_vars_s), dim=0)

            all_swag_means.append(swag_mean)
            all_swag_within_vars.append(swag_within_var)
            all_swag_aleatorics.append(swag_aleatoric)

        # Stack across K modes: [n_models, batch_size, output_dim]
        all_preds = torch.stack(all_swag_means)
        mean = all_preds.mean(0)

        # Between-mode epistemic (variance across model modes/optima)
        between_mode_epistemic = torch.var(all_preds, dim=0, correction=correction)

        # Average within-mode epistemic
        within_mode_epistemic = torch.stack(all_swag_within_vars).mean(0)

        # Total epistemic uncertainty = between-mode + within-mode
        epistemic_var = between_mode_epistemic + within_mode_epistemic

        # Aleatoric uncertainty: average aleatoric across modes
        if is_heteroscedastic:
            aleatoric_var = torch.stack(all_swag_aleatorics).mean(0)
        else:
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

        for swag_model_module in self.swag_models:
            swag_model = cast(SWAG, swag_model_module)
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
