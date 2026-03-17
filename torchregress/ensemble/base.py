"""
Base classes for ensemble models.

This module provides foundation classes and abstractions for all ensemble techniques
in the torchregress library.
"""

from copy import deepcopy
from typing import Any, Dict, Union

import torch
import torch.nn as nn


class BaseEnsembleModel(nn.Module):
    """
    Base class for ensemble models.

    This class provides common functionality for different ensemble techniques.

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
        super().__init__()
        self.ensemble_size = ensemble_size
        self.device = device

        # Create ensemble members
        self.models = nn.ModuleList()
        for i in range(ensemble_size):
            if isinstance(base_model, type):
                # If base_model is a class, instantiate it with kwargs
                model = base_model(**base_model_kwargs)
            else:
                # Otherwise, make a deep copy of the provided instance
                model = deepcopy(base_model)
            self.models.append(model)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, list[Any]]:
        """
        Forward pass computes predictions from all ensemble members.

        Args:
            x: Input tensor [batch_size, ...]

        Returns:
            Stacked predictions from each ensemble member [ensemble_size, batch_size, ...]
            for tensor outputs, otherwise a list of per-member outputs.
        """
        outputs = [model(x) for model in self.models]
        if outputs and isinstance(outputs[0], torch.Tensor):
            return torch.stack(outputs)
        return outputs

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make prediction with uncertainty estimates.

        Args:
            x: Input tensor [batch_size, ...]

        Returns:
            Dictionary with mean and variance of predictions.
        """
        with torch.no_grad():
            # Get predictions from all ensemble members
            stacked_preds = self.forward(x)
            if isinstance(stacked_preds, list):
                if not all(isinstance(pred, torch.Tensor) for pred in stacked_preds):
                    raise ValueError(
                        "BaseEnsembleModel.predict expects tensor outputs. "
                        "Use a specialized ensemble for structured outputs."
                    )
                stacked_preds = torch.stack(stacked_preds)

            # Calculate mean across ensemble dimension
            mean = torch.mean(stacked_preds, dim=0)

            # Calculate variance across ensemble dimension
            variance = torch.var(stacked_preds, dim=0, unbiased=True)

            return {"mean": mean, "variance": variance}

    def predict_full_covariance(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make prediction with full-output covariance estimation.

        Args:
            x: Input tensor [batch_size, ...]

        Returns:
            Dictionary with:
                - 'mean': [batch_size, output_dim]
                - 'covariance': [batch_size, output_dim, output_dim]
        """
        with torch.no_grad():
            preds = self.forward(x)
            if isinstance(preds, list):
                if not all(isinstance(pred, torch.Tensor) for pred in preds):
                    raise ValueError(
                        "BaseEnsembleModel.predict_full_covariance expects tensor outputs. "
                        "Use a specialized ensemble for structured outputs."
                    )
                preds = torch.stack(preds)
            stacked = preds  # [ensemble_size, batch, dim]
            mean = torch.mean(stacked, dim=0)
            # Compute sample covariance across ensemble members
            # stacked => [M, B, D] -> [B, M, D]
            p = stacked.permute(1, 0, 2)
            p_centered = p - mean.unsqueeze(1)
            cov = torch.einsum("bmd,bnd->bmn", p_centered, p_centered) / (self.ensemble_size - 1)
            return {"mean": mean, "covariance": cov}

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        epochs: int = 10,
        lr: float = 1e-3,
        optimizer_cls: type = torch.optim.Adam,
        verbose: bool = True,
        device: Union[str, torch.device, None] = None,
    ) -> Dict[str, list]:
        """
        Train each ensemble member independently.
        """
        device = device or self.device
        member_histories = []

        for model in self.models:
            model.to(device)

        if not hasattr(self, "_optimizers") or getattr(self, "_optimizer_cls", None) is not optimizer_cls:
            self._optimizers = [optimizer_cls(model.parameters(), lr=lr) for model in self.models]
            self._optimizer_cls = optimizer_cls
        else:
            for opt in self._optimizers:
                for param_group in opt.param_groups:
                    param_group["lr"] = lr

        for idx, model in enumerate(self.models):
            optimizer = self._optimizers[idx]
            history = []

            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                batch_count = 0

                for batch in train_loader:
                    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                        x, y = batch[0], batch[1]
                    else:
                        raise ValueError("train_loader must yield (inputs, targets) tuples")

                    x = x.to(device)
                    y = y.to(device)

                    optimizer.zero_grad()
                    preds = model(x)
                    loss = criterion(preds, y)
                    loss.backward()
                    optimizer.step()

                    running_loss += float(loss.detach().item())
                    batch_count += 1

                epoch_loss = running_loss / max(batch_count, 1)
                history.append(epoch_loss)

                if verbose:
                    print(
                        f"Member {idx + 1}/{self.ensemble_size} "
                        f"Epoch {epoch + 1}/{epochs} "
                        f"Loss {epoch_loss:.6f}"
                    )

            member_histories.append(history)

        return {"member_histories": member_histories}
