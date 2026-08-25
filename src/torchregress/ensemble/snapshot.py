"""Snapshot ensembles: cyclic cosine-annealed LR snapshots (Huang et al., 2017).

The base model is trained through ``n_snapshots`` cosine annealing cycles; a
``state_dict`` copy is stored at each cycle minimum. Uncertainty prediction
mirrors the :class:`~torchregress.ensemble.mc_dropout.MCDropoutWrapper` API
(``predict_with_uncertainty`` / ``predict_interval`` trio).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader


class SnapshotEnsemble(nn.Module):
    """
    Collect model snapshots at the minima of cyclic cosine learning-rate
    annealing and treat them as an ensemble.

    Args:
        base_model: The network whose snapshots are collected.
        n_snapshots: Number of snapshots (= number of LR cycles).
        cycle_epochs: Epochs per cosine cycle.
        lr_max: Peak learning rate at each cycle start.
        lr_min: Minimum learning rate at each cycle end (snapshot point).

    Reference: Huang et al., "Snapshot Ensembles: Train 1, Get M for Free"
    (ICLR 2017).
    """

    def __init__(
        self,
        base_model: nn.Module,
        n_snapshots: int = 5,
        cycle_epochs: int = 40,
        lr_max: float = 1e-2,
        lr_min: float = 1e-5,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.n_snapshots = int(n_snapshots)
        self.cycle_epochs = int(cycle_epochs)
        self.lr_max = float(lr_max)
        self.lr_min = float(lr_min)
        # state_dict copies taken at cycle minima; a plain attribute so the
        # wrapped module's parameters/buffers are not re-registered here.
        self.snapshots: list[dict[str, Tensor]] = []

    def _cosine_lr(self, epoch_in_cycle: int) -> float:
        t = min(epoch_in_cycle / max(self.cycle_epochs, 1), 1.0)
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + math.cos(math.pi * t))

    def _resolve_device(self) -> torch.device:
        for param in self.base_model.parameters():
            return param.device
        return torch.device("cpu")

    def fit(
        self,
        dataloader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[nn.Module] = None,
        epochs: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> "SnapshotEnsemble":
        """Train with cyclic cosine-annealed LR, snapshotting at each cycle minimum."""
        total_epochs = epochs if epochs is not None else self.n_snapshots * self.cycle_epochs
        opt = (
            optimizer
            if optimizer is not None
            else torch.optim.AdamW(self.base_model.parameters(), lr=self.lr_max)
        )
        criterion = loss_fn if loss_fn is not None else nn.MSELoss()
        target_device = torch.device(device) if device is not None else self._resolve_device()

        self.snapshots = []
        self.base_model.to(target_device)
        for epoch in range(total_epochs):
            lr = self._cosine_lr(epoch % self.cycle_epochs)
            for group in opt.param_groups:
                group["lr"] = lr
            self.base_model.train()
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(target_device)
                y_batch = y_batch.to(target_device)
                opt.zero_grad()
                loss = criterion(self.base_model(x_batch), y_batch)
                loss.backward()
                opt.step()
            if (epoch + 1) % self.cycle_epochs == 0 and len(self.snapshots) < self.n_snapshots:
                self.snapshots.append(
                    {
                        name: tensor.detach().clone()
                        for name, tensor in self.base_model.state_dict().items()
                    }
                )
        return self

    def mc_forward(self, x: Tensor, n_samples: Optional[int] = None) -> Tensor:
        """
        Predict with every stored snapshot.

        Returns:
            Stacked predictions [n_snapshots, batch_size, output_dim].
        """
        if not self.snapshots:
            raise RuntimeError("SnapshotEnsemble has no snapshots; call fit() first")
        members = self.snapshots if n_samples is None else self.snapshots[: int(n_samples)]
        was_training = self.base_model.training
        self.base_model.eval()
        try:
            with torch.no_grad():
                preds = torch.stack([self._forward_with(sd, x) for sd in members], dim=0)
        finally:
            self.base_model.train(was_training)
        return preds

    def _forward_with(self, state: dict[str, Tensor], x: Tensor) -> Tensor:
        """Forward one snapshot through the base model without mutating it."""
        backup = {
            name: tensor.detach().clone() for name, tensor in self.base_model.state_dict().items()
        }
        try:
            self.base_model.load_state_dict(state)
            output = self.base_model(x)
        finally:
            self.base_model.load_state_dict(backup)
        assert isinstance(output, Tensor), f"expected Tensor from forward, got {type(output)!r}"
        return output

    def predict_with_uncertainty(
        self,
        x: Tensor,
        n_samples: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Make prediction with uncertainty estimate.

        Returns:
            Tuple of (mean, std) predictions over snapshot members.
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
        Compute prediction intervals using snapshot quantiles.

        Args:
            x: Input tensor [batch_size, ...]
            confidence: Confidence level (default 0.95)
            n_samples: Number of snapshot members to use (default all)

        Returns:
            Tuple of (lower, upper) bounds.
        """
        samples = self.mc_forward(x, n_samples)
        alpha = 1 - confidence
        lower = torch.quantile(samples, alpha / 2, dim=0)
        upper = torch.quantile(samples, 1 - alpha / 2, dim=0)
        return lower, upper

    def forward(self, x: Tensor) -> Tensor:
        """Ensemble mean prediction."""
        mean, _ = cast(Tuple[Tensor, Tensor], self.predict_with_uncertainty(x))
        return mean


__all__ = ["SnapshotEnsemble"]
