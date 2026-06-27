"""
MSE warmup → FunctionalEIV_MC trainer.

This module provides a lightweight training wrapper that pre-trains a model
with MSE for a configurable number of epochs, then switches to
``FunctionalEIVLoss(mode="mc")`` for the remainder.  On linear EIV data,
the warmup helps where pure MC struggles at high noise (modestly improving
RMSE_clean at σx=2.0) without degrading the MC advantage at σx=1.0.
On nonlinear data, warmup does not rescue MC but does not hurt either.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from ..losses.eiv import FunctionalEIVLoss
from ..utils.validation import check_tensor


class WarmupMCTrainer:
    """MSE warmup → FunctionalEIV_MC training wrapper.

    Pre-trains a model with plain MSE for ``warmup_epochs``, then switches to
    ``FunctionalEIVLoss(mode="mc")`` for the remaining epochs.  The same
    Adam optimizer (with momentum) is shared across both phases.

    Designed for linear EIV data where pure MC is competitive — warmup
    avoids early-phase MC gradient noise while preserving the MC advantage.

    Args:
        model_factory: Callable returning a freshly initialized model.
        sigma_x: Known input noise standard deviation.
        sigma_y: Optional target noise standard deviation.
        total_epochs: Total training epochs (warmup + MC).
        warmup_epochs: Number of MSE-only epochs before switching to MC.
            Must be less than ``total_epochs``.
        lr: Learning rate for Adam optimizer.
        batch_size: Mini-batch size.
        n_mc_samples: Number of MC perturbation samples for the MC phase.

    Example:
        >>> import torch
        >>> from torchregress.algorithms import WarmupMCTrainer
        >>>
        >>> def make_model():
        ...     return nn.Sequential(nn.Linear(6, 32), nn.ReLU(),
        ...                          nn.Linear(32, 32), nn.ReLU(),
        ...                          nn.Linear(32, 1))
        >>>
        >>> trainer = WarmupMCTrainer(
        ...     model_factory=make_model,
        ...     sigma_x=1.0,
        ...     total_epochs=60,
        ...     warmup_epochs=20,
        ... )
        >>> trainer.fit(X_train, y_train)
        >>> preds = trainer.predict(X_test)
    """

    def __init__(
        self,
        *,
        model_factory: Callable[[], nn.Module],
        sigma_x: float | torch.Tensor,
        sigma_y: float | torch.Tensor | None = None,
        total_epochs: int = 60,
        warmup_epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 64,
        n_mc_samples: int = 20,
    ) -> None:
        self.model_factory = model_factory
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.total_epochs = int(total_epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.n_mc_samples = int(n_mc_samples)

        self.model: nn.Module | None = None

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> "WarmupMCTrainer":
        """Fit the model with MSE warmup → FunctionalEIV_MC.

        Args:
            X: Noisy observed inputs, shape ``(N, D)``.
            y: Observed targets, shape ``(N, 1)`` or ``(N,)``.

        Returns:
            self (for chaining).
        """
        check_tensor(X, "X")
        check_tensor(y, "y")
        if X.ndim != 2:
            raise ValueError("X must be a 2D tensor (N, D)")
        if y.dim() == 1:
            y = y.unsqueeze(1)

        device = X.device
        model = self.model_factory().to(device)
        opt = Adam(model.parameters(), lr=self.lr)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Single loop, swap loss after warmup boundary
        mc_loss = FunctionalEIVLoss(
            model=model,
            sigma_x=self.sigma_x,
            sigma_y=self.sigma_y,
            mode="mc",
            n_samples=self.n_mc_samples,
        )
        model.train()
        for epoch in range(self.total_epochs):
            use_mc = epoch >= self.warmup_epochs
            for Xb, yb in loader:
                opt.zero_grad(set_to_none=True)
                if use_mc:
                    loss = mc_loss(Xb, yb)
                else:
                    loss = ((model(Xb) - yb) ** 2).mean()
                loss.backward()
                opt.step()

        model.eval()
        self.model = model
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict with the fitted model.

        Args:
            X: Input tensor, shape ``(N, D)``.

        Returns:
            Predictions, shape ``(N, 1)``.
        """
        check_tensor(X, "X")
        if self.model is None:
            raise RuntimeError("WarmupMCTrainer must be fit before calling predict")
        with torch.no_grad():
            return self.model(X.to(next(self.model.parameters()).device))
