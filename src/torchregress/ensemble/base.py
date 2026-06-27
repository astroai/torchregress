"""
Base classes for ensemble models.

This module provides foundation classes and abstractions for all ensemble techniques
in the torchregress library.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch
from torch import nn

Optimizer = torch.optim.Optimizer
OptimizerLike = Optimizer | tuple[Optimizer, ...]


def _optimizer_like_zero_grad(opt: OptimizerLike) -> None:
    for o in opt if isinstance(opt, tuple) else (opt,):
        o.zero_grad()


def _optimizer_like_step(opt: OptimizerLike) -> None:
    for o in opt if isinstance(opt, tuple) else (opt,):
        o.step()


def _optimizer_like_set_lr(opt: OptimizerLike, lr: float) -> None:
    for o in opt if isinstance(opt, tuple) else (opt,):
        for param_group in o.param_groups:
            param_group["lr"] = lr


@dataclass(frozen=True)
class EnsembleFitConfig:
    """
    Configuration options for training an ensemble model.

    Parameters
    ----------
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    optimizer_cls : type
        The optimizer class to use (defaults to torch.optim.Adam).
    optimizer_kwargs : Optional[dict[str, Any]]
        Additional keyword arguments for the optimizer.
    optimizer_factory : Optional[Callable[[nn.Module], OptimizerLike]]
        A factory function to instantiate custom optimizer(s) for a member model.
        If provided, `optimizer_cls` and `optimizer_kwargs` are ignored.
    verbose : bool
        Whether to print training progress.
    device : Optional[Union[str, torch.device]]
        The device to use for training.
    """

    epochs: int = 10
    lr: float = 1e-3
    optimizer_cls: type = torch.optim.Adam
    optimizer_kwargs: Optional[dict[str, Any]] = None
    optimizer_factory: Optional[Callable[[nn.Module], OptimizerLike]] = None
    verbose: bool = True
    device: Union[str, torch.device, None] = None


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
        base_model: Union[nn.Module, type, None] = None,
        ensemble_size: int = 5,
        device: str = "cpu",
        member_factory: Optional[Callable[[int, Optional[int]], nn.Module]] = None,
        base_seed: Optional[int] = None,
        reset_parameters: bool = True,
        **base_model_kwargs: Any,
    ) -> None:
        super().__init__()
        self.ensemble_size = ensemble_size
        self.device = device

        # Create ensemble members
        self.models = nn.ModuleList()
        for i in range(ensemble_size):
            seed = base_seed + i if base_seed is not None else None
            if member_factory is not None:
                if seed is not None:
                    torch.manual_seed(seed)
                model = member_factory(i, seed)
            elif isinstance(base_model, type):
                # If base_model is a class, instantiate it with kwargs
                if seed is not None:
                    torch.manual_seed(seed)
                model = base_model(**base_model_kwargs)
            elif isinstance(base_model, nn.Module):
                # Otherwise, make a deep copy of the provided instance
                model = deepcopy(base_model)
                if reset_parameters:
                    self._reset_model_parameters(model, seed)
            else:
                raise ValueError("Either base_model or member_factory must be provided.")
            self.models.append(model)

        # Check if parameters of different members are identical
        if ensemble_size > 1 and len(self.models) > 0:
            all_identical = True
            first_model_params = list(self.models[0].parameters())
            if len(first_model_params) > 0:
                for m in self.models[1:]:
                    for p1, p2 in zip(first_model_params, m.parameters()):
                        if not torch.equal(p1, p2):
                            all_identical = False
                            break
                    if not all_identical:
                        break
                if all_identical:
                    import warnings

                    warnings.warn(
                        "All ensemble members have identical parameter values. "
                        "Make sure to use different initializations or seeds to "
                        "promote functional diversity.",
                        UserWarning,
                        stacklevel=2,
                    )

    def _reset_model_parameters(self, model: nn.Module, seed: Optional[int] = None) -> None:
        if seed is not None:
            torch.manual_seed(seed)
        for module in model.modules():
            if hasattr(module, "reset_parameters") and callable(module.reset_parameters):
                module.reset_parameters()

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

    def predict(self, x: torch.Tensor, correction: int = 0) -> Dict[str, torch.Tensor]:
        """
        Make prediction with uncertainty estimates.

        Args:
            x: Input tensor [batch_size, ...]
            correction: Bessel's correction setting (0 for population/predictive-mixture
                variance, 1 for sample variance). Default: 0

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
            variance = torch.var(stacked_preds, dim=0, correction=correction)

            return {"mean": mean, "variance": variance}

    def predict_full_covariance(
        self, x: torch.Tensor, correction: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Make prediction with full-output covariance estimation.

        Args:
            x: Input tensor [batch_size, ...]
            correction: Bessel's correction setting. Default: 0

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
            # Compute covariance across ensemble members
            # stacked => [M, B, D] -> [B, M, D]
            p = stacked.permute(1, 0, 2)
            p_centered = p - mean.unsqueeze(1)
            # Correct contraction to output dimensions: [B, D, D] instead of [B, M, M]
            cov = torch.einsum("bmd,bme->bde", p_centered, p_centered)
            denom = max(self.ensemble_size - correction, 1)
            return {"mean": mean, "covariance": cov / denom}

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        epochs: int = 10,
        lr: float = 1e-3,
        optimizer_cls: type = torch.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        optimizer_factory: Optional[Callable[[nn.Module], OptimizerLike]] = None,
        verbose: bool = True,
        device: Union[str, torch.device, None] = None,
    ) -> Dict[str, list]:
        """
        Train each ensemble member independently.

        When ``optimizer_factory`` is set, it is called as ``factory(model)`` for
        each member and must return a single ``torch.optim.Optimizer`` or a tuple
        of optimizers (e.g. AdamW + Muon). In that case ``optimizer_cls`` and
        ``optimizer_kwargs`` are ignored.
        """

        cfg = EnsembleFitConfig(
            epochs=epochs,
            lr=lr,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            optimizer_factory=optimizer_factory,
            verbose=verbose,
            device=device,
        )
        device = cfg.device or self.device

        member_histories = []
        optimizer_kwargs = dict(cfg.optimizer_kwargs or {})
        optimizer_signature = tuple(
            sorted((str(key), repr(value)) for key, value in optimizer_kwargs.items())
        )

        for model in self.models:
            model.to(device)

        if cfg.optimizer_factory is not None:
            self._optimizers = [cfg.optimizer_factory(model) for model in self.models]
            self._optimizer_uses_factory = True
        elif (
            not hasattr(self, "_optimizers")
            or getattr(self, "_optimizer_cls", None) is not cfg.optimizer_cls
            or getattr(self, "_optimizer_kwargs_signature", None) != optimizer_signature
            or getattr(self, "_optimizer_uses_factory", False)
        ):
            self._optimizers = [
                cfg.optimizer_cls(model.parameters(), lr=cfg.lr, **optimizer_kwargs)
                for model in self.models
            ]
            self._optimizer_cls = cfg.optimizer_cls
            self._optimizer_kwargs_signature = optimizer_signature
            self._optimizer_uses_factory = False
        else:
            for opt in self._optimizers:
                _optimizer_like_set_lr(opt, cfg.lr)

        for idx, model in enumerate(self.models):
            optimizer = self._optimizers[idx]
            history = self._train_single_member(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                criterion=criterion,
                device=device,
                cfg=cfg,
                idx=idx,
            )
            member_histories.append(history)

        return {"member_histories": member_histories}

    def _train_single_member(
        self,
        model: nn.Module,
        optimizer: OptimizerLike,
        train_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: Union[str, torch.device],
        cfg: EnsembleFitConfig,
        idx: int,
    ) -> list[float]:
        """Train a single member of the ensemble for the given number of epochs."""
        history = []

        for epoch in range(cfg.epochs):
            epoch_loss = self._train_member_epoch(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                criterion=criterion,
                device=device,
            )
            history.append(epoch_loss)

            if cfg.verbose:
                print(
                    f"Member {idx + 1}/{self.ensemble_size} "
                    f"Epoch {epoch + 1}/{cfg.epochs} "
                    f"Loss {epoch_loss:.6f}"
                )

        return history

    def _train_member_epoch(
        self,
        model: nn.Module,
        optimizer: OptimizerLike,
        train_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: Union[str, torch.device],
    ) -> float:
        """Run a single training epoch for an ensemble member."""
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

            _optimizer_like_zero_grad(optimizer)
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            _optimizer_like_step(optimizer)

            running_loss += float(loss.detach().item())
            batch_count += 1

        return running_loss / max(batch_count, 1)
