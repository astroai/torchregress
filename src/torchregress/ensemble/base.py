"""
Base classes for ensemble models.

This module provides foundation classes and abstractions for all ensemble techniques
in the torchregress library.
"""

from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any, Dict, Union

import torch
import torch.nn as nn

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

    def _setup_optimizers(
        self,
        lr: float,
        optimizer_cls: type,
        optimizer_kwargs: dict[str, Any],
        optimizer_factory: Callable[[nn.Module], OptimizerLike] | None,
    ) -> None:
        """Set up optimizers for all ensemble members."""
        optimizer_signature = tuple(
            sorted((str(key), repr(value)) for key, value in optimizer_kwargs.items())
        )

        if optimizer_factory is not None:
            self._optimizers = [optimizer_factory(model) for model in self.models]
            self._optimizer_uses_factory = True
        elif (
            not hasattr(self, "_optimizers")
            or getattr(self, "_optimizer_cls", None) is not optimizer_cls
            or getattr(self, "_optimizer_kwargs_signature", None) != optimizer_signature
            or getattr(self, "_optimizer_uses_factory", False)
        ):
            self._optimizers = [
                optimizer_cls(model.parameters(), lr=lr, **optimizer_kwargs)
                for model in self.models
            ]
            self._optimizer_cls = optimizer_cls
            self._optimizer_kwargs_signature = optimizer_signature
            self._optimizer_uses_factory = False
        else:
            for opt in self._optimizers:
                _optimizer_like_set_lr(opt, lr)

    def _train_epoch(
        self,
        model: nn.Module,
        optimizer: OptimizerLike,
        train_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: Union[str, torch.device],
        augmenter: Any | None,
        adversarial_loss_weight: float,
        batch_regularizer: Callable[[torch.Tensor, Sequence[torch.Tensor]], torch.Tensor] | None,
    ) -> tuple[float, float, float]:
        """Train a single member for one epoch."""
        model.train()
        running_loss = 0.0
        running_clean_loss = 0.0
        running_adversarial_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
                extra_tensors: tuple[torch.Tensor, ...] = tuple(t.to(device) for t in batch[2:])
            else:
                raise ValueError("train_loader must yield (inputs, targets) tuples")

            x = x.to(device)
            y = y.to(device)

            _optimizer_like_zero_grad(optimizer)
            preds = model(x)
            clean_loss = criterion(preds, y)
            loss = clean_loss
            if batch_regularizer is not None:
                loss = loss + batch_regularizer(preds, extra_tensors)

            if augmenter is not None:
                x_adv, _ = augmenter(x, y)
                preds_adv = model(x_adv)
                adversarial_loss = criterion(preds_adv, y)
                loss = clean_loss + (adversarial_loss_weight * adversarial_loss)
                running_adversarial_loss += float(adversarial_loss.detach().item())

            loss.backward()
            _optimizer_like_step(optimizer)

            running_loss += float(loss.detach().item())
            running_clean_loss += float(clean_loss.detach().item())
            batch_count += 1

        epoch_loss = running_loss / max(batch_count, 1)
        epoch_clean_loss = running_clean_loss / max(batch_count, 1)
        epoch_adversarial_loss = 0.0
        if augmenter is not None:
            epoch_adversarial_loss = running_adversarial_loss / max(batch_count, 1)

        return epoch_loss, epoch_clean_loss, epoch_adversarial_loss

    def _create_augmenter(
        self,
        model: nn.Module,
        criterion: nn.Module,
        adversarial_training: bool,
        adversarial_epsilon: float,
        adversarial_steps: int,
        adversarial_alpha: float | None,
        adversarial_probability: float,
        adversarial_loss_weight: float,
        adversarial_random_start: bool,
    ) -> Any | None:
        """Create an Adversarial augmenter if adversarial training is enabled."""
        if not (adversarial_training and adversarial_loss_weight > 0):
            return None

        from torchregress.utils.augment import Adversarial

        return Adversarial(
            model=model,
            loss_fn=criterion,
            epsilon=adversarial_epsilon,
            steps=adversarial_steps,
            alpha=adversarial_alpha,
            probability=adversarial_probability,
            random_start=adversarial_random_start,
        )

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        epochs: int = 10,
        lr: float = 1e-3,
        optimizer_cls: type = torch.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        optimizer_factory: Callable[[nn.Module], OptimizerLike] | None = None,
        verbose: bool = True,
        device: Union[str, torch.device, None] = None,
        adversarial_training: bool = False,
        adversarial_epsilon: float = 0.0,
        adversarial_steps: int = 1,
        adversarial_alpha: float | None = None,
        adversarial_probability: float = 1.0,
        adversarial_loss_weight: float = 1.0,
        adversarial_random_start: bool = False,
        batch_regularizer: (
            Callable[[torch.Tensor, Sequence[torch.Tensor]], torch.Tensor] | None
        ) = None,
    ) -> Dict[str, list]:
        """
        Train each ensemble member independently.

        When ``adversarial_training`` is enabled, each member is trained on the
        standard objective plus an optional FGSM/PGD-style adversarial term using
        the same criterion. This matches the original deep-ensemble recipe while
        keeping the ensemble members otherwise independent.

        When ``optimizer_factory`` is set, it is called as ``factory(model)`` for
        each member and must return a single ``torch.optim.Optimizer`` or a tuple
        of optimizers (e.g. AdamW + Muon). In that case ``optimizer_cls`` and
        ``optimizer_kwargs`` are ignored.

        Optional ``batch_regularizer`` adds a per-batch term after ``criterion``
        outputs (e.g. population-level stacked n(z) penalties). It is called as
        ``batch_regularizer(member_logits, extra_tensors)`` where ``extra_tensors``
        is ``batch[2:]`` from each loader batch (empty tuple if only inputs and
        targets are present). The regularizer should return a scalar tensor.
        It is applied to the clean forward pass only (not recomputed on adversarial
        inputs).
        """

        device = device or self.device
        member_histories = []
        member_clean_histories = []
        member_adversarial_histories = []
        optimizer_kwargs = dict(optimizer_kwargs or {})

        if adversarial_loss_weight < 0:
            raise ValueError(
                f"adversarial_loss_weight must be non-negative, got {adversarial_loss_weight}"
            )

        for model in self.models:
            model.to(device)

        self._setup_optimizers(lr, optimizer_cls, optimizer_kwargs, optimizer_factory)

        for idx, model in enumerate(self.models):
            optimizer = self._optimizers[idx]
            history = []
            clean_history = []
            adversarial_history = []

            augmenter = self._create_augmenter(
                model,
                criterion,
                adversarial_training,
                adversarial_epsilon,
                adversarial_steps,
                adversarial_alpha,
                adversarial_probability,
                adversarial_loss_weight,
                adversarial_random_start,
            )

            for epoch in range(epochs):
                epoch_loss, epoch_clean_loss, epoch_adv_loss = self._train_epoch(
                    model,
                    optimizer,
                    train_loader,
                    criterion,
                    device,  # type: ignore[arg-type]
                    augmenter,
                    adversarial_loss_weight,
                    batch_regularizer,
                )

                history.append(epoch_loss)
                clean_history.append(epoch_clean_loss)
                if augmenter is not None:
                    adversarial_history.append(epoch_adv_loss)

                if verbose:
                    message = (
                        f"Member {idx + 1}/{self.ensemble_size} "
                        f"Epoch {epoch + 1}/{epochs} "
                        f"Loss {epoch_loss:.6f}"
                    )
                    if augmenter is not None:
                        message += f" Clean {epoch_clean_loss:.6f} Adv {epoch_adv_loss:.6f}"
                    print(message)

            member_histories.append(history)
            member_clean_histories.append(clean_history)
            if augmenter is not None:
                member_adversarial_histories.append(adversarial_history)

        result: Dict[str, list] = {"member_histories": member_histories}
        if adversarial_training and adversarial_loss_weight > 0:
            result["member_clean_histories"] = member_clean_histories
            result["member_adversarial_histories"] = member_adversarial_histories
        return result
