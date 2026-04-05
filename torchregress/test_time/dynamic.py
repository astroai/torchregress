"""Dynamic ensemble helpers for online/test-time adaptation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ParameterEMA:
    """Exponential moving average over model parameters."""

    decay: float = 0.99

    def __post_init__(self) -> None:
        self.shadow: dict[str, torch.Tensor] = {}

    def initialize(self, model: torch.nn.Module) -> None:
        self.shadow = {name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad}

    def update(self, model: torch.nn.Module) -> None:
        if not self.shadow:
            self.initialize(model)
            return
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: torch.nn.Module) -> None:
        if not self.shadow:
            raise RuntimeError("EMA state is empty; call initialize() or update() first")
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])


__all__ = ["ParameterEMA"]
