"""Target-space transforms for regression workflows."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _require_positive_support(x: Tensor, *, eps: float, name: str) -> Tensor:
    min_allowed = -float(eps)
    if torch.any(x < min_allowed):
        raise ValueError(f"{name} requires inputs >= {-eps:g}; received values below support")
    return x


class TargetTransform:
    """Bidirectional tensor transform used by transformed-target losses."""

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def inverse(self, y: Tensor) -> Tensor:
        raise NotImplementedError


@dataclass(frozen=True)
class IdentityTransform(TargetTransform):
    """No-op transform."""

    def forward(self, x: Tensor) -> Tensor:
        return x

    def inverse(self, y: Tensor) -> Tensor:
        return y


@dataclass(frozen=True)
class LogTransform(TargetTransform):
    """Positive-support log transform."""

    eps: float = 1e-6

    def forward(self, x: Tensor) -> Tensor:
        _require_positive_support(x, eps=self.eps, name="LogTransform")
        return torch.log(x + self.eps)

    def inverse(self, y: Tensor) -> Tensor:
        return torch.clamp(torch.exp(y) - self.eps, min=0.0)


@dataclass(frozen=True)
class BoxCoxTransform(TargetTransform):
    """Positive-support Box-Cox transform."""

    lam: float = 0.0
    eps: float = 1e-6

    def forward(self, x: Tensor) -> Tensor:
        _require_positive_support(x, eps=self.eps, name="BoxCoxTransform")
        shifted = x + self.eps
        if abs(self.lam) < 1e-8:
            return torch.log(shifted)
        return (torch.pow(shifted, self.lam) - 1.0) / self.lam

    def inverse(self, y: Tensor) -> Tensor:
        if abs(self.lam) < 1e-8:
            return torch.clamp(torch.exp(y) - self.eps, min=0.0)
        base = torch.clamp(y * self.lam + 1.0, min=0.0)
        return torch.clamp(torch.pow(base, 1.0 / self.lam) - self.eps, min=0.0)


@dataclass(frozen=True)
class SqrtTransform(TargetTransform):
    """Positive-support square-root transform."""

    eps: float = 1e-6

    def forward(self, x: Tensor) -> Tensor:
        _require_positive_support(x, eps=self.eps, name="SqrtTransform")
        return torch.sqrt(x + self.eps)

    def inverse(self, y: Tensor) -> Tensor:
        return torch.clamp(y.square() - self.eps, min=0.0)


@dataclass(frozen=True)
class YeoJohnsonTransform(TargetTransform):
    """Signed-target power transform."""

    lam: float = 1.0

    def forward(self, x: Tensor) -> Tensor:
        pos = x >= 0
        out = torch.empty_like(x)
        if abs(self.lam) < 1e-8:
            out[pos] = torch.log1p(x[pos])
        else:
            out[pos] = ((x[pos] + 1.0).pow(self.lam) - 1.0) / self.lam

        neg_lam = 2.0 - self.lam
        if abs(neg_lam) < 1e-8:
            out[~pos] = -torch.log1p(-x[~pos])
        else:
            out[~pos] = -(((1.0 - x[~pos]).pow(neg_lam) - 1.0) / neg_lam)
        return out

    def inverse(self, y: Tensor) -> Tensor:
        pos = y >= 0
        out = torch.empty_like(y)
        if abs(self.lam) < 1e-8:
            out[pos] = torch.expm1(y[pos])
        else:
            out[pos] = torch.clamp(self.lam * y[pos] + 1.0, min=0.0).pow(1.0 / self.lam) - 1.0

        neg_lam = 2.0 - self.lam
        if abs(neg_lam) < 1e-8:
            out[~pos] = 1.0 - torch.exp(-y[~pos])
        else:
            out[~pos] = 1.0 - torch.clamp(1.0 - neg_lam * y[~pos], min=0.0).pow(1.0 / neg_lam)
        return out


def make_target_transform(name: str, **kwargs: float) -> TargetTransform:
    """Create a named target transform."""
    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"identity", "none"}:
        return IdentityTransform()
    if normalized == "log":
        return LogTransform(eps=float(kwargs.get("eps", 1e-6)))
    if normalized in {"boxcox", "box_cox"}:
        return BoxCoxTransform(
            lam=float(kwargs.get("lam", 0.0)),
            eps=float(kwargs.get("eps", 1e-6)),
        )
    if normalized in {"sqrt", "square_root"}:
        return SqrtTransform(eps=float(kwargs.get("eps", 1e-6)))
    if normalized in {"yeojohnson", "yeo_johnson"}:
        return YeoJohnsonTransform(lam=float(kwargs.get("lam", 1.0)))
    raise ValueError(f"Unknown target transform {name!r}")


__all__ = [
    "TargetTransform",
    "IdentityTransform",
    "LogTransform",
    "BoxCoxTransform",
    "SqrtTransform",
    "YeoJohnsonTransform",
    "make_target_transform",
]
