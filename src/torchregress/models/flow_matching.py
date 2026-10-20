"""Lean conditional flow matching (CFM) for tabular regression.

This is a simulation-free continuous-normalizing-flow objective: regress the
conditional vector field along an optimal-transport interpolation between
Gaussian noise and targets.  Unlike an exact normalizing flow it does not give
a closed-form likelihood, but its sampler is a short Euler ODE and it handles
multimodal and multi-target conditional densities without Jacobians.

Reference: Lipman et al. (2022), "Flow Matching for Generative Modeling";
Tong et al. (2023), "Improving and generalizing flow-based generative models
with minibatch optimal transport".
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class FlowMatchingMLP(nn.Module):
    """Time-conditioned MLP velocity field ``v(x, t, context)``."""

    time_frequencies: Tensor

    def __init__(
        self,
        n_features: int,
        context_dim: int,
        *,
        hidden: tuple[int, ...] = (128, 128),
        time_embedding: int = 16,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.context_dim = int(context_dim)
        self.time_embedding = int(time_embedding)
        widths = (n_features + context_dim + time_embedding, *hidden, n_features)
        layers: list[nn.Module] = []
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i + 1]))
            if i < len(widths) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)
        self.register_buffer("time_frequencies", torch.arange(1, time_embedding // 2 + 1).float())

    def _time_features(self, t: Tensor) -> Tensor:
        arg = t.reshape(-1, 1) * self.time_frequencies.to(t.device, t.dtype) * torch.pi
        return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)

    def forward(self, x: Tensor, t: Tensor, context: Tensor) -> Tensor:
        if x.dim() != 2 or context.dim() != 2:
            raise ValueError("x and context must be 2-D")
        if x.shape[0] != context.shape[0] or t.numel() != x.shape[0]:
            raise ValueError("x, t, and context must share the batch dimension")
        return self.net(torch.cat([x, self._time_features(t.to(x.dtype)), context], dim=-1))


def cfm_loss(
    velocity: nn.Module,
    context: Tensor,
    target: Tensor,
    *,
    generator: Optional[torch.Generator] = None,
    eps: float = 1e-3,
) -> Tensor:
    """OT-CFM regression loss, averaged over the batch.

    At time ``t ~ U(eps, 1-eps)`` interpolate ``x_t=(1-t)x0+t x1`` and regress
    the velocity ``x1-x0``.
    """
    if target.dim() == 1:
        target = target.unsqueeze(-1)
    if context.dim() != 2 or target.dim() != 2 or target.shape[0] != context.shape[0]:
        raise ValueError("context and target must be [B, C] and [B, D]")
    n, d = target.shape
    x0 = torch.randn(target.shape, dtype=target.dtype, device=target.device, generator=generator)
    t = eps + (1.0 - 2.0 * eps) * torch.rand(
        n, dtype=target.dtype, device=target.device, generator=generator
    )
    xt = (1.0 - t[:, None]) * x0 + t[:, None] * target
    v_target = target - x0
    v_pred = velocity(xt, t, context)
    return torch.mean((v_pred - v_target) ** 2)


@torch.no_grad()
def sample_cfm(
    velocity: nn.Module,
    context: Tensor,
    *,
    n_samples: int = 128,
    steps: int = 32,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Euler-integrate the learned conditional flow.

    Returns ``[B, n_samples, D]`` samples from the conditional density.
    """
    if context.dim() != 2:
        raise ValueError("context must be [B, C]")
    b, c = context.shape
    d = int(getattr(velocity, "n_features"))
    x = torch.randn(
        b * n_samples, d, dtype=context.dtype, device=context.device, generator=generator
    )
    ctx = context.repeat_interleave(n_samples, dim=0)
    dt = 1.0 / float(steps)
    for step in range(int(steps)):
        t = torch.full((x.shape[0],), step * dt, dtype=x.dtype, device=x.device)
        x = x + dt * velocity(x, t, ctx)
    return x.reshape(b, n_samples, d)
