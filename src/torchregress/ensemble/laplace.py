"""Full-network diagonal Laplace approximation via the empirical Fisher.

Extends last-layer Laplace (see
``torchregress.algorithms.heteroscedastic_laplace``) to ALL network
parameters: a diagonal empirical Fisher is accumulated per-parameter over a
dataloader, and the posterior is approximated as

    theta ~ N(theta_hat, diag(F + damping)^{-1})

so the posterior standard deviation of each parameter is
``sqrt(1 / (F + damping))``.
"""

from __future__ import annotations

from typing import Callable, Iterator, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader


class FullNetworkLaplace(nn.Module):
    """
    Diagonal empirical-Fisher Laplace approximation over all parameters.

    Args:
        model: The trained network.
        fisher_type: Only ``"empirical"`` is supported.
        damping: Added to every Fisher diagonal entry (Tikhonov-style prior).
        n_samples: Default number of MC posterior samples at prediction time.
    """

    def __init__(
        self,
        model: nn.Module,
        fisher_type: str = "empirical",
        damping: float = 1e-3,
        n_samples: int = 100,
    ) -> None:
        super().__init__()
        if fisher_type != "empirical":
            raise ValueError(f"fisher_type must be 'empirical', got {fisher_type!r}")
        self.model = model
        self.fisher_type = fisher_type
        self.damping = float(damping)
        self.n_samples = int(n_samples)
        self._params: dict[str, Tensor] = {
            name: p.detach() for name, p in model.named_parameters() if p.requires_grad
        }
        # Named ``_buffer_map`` (not ``_buffers``): nn.Module declares
        # ``_buffers: dict[str, Optional[Tensor]]``, and the base-class
        # declaration leaks into ty's view of the shadowing attribute.
        self._buffer_map: dict[str, Tensor] = {
            name: b.detach() for name, b in model.named_buffers() if b is not None
        }
        self.fisher_diag: Optional[dict[str, Tensor]] = None
        self.posterior_std: Optional[dict[str, Tensor]] = None
        self.is_fitted = False

    def fit(
        self,
        dataloader: DataLoader,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> "FullNetworkLaplace":
        """
        Accumulate the diagonal empirical Fisher over ``dataloader``.

        Args:
            dataloader: Yields ``(x, y)`` batches on any device.
            loss_fn: Must return a PER-SAMPLE loss tensor of shape ``[B]`` or
                ``[B, ...]`` (e.g. a ``reduction='none'`` loss); scalar losses
                cannot be decomposed into per-sample gradient squares.
        """
        from torch.func import functional_call, vmap

        device = next(iter(self._params.values())).device if self._params else torch.device("cpu")

        def per_sample_loss(
            params: dict[str, Tensor], buffers: dict[str, Tensor], x: Tensor, y: Tensor
        ) -> Tensor:
            out = functional_call(self.model, (params, buffers), (x,))
            loss = loss_fn(out, y)
            if loss.dim() == 0:
                raise ValueError(
                    "loss_fn must return per-sample losses ([B] or [B, ...]); "
                    "pass a reduction='none' loss"
                )
            # torch.func.grad requires a scalar; under vmap each call sees a
            # single sample, so the flat sum recovers that sample's loss.
            return loss.reshape(loss.shape[0], -1).sum()

        grad_fn = torch.func.grad(per_sample_loss, argnums=0)
        batched_grad_fn = vmap(grad_fn, in_dims=(None, None, 0, 0))

        fisher = {name: torch.zeros_like(p) for name, p in self._params.items()}
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            grads = batched_grad_fn(self._params, self._buffer_map, x_batch, y_batch)
            with torch.no_grad():
                for name, g in grads.items():
                    fisher[name] += g.pow(2).sum(dim=0)

        self.fisher_diag = {name: f + self.damping for name, f in fisher.items()}
        self.posterior_std = {name: torch.rsqrt(f) for name, f in self.fisher_diag.items()}
        self.is_fitted = True
        return self

    def _sample_parameters(self, n_samples: int) -> Iterator[dict[str, Tensor]]:
        assert self.posterior_std is not None
        for _ in range(n_samples):
            yield {
                name: theta + torch.randn_like(theta) * self.posterior_std[name]
                for name, theta in self._params.items()
            }

    def _check_fitted(self) -> None:
        if not self.is_fitted or self.posterior_std is None:
            raise RuntimeError("FullNetworkLaplace is not fitted; call fit() first")

    def mc_forward(self, x: Tensor, n_samples: Optional[int] = None) -> Tensor:
        """
        MC-sample parameter noise over ALL parameters and forward each draw.

        Returns:
            Stacked predictions [n_samples, batch_size, output_dim].
        """
        from torch.func import functional_call

        self._check_fitted()
        n = n_samples or self.n_samples
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                preds = [
                    functional_call(self.model, (sampled, self._buffer_map), (x,))
                    for sampled in self._sample_parameters(n)
                ]
                stacked = torch.stack(preds, dim=0)
        finally:
            self.model.train(was_training)
        assert isinstance(stacked, Tensor)
        return stacked

    def predict_with_uncertainty(
        self,
        x: Tensor,
        n_samples: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Mean and std over MC posterior samples."""
        samples = self.mc_forward(x, n_samples)
        return samples.mean(dim=0), samples.std(dim=0)

    def predict_interval(
        self,
        x: Tensor,
        confidence: float = 0.95,
        n_samples: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Empirical prediction interval from MC posterior sample quantiles."""
        samples = self.mc_forward(x, n_samples)
        alpha = 1 - confidence
        lower = torch.quantile(samples, alpha / 2, dim=0)
        upper = torch.quantile(samples, 1 - alpha / 2, dim=0)
        return lower, upper

    def forward(self, x: Tensor) -> Tensor:
        """Posterior mean prediction."""
        mean, _ = self.predict_with_uncertainty(x)
        return mean


__all__ = ["FullNetworkLaplace"]
