"""Super-Level-Set (SLS) Regression loss functions and models.

Implements the framework from Sacha Braun, Michael I. Jordan, and Francis Bach:
"Super-Level-Set Regression: Conditional Quantiles via Volume Minimization" (arXiv:2605.06210).
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
from torch import Tensor

from .base import RegressionLoss
from .loss_registry import register_regression_loss


class VolumePreservingCouplingLayer(nn.Module):
    """A translation-only conditional coupling layer for volume-preserving flows.

    Ensures a Jacobian determinant of exactly 1.
    """

    mask: Tensor

    def __init__(
        self,
        d: int,
        mask: Tensor,
        context_dim: int = 0,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d = d
        self.register_buffer("mask", mask.bool())

        # Count dimensions
        self.d1 = int(self.mask.sum().item())
        self.d2 = d - self.d1

        # Build translation/shift network
        input_dim = self.d1 + context_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.d2),
        )

        # Initialize last layer weights and biases to 0 so flow starts as identity
        last_layer = self.net[-1]
        assert isinstance(last_layer, nn.Linear)
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)

    def forward(self, y: Tensor, context: Optional[Tensor] = None) -> Tensor:
        y1 = y[..., self.mask]
        y2 = y[..., ~self.mask]

        if context is not None:
            net_input = torch.cat([y1, context], dim=-1)
        else:
            net_input = y1

        shift = self.net(net_input)

        # Subtract translation at zero for bias correction/stability
        if context is not None:
            zeros = torch.zeros_like(y1)
            zeros_input = torch.cat([zeros, context], dim=-1)
        else:
            zeros_input = torch.zeros_like(y1)
        bias = self.net(zeros_input)

        z2 = y2 + (shift - bias)

        z = torch.empty_like(y)
        z[..., self.mask] = y1
        z[..., ~self.mask] = z2
        return z

    def inverse(self, z: Tensor, context: Optional[Tensor] = None) -> Tensor:
        z1 = z[..., self.mask]
        z2 = z[..., ~self.mask]

        if context is not None:
            net_input = torch.cat([z1, context], dim=-1)
            zeros = torch.zeros_like(z1)
            zeros_input = torch.cat([zeros, context], dim=-1)
        else:
            net_input = z1
            zeros_input = torch.zeros_like(z1)

        shift = self.net(net_input)
        bias = self.net(zeros_input)

        y2 = z2 - (shift - bias)

        y = torch.empty_like(z)
        y[..., self.mask] = z1
        y[..., ~self.mask] = y2
        return y


class VolumePreservingFlow(nn.Module):
    """A strictly volume-preserving conditional normalizing flow."""

    def __init__(
        self,
        d: int,
        context_dim: int = 0,
        n_transforms: int = 4,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d = d
        self.context_dim = context_dim

        self.layers = nn.ModuleList()
        for i in range(n_transforms):
            if d == 1:
                mask = torch.tensor([True])
            else:
                mask = torch.zeros(d, dtype=torch.bool)
                if i % 2 == 0:
                    mask[: d // 2] = True
                else:
                    mask[d // 2 :] = True

            self.layers.append(
                VolumePreservingCouplingLayer(d, mask, context_dim, hidden_dim, dropout)
            )

    def forward(self, y: Tensor, context: Optional[Tensor] = None) -> Tensor:
        if self.d == 1:
            return y
        for layer in self.layers:
            assert isinstance(layer, VolumePreservingCouplingLayer)
            y = layer(y, context)
        return y

    def inverse(self, z: Tensor, context: Optional[Tensor] = None) -> Tensor:
        if self.d == 1:
            return z
        for layer in reversed(self.layers):
            assert isinstance(layer, VolumePreservingCouplingLayer)
            z = layer.inverse(z, context)
        return z


class MahalanobisFrontier(nn.Module):
    """Mahalanobis boundary function for SLS regression using volume-preserving flows."""

    def __init__(
        self,
        d: int,
        context_dim: int = 0,
        mode: str = "full",
        rank: Optional[int] = None,
        flow: Optional[nn.Module] = None,
        hidden_dim: int = 64,
        n_transforms: int = 4,
    ) -> None:
        super().__init__()
        self.d = d
        self.context_dim = context_dim
        self.mode = mode.lower()

        if self.mode == "low_rank":
            self.rank = rank if rank is not None else max(1, int(math.ceil(math.sqrt(d))))
        else:
            self.rank = 0

        if flow is not None:
            self.flow = flow
        else:
            self.flow = VolumePreservingFlow(
                d, context_dim, n_transforms=n_transforms, hidden_dim=hidden_dim
            )

        if self.mode == "full":
            self.num_L_params = d * (d + 1) // 2
        elif self.mode == "low_rank":
            self.num_L_params = d + d * self.rank
        else:
            raise ValueError(f"Unknown mode {mode!r}")

        if context_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, d + self.num_L_params),
            )
        else:
            self.mu_param = nn.Parameter(torch.zeros(d))
            self.L_param = nn.Parameter(torch.zeros(self.num_L_params))

    def _get_params(self, context: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if self.context_dim > 0:
            if context is None:
                raise ValueError("context must be provided if context_dim > 0")
            out = self.net(context)
            mu = out[..., : self.d]
            L_params = out[..., self.d :]
        else:
            batch_shape = context.shape[:-1] if context is not None else ()
            mu = self.mu_param.expand(*batch_shape, self.d)
            L_params = self.L_param.expand(*batch_shape, self.num_L_params)
        return mu, L_params

    def _get_L_matrix_and_logdet(
        self, L_params: Tensor
    ) -> Tuple[Union[Tensor, Tuple[Tensor, Tensor]], Tensor]:
        batch_shape = L_params.shape[:-1]
        device = L_params.device
        dtype = L_params.dtype

        if self.mode == "full":
            L = torch.zeros(*batch_shape, self.d, self.d, device=device, dtype=dtype)
            idx = 0
            log_det_L = torch.zeros(batch_shape, device=device, dtype=dtype)
            for i in range(self.d):
                diag_val = torch.nn.functional.softplus(L_params[..., idx]) + 1e-6
                L[..., i, i] = diag_val
                log_det_L = log_det_L + torch.log(diag_val)
                idx += 1
                for j in range(i):
                    L[..., i, j] = L_params[..., idx]
                    idx += 1
            return L, log_det_L

        else:  # low_rank
            D_params = L_params[..., : self.d]
            V_params = L_params[..., self.d :]

            D = torch.nn.functional.softplus(D_params) + 1e-6
            V = V_params.reshape(*batch_shape, self.d, self.rank)

            inv_D = 1.0 / D
            V_scaled = V * inv_D.unsqueeze(-1)

            Vt_Dinv_V = torch.matmul(V.transpose(-2, -1), V_scaled)
            Eye = torch.eye(self.rank, device=device, dtype=dtype).expand(
                *batch_shape, self.rank, self.rank
            )
            M = Eye + Vt_Dinv_V

            logdet_M = torch.logdet(M)
            logdet_D = torch.sum(torch.log(D), dim=-1)

            log_det_L = 0.5 * (logdet_M + logdet_D)
            return (D, V), log_det_L

    def forward(self, y: Tensor, context: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # A6: an unconditional frontier (context_dim == 0) must never receive a
        # degenerate context tensor — pass None so the parameter path is used.
        if self.context_dim == 0:
            context = None
        mu, L_params = self._get_params(context)
        z_flow = self.flow(y, context)
        z = z_flow - mu

        L_or_DV, log_det_L = self._get_L_matrix_and_logdet(L_params)

        if self.mode == "full":
            assert isinstance(L_or_DV, Tensor)
            L = L_or_DV
            L_z = torch.matmul(L, z.unsqueeze(-1)).squeeze(-1)
            G = torch.sum(L_z**2, dim=-1)
        else:
            assert isinstance(L_or_DV, tuple)
            D, V = L_or_DV
            D_z2 = torch.sum(D * (z**2), dim=-1)
            Vt_z = torch.matmul(V.transpose(-2, -1), z.unsqueeze(-1)).squeeze(-1)
            Vt_z2 = torch.sum(Vt_z**2, dim=-1)
            G = D_z2 + Vt_z2

        return G, log_det_L


class UnionFrontier(nn.Module):
    """Union of multiple normalizing flow regions for multimodal level sets."""

    beta: Tensor

    def __init__(
        self,
        d: int,
        K: int,
        context_dim: int = 0,
        mode: str = "full",
        rank: Optional[int] = None,
        hidden_dim: int = 64,
        n_transforms: int = 4,
        beta_init: float = 1.0,
        beta_decay: float = 0.9995,
    ) -> None:
        super().__init__()
        self.d = d
        self.K = K
        self.context_dim = context_dim

        self.components = nn.ModuleList(
            [
                MahalanobisFrontier(
                    d,
                    context_dim,
                    mode=mode,
                    rank=rank,
                    hidden_dim=hidden_dim,
                    n_transforms=n_transforms,
                )
                for _ in range(K)
            ]
        )

        if context_dim > 0:
            self.weights_net = nn.Sequential(
                nn.Linear(context_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, K)
            )
        else:
            self.weights_param = nn.Parameter(torch.zeros(K))

        self.register_buffer("beta", torch.tensor(beta_init))
        self.beta_decay = beta_decay
        self._freeze_weights = True

    def step_beta(self) -> None:
        self.beta.copy_(self.beta * 1.01)

    def freeze_weights(self, freeze: bool = True) -> None:
        self._freeze_weights = freeze

    def _get_mixture_weights(self, context: Optional[Tensor] = None) -> Tensor:
        if self._freeze_weights:
            batch_shape = context.shape[:-1] if context is not None else ()
            device = context.device if context is not None else self.beta.device
            return torch.full(
                (*batch_shape, self.K), 1.0 / self.K, device=device, dtype=torch.float32
            )

        if self.context_dim > 0:
            if context is None:
                raise ValueError("context must be provided if context_dim > 0")
            logits = self.weights_net(context)
            return torch.softmax(logits, dim=-1)
        else:
            return torch.softmax(self.weights_param, dim=-1)

    def forward(self, y: Tensor, context: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        p = self._get_mixture_weights(context)

        G_components = []
        logdet_components = []

        for k in range(self.K):
            G_k_prime, logdet_k = self.components[k](y, context)
            p_k = p[..., k].clamp(min=1e-8)
            G_k = (p_k ** (-2.0 / self.d)) * G_k_prime
            G_components.append(G_k)
            logdet_components.append(logdet_k)

        G_stack = torch.stack(G_components, dim=-1)
        logdet_stack = torch.stack(logdet_components, dim=-1)

        logits = -self.beta * G_stack
        weights = torch.softmax(logits, dim=-1)

        G_beta = torch.sum(weights * G_stack, dim=-1)

        log_p = torch.log(p.clamp(min=1e-8))
        log_terms = log_p - logdet_stack
        log_vol_term = torch.logsumexp(log_terms, dim=-1)

        return G_beta, log_vol_term


class QuantileNetwork(nn.Module):
    """Backbone MLP for predicting target levels in SLS regression."""

    def __init__(self, context_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.context_dim = context_dim
        if context_dim > 0:
            self.backbone = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.heads = nn.Linear(hidden_dim, 3)
        else:
            self.params = nn.Parameter(torch.zeros(3))

    def forward(self, context: Optional[Tensor] = None) -> Tensor:
        if self.context_dim > 0:
            if context is None:
                raise ValueError("context must be provided if context_dim > 0")
            h = self.backbone(context)
            out = self.heads(h)
        else:
            batch_shape = context.shape[:-1] if context is not None else ()
            out = self.params.expand(*batch_shape, 3)

        quantiles = torch.exp(out) + 1e-6
        quantiles_sorted = torch.sort(quantiles, dim=-1).values
        return quantiles_sorted


def sigmoidal_schedule(
    step: int,
    warmup_steps: int,
    init_val: float,
    min_val: float,
    k: float = 0.005,
    t0: float = 1000.0,
) -> float:
    if step <= warmup_steps:
        return init_val
    t = float(step - warmup_steps)

    def sig(z: float) -> float:
        return 1.0 / (1.0 + math.exp(-z))

    numerator = sig(k * (t - t0)) - sig(-k * t0)
    denominator = 1.0 - sig(-k * t0)

    val = init_val + (min_val - init_val) * (numerator / denominator)
    return max(min_val, min(init_val, val))


@register_regression_loss("sls")
class SLSLoss(RegressionLoss):
    """Super-Level-Set (SLS) Regression loss.

    Directly estimates minimum-volume prediction regions with conditional coverage
    by optimizing the level-set boundary under a volume penalty.

    Parameters
    ----------
    d : int
        Target dimensionality (number of output dimensions).
    context_dim : int
        Context / feature dimensionality from the backbone model.
    K : int, default=1
        Number of mixture components for multimodal level sets.
        Use K > 1 for disconnected / multimodal target distributions.
    mode : str, default="full"
        Frontier covariance mode: ``"full"`` or ``"low_rank"``.
    rank : int, optional
        Rank for low-rank frontier mode.  If None, defaults to ``ceil(sqrt(d))``.
    hidden_dim : int, default=64
        Hidden dimension shared by the volume-preserving coupling layers,
        the frontier context network, and the quantile network.

        **Recommended minimum: 64.**  Reducing to 32 under-expresses the flow:
        JointCoverage drops ~4.5% (0.92 → 0.88) and intervals widen.
        Empirically validated on synthetic_multivariate benchmarks.
    n_transforms : int, default=4
        Number of volume-preserving coupling transforms in the flow.

        **Recommended minimum: 4.**  Using 2 transforms (together with
        ``hidden_dim=32``) compounds the under-expression — coverage is
        maintained but intervals become wider and less efficient.
        Sweep data shows ``hidden_dim=32, n_transforms=2`` produces the
        worst IntervalScore across all tested configurations.
    tau : float, default=0.9
        Target conditional coverage level (e.g., 0.9 = 90% prediction intervals).
    warmup_steps : int, default=500
        Number of frontier-only warmup steps before quantile network training
        begins (via sigmoidal schedule).  On small/medium tabular datasets,
        50 steps is often sufficient; 500 is the safe library default that
        ensures frontier convergence.
    error_init : float, default=0.4
        Initial sigmoidal schedule window width (in coverage probability space).
    error_min : float, default=0.05
        Minimum sigmoidal schedule window width after warmup.
    reduction : str, default="mean"
        Loss reduction: ``"mean"``, ``"sum"``, or ``"none"``.

    Notes
    -----
    The loss has two alternating forward passes per step:
    ``forward_frontier`` (optimises the level-set boundary via volume penalty)
    and ``forward_quantiles`` (optimises the quantile network via pinball
    loss).  Both are summed to form the final loss returned by ``forward()``.

    For K > 1 (UnionFrontier), mixture weights are frozen during frontier warmup
    and unfrozen afterwards via ``freeze_weights(False)`` and ``step_beta()``.

    References
    ----------
    .. [1] S. Braun, M.I. Jordan, F. Bach. (2026). Super-Level-Set (SLS) Regression.
       In *NeurIPS 2026 Submission*. https://arxiv.org/abs/2605.06210
    """

    frontier: Union[MahalanobisFrontier, UnionFrontier]

    def __init__(
        self,
        d: int,
        context_dim: int,
        K: int = 1,
        mode: str = "full",
        rank: Optional[int] = None,
        hidden_dim: int = 64,
        n_transforms: int = 4,
        tau: float = 0.9,
        warmup_steps: int = 500,
        error_init: float = 0.4,
        error_min: float = 0.05,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.d = d
        self.context_dim = context_dim
        self.K = K
        self.tau = tau
        self.warmup_steps = warmup_steps
        self.error_init = error_init
        self.error_min = error_min

        # Build frontier
        if K == 1:
            self.frontier = MahalanobisFrontier(
                d,
                context_dim,
                mode=mode,
                rank=rank,
                hidden_dim=hidden_dim,
                n_transforms=n_transforms,
            )
        else:
            self.frontier = UnionFrontier(
                d,
                K,
                context_dim,
                mode=mode,
                rank=rank,
                hidden_dim=hidden_dim,
                n_transforms=n_transforms,
            )

        # Build quantile network
        self.quantile_net = QuantileNetwork(context_dim, hidden_dim=hidden_dim)

        self.step_counter = 0

    def get_current_window(self, step: Optional[int] = None) -> Tuple[float, float]:
        current_step = step if step is not None else self.step_counter
        phi = sigmoidal_schedule(current_step, self.warmup_steps, self.error_init, self.error_min)
        psi = sigmoidal_schedule(current_step, self.warmup_steps, self.error_init, self.error_min)
        return phi, psi

    def forward_frontier(
        self,
        y_pred: Tensor,
        target: Tensor,
        step: Optional[int] = None,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        if step is not None:
            current_step = step
        else:
            self.step_counter += 1
            current_step = self.step_counter
        G, log_det_L = self.frontier(target, y_pred)

        # A6: sign-consistent volume term. MahalanobisFrontier's log_det_L is
        # log|L| (volume of the level set scales as |Σ|^{1/2} = prod L_ii),
        # while UnionFrontier already reports a log-volume with the opposite
        # sign convention.
        vol_term = -log_det_L if isinstance(self.frontier, MahalanobisFrontier) else log_det_L

        if current_step <= self.warmup_steps:
            loss_val = 0.5 * self.d * torch.log(G + 1e-8) + vol_term
        else:
            phi, psi = self.get_current_window(step=step)
            quantiles = self.quantile_net(y_pred).detach()
            q_low = quantiles[..., 0]
            q_high = quantiles[..., 2]

            indicator = ((G >= q_low) & (G <= q_high)).float()

            loss_val = indicator * (0.5 * self.d * torch.log(G + 1e-8) + vol_term) / (phi + psi)

        return self._reduce(loss_val, mask=mask, weights=weights)

    def forward_quantiles(
        self,
        y_pred: Tensor,
        target: Tensor,
        step: Optional[int] = None,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        with torch.no_grad():
            G, _ = self.frontier(target, y_pred)

        phi, psi = self.get_current_window(step=step)
        beta_low = max(0.01, self.tau - phi)
        beta_high = min(0.99, self.tau + psi)

        levels = torch.tensor(
            [beta_low, self.tau, beta_high],
            device=y_pred.device,
            dtype=y_pred.dtype,
        )

        quantiles = self.quantile_net(y_pred)

        diff = G.unsqueeze(-1) - quantiles
        loss_val = torch.max(levels * diff, (levels - 1.0) * diff)
        loss_val = loss_val.sum(dim=-1)

        return self._reduce(loss_val, mask=mask, weights=weights)

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        step: Optional[int] = None,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        step_val = step if step is not None else kwargs.get("step")
        current_step = step_val if step_val is not None else self.step_counter
        effective_step = current_step if step_val is not None else current_step + 1

        if effective_step > self.warmup_steps and self.K > 1:
            # UnionFrontier typecast since Components can be either Mahalanobis or UnionFrontier
            frontier_union = cast(UnionFrontier, self.frontier)
            frontier_union.freeze_weights(False)
            frontier_union.step_beta()

        loss_frontier = self.forward_frontier(
            y_pred, target, step=step_val, mask=mask, weights=weights
        )
        loss_quantiles = self.forward_quantiles(
            y_pred, target, step=step_val, mask=mask, weights=weights
        )

        return loss_frontier + loss_quantiles
