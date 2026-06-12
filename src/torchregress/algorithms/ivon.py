"""
Improved Variational Online Newton (IVON) optimizer.

Derived from the Bayesian Learning Rule framework (Khan & Rue, 2023).
"""

from __future__ import annotations

from contextlib import contextmanager
from math import pow
from typing import Any, Callable, Generator, Tuple

import torch
import torch.distributed as dist
import torch.optim
from torch import Tensor

ClosureType = Callable[[], Tensor]


def _welford_mean(avg: Tensor | None, newval: Tensor, count: int) -> Tensor:
    return newval if avg is None else avg + (newval - avg) / count


class IVON(torch.optim.Optimizer):
    """
    Improved Variational Online Newton (IVON) optimizer.

    IVON is a natural-gradient-based optimization algorithm derived from the
    Bayesian Learning Rule framework. It fits a Gaussian variational posterior
    q(θ) = N(θ | μ, Σ) over model parameters by tracking first moments (momentum)
    and second moments (Hessian/precision estimates).

    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize or dicts defining parameter groups.
    lr : float
        Learning rate.
    ess : float
        Effective sample size (often set to the training dataset size).
    hess_init : float, default=1.0
        Initial value for the Hessian/precision diagonal.
    beta1 : float, default=0.9
        Exponential decay rate for the first moment (momentum) updates.
    beta2 : float, default=0.99999
        Exponential decay rate for the second moment (Hessian) updates.
    weight_decay : float, default=1e-4
        L2 regularization / weight decay coefficient.
    mc_samples : int, default=1
        Number of Monte Carlo samples to draw for gradient estimation in `step()`.
    hess_approx : str, default='price'
        Hessian approximation method: 'price' (using Price's theorem/Stein's lemma)
        or 'gradsq' (using the squared gradients, similar to RMSprop/Adam).
    clip_radius : float, default=inf
        Maximum absolute value for gradient/momentum clipping.
    sync : bool, default=False
        Whether to synchronize sample statistics across distributed workers.
    debias : bool, default=True
        Whether to apply bias correction to the momentum updates.
    rescale_lr : bool, default=True
        Whether to rescale the learning rate by (hess_init + weight_decay).

    Examples
    --------
    >>> import torch
    >>> import torch.nn as nn
    >>> from torchregress.algorithms.ivon import IVON
    >>>
    >>> model = nn.Linear(10, 1)
    >>> optimizer = IVON(model.parameters(), lr=0.1, ess=100.0)
    >>>
    >>> x = torch.randn(16, 10)
    >>> y = torch.randn(16, 1)
    >>>
    >>> for _ in range(5):
    ...     with optimizer.sampled_params(train=True):
    ...         optimizer.zero_grad()
    ...         out = model(x)
    ...         loss = (out - y).pow(2).mean()
    ...         loss.backward()
    ...     optimizer.step()

    References
    ----------
    .. [1] Khan, M. E., & Rue, H. (2023). The Bayesian Learning Rule.
       In *Journal of Machine Learning Research*, 24(214), 1-45.
       https://arxiv.org/abs/2107.04562
    .. [2] Shen, Y., et al. (2024). Variational Learning is Effective for Large Deep Networks.
       In *ICML 2024*. https://arxiv.org/abs/2402.17641
    """

    hessian_approx_methods = (
        "price",
        "gradsq",
    )

    def __init__(
        self,
        params: Any,
        lr: float,
        ess: float,
        hess_init: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.99999,
        weight_decay: float = 1e-4,
        mc_samples: int = 1,
        hess_approx: str = "price",
        clip_radius: float = float("inf"),
        sync: bool = False,
        debias: bool = True,
        rescale_lr: bool = True,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 1 <= mc_samples:
            raise ValueError(f"Invalid number of MC samples: {mc_samples}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        if not 0.0 < hess_init:
            raise ValueError(f"Invalid Hessian initialization: {hess_init}")
        if not 0.0 < ess:
            raise ValueError(f"Invalid effective sample size: {ess}")
        if not 0.0 < clip_radius:
            raise ValueError(f"Invalid clipping radius: {clip_radius}")
        if not 0.0 <= beta1 <= 1.0:
            raise ValueError(f"Invalid beta1 parameter: {beta1}")
        if not 0.0 <= beta2 <= 1.0:
            raise ValueError(f"Invalid beta2 parameter: {beta2}")
        if hess_approx not in self.hessian_approx_methods:
            raise ValueError(f"Invalid hess_approx parameter: {hess_approx}")

        defaults = dict(
            lr=lr,
            mc_samples=mc_samples,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            hess_init=hess_init,
            ess=ess,
            clip_radius=clip_radius,
        )
        super().__init__(params, defaults)

        self.mc_samples = mc_samples
        self.hess_approx = hess_approx
        self.sync = sync
        self._numel, self._device, self._dtype = self._get_param_configs()
        self.current_step = 0
        self.debias = debias
        self.rescale_lr = rescale_lr

        self._reset_samples()
        self._init_buffers()

    @property
    def opt_state(self) -> dict[str, Any]:
        return self.state  # type: ignore[return-value]

    def _get_param_configs(self) -> Tuple[int, torch.device, torch.dtype]:
        all_params = []
        for pg in self.param_groups:
            pg["numel"] = sum(p.numel() for p in pg["params"] if p is not None)
            all_params += [p for p in pg["params"] if p is not None]
        if len(all_params) == 0:
            return 0, torch.device("cpu"), torch.get_default_dtype()
        devices = {p.device for p in all_params}
        if len(devices) > 1:
            raise ValueError(f"Parameters are on different devices: {[str(d) for d in devices]}")
        device = next(iter(devices))
        dtypes = {p.dtype for p in all_params}
        if len(dtypes) > 1:
            raise ValueError(f"Parameters are on different dtypes: {[str(d) for d in dtypes]}")
        dtype = next(iter(dtypes))
        total = sum(pg["numel"] for pg in self.param_groups)
        return total, device, dtype

    def _reset_samples(self) -> None:
        self.opt_state["count"] = 0
        self.opt_state["avg_grad"] = None
        self.opt_state["avg_nxg"] = None
        self.opt_state["avg_gsq"] = None

    def _init_buffers(self) -> None:
        for group in self.param_groups:
            hess_init, numel = group["hess_init"], group["numel"]

            group["momentum"] = torch.zeros(numel, device=self._device, dtype=self._dtype)
            group["hess"] = torch.zeros(numel, device=self._device, dtype=self._dtype).add(
                torch.as_tensor(hess_init)
            )

    @contextmanager
    def sampled_params(self, train: bool = False) -> Generator[None, None, None]:
        param_avg, noise = self._sample_params()
        try:
            yield
        finally:
            self._restore_param_average(train, param_avg, noise)

    def _restore_param_average(self, train: bool, param_avg: Tensor, noise: Tensor) -> None:
        param_grads = []
        offset = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p is None:
                    continue

                p_slice = slice(offset, offset + p.numel())

                p.data = param_avg[p_slice].view(p.shape)
                if train:
                    if p.grad is not None:
                        param_grads.append(p.grad.flatten())
                    else:
                        param_grads.append(torch.zeros_like(p).flatten())
                offset += p.numel()
        assert offset == self._numel

        if train:
            grad_sample = torch.cat(param_grads, 0)
            count = self.opt_state["count"] + 1
            self.opt_state["count"] = count
            self.opt_state["avg_grad"] = _welford_mean(
                self.opt_state["avg_grad"], grad_sample, count
            )
            if self.hess_approx == "price":
                self.opt_state["avg_nxg"] = _welford_mean(
                    self.opt_state["avg_nxg"], noise * grad_sample, count
                )
            elif self.hess_approx == "gradsq":
                self.opt_state["avg_gsq"] = _welford_mean(
                    self.opt_state["avg_gsq"], grad_sample.square(), count
                )

    @torch.no_grad()
    def step(self, closure: ClosureType | None = None) -> Tensor | None:  # type: ignore[override]
        loss = None
        if closure is not None:
            losses = []
            for _ in range(self.mc_samples):
                with torch.enable_grad():
                    loss_val = closure()
                losses.append(loss_val)
            loss = torch.stack(losses).mean()

        if self.sync and dist.is_initialized():
            self._sync_samples()
        self._update()
        self._reset_samples()
        return loss

    def _sync_samples(self) -> None:
        world_size = dist.get_world_size()
        if self.opt_state["avg_grad"] is not None:
            dist.all_reduce(self.opt_state["avg_grad"])
            self.opt_state["avg_grad"].div_(world_size)
        if self.opt_state["avg_nxg"] is not None:
            dist.all_reduce(self.opt_state["avg_nxg"])
            self.opt_state["avg_nxg"].div_(world_size)

    def _sample_params(self) -> Tuple[Tensor, Tensor]:
        noise_samples = []
        param_avgs = []

        offset = 0
        for group in self.param_groups:
            gnumel = group["numel"]
            noise_sample = (
                torch.randn(gnumel, device=self._device, dtype=self._dtype)
                / (group["ess"] * (group["hess"] + group["weight_decay"])).sqrt()
            )
            noise_samples.append(noise_sample)

            goffset = 0
            for p in group["params"]:
                if p is None:
                    continue

                p_avg = p.data.flatten()
                numel = p.numel()
                p_noise = noise_sample[goffset : goffset + numel]

                param_avgs.append(p_avg)
                p.data = (p_avg + p_noise).view(p.shape)
                goffset += numel
                offset += numel
            assert goffset == group["numel"]
        assert offset == self._numel

        return torch.cat(param_avgs, 0), torch.cat(noise_samples, 0)

    def _update(self) -> None:
        self.current_step += 1

        offset = 0
        for group in self.param_groups:
            lr = group["lr"]
            b1 = group["beta1"]
            b2 = group["beta2"]
            pg_slice = slice(offset, offset + group["numel"])

            param_avg = torch.cat([p.flatten() for p in group["params"] if p is not None], 0)

            avg_grad = self.opt_state["avg_grad"]
            if avg_grad is None:
                avg_grad = torch.zeros(group["numel"], device=self._device, dtype=self._dtype)
            else:
                avg_grad = avg_grad[pg_slice]

            group["momentum"] = self._new_momentum(avg_grad, group["momentum"], b1)

            group["hess"] = self._new_hess(
                self.hess_approx,
                group["hess"],
                self.opt_state["avg_nxg"],
                self.opt_state["avg_gsq"],
                pg_slice,
                group["ess"],
                b2,
                group["weight_decay"],
            )

            rescaled_lr = (
                lr * (group["hess_init"] + group["weight_decay"]) if self.rescale_lr else lr
            )
            debias_factor = 1.0 - pow(b1, float(self.current_step)) if self.debias else 1.0

            param_avg = self._new_param_averages(
                param_avg,
                group["hess"],
                group["momentum"],
                rescaled_lr,
                group["weight_decay"],
                group["clip_radius"],
                debias_factor,
                group["hess_init"],
            )

            pg_offset = 0
            for p in group["params"]:
                if p is not None:
                    p.data = param_avg[pg_offset : pg_offset + p.numel()].view(p.shape)
                    pg_offset += p.numel()
            assert pg_offset == group["numel"]
            offset += group["numel"]
        assert offset == self._numel

    @staticmethod
    def _get_nll_hess(
        method: str, hess: Tensor, avg_nxg: Tensor | None, avg_gsq: Tensor | None, pg_slice: slice
    ) -> Tensor:
        if method == "price":
            if avg_nxg is None:
                return torch.zeros_like(hess)
            return avg_nxg[pg_slice] * hess
        elif method == "gradsq":
            if avg_gsq is None:
                return torch.zeros_like(hess)
            return avg_gsq[pg_slice]
        else:
            raise NotImplementedError(f"unknown hessian approx.: {method}")

    @staticmethod
    def _new_momentum(avg_grad: Tensor, m: Tensor, b1: float) -> Tensor:
        return b1 * m + (1.0 - b1) * avg_grad

    @staticmethod
    def _new_hess(
        method: str,
        hess: Tensor,
        avg_nxg: Tensor | None,
        avg_gsq: Tensor | None,
        pg_slice: slice,
        ess: float,
        beta2: float,
        wd: float,
    ) -> Tensor:
        f = IVON._get_nll_hess(method, hess + wd, avg_nxg, avg_gsq, pg_slice) * ess
        return (
            beta2 * hess
            + (1.0 - beta2) * f
            + (0.5 * (1.0 - beta2) ** 2) * (hess - f).square() / (hess + wd)
        )

    @staticmethod
    def _new_param_averages(
        param_avg: Tensor,
        hess: Tensor,
        momentum: Tensor,
        lr: float,
        wd: float,
        clip_radius: float,
        debias: float,
        hess_init: float,
    ) -> Tensor:
        return param_avg - lr * torch.clip(
            (momentum / debias + wd * param_avg) / (hess + wd),
            min=-clip_radius,
            max=clip_radius,
        )
