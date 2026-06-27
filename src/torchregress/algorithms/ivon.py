"""
Improved Variational Online Newton (IVON) optimizer.

Derived from the Bayesian Learning Rule framework (Khan & Rue, 2023).
"""

from __future__ import annotations

from contextlib import contextmanager
from math import pow
from typing import Any, Callable, Generator, cast

import torch
import torch.optim
from torch import Tensor


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
        if hess_approx not in ("price", "gradsq"):
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

        self._reset_samples()
        self._init_buffers()

    def _get_param_configs(self) -> tuple[int, torch.device, torch.dtype]:
        for pg in self.param_groups:
            pg["numel"] = sum(p.numel() for p in pg["params"] if p is not None)
        all_params = [p for pg in self.param_groups for p in pg["params"] if p is not None]
        if not all_params:
            return 0, torch.device("cpu"), torch.get_default_dtype()
        device = next(iter({p.device for p in all_params}))
        dtype = next(iter({p.dtype for p in all_params}))
        total = sum(pg["numel"] for pg in self.param_groups)
        return total, device, dtype

    def _reset_samples(self) -> None:
        s: dict[str, Any] = cast("dict[str, Any]", self.state)
        s["count"] = 0
        s["avg_grad"] = None
        s["avg_nxg"] = None
        s["avg_gsq"] = None

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
            s: dict[str, Any] = cast("dict[str, Any]", self.state)
            grad_sample = torch.cat(param_grads, 0)
            count = s["count"] + 1
            s["count"] = count
            s["avg_grad"] = _welford_mean(s["avg_grad"], grad_sample, count)
            if self.hess_approx == "price":
                s["avg_nxg"] = _welford_mean(s["avg_nxg"], noise * grad_sample, count)
            elif self.hess_approx == "gradsq":
                s["avg_gsq"] = _welford_mean(s["avg_gsq"], grad_sample.square(), count)

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:  # type: ignore[override]
        loss = None
        if closure is not None:
            losses = []
            for _ in range(self.mc_samples):
                with torch.enable_grad():
                    loss_val = closure()
                losses.append(loss_val)
            loss = torch.stack(losses).mean()

        self._update()
        self._reset_samples()
        return loss

    def _sample_params(self) -> tuple[Tensor, Tensor]:
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
        state: dict[str, Any] = cast("dict[str, Any]", self.state)
        self.current_step += 1
        debias = self.current_step
        offset = 0
        for group in self.param_groups:
            lr = group["lr"]
            b1 = group["beta1"]
            b2 = group["beta2"]
            pg_slice = slice(offset, offset + group["numel"])

            param_avg = torch.cat([p.flatten() for p in group["params"] if p is not None], 0)

            avg_grad = state["avg_grad"]
            if avg_grad is None:
                avg_grad = torch.zeros(group["numel"], device=self._device, dtype=self._dtype)
            else:
                avg_grad = avg_grad[pg_slice]

            group["momentum"] = b1 * group["momentum"] + (1.0 - b1) * avg_grad

            hess_wd = group["hess"] + group["weight_decay"]
            if self.hess_approx == "price":
                avg_nxg = state["avg_nxg"]
                f = (
                    avg_nxg[pg_slice] * hess_wd
                    if avg_nxg is not None
                    else torch.zeros_like(group["hess"])
                ) * group["ess"]  # noqa: E501
            elif self.hess_approx == "gradsq":
                avg_gsq = state["avg_gsq"]
                f = (
                    avg_gsq[pg_slice] if avg_gsq is not None else torch.zeros_like(group["hess"])
                ) * group["ess"]  # noqa: E501
            else:
                raise NotImplementedError(f"unknown hessian approx.: {self.hess_approx}")
            group["hess"] = (
                b2 * group["hess"]
                + (1.0 - b2) * f
                + (0.5 * (1.0 - b2) ** 2) * (group["hess"] - f).square() / hess_wd
            )  # noqa: E501

            rescaled_lr = lr * (group["hess_init"] + group["weight_decay"])
            debias_factor = 1.0 - pow(b1, float(debias))

            param_avg = param_avg - rescaled_lr * torch.clip(
                (group["momentum"] / debias_factor + group["weight_decay"] * param_avg)
                / (group["hess"] + group["weight_decay"]),  # noqa: E501
                min=-group["clip_radius"],
                max=group["clip_radius"],
            )

            pg_offset = 0
            for p in group["params"]:
                if p is not None:
                    p.data = param_avg[pg_offset : pg_offset + p.numel()].view(p.shape)
                    pg_offset += p.numel()
            assert pg_offset == group["numel"]
            offset += group["numel"]
        assert offset == self._numel
