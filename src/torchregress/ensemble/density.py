"""
Mixtures of conditional density models (deep ensembles of flows).

A :class:`DensityEnsemble` combines ``M`` independently trained conditional
density heads (normalizing flows, or any head with ``log_prob``/``sample``) into
the mixture

.. math::

    p(y \\mid x) = \\sum_{m=1}^{M} \\pi_m\\, p_m(y \\mid x),

with equal weights ``pi_m = 1/M`` by default.  :meth:`DensityEnsemble.fit_weights`
fits the weights by maximum likelihood on held-out data and, following the
*wifi ensemble* idea of Benevedes & Thaler (2025) applied to densities instead
of ratios, returns a sandwich covariance for them, so that
:meth:`DensityEnsemble.log_prob_std` gives a frequentist standard error on
``log p(y | x)``.

Each member is a full conditional density, so the ensemble supports an
entropy decomposition of the predictive uncertainty (Depeweg et al. 2018):

.. math::

    \\underbrace{H\\bigl[p(\\cdot \\mid x)\\bigr]}_{\\text{total}}
    = \\underbrace{\\frac{1}{M}\\sum_m H\\bigl[p_m(\\cdot \\mid x)\\bigr]}_{\\text{aleatoric}}
    + \\underbrace{\\frac{1}{M}\\sum_m
      \\mathrm{KL}\\bigl(p_m(\\cdot \\mid x)\\,\\|\\,p(\\cdot \\mid x)\\bigr)}_{\\text{epistemic}}.

The epistemic term is the mutual information between ``y`` and the member
index (the generalised Jensen-Shannon divergence of the members).  It is
non-negative and works for multimodal, multi-target densities where a
variance decomposition would be misleading.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Optional, Protocol, cast

import torch
from torch import Tensor, nn

__all__ = ["DensityEnsemble"]


class _DensityHead(Protocol):
    """Flow-head API: ``log_prob(context, y) -> [N]``, ``sample(context, n) -> [N, S, D]``."""

    def log_prob(self, context: Tensor, target: Tensor) -> Tensor: ...

    def sample(self, context: Tensor, n_samples: int = 1) -> Tensor: ...


class DensityEnsemble(nn.Module):
    """Mixture of conditional density models (equal or fitted weights).

    Args:
        heads: ``M >= 2`` density heads.  Each must provide
            ``log_prob(context, y) -> [N]`` and
            ``sample(context, n_samples) -> [N, S, D]`` (or ``[N, D]`` for one
            draw), as :class:`~torchregress.losses.NormalizingFlowLoss` does.
        encoders: Optional per-member context networks mapping ``x`` to that
            member's flow context.  ``None`` passes ``x`` unchanged to every
            head (a flow conditioned directly on the features).

    Members are trained independently (different seeds, bootstraps or
    hyper-parameters); this class only combines them.  The mixture weights
    start equal and can be fitted on held-out data with :meth:`fit_weights`.  ``log_prob`` and
    ``sample`` follow the flow-head convention, so the ensemble works with
    :func:`~torchregress.metrics.model_hpd_level`, and ``forward(x, y)`` is the
    per-sample negative log-likelihood, so the ensemble itself is a valid score
    for :meth:`~torchregress.losses.ConformalizedDensity.calibrate`.

    Example:
        >>> from torchregress.losses import create_flow_loss
        >>> heads = [
        ...     create_flow_loss(n_features=2, context_dim=3, flow_type="nsf") for _ in range(3)
        ... ]
        >>> ensemble = DensityEnsemble(heads)
        >>> x, y = torch.randn(8, 3), torch.randn(8, 2)
        >>> ensemble.log_prob(x, y).shape
        torch.Size([8])
    """

    def __init__(
        self,
        heads: Sequence[nn.Module],
        encoders: Optional[Sequence[nn.Module]] = None,
    ) -> None:
        super().__init__()
        heads = list(heads)
        if len(heads) < 2:
            raise ValueError(f"DensityEnsemble needs at least two heads, got {len(heads)}")
        for index, head in enumerate(heads):
            if not isinstance(head, nn.Module):
                raise TypeError(f"heads[{index}] must be a torch.nn.Module, got {type(head)}")
            if not (
                callable(getattr(head, "log_prob", None))
                and callable(getattr(head, "sample", None))
            ):
                raise TypeError(
                    f"heads[{index}] must define log_prob(context, y) and sample(context, n)"
                )
        if encoders is None:
            encoder_list: list[nn.Module] = [nn.Identity() for _ in heads]
        else:
            encoder_list = list(encoders)
            if len(encoder_list) != len(heads):
                raise ValueError(
                    f"encoders must match heads in length, got {len(encoder_list)} and {len(heads)}"
                )
        self.heads = nn.ModuleList(heads)
        self.encoders = nn.ModuleList(encoder_list)
        self.register_buffer("mixture_weights", torch.full((len(heads),), 1.0 / len(heads)))
        self.weight_covariance: Optional[Tensor] = None

    def _head(self, index: int) -> _DensityHead:
        return cast(_DensityHead, self.heads[index])

    @property
    def n_members(self) -> int:
        """Number of ensemble members ``M``."""
        return len(self.heads)

    def member_contexts(self, x: Tensor) -> list[Tensor]:
        """Per-member flow contexts ``[encoder_m(x) for m]``."""
        return [encoder(x) for encoder in self.encoders]

    def member_log_prob(self, x: Tensor, target: Tensor) -> Tensor:
        """Member log-densities ``log p_m(y | x)``, shape ``[M, N]``."""
        return torch.stack(
            [
                self._head(index).log_prob(context, target).reshape(-1)
                for index, context in enumerate(self.member_contexts(x))
            ]
        )

    def _log_weights(self, like: Tensor) -> Tensor:
        weights = cast(Tensor, self.mixture_weights).to(device=like.device, dtype=like.dtype)
        return torch.log(weights.clamp_min(torch.finfo(like.dtype).tiny))

    def log_prob(self, x: Tensor, target: Tensor) -> Tensor:
        """Mixture log-density ``log sum_m pi_m p_m(y | x)``, shape ``[N]``."""
        member = self.member_log_prob(x, target)
        return torch.logsumexp(member + self._log_weights(member).unsqueeze(1), dim=0)

    def fit_weights(
        self,
        x: Tensor,
        target: Tensor,
        *,
        max_iter: int = 1000,
        tol: float = 1e-10,
    ) -> "DensityEnsemble":
        """Fit the mixture weights by maximum likelihood, with a sandwich covariance.

        Maximises ``(1/N) sum_j log sum_m pi_m p_m(y_j | x_j)`` over the simplex
        by EM (the objective is concave in ``pi``), with the heads frozen.  Use
        data that was **not** used to train the heads, as in wifi ensembles
        (Benevedes & Thaler 2025): the covariance assumes fixed members.

        The covariance of the free weights ``pi_1..pi_{M-1}`` (``pi_M = 1 -
        sum``) is the M-estimator sandwich ``A^{-1} V A^{-1} / N`` with
        ``A = mean_j s_j s_j^T`` and ``V = cov_j(s_j)`` for the per-sample
        scores ``s_{jm} = (p_m - p_M) / p``.  It is stored, lifted to all
        ``M`` weights, as :attr:`weight_covariance`.  It is valid when the
        fitted weights are away from 0; a weight driven to zero signals a
        member the data does not support.

        Args:
            x: Held-out inputs ``[N, ...]``.
            target: Held-out targets ``[N, D]``.
            max_iter: Maximum EM iterations.
            tol: Stop when the mean log-likelihood improves by less than this.

        Returns:
            ``self``.
        """
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError(f"max_iter must be a positive integer, got {max_iter!r}")
        with torch.no_grad():
            member = self.member_log_prob(x, target).to(torch.float64)  # [M, N]
            if not torch.isfinite(member).all():
                raise ValueError("member log-densities must be finite on the fitting data")
            n_members, n_rows = member.shape
            log_pi = torch.full(
                (n_members,), -math.log(n_members), dtype=torch.float64, device=member.device
            )
            previous = -math.inf
            for _ in range(max_iter):
                joint = member + log_pi.unsqueeze(1)
                log_mix = torch.logsumexp(joint, dim=0)
                current = float(log_mix.mean())
                log_pi = torch.logsumexp(joint - log_mix, dim=1) - math.log(n_rows)
                if current - previous < tol:
                    break
                previous = current
            weights = log_pi.exp()
            weights = weights / weights.sum()

            log_mix = torch.logsumexp(member + weights.log().unsqueeze(1), dim=0)
            ratio = torch.exp(member - log_mix)  # p_m / p, [M, N]
            scores = (ratio[:-1] - ratio[-1:]).T  # [N, M - 1]
            information = scores.T @ scores / n_rows
            centred = scores - scores.mean(dim=0, keepdim=True)
            meat = centred.T @ centred / max(n_rows - 1, 1)
            bread = torch.linalg.pinv(information, hermitian=True)
            free_covariance = bread @ meat @ bread / n_rows
            lift = torch.cat(
                [
                    torch.eye(n_members - 1, dtype=torch.float64, device=member.device),
                    -torch.ones(1, n_members - 1, dtype=torch.float64, device=member.device),
                ]
            )
            covariance = lift @ free_covariance @ lift.T
        buffer = cast(Tensor, self.mixture_weights)
        buffer.copy_(weights.to(dtype=buffer.dtype, device=buffer.device))
        self.weight_covariance = 0.5 * (covariance + covariance.T)
        return self

    def log_prob_std(self, x: Tensor, target: Tensor) -> Tensor:
        """Standard error of :meth:`log_prob` from the fitted-weight covariance, ``[N]``.

        ``Var[log p(y | x)] = g^T Sigma_pi g`` with ``g_m = p_m(y | x) / p(y | x)``.
        Requires :meth:`fit_weights`.  This is the uncertainty from estimating
        the weights only; it does not include errors of the frozen members.
        """
        if self.weight_covariance is None:
            raise RuntimeError("call fit_weights() before log_prob_std()")
        with torch.no_grad():
            member = self.member_log_prob(x, target).to(torch.float64)
            log_mix = torch.logsumexp(member + self._log_weights(member).unsqueeze(1), dim=0)
            gradient = torch.exp(member - log_mix).T  # [N, M]
            covariance = self.weight_covariance.to(gradient.device)
            variance = ((gradient @ covariance) * gradient).sum(dim=1)
        return variance.clamp_min(0.0).sqrt()

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        """Mixture negative log-likelihood per sample, ``[N]`` (evaluation only)."""
        return -self.log_prob(x, target)

    def _member_samples(self, contexts: list[Tensor], n_samples: int) -> Tensor:
        """Draws from every member, shape ``[M, N, S, D]``."""
        draws = []
        for index, context in enumerate(contexts):
            sample = self._head(index).sample(context, n_samples=n_samples)
            if sample.dim() == 2:
                sample = sample.unsqueeze(1)
            draws.append(sample)
        return torch.stack(draws)

    def sample(self, x: Tensor, n_samples: int = 1) -> Tensor:
        """Draws from the mixture, ``[N, S, D]`` (``[N, D]`` when ``n_samples == 1``).

        Each draw picks member ``m`` with probability ``pi_m``.  For simplicity every
        member produces ``n_samples`` draws and one is kept per slot, so the cost
        is ``M`` times that of a single flow.
        """
        if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
            raise ValueError(f"n_samples must be a positive integer, got {n_samples!r}")
        member_draws = self._member_samples(self.member_contexts(x), n_samples)
        n_members, n_rows, _, n_features = member_draws.shape
        weights = cast(Tensor, self.mixture_weights).to(member_draws.device, torch.float64)
        choice = torch.multinomial(weights, n_rows * n_samples, replacement=True).reshape(
            n_rows, n_samples
        )
        index = choice.reshape(1, n_rows, n_samples, 1).expand(1, n_rows, n_samples, n_features)
        samples = member_draws.gather(0, index).squeeze(0)
        return samples.squeeze(1) if n_samples == 1 else samples

    def entropy_decomposition(self, x: Tensor, n_samples: int = 128) -> dict[str, Tensor]:
        """Monte-Carlo entropy decomposition of the mixture, each entry ``[N]``.

        Returns ``total`` (differential entropy of the mixture), ``aleatoric``
        (weighted mean member entropy) and ``epistemic`` (weighted mean KL of
        the members to the mixture, i.e. their mutual information with ``y``), with
        ``total = aleatoric + epistemic``.  Differential entropies can be
        negative and depend on the target scaling; ``epistemic`` is
        scale-invariant and non-negative in expectation (the Monte-Carlo
        estimate can dip slightly below zero when members agree).

        Draws ``n_samples`` from every member and scores them under every
        member, so the cost is ``M**2 * n_samples`` log-density evaluations per
        row.  Runs under ``torch.no_grad()``.
        """
        if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 2:
            raise ValueError(f"n_samples must be an integer >= 2, got {n_samples!r}")
        with torch.no_grad():
            contexts = self.member_contexts(x)
            draws = self._member_samples(contexts, n_samples)  # [M, N, S, D]
            n_members, n_rows, _, n_features = draws.shape
            flat = draws.permute(1, 0, 2, 3).reshape(n_rows, n_members * n_samples, n_features)
            flat = flat.reshape(n_rows * n_members * n_samples, n_features)
            repeat = n_members * n_samples
            scores = torch.stack(
                [
                    self._head(index)
                    .log_prob(context.repeat_interleave(repeat, dim=0), flat)
                    .reshape(n_rows, n_members, n_samples)
                    for index, context in enumerate(contexts)
                ]
            )  # [scoring member, N, drawing member, S]
            log_weights = self._log_weights(scores)
            mixture = torch.logsumexp(scores + log_weights.reshape(-1, 1, 1, 1), dim=0)  # [N, M, S]
            own = torch.diagonal(scores, dim1=0, dim2=2).permute(0, 2, 1)  # [N, M, S]
            weights = log_weights.exp().reshape(1, -1)
            aleatoric = -(own.mean(dim=2) * weights).sum(dim=1)
            total = -(mixture.mean(dim=2) * weights).sum(dim=1)
        return {"total": total, "aleatoric": aleatoric, "epistemic": total - aleatoric}
