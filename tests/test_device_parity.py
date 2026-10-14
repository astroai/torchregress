"""CPU vs. accelerator parity for public losses and metrics.

Every test in this module carries the ``cuda`` marker.  By default the target
device is ``cuda`` (the tests are skipped when CUDA is unavailable, see
``conftest.py``).  Set ``TORCHREGRESS_PARITY_DEVICE`` to choose another target;
in particular ``TORCHREGRESS_PARITY_DEVICE=cpu pytest -m cuda`` compares CPU
with CPU, which still validates the input builders below and catches dtype
bugs (e.g. a float32 tensor created inside ``forward`` for float64 inputs).

For each curated loss / metric a small deterministic float64 case is built on
CPU, evaluated on CPU (value, and for losses the gradient w.r.t. ``y_pred``),
then the inputs and the module are moved to the target device and re-evaluated.
A float32 variant uses looser tolerances.

Known library bugs are recorded as strict ``xfail`` entries in
the ``XFAIL_*`` tables rather than being fixed here.
"""

from __future__ import annotations

import copy
import math
import os
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchregress.losses as L
import torchregress.metrics as M

pytestmark = pytest.mark.cuda

TARGET_DEVICE = os.environ.get("TORCHREGRESS_PARITY_DEVICE", "cuda")

N = 8  # batch size used by the builders
D = 3  # feature dimension used by the multivariate builders

# (rtol, atol) per dtype.
TOLERANCES: dict[torch.dtype, tuple[float, float]] = {
    torch.float64: (1e-5, 1e-6),
    torch.float32: (1e-3, 1e-4),
}
DTYPES = [torch.float64, torch.float32]
DTYPE_IDS = ["f64", "f32"]

Built = tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]
Builder = Callable[["_Rng"], Built]


# --------------------------------------------------------------------------- #
# Deterministic input helpers (always float64 on CPU)
# --------------------------------------------------------------------------- #
class _Rng:
    """Seeded CPU generator producing float64 tensors."""

    def __init__(self, seed: int = 0) -> None:
        self.g = torch.Generator(device="cpu").manual_seed(seed)

    def normal(self, *shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=self.g, dtype=torch.float64)

    def uniform(self, *shape: int, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
        return lo + (hi - lo) * torch.rand(*shape, generator=self.g, dtype=torch.float64)

    def positive(self, *shape: int) -> torch.Tensor:
        return self.uniform(*shape, lo=0.5, hi=2.0)

    def counts(self, *shape: int, high: int = 6) -> torch.Tensor:
        return torch.randint(0, high, shape, generator=self.g).to(torch.float64)

    def mask(self, *shape: int) -> torch.Tensor:
        m = torch.rand(*shape, generator=self.g) > 0.25
        m.view(-1)[0] = True
        return m

    def ints(self, *shape: int, high: int) -> torch.Tensor:
        return torch.randint(0, high, shape, generator=self.g)

    def spd(self, n: int, d: int) -> torch.Tensor:
        a = self.normal(n, d, d)
        eye = torch.eye(d, dtype=torch.float64, device="cpu")
        return a @ a.transpose(-1, -2) / d + eye

    def sorted_cols(self, n: int, k: int, *tail: int) -> torch.Tensor:
        """``[n, k, *tail]`` with strictly increasing values along dim 1."""
        base = torch.cumsum(self.uniform(n, k, *tail, lo=0.2, hi=1.0), dim=1)
        return base - base.mean(dim=1, keepdim=True)


def _as(obj: Any, device: str | torch.device, dtype: torch.dtype) -> Any:
    """Recursively move tensors to ``device``; cast floating tensors to ``dtype``."""
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point():
            return obj.detach().to(device=device, dtype=dtype)
        return obj.detach().to(device=device)
    if isinstance(obj, dict):
        return {k: _as(v, device, dtype) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_as(v, device, dtype) for v in obj)
    if isinstance(obj, list):
        return [_as(v, device, dtype) for v in obj]
    return obj


def _leaves(obj: Any) -> list[torch.Tensor]:
    """Flatten nested outputs into a list of tensors (python scalars become tensors)."""
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, bool | int | float):
        return [torch.tensor(float(obj), dtype=torch.float64)]
    if isinstance(obj, dict):
        out: list[torch.Tensor] = []
        for k in sorted(obj, key=str):
            out.extend(_leaves(obj[k]))
        return out
    if isinstance(obj, tuple | list):
        out = []
        for v in obj:
            out.extend(_leaves(v))
        return out
    if hasattr(obj, "__dict__"):
        return _leaves({k: v for k, v in vars(obj).items() if not k.startswith("_")})
    raise TypeError(f"cannot flatten output of type {type(obj).__name__}")


class _MetricRunner(nn.Module):
    """Run a stateful torchmetrics-style metric as ``reset -> update -> compute``."""

    def __init__(self, metric: nn.Module) -> None:
        super().__init__()
        self.metric = metric

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self.metric.reset()
        self.metric.update(*args, **kwargs)
        return self.metric.compute()


def _prepare(fn: Callable[..., Any], device: str, dtype: torch.dtype) -> Callable[..., Any]:
    """Deep-copy ``fn`` (so both devices start from identical state) and move it."""
    fn = copy.deepcopy(fn)
    if isinstance(fn, nn.Module):
        fn = fn.to(device)
        for m in fn.modules():
            if hasattr(m, "set_dtype"):  # torchmetrics.Metric
                m.set_dtype(dtype)
        fn = fn.to(dtype)
    return fn


def _evaluate(
    build: Builder,
    device: str,
    dtype: torch.dtype,
    *,
    grad: bool,
) -> tuple[Any, list[torch.Tensor | None]]:
    """Build a case on CPU, move it to ``device`` / ``dtype`` and evaluate it."""
    fn, args, kwargs = build(_Rng())
    fn = _prepare(fn, device, dtype)
    args = _as(args, device, dtype)
    kwargs = _as(kwargs, device, dtype)
    inputs: list[torch.Tensor] = []
    if grad and args:

        def _track(t: torch.Tensor) -> torch.Tensor:
            if t.is_floating_point():
                t = t.clone().requires_grad_(True)
                inputs.append(t)
            return t

        first = args[0]
        if isinstance(first, torch.Tensor):
            first = _track(first)
        elif isinstance(first, tuple | list):
            first = type(first)(_track(t) if isinstance(t, torch.Tensor) else t for t in first)
        args = (first, *args[1:])
    out = fn(*args, **kwargs)
    grads: list[torch.Tensor | None] = []
    if inputs:
        value = out if isinstance(out, torch.Tensor) else _leaves(out)[0]
        if value.requires_grad:
            grads = list(torch.autograd.grad(value.sum(), inputs, allow_unused=True))
        else:
            grads = [None] * len(inputs)
    return out, grads


def _assert_parity(build: Builder, dtype: torch.dtype, *, grad: bool) -> None:
    rtol, atol = TOLERANCES[dtype]
    ref, ref_grads = _evaluate(build, "cpu", dtype, grad=grad)
    got, got_grads = _evaluate(build, TARGET_DEVICE, dtype, grad=grad)
    ref_leaves, got_leaves = _leaves(ref), _leaves(got)
    assert len(ref_leaves) == len(got_leaves)
    for a, b in zip(ref_leaves, got_leaves, strict=True):
        assert a.shape == b.shape
        torch.testing.assert_close(
            b.detach().cpu().to(a.dtype), a.detach(), rtol=rtol, atol=atol, equal_nan=True
        )
    assert len(ref_grads) == len(got_grads)
    for a, b in zip(ref_grads, got_grads, strict=True):
        assert (a is None) == (b is None)
        if a is not None and b is not None:
            torch.testing.assert_close(
                b.detach().cpu().to(a.dtype), a.detach(), rtol=rtol, atol=atol, equal_nan=True
            )


def _assert_device_and_dtype(build: Builder, dtype: torch.dtype, *, grad: bool) -> None:
    """Outputs (and gradients) live on the target device and keep the input dtype.

    The dtype check only applies when the case has floating-point tensor inputs: rates
    computed from purely integer inputs (e.g. ordinal labels) have no dtype to preserve.
    """
    out, grads = _evaluate(build, TARGET_DEVICE, dtype, grad=grad)
    _, args, kwargs = build(_Rng())
    has_float_input = any(t.is_floating_point() for t in _leaves_tensors_only((args, kwargs)))
    target = torch.device(TARGET_DEVICE)
    for t in [*_leaves_tensors_only(out), *(g for g in grads if g is not None)]:
        assert t.device.type == target.type, f"output on {t.device}, expected {target}"
        if t.is_floating_point() and has_float_input:
            assert t.dtype == dtype, f"output dtype {t.dtype} != input dtype {dtype}"


def _leaves_tensors_only(obj: Any) -> list[torch.Tensor]:
    """Like ``_leaves`` but drops python scalars (they carry no device / dtype)."""
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, dict):
        return [t for k in sorted(obj, key=str) for t in _leaves_tensors_only(obj[k])]
    if isinstance(obj, tuple | list):
        return [t for v in obj for t in _leaves_tensors_only(v)]
    if hasattr(obj, "__dict__") and not isinstance(obj, bool | int | float):
        return _leaves_tensors_only({k: v for k, v in vars(obj).items() if not k.startswith("_")})
    return []


# --------------------------------------------------------------------------- #
# Loss cases:  name -> builder(rng) -> (module_or_fn, args, kwargs)
# The gradient is checked w.r.t. the floating tensors in ``args[0]`` (y_pred).
# --------------------------------------------------------------------------- #
def _point(cls: Callable[..., Any], **init: Any) -> Builder:
    """Point-prediction loss on ``[N, 1]`` with a mask and weights."""

    def build(r: _Rng) -> Built:
        return (
            cls(**init),
            (r.normal(N, 1), r.normal(N, 1)),
            {"mask": r.mask(N, 1), "weights": r.uniform(N, 1, lo=0.5, hi=1.5)},
        )

    return build


def _positive_rate(cls: Callable[..., Any], *, shape: int = 2, **init: Any) -> Builder:
    """Count-like losses: positive predictions, integer-valued float targets."""

    def build(r: _Rng) -> Built:
        return cls(**init), (r.positive(N, shape), r.counts(N, shape)), {}

    return build


def _family(cls: Callable[..., Any], k: int) -> Builder:
    """Flexible-shape families: ``y_pred`` has ``k`` unconstrained parameters."""

    def build(r: _Rng) -> Built:
        return (
            cls(),
            (0.5 * r.normal(N, k), r.normal(N, 1)),
            {"weights": r.uniform(N, 1, lo=0.5, hi=1.5)},
        )

    return build


def _quantile_stack(cls: Callable[..., Any], levels: list[float], **init: Any) -> Builder:
    def build(r: _Rng) -> Built:
        return (
            cls(levels, **init),
            (r.sorted_cols(N, len(levels), 2), r.normal(N, 2)),
            {"mask": r.mask(N, 2), "weights": r.uniform(N, 2, lo=0.5, hi=1.5)},
        )

    return build


def _mdn(cls: Callable[..., Any], cov: str) -> Builder:
    def build(r: _Rng) -> Built:
        k, f = 3, 2
        width = k + 2 * k * f if cov == "diagonal" else k + k * f + k * f * (f + 1) // 2
        return (
            cls(n_components=k, n_features=f, covariance_type=cov),
            (
                0.5 * r.normal(N, width),
                r.normal(N, f),
            ),
            {},
        )

    return build


def _censor_inputs(r: _Rng) -> tuple[torch.Tensor, torch.Tensor]:
    target = r.positive(N, 1)
    censoring = r.ints(N, 1, high=3) - 1  # {-1, 0, 1}
    return target, censoring


def _censored_gaussian(r: _Rng) -> Built:
    target, censoring = _censor_inputs(r)
    return (
        L.CensoredGaussianNLLLoss(),
        ((r.normal(N, 1), 0.3 * r.normal(N, 1)), target),
        {"censoring": censoring},
    )


def _aft(r: _Rng) -> Built:
    target, censoring = _censor_inputs(r)
    return (
        L.AFTLoss(),
        ((r.normal(N, 1), 0.3 * r.normal(N, 1)), target),
        {"censoring": censoring},
    )


def _censored_quantile(r: _Rng) -> Built:
    target, censoring = _censor_inputs(r)
    return L.CensoredQuantileLoss(quantile=0.7), (r.normal(N, 1), target), {"censoring": censoring}


def _interval_censored(r: _Rng) -> Built:
    target = r.normal(N, 1)
    censoring = torch.zeros(N, 1, dtype=torch.int64)
    return (
        L.CensoredGaussianNLLLoss(),
        ((r.normal(N, 1), 0.3 * r.normal(N, 1)), target),
        {"censoring": censoring, "lower_bound": target - 0.5, "upper_bound": target + 0.5},
    )


def _gaussian_packed(r: _Rng) -> Built:
    return L.GaussianNLLLoss(), (r.normal(N, 2), r.normal(N, 1)), {"weights": r.uniform(N, 1)}


def _gaussian_tuple(r: _Rng) -> Built:
    return (
        L.GaussianNLLLoss(log_variance=False),
        ((r.normal(N, D), r.positive(N, D)), r.normal(N, D)),
        {"mask": r.mask(N, D)},
    )


def _gaussian_fixed_var(r: _Rng) -> Built:
    return L.GaussianNLLLoss(fixed_variance=0.5), (r.normal(N, 1), r.normal(N, 1)), {}


def _gaussian_crps(r: _Rng) -> Built:
    return L.GaussianCRPSLoss(), (r.normal(N, 2), r.normal(N, 1)), {}


def _beta_nll(r: _Rng) -> Built:
    return L.BetaNLLLoss(beta=0.5), (r.normal(N, 2), r.normal(N, 1)), {}


def _faithful(r: _Rng) -> Built:
    return L.FaithfulGaussianLoss(), (r.normal(N, 2), r.normal(N, 1)), {}


def _mvn(r: _Rng) -> Built:
    return (
        L.MultivariateGaussianLoss(),
        (r.normal(N, D), r.normal(N, D)),
        {"covariance_matrices": r.spd(N, D)},
    )


def _lowrank(r: _Rng) -> Built:
    return (
        L.LowRankGaussianLoss(),
        (r.normal(N, D), r.normal(N, D)),
        {"cov_factor": 0.5 * r.normal(N, D, 2), "cov_diag": r.positive(N, D)},
    )


def _gw_bound(r: _Rng) -> Built:
    return (
        L.GaussianWassersteinBoundLoss(),
        (r.normal(N, D), r.normal(N, D)),
        {"pred_covariance": r.spd(N, D), "target_covariance": r.spd(N, D)},
    )


def _noisy_target_nll(r: _Rng) -> Built:
    return (
        L.NoisyTargetGaussianNLL(),
        ((r.normal(N, 1), r.positive(N, 1)), r.normal(N, 1)),
        {"target_variance": 0.1 * r.positive(N, 1)},
    )


def _pseudo_label_nll(r: _Rng) -> Built:
    return (
        L.PseudoLabelNLL(),
        ((r.normal(N, 1), r.positive(N, 1)), r.normal(N, 1)),
        {
            "pseudo_target": r.normal(N, 1),
            "pseudo_confidence": r.uniform(N, 1),
            "label_mask": r.mask(N, 1),
        },
    )


def _evidential(r: _Rng) -> Built:
    return L.EvidentialRegressionLoss(), (r.normal(N, 4), r.normal(N, 1)), {}


def _evidential_tuple(r: _Rng) -> Built:
    return (
        L.EvidentialRegressionLoss(unconstrained_inputs=False),
        (
            (r.normal(N, 1), r.positive(N, 1), 1.0 + r.positive(N, 1), r.positive(N, 1)),
            r.normal(N, 1),
        ),
        {},
    )


def _zip(r: _Rng) -> Built:
    return (
        L.ZeroInflatedPoissonNLLLoss(),
        (0.5 * r.normal(N, 2), r.counts(N, 2)),
        {"pi_logits": r.normal(N, 2)},
    )


def _poisson_gaussian(cls: Callable[..., Any], **init: Any) -> Builder:
    def build(r: _Rng) -> Built:
        return cls(**init), (0.5 * r.normal(N, 2) + 2.0, r.positive(N, 2) * 8.0), {}

    return build


def _conformal(method: str) -> Builder:
    def build(r: _Rng) -> Built:
        if method == "split":
            return L.ConformalLoss(method="split"), (r.normal(N, 1), r.normal(N, 1)), {}
        pred = r.normal(N, 1) + torch.tensor([-1.0, 1.0], dtype=torch.float64)
        return L.ConformalLoss(method=method, alpha=0.2), (pred, r.normal(N, 1)), {}

    return build


def _transformed(cls: Callable[..., Any], **init: Any) -> Builder:
    def build(r: _Rng) -> Built:
        return (
            cls(**init),
            (r.normal(N, 1).abs() + 0.5, r.positive(N, 1)),
            {"mask": r.mask(N, 1)},
        )

    return build


def _balanced_mse(r: _Rng) -> Built:
    edges = torch.linspace(-3.0, 3.0, 5, dtype=torch.float64)
    loss = L.BalancedMSELoss(edges).fit(r.normal(64, 1))
    return loss, (r.normal(N, 1), r.normal(N, 1)), {}


def _bin_reweighted(r: _Rng) -> Built:
    loss = L.BinReweightedMSELoss(num_bins=4).fit(r.normal(64, 1))
    return loss, (r.normal(N, 1), r.normal(N, 1)), {}


def _focal_r(r: _Rng) -> Built:
    return L.FocalRLoss(), (r.normal(N, 1), r.normal(N, 1)), {}


def _lds(r: _Rng) -> Built:
    loss = L.LDSLoss()
    loss.fit(r.normal(64, 1))
    return loss, (r.normal(N, 1), r.normal(N, 1)), {}


def _density_weighted(r: _Rng) -> Built:
    loss = L.DensityWeightedLoss()
    loss.fit_density(r.normal(64, 1))
    return loss, (r.normal(N, 1), r.normal(N, 1)), {}


def _propensity(r: _Rng) -> Built:
    return (
        L.PropensityWeightedLoss(),
        (r.normal(N, 1), r.normal(N, 1)),
        {"propensity": r.uniform(N, 1, lo=0.2, hi=0.9), "observed": r.mask(N, 1)},
    )


def _consistency(r: _Rng) -> Built:
    return (
        L.ConsistencyRegLoss(),
        (r.normal(N, 1), r.normal(N, 1)),
        {"teacher_pred": r.normal(N, 1)},
    )


def _pseudo_consistency(r: _Rng) -> Built:
    return (
        L.PseudoLabelConsistencyLoss(),
        (r.normal(N, 1), r.normal(N, 1)),
        {
            "pseudo_target": r.normal(N, 1),
            "pseudo_confidence": r.uniform(N, 1),
            "teacher_pred": r.normal(N, 1),
            "label_mask": r.mask(N, 1),
        },
    )


def _ordinal(cls: Callable[..., Any], *, cumulative: bool) -> Builder:
    def build(r: _Rng) -> Built:
        k = 5
        width = k - 1 if cumulative else k
        return cls(), (r.normal(N, width), r.ints(N, high=k)), {}

    return build


def _wasserstein1d(r: _Rng) -> Built:
    return (
        L.DiscreteWasserstein1Loss(bin_edges=torch.linspace(-2.0, 2.0, 6, dtype=torch.float64)),
        (r.normal(N, 5), r.normal(N, 1)),
        {},
    )


def _rank_n_contrast(r: _Rng) -> Built:
    return L.RankNContrastLoss(), (r.normal(N, 4), r.normal(N)), {}


def _tweedie_fn(r: _Rng) -> Built:
    return L.tweedie_loss, (r.positive(N, 1), r.positive(N, 1)), {"p": 1.4}


def _quantile_fn(r: _Rng) -> Built:
    return L.quantile_loss, (r.normal(N, 1), r.normal(N, 1)), {"quantile": 0.3}


def _expectile_fn(r: _Rng) -> Built:
    return L.expectile_loss, (r.normal(N, 1), r.normal(N, 1)), {"expectile": 0.3}


def _beta_nll_fn(r: _Rng) -> Built:
    return L.beta_nll_loss, (r.normal(N, 2), r.normal(N, 1)), {"beta": 0.5}


def _skew_normal_fn(r: _Rng) -> Built:
    return L.skew_normal_nll, (0.5 * r.normal(N, 3), r.normal(N, 1)), {}


def _gev_fn(r: _Rng) -> Built:
    return L.gev_nll, (0.5 * r.normal(N, 3), r.normal(N, 1)), {}


def _sqr(r: _Rng) -> Built:
    return L.SQRLoss(n_levels=8), (r.sorted_cols(N, 8), r.normal(N, 1)), {}


def _adaptive_robust(r: _Rng) -> Built:
    return L.AdaptiveRobustLoss(), (r.normal(N, 1), r.normal(N, 1)), {"mask": r.mask(N, 1)}


def _cvar(r: _Rng) -> Built:
    return L.CVaRLoss(alpha=0.25), (r.normal(N), r.normal(N)), {}


def _beta_regression(r: _Rng) -> Built:
    return L.BetaRegressionNLLLoss(), (0.5 * r.normal(N, 2), r.uniform(N, 1, lo=0.1, hi=0.9)), {}


LOSS_CASES: dict[str, Builder] = {
    # Weighted point losses
    "WeightedMSELoss": _point(L.WeightedMSELoss),
    "WeightedL1Loss": _point(L.WeightedL1Loss),
    "WeightedHuberLoss": _point(L.WeightedHuberLoss),
    # Gaussian family
    "GaussianNLLLoss[packed]": _gaussian_packed,
    "GaussianNLLLoss[tuple,multivariate]": _gaussian_tuple,
    "GaussianNLLLoss[fixed_variance]": _gaussian_fixed_var,
    "GaussianCRPSLoss": _gaussian_crps,
    "BetaNLLLoss": _beta_nll,
    "beta_nll_loss": _beta_nll_fn,
    "FaithfulGaussianLoss": _faithful,
    "MultivariateGaussianLoss": _mvn,
    "LowRankGaussianLoss": _lowrank,
    "GaussianWassersteinBoundLoss": _gw_bound,
    "StudentTLoss": _point(L.StudentTLoss, nu=3.0),
    "NoisyTargetGaussianNLL": _noisy_target_nll,
    "PseudoLabelNLL": _pseudo_label_nll,
    # Mixture density networks
    "MDNLoss[diagonal]": _mdn(L.MDNLoss, "diagonal"),
    "MixtureDensityLoss[diagonal]": _mdn(L.MixtureDensityLoss, "diagonal"),
    "MixtureDensityLoss[full]": _mdn(L.MixtureDensityLoss, "full"),
    # Quantile / expectile
    "QuantileLoss": _point(L.QuantileLoss, quantile=0.3),
    "quantile_loss": _quantile_fn,
    "MultiQuantileLoss": _quantile_stack(L.MultiQuantileLoss, [0.1, 0.5, 0.9]),
    "QuantileCrossoverLoss": _quantile_stack(L.QuantileCrossoverLoss, [0.1, 0.5, 0.9]),
    "ExpectileLoss": _point(L.ExpectileLoss, expectile=0.3),
    "expectile_loss": _expectile_fn,
    "AsymmetricLeastSquaresLoss": _point(L.AsymmetricLeastSquaresLoss, expectile=0.7),
    "MultiExpectileLoss": _quantile_stack(L.MultiExpectileLoss, [0.1, 0.5, 0.9]),
    "ExpectileCrossoverLoss": _quantile_stack(L.ExpectileCrossoverLoss, [0.1, 0.5, 0.9]),
    "SQRLoss": _sqr,
    # Robust
    "BarronLoss": _point(L.BarronLoss, alpha=0.5),
    "AdaptiveRobustLoss": _adaptive_robust,
    "CauchyLoss": _point(L.CauchyLoss),
    "CharbonnierLoss": _point(L.CharbonnierLoss),
    "LogCoshLoss": _point(L.LogCoshLoss),
    "PseudoHuberLoss": _point(L.PseudoHuberLoss),
    "TukeyBiweightLoss": _point(L.TukeyBiweightLoss),
    "CVaRLoss": _cvar,
    # Counts / positive support
    "PoissonDevianceLoss": _positive_rate(L.PoissonDevianceLoss, log_input=False),
    "PoissonLikelihoodRatioLoss": _positive_rate(L.PoissonLikelihoodRatioLoss, log_input=False),
    "ZeroInflatedPoissonNLLLoss": _zip,
    "NegativeBinomialNLLLoss": _positive_rate(L.NegativeBinomialNLLLoss, learn_theta=False),
    "NegativeBinomialNLLLoss[learn_theta]": _positive_rate(
        L.NegativeBinomialNLLLoss, learn_theta=True
    ),
    "TweedieLoss": _positive_rate(L.TweedieLoss, p=1.5, link="identity"),
    "tweedie_loss": _tweedie_fn,
    "GammaLoss": _positive_rate(L.GammaLoss, link="identity"),
    "InverseGaussianLoss": _positive_rate(L.InverseGaussianLoss, link="identity"),
    "CompoundPoissonLoss": _positive_rate(L.CompoundPoissonLoss, link="identity"),
    "PoissonGaussianMixtureLoss": _poisson_gaussian(
        L.PoissonGaussianMixtureLoss, initial_variance=0.1, log_input=False
    ),
    "PoissonGaussianLikelihoodRatioLoss": _poisson_gaussian(
        L.PoissonGaussianLikelihoodRatioLoss, log_input=False
    ),
    "EnhancedPoissonGaussianMixtureLoss": _poisson_gaussian(
        L.EnhancedPoissonGaussianMixtureLoss, gain=1.5, read_noise=0.2, shot_noise=0.1
    ),
    # Censored
    "CensoredGaussianNLLLoss": _censored_gaussian,
    "CensoredGaussianNLLLoss[interval]": _interval_censored,
    "CensoredQuantileLoss": _censored_quantile,
    "AFTLoss": _aft,
    # Evidential
    "EvidentialRegressionLoss": _evidential,
    "EvidentialRegressionLoss[constrained tuple]": _evidential_tuple,
    # Flexible families
    "SkewNormalNLLLoss": _family(L.SkewNormalNLLLoss, 3),
    "skew_normal_nll": _skew_normal_fn,
    "GEVNLLLoss": _family(L.GEVNLLLoss, 3),
    "gev_nll": _gev_fn,
    "JohnsonSUNLLLoss": _family(L.JohnsonSUNLLLoss, 4),
    "SinhArcsinhNLLLoss": _family(L.SinhArcsinhNLLLoss, 4),
    "SkewTLoss": _family(L.SkewTLoss, 4),
    "AsymmetricLaplaceNLLLoss": _family(L.AsymmetricLaplaceNLLLoss, 3),
    "BetaRegressionNLLLoss": _beta_regression,
    # Imbalanced regression
    "BalancedMSELoss": _balanced_mse,
    "BinReweightedMSELoss": _bin_reweighted,
    "FocalRLoss": _focal_r,
    "LDSLoss": _lds,
    "DensityWeightedLoss": _density_weighted,
    "PropensityWeightedLoss": _propensity,
    # Semi-supervised
    "ConsistencyRegLoss": _consistency,
    "PseudoLabelConsistencyLoss": _pseudo_consistency,
    # Conformal-style training losses
    "ConformalLoss[split]": _conformal("split"),
    "ConformalLoss[cqr]": _conformal("cqr"),
    # Target transforms
    "LogTransformLoss": _transformed(L.LogTransformLoss),
    "SqrtTransformLoss": _transformed(L.SqrtTransformLoss),
    "BoxCoxTransformLoss": _transformed(L.BoxCoxTransformLoss, lam=0.5),
    "YeoJohnsonTransformLoss": _transformed(L.YeoJohnsonTransformLoss, lam=0.5),
    # Ordinal / distributional-target
    "CORALLoss": _ordinal(L.CORALLoss, cumulative=True),
    "CumulativeLinkLoss": _ordinal(L.CumulativeLinkLoss, cumulative=True),
    "OrdinalCrossEntropyLoss": _ordinal(L.OrdinalCrossEntropyLoss, cumulative=False),
    "DiscreteWasserstein1Loss": _wasserstein1d,
    "RankNContrastLoss": _rank_n_contrast,
}

# Strict xfails for genuine library device/dtype bugs, keyed by case name (applies to the
# float64 variant: ``f32`` inputs cannot expose float32 drift).  The *_PARITY tables apply
# to the value/gradient parity tests, the *_DTYPE tables to the output device/dtype test.
XFAIL_LOSSES_PARITY: dict[str, str] = {}
XFAIL_LOSSES_DTYPE: dict[str, str] = {}


def _params(cases: dict[str, Builder], xfail: dict[str, str]) -> list[Any]:
    params = []
    for name in cases:
        for dtype, dtype_id in zip(DTYPES, DTYPE_IDS, strict=True):
            marks = []
            if name in xfail and dtype is torch.float64:
                marks.append(pytest.mark.xfail(strict=True, reason=f"device bug: {xfail[name]}"))
            params.append(pytest.param(name, dtype, id=f"{name}-{dtype_id}", marks=marks))
    return params


# --------------------------------------------------------------------------- #
# Metric cases
# --------------------------------------------------------------------------- #
def _quantile_dict(r: _Rng, y: torch.Tensor) -> dict[float, torch.Tensor]:
    levels = [0.1, 0.25, 0.5, 0.75, 0.9]
    spread = torch.tensor([-1.28, -0.67, 0.0, 0.67, 1.28], dtype=torch.float64)
    center = y + 0.3 * r.normal(*y.shape)
    return {
        q: center + float(s) * (1.0 + 0.2 * r.uniform(*y.shape))
        for q, s in zip(levels, spread, strict=True)
    }


def _metric(fn: Callable[..., Any], *make: Callable[[_Rng], Any], **kwargs: Any) -> Builder:
    """Function metric; ``make`` are per-argument factories."""

    def build(r: _Rng) -> Built:
        return fn, tuple(m(r) for m in make), kwargs

    return build


def _stateful(cls: Callable[..., Any], *make: Callable[[_Rng], Any], **init: Any) -> Builder:
    """Stateful metric run as reset -> update -> compute."""

    def build(r: _Rng) -> Built:
        return _MetricRunner(cls(**init)), tuple(m(r) for m in make), {}

    return build


def _n1(r: _Rng) -> torch.Tensor:
    return r.normal(N, 1)


def _n(r: _Rng) -> torch.Tensor:
    return r.normal(N)


def _nd(r: _Rng) -> torch.Tensor:
    return r.normal(N, D)


def _pos(r: _Rng) -> torch.Tensor:
    return r.positive(N)


def _samples(r: _Rng) -> torch.Tensor:
    return r.normal(16, N)


def _samples_nd(r: _Rng) -> torch.Tensor:
    return r.normal(16, N, D)


def _interval(r: _Rng) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(lower, upper, y_true)`` with ``lower <= upper`` and a mix of hits and misses."""
    y = r.normal(N)
    centre = y + 0.5 * r.normal(N)
    width = r.uniform(N, lo=0.2, hi=1.5)
    return centre - width, centre + width, y


def _pit_case(r: _Rng) -> Built:
    def run(loc: torch.Tensor, scale: torch.Tensor, y: torch.Tensor) -> Any:
        dist = torch.distributions.Normal(loc, scale)
        return M.probability_integral_transform(dist.cdf, y, n_bins=5, return_histogram=True)

    return run, (r.normal(N), r.positive(N), r.normal(N)), {}


def _ks_uniform(r: _Rng) -> Built:
    return M.kolmogorov_smirnov_uniform_statistic, (r.uniform(32),), {}


def _grid(r: _Rng) -> tuple[torch.Tensor, torch.Tensor]:
    support = torch.linspace(-4.0, 4.0, 41, dtype=torch.float64)
    centers = r.normal(N, 1)
    density = torch.exp(-0.5 * (support[None, :] - centers) ** 2) / math.sqrt(2 * math.pi)
    return support, density


def _hpd_cov(r: _Rng) -> Built:
    support, density = _grid(r)
    return M.highest_posterior_density_coverage, (support, density, r.normal(N)), {"alpha": 0.2}


def _hpd_level(r: _Rng) -> Built:
    support, density = _grid(r)
    return M.highest_posterior_density_level, (support, density, r.normal(N)), {}


def _cde_loss(r: _Rng) -> Built:
    support, density = _grid(r)
    return M.conditional_density_estimation_loss, (support, density, r.normal(N)), {}


def _calibration_report(r: _Rng) -> Built:
    # Explicit samples/quantiles: drawing from a Distribution would use device-specific RNG.
    y = r.normal(N)
    return (
        M.calibration_metrics_report,
        (r.normal(16, N), y),
        {"y_pred_quantiles": _quantile_dict(r, y), "n_bins": 5},
    )


def _distribution_report(r: _Rng) -> Built:
    y = r.normal(N)
    support, density = _grid(r)
    return (
        M.distribution_metrics_report,
        (),
        {
            "y_true": y,
            "y_pred_quantiles": _quantile_dict(r, y),
            "samples": r.normal(16, N),
            "support": support,
            "density": density,
        },
    )


def _ece_qdict(r: _Rng) -> Built:
    y = r.normal(N)
    return M.expected_calibration_error, (_quantile_dict(r, y), y), {}


def _crps_q(r: _Rng) -> Built:
    y = r.normal(N)
    return M.continuous_ranked_probability_score, (_quantile_dict(r, y), y), {}


def _pinball_q(r: _Rng) -> Built:
    y = r.normal(N)
    return M.pinball_metric, (_quantile_dict(r, y), y), {}


def _crps_cls(r: _Rng) -> Built:
    y = r.normal(N)
    return _MetricRunner(M.ContinuousRankedProbabilityScore()), (_quantile_dict(r, y), y), {}


def _pinball_cls(r: _Rng) -> Built:
    y = r.normal(N)
    return _MetricRunner(M.PinballMetric()), (_quantile_dict(r, y), y), {}


def _ece_cls(r: _Rng) -> Built:
    y = r.normal(N)
    return _MetricRunner(M.ExpectedCalibrationError()), (_quantile_dict(r, y), y), {}


def _ensemble_mv(r: _Rng) -> tuple[torch.Tensor, torch.Tensor]:
    return r.normal(5, N), r.positive(5, N)


def _typicality(r: _Rng) -> Built:
    # ``x`` is passed so that no random samples are drawn (RNG streams differ per device).
    return M.typicality_score, ((r.normal(N, D), r.positive(N, D)),), {"x": r.normal(N, D)}


def _interval_metric(fn: Callable[..., Any], *, y: bool, **kwargs: Any) -> Builder:
    def build(r: _Rng) -> Built:
        lo, hi, y_true = _interval(r)
        return fn, (lo, hi, y_true) if y else (lo, hi), kwargs

    return build


def _interval_cls(cls: Callable[..., Any], *, y: bool) -> Builder:
    def build(r: _Rng) -> Built:
        lo, hi, y_true = _interval(r)
        return _MetricRunner(cls()), (lo, hi, y_true) if y else (lo, hi), {}

    return build


def _sharpness(r: _Rng) -> Built:
    lo, hi, _ = _interval(r)
    return M.sharpness, (torch.stack([lo, hi], dim=-1),), {}


def _regress_report(r: _Rng) -> Built:
    return M.regression_metrics_report, (r.normal(N), r.normal(N)), {}


def _interval_report(r: _Rng) -> Built:
    lo, hi, y = _interval(r)
    return M.interval_metrics_report, ({"m": {"lower": lo, "upper": hi}}, y), {}


def _ood_report(r: _Rng) -> Built:
    return (
        M.ood_metrics_report,
        (),
        {
            "x_test": r.normal(N, D),
            "x_reference": r.normal(2 * N, D),
            "mean": r.normal(D),
            "cov": r.spd(1, D)[0],
            "samples": r.normal(16, N, D),
        },
    )


def _risk_cov_fn(r: _Rng) -> Built:
    return M.risk_coverage_curve, (r.normal(N), r.normal(N), r.positive(N)), {"n_points": 5}


def _ordinal_metric(fn: Callable[..., Any], **kw: Any) -> Builder:
    def build(r: _Rng) -> Built:
        return fn, (r.ints(N, high=5), r.ints(N, high=5)), kw

    return build


def _ordinal_logits(fn: Callable[..., Any], **kw: Any) -> Builder:
    def build(r: _Rng) -> Built:
        return fn, (r.normal(N, 4), r.ints(N, high=5)), {"encoding": "cumulative_logits", **kw}

    return build


def _censored_metric(fn: Callable[..., Any]) -> Builder:
    def build(r: _Rng) -> Built:
        return fn, (r.normal(N), r.positive(N)), {"censoring": r.ints(N, high=3) - 1}

    return build


def _mahalanobis_fn(r: _Rng) -> Built:
    return M.mahalanobis_distance, (r.normal(N, D), r.normal(D), r.spd(1, D)[0]), {}


def _mahalanobis_cls(r: _Rng) -> Built:
    return _MetricRunner(M.MahalanobisDistance()), (r.normal(N, D), r.normal(D), r.spd(1, D)[0]), {}


def _tac(r: _Rng) -> Built:
    return M.task_agnostic_correlations, (r.normal(N, D), r.normal(N, D), r.spd(N, D)), {}


def _wasserstein_gaussian(r: _Rng) -> Built:
    return M.wasserstein_gaussian_p2, (r.normal(N), r.positive(N), r.normal(N), r.positive(N)), {}


def _noisy_nll(r: _Rng) -> Built:
    return (
        M.noisy_target_gaussian_nll,
        (r.normal(N), r.positive(N), r.normal(N), 0.1 * r.positive(N)),
        {},
    )


def _uncertain_report(r: _Rng) -> Built:
    return (
        M.uncertain_gt_metrics_report,
        (),
        {
            "pred_mean": r.normal(N),
            "pred_variance": r.positive(N),
            "target": r.normal(N),
            "target_variance": 0.1 * r.positive(N),
            "teacher_pred": r.normal(N),
            "pseudo_confidence": r.uniform(N),
        },
    )


def _gnll_ens_cls(r: _Rng) -> Built:
    return _MetricRunner(M.GaussianNLLEnsemble()), (*_ensemble_mv(r), r.normal(N)), {}


def _eim_cls(r: _Rng) -> Built:
    return _MetricRunner(M.EnsembleIntervalMetrics(alpha=0.2)), (*_ensemble_mv(r), r.normal(N)), {}


def _rcc_cls(r: _Rng) -> Built:
    return (
        _MetricRunner(M.RiskCoverageCurve(n_points=5)),
        (r.normal(N), r.normal(N), r.positive(N)),
        {},
    )


def _rejection_cls(r: _Rng) -> Built:
    return (
        _MetricRunner(M.RejectionPolicy(fraction=0.5)),
        (r.normal(N), r.normal(N), r.positive(N)),
        {},
    )


def _pcls(cls: Callable[..., Any], *make: Callable[[_Rng], Any], **init: Any) -> Builder:
    return _stateful(cls, *make, **init)


METRIC_CASES: dict[str, Builder] = {
    # Point metrics (functions)
    "mean_squared_error": _metric(M.mean_squared_error, _n, _n),
    "mean_absolute_error": _metric(M.mean_absolute_error, _n, _n),
    "rmse": _metric(M.rmse, _n, _n),
    "median_absolute_error": _metric(M.median_absolute_error, _n, _n),
    "huber_loss": _metric(M.huber_loss, _n, _n, delta=0.7),
    "r2_score": _metric(M.r2_score, _n, _n),
    "bias": _metric(M.bias, _n, _n),
    "attenuation_factor": _metric(M.attenuation_factor, _n, _n),
    "trimmed_mean_squared_error": _metric(M.trimmed_mean_squared_error, _n, _n),
    "median_absolute_deviation": _metric(M.median_absolute_deviation, _n, _n),
    "normalized_rmse": _metric(M.normalized_rmse, _n, _n),
    "tail_mae": _metric(M.tail_mae, _n, _n, quantile=0.8),
    "tail_rmse": _metric(M.tail_rmse, _n, _n, quantile=0.8),
    "regression_metrics_report": _regress_report,
    # Point metrics (stateful classes)
    "HuberMetric": _pcls(M.HuberMetric, _n, _n),
    "MedianAbsoluteError": _pcls(M.MedianAbsoluteError, _n, _n),
    "MedianAbsoluteDeviation": _pcls(M.MedianAbsoluteDeviation, _n, _n),
    "NormalizedMedianAbsoluteDeviation": _pcls(M.NormalizedMedianAbsoluteDeviation, _n, _n),
    "NormalizedRMSE": _pcls(M.NormalizedRMSE, _n, _n),
    "OutlierFraction": _pcls(M.OutlierFraction, _n, _n),
    "TrimmedMeanSquaredError": _pcls(M.TrimmedMeanSquaredError, _n, _n),
    "MultivariateMAE": _pcls(M.MultivariateMAE, _nd, _nd),
    "MultivariateRMSE": _pcls(M.MultivariateRMSE, _nd, _nd),
    # Probabilistic scores
    "crps_gaussian": _metric(M.crps_gaussian, _n, _n, _pos),
    "gaussian_nll": _metric(M.gaussian_nll, _n, _n, _pos),
    "dss_score": _metric(M.dss_score, _n, _pos, _n),
    "crps_from_samples": _metric(M.crps_from_samples, _samples, _n),
    "energy_score": _metric(M.energy_score, _samples_nd, _nd),
    "variogram_score": _metric(M.variogram_score, _samples_nd, _nd),
    "vario_score": _metric(M.vario_score, _samples_nd, _nd),
    "continuous_ranked_probability_score": _crps_q,
    "pinball_metric": _pinball_q,
    "pinball_loss": _metric(M.pinball_loss, lambda r: 0.3, _n, _n),
    "wasserstein_gaussian_p2": _wasserstein_gaussian,
    "ContinuousRankedProbabilityScore": _crps_cls,
    "PinballMetric": _pinball_cls,
    "DawidSebastianiScore": _pcls(M.DawidSebastianiScore, _n, _pos, _n),
    "EnergyScore": _pcls(M.EnergyScore, _samples_nd, _nd),
    "VariogramScore": _pcls(M.VariogramScore, _samples_nd, _nd),
    "VarioScore": _pcls(M.VarioScore, _samples_nd, _nd),
    "WassersteinGaussian": _pcls(M.WassersteinGaussian, _n, _pos, _n),
    # Density / PIT / HPD
    "probability_integral_transform": _pit_case,
    "kolmogorov_smirnov_uniform_statistic": _ks_uniform,
    "highest_posterior_density_coverage": _hpd_cov,
    "highest_posterior_density_level": _hpd_level,
    "conditional_density_estimation_loss": _cde_loss,
    # Calibration
    "expected_calibration_error": _ece_qdict,
    "ExpectedCalibrationError": _ece_cls,
    "marginal_calibration_error": _metric(M.marginal_calibration_error, _samples, _n, n_bins=5),
    "MarginalCalibrationError": _pcls(M.MarginalCalibrationError, _samples, _n, n_bins=5),
    "calibration_score": _metric(M.calibration_score, _n, _n, _pos, n_levels=9),
    "calibration_metrics_report": _calibration_report,
    "distribution_metrics_report": _distribution_report,
    # Intervals
    "interval_score": _interval_metric(M.interval_score, y=True),
    "prediction_interval_coverage_probability": _interval_metric(
        M.prediction_interval_coverage_probability, y=True
    ),
    "prediction_interval_coverage": _interval_metric(M.prediction_interval_coverage, y=True),
    "sharpness": _sharpness,
    "interval_metrics_report": _interval_report,
    "IntervalScore": _interval_cls(M.IntervalScore, y=True),
    "PredictionIntervalCoverageProbability": _interval_cls(
        M.PredictionIntervalCoverageProbability, y=True
    ),
    "MeanPredictionIntervalWidth": _interval_cls(M.MeanPredictionIntervalWidth, y=False),
    "Sharpness": _interval_cls(M.Sharpness, y=False),
    # Ensembles
    "ensemble_mean": _metric(M.ensemble_mean, lambda r: r.normal(5, N)),
    "ensemble_std": _metric(M.ensemble_std, lambda r: r.normal(5, N)),
    "ensemble_statistics": _metric(M.ensemble_statistics, lambda r: r.normal(5, N)),
    "ensemble_variance_decomposition": _metric(
        M.ensemble_variance_decomposition, lambda r: r.normal(5, N), lambda r: r.positive(5, N)
    ),
    "uncertainty_decomposition": _metric(
        M.uncertainty_decomposition, lambda r: r.normal(5, N), lambda r: r.positive(5, N)
    ),
    "gaussian_nll_ensemble": _metric(
        M.gaussian_nll_ensemble, lambda r: r.normal(5, N), lambda r: r.positive(5, N), _n
    ),
    "ensemble_interval_bounds": _metric(
        M.ensemble_interval_bounds, lambda r: r.normal(5, N), lambda r: r.positive(5, N), alpha=0.2
    ),
    "ensemble_interval_metrics": _metric(
        M.ensemble_interval_metrics,
        lambda r: r.normal(5, N),
        lambda r: r.positive(5, N),
        _n,
        alpha=0.2,
    ),
    "GaussianNLLEnsemble": _gnll_ens_cls,
    "EnsembleIntervalMetrics": _eim_cls,
    # Selective prediction
    "risk_coverage_curve": _risk_cov_fn,
    "RiskCoverageCurve": _rcc_cls,
    "RejectionPolicy": _rejection_cls,
    # OOD
    "mahalanobis_distance": _mahalanobis_fn,
    "MahalanobisDistance": _mahalanobis_cls,
    "entropy_score": _metric(M.entropy_score, _samples_nd, n_bins=5),
    "EntropyScore": _pcls(M.EntropyScore, _samples_nd, n_bins=5),
    "kernel_density_score": _metric(
        M.kernel_density_score, _nd, lambda r: r.normal(2 * N, D), bandwidth=0.8
    ),
    "KernelDensityScore": _pcls(
        M.KernelDensityScore, _nd, lambda r: r.normal(2 * N, D), bandwidth=0.8
    ),
    "typicality_score": _typicality,
    "ood_metrics_report": _ood_report,
    # Multivariate
    "task_agnostic_correlations": _tac,
    # Ordinal
    "ordinal_accuracy": _ordinal_metric(M.ordinal_accuracy),
    "mean_absolute_class_error": _ordinal_metric(M.mean_absolute_class_error),
    "quadratic_weighted_kappa": _ordinal_metric(M.quadratic_weighted_kappa, num_classes=5),
    "ordinal_accuracy[cumulative_logits]": _ordinal_logits(M.ordinal_accuracy),
    # Censored
    "observed_mae": _censored_metric(M.observed_mae),
    "concordance_index": _censored_metric(M.concordance_index),
    "censoring_rate": _metric(M.censoring_rate, lambda r: r.ints(N, high=3) - 1),
    # Uncertain ground truth / semi-supervised
    "noisy_target_gaussian_nll": _noisy_nll,
    "consistency_error": _metric(M.consistency_error, _n, _n),
    "pseudo_label_acceptance_rate": _metric(M.pseudo_label_acceptance_rate, lambda r: r.uniform(N)),
    "uncertain_gt_metrics_report": _uncertain_report,
}

XFAIL_METRICS_PARITY: dict[str, str] = {}
XFAIL_METRICS_DTYPE: dict[str, str] = {}


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("name", "dtype"), _params(LOSS_CASES, XFAIL_LOSSES_PARITY))
def test_loss_value_and_gradient_parity(name: str, dtype: torch.dtype) -> None:
    _assert_parity(LOSS_CASES[name], dtype, grad=True)


@pytest.mark.parametrize(("name", "dtype"), _params(METRIC_CASES, XFAIL_METRICS_PARITY))
def test_metric_value_parity(name: str, dtype: torch.dtype) -> None:
    _assert_parity(METRIC_CASES[name], dtype, grad=False)


@pytest.mark.parametrize(("name", "dtype"), _params(LOSS_CASES, XFAIL_LOSSES_DTYPE))
def test_loss_outputs_on_target_device_with_input_dtype(name: str, dtype: torch.dtype) -> None:
    _assert_device_and_dtype(LOSS_CASES[name], dtype, grad=True)


@pytest.mark.parametrize(("name", "dtype"), _params(METRIC_CASES, XFAIL_METRICS_DTYPE))
def test_metric_outputs_on_target_device_with_input_dtype(name: str, dtype: torch.dtype) -> None:
    _assert_device_and_dtype(METRIC_CASES[name], dtype, grad=False)


def test_case_tables_are_large_enough() -> None:
    assert len({n.split("[")[0] for n in LOSS_CASES}) >= 40
    assert len({n.split("[")[0] for n in METRIC_CASES}) >= 25
