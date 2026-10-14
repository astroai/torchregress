"""Device / dtype placement of functional wrappers around ``torchmetrics.Metric``.

Functional wrappers (``gaussian_nll_ensemble``, ``dss_score`` ...) build a
``Metric`` whose states are created on the CPU in the default float dtype.  They
must move that metric to the inputs' device and dtype *before* ``update`` --
otherwise CUDA inputs fail with a device mismatch and float64 inputs are
silently accumulated (and returned) in float32.  Real CUDA coverage lives in
``tests/test_device_parity.py``; these tests run on CPU-only machines.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest
import torch
from torchmetrics import Metric

import torchregress.metrics.decision as decision_mod
import torchregress.metrics.distribution as distribution_mod
import torchregress.metrics.ensemble as ensemble_mod
import torchregress.metrics.point as point_mod
import torchregress.metrics.tac as tac_mod
from torchregress.metrics.ensemble import EnsembleIntervalMetrics
from torchregress.metrics.utils import float_dtype, prepare_functional_metric

N = 12
DTYPES = [torch.float64, torch.float32]


def _g(dtype: torch.dtype, *shape: int, positive: bool = False, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(*shape, generator=gen, dtype=torch.float64)
    return (x.abs() + 0.5 if positive else x).to(dtype)


def _cov(dtype: torch.dtype, b: int, d: int) -> torch.Tensor:
    a = _g(torch.float64, b, d, d, seed=3)
    return (a @ a.transpose(-1, -2) + torch.eye(d, dtype=torch.float64)).to(dtype)


# (module holding the wrapper's ``prepare_functional_metric`` name, wrapper, args factory)
Case = tuple[ModuleType, Callable[..., Any], Callable[[torch.dtype], tuple[Any, ...]]]
CASES: dict[str, Case] = {
    "gaussian_nll_ensemble": (
        ensemble_mod,
        ensemble_mod.gaussian_nll_ensemble,
        lambda t: (_g(t, 5, N), _g(t, 5, N, positive=True, seed=1), _g(t, N, seed=2)),
    ),
    "ensemble_interval_metrics": (
        ensemble_mod,
        ensemble_mod.ensemble_interval_metrics,
        lambda t: (_g(t, 5, N), _g(t, 5, N, positive=True, seed=1), _g(t, N, seed=2)),
    ),
    "task_agnostic_correlations": (
        tac_mod,
        tac_mod.task_agnostic_correlations,
        lambda t: (_g(t, 6, 3), _g(t, 6, 3, seed=1), _cov(t, 6, 3)),
    ),
    "risk_coverage_curve": (
        decision_mod,
        decision_mod.risk_coverage_curve,
        lambda t: (_g(t, N), _g(t, N, seed=1), _g(t, N, positive=True, seed=2)),
    ),
    "dss_score": (
        distribution_mod,
        distribution_mod.dss_score,
        lambda t: (_g(t, N), _g(t, N, positive=True, seed=1), _g(t, N, seed=2)),
    ),
    "vario_score": (
        distribution_mod,
        distribution_mod.vario_score,
        lambda t: (_g(t, 7, N), _g(t, N, seed=2)),
    ),
    "pinball_metric": (
        distribution_mod,
        distribution_mod.pinball_metric,
        lambda t: ({0.1: _g(t, N) - 1, 0.5: _g(t, N), 0.9: _g(t, N) + 1}, _g(t, N, seed=2)),
    ),
    "regression_metrics_report": (
        point_mod,
        point_mod.regression_metrics_report,
        lambda t: (_g(t, N), _g(t, N, seed=2)),
    ),
    "r2_score": (point_mod, point_mod.r2_score, lambda t: (_g(t, N), _g(t, N, seed=2))),
}


def _tensors(obj: Any) -> list[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, dict):
        return [t for v in obj.values() for t in _tensors(v)]
    if isinstance(obj, (list, tuple)):
        return [t for v in obj for t in _tensors(v)]
    return []


@pytest.mark.parametrize("dtype", DTYPES, ids=["f64", "f32"])
@pytest.mark.parametrize("name", list(CASES))
def test_wrapper_metric_is_placed_on_input_device_and_dtype_before_update(
    name: str, dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The metric the wrapper updates has every state on the inputs' device/dtype."""
    module, fn, make_args = CASES[name]
    args = make_args(dtype)
    device = _tensors(args)[0].device

    real_prepare = module.prepare_functional_metric
    prepared: list[Metric] = []
    states_at_update: list[tuple[torch.device, torch.dtype]] = []
    update_calls: list[Metric] = []

    def spying_prepare(metric: Metric, *inputs: Any) -> Metric:
        out = real_prepare(metric, *inputs)
        prepared.append(out)
        for m in out.modules():
            if isinstance(m, Metric):
                real_update = m.update

                def checked(*a: Any, _real: Any = real_update, _m: Metric = m, **k: Any) -> Any:
                    update_calls.append(_m)
                    for key in _m._defaults:
                        for t in _tensors(getattr(_m, key)):
                            states_at_update.append((t.device, t.dtype))
                    return _real(*a, **k)

                m.update = checked  # ty: ignore[invalid-assignment]
        return out

    monkeypatch.setattr(module, "prepare_functional_metric", spying_prepare)
    fn(*args)

    assert prepared, f"{name} did not route its Metric through prepare_functional_metric"
    assert update_calls, "update() was never called on a prepared metric"
    for state_device, state_dtype in states_at_update:
        assert state_device == device
        # Integer counters keep their exact int64 dtype unless the inputs force a conversion.
        assert state_dtype == dtype or not state_dtype.is_floating_point


@pytest.mark.parametrize("dtype", DTYPES, ids=["f64", "f32"])
def test_tensor_returning_wrappers_keep_input_dtype(dtype: torch.dtype) -> None:
    means, variances, y = (
        _g(dtype, 5, N),
        _g(dtype, 5, N, positive=True, seed=1),
        _g(dtype, N, seed=2),
    )

    nll = ensemble_mod.gaussian_nll_ensemble(means, variances, y)
    assert nll.dtype == dtype

    interval = ensemble_mod.ensemble_interval_metrics(means, variances, y)
    assert interval["interval_score"].dtype == dtype
    assert interval["picp"].dtype == dtype

    lower, upper = ensemble_mod.ensemble_interval_bounds(means, variances)
    assert lower.dtype == upper.dtype == dtype

    tac = tac_mod.task_agnostic_correlations(
        _g(dtype, 6, 3), _g(dtype, 6, 3, seed=1), _cov(dtype, 6, 3)
    )
    assert tac.dtype == dtype

    curve = decision_mod.risk_coverage_curve(
        _g(dtype, N), _g(dtype, N, seed=1), _g(dtype, N, seed=2)
    )
    assert curve["coverage"].dtype == curve["risk"].dtype == curve["aurc"].dtype == dtype


def test_float64_wrapper_matches_direct_computation() -> None:
    """float64 inputs are accumulated in float64 (no float32 round-trip)."""
    means = _g(torch.float64, 5, N)
    variances = _g(torch.float64, 5, N, positive=True, seed=1)
    y = _g(torch.float64, N, seed=2)
    total_var = means.var(dim=0, unbiased=False) + variances.mean(dim=0)
    expected = (
        0.5 * (torch.log(2 * torch.pi * total_var) + (y - means.mean(0)) ** 2 / total_var)
    ).mean()
    torch.testing.assert_close(
        ensemble_mod.gaussian_nll_ensemble(means, variances, y), expected, rtol=1e-12, atol=1e-12
    )


def test_prepare_functional_metric_moves_metric_and_children_to_input_device() -> None:
    """Runs against the ``meta`` device so the device move is observable without a GPU."""
    ref = torch.empty(4, dtype=torch.float64, device="meta")
    metric = prepare_functional_metric(EnsembleIntervalMetrics(), ref)
    seen = 0
    for m in metric.modules():
        if isinstance(m, Metric):
            for key in m._defaults:
                state = getattr(m, key)
                assert state.device.type == "meta"
                assert state.dtype == torch.float64
                seen += 1
    assert seen == 4  # children carry score/total and covered/total


def test_prepare_functional_metric_finds_tensors_in_nested_inputs() -> None:
    quantiles = {0.5: torch.empty(3, dtype=torch.float64, device="meta")}
    metric = prepare_functional_metric(distribution_mod.PinballMetric(), quantiles)
    assert metric.loss_sum.device.type == "meta"  # ty: ignore[unresolved-attribute]
    assert metric.loss_sum.dtype == torch.float64  # ty: ignore[unresolved-attribute]


def test_prepare_functional_metric_keeps_default_dtype_states_and_without_tensors() -> None:
    metric = prepare_functional_metric(distribution_mod.PinballMetric(), torch.zeros(3))
    assert metric.total.dtype == torch.int64  # ty: ignore[unresolved-attribute]  # exact counter
    untouched = prepare_functional_metric(distribution_mod.PinballMetric())
    assert untouched.loss_sum.dtype == torch.get_default_dtype()  # ty: ignore[unresolved-attribute]


def test_float_dtype_promotes_floating_inputs_and_falls_back_to_default() -> None:
    f32, f64 = torch.zeros(1), torch.zeros(1, dtype=torch.float64)
    assert float_dtype(f32, f64) == torch.float64
    assert float_dtype(f32) == torch.float32
    assert float_dtype(torch.zeros(1, dtype=torch.int64), torch.zeros(1, dtype=torch.bool)) == (
        torch.get_default_dtype()
    )
    assert float_dtype() == torch.get_default_dtype()
