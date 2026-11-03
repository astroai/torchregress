"""Flow sampling and spline inverses must not strand tensors in reference cycles.

torch's ``Transform.inv`` caches its inverse with a strong reference, so each
``t.inv(y)`` on a freshly built transform leaves a ``t <-> inverse`` cycle
holding the transform's parameters.  zuko builds such transforms per sampling
call and per autoregressive pass; without an explicit young-generation
collection the spline parameters (about 11 KiB per 6-D NSF draw) stay on the
device until Python's object-count-driven GC runs, which on CUDA meant an
out-of-memory error after a few chunks of conformal-density evaluation.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from typing import Any

import pytest
import torch

pytest.importorskip("zuko")

from torchregress.losses import ContrastiveFlowLoss, NormalizingFlowLoss, create_flow_model
from torchregress.models.copula_spline import GaussianCopulaSplineHead
from torchregress.utils.pytorch_compat import _collect_transform_cycles


def _cyclic_tensor_bytes(fn: Callable[[], Any]) -> int:
    """Bytes of tensors that only a cyclic collection would free after ``fn()``."""
    fn()  # warm-up: first calls import lazily and leave unrelated garbage
    gc.collect()
    result = fn()
    flags = gc.get_debug()
    gc.set_debug(gc.DEBUG_SAVEALL)
    try:
        gc.collect()
        found = sum(
            obj.untyped_storage().nbytes() for obj in gc.garbage if isinstance(obj, torch.Tensor)
        )
    finally:
        gc.garbage.clear()
        gc.set_debug(flags)
    del result
    return found


def _flow_loss(kind: type, flow_type: str) -> NormalizingFlowLoss:
    torch.manual_seed(0)
    flow = create_flow_model(
        n_features=6, context_dim=16, flow_type=flow_type, n_transforms=3, hidden_features=[16, 16]
    )
    return kind(flow=flow)


@pytest.mark.parametrize("kind", [NormalizingFlowLoss, ContrastiveFlowLoss])
@pytest.mark.parametrize("flow_type", ["nsf", "maf"])
@pytest.mark.parametrize("n_samples", [1, 64])
def test_flow_sample_leaves_no_cyclic_tensors(kind: type, flow_type: str, n_samples: int) -> None:
    loss = _flow_loss(kind, flow_type)
    context = torch.randn(32, 16)
    with torch.no_grad():
        stranded = _cyclic_tensor_bytes(lambda: loss.sample(context, n_samples=n_samples))
    assert stranded == 0


@pytest.mark.parametrize("n_samples", [1, 5])
def test_flow_sample_matches_direct_distribution_draws(n_samples: int) -> None:
    loss = _flow_loss(NormalizingFlowLoss, "nsf")
    context = torch.randn(8, 16)
    with torch.no_grad():
        torch.manual_seed(3)
        got = loss.sample(context, n_samples=n_samples)
        torch.manual_seed(3)
        ref = loss.flow(context).sample((n_samples,)).transpose(0, 1)
    if n_samples == 1:
        ref = ref.squeeze(1)
    assert torch.equal(got, ref)


def test_copula_spline_log_probs_leave_no_cyclic_tensors() -> None:
    torch.manual_seed(1)
    head = GaussianCopulaSplineHead(8, 4, width=16)
    context = torch.randn(64, 8)
    value = torch.randn(64, 4)
    mask = torch.rand(64, 4) > 0.3
    sigma = torch.full_like(value, 0.1)
    nodes = torch.randn(8, 4)

    def intrinsic() -> None:
        head.zero_grad()
        head.intrinsic_log_prob(context, value, mask).sum().backward()

    def observed() -> None:
        head.zero_grad()
        head.observed_log_prob(context, value, sigma, mask, nodes)[0].sum().backward()

    assert _cyclic_tensor_bytes(intrinsic) == 0
    assert _cyclic_tensor_bytes(observed) == 0


@pytest.mark.parametrize("enabled", [True, False])
def test_collect_transform_cycles_restores_gc_state(enabled: bool) -> None:
    was_enabled = gc.isenabled()
    try:
        gc.enable() if enabled else gc.disable()
        with _collect_transform_cycles():
            assert not gc.isenabled()
        assert gc.isenabled() is enabled
        with pytest.raises(RuntimeError, match="boom"):
            with _collect_transform_cycles():
                raise RuntimeError("boom")
        assert gc.isenabled() is enabled
    finally:
        gc.enable() if was_enabled else gc.disable()
