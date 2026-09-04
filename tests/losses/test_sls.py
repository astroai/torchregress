from __future__ import annotations

import pytest
import torch

from torchregress.losses.sls import (
    _SLS_STEP_CAP,
    MahalanobisFrontier,
    SLSLoss,
    UnionFrontier,
    VolumePreservingCouplingLayer,
    VolumePreservingFlow,
    sigmoidal_schedule,
)


def test_volume_preserving_coupling_layer_invertibility() -> None:
    d = 4
    mask = torch.tensor([True, True, False, False])
    layer = VolumePreservingCouplingLayer(d, mask, context_dim=2, hidden_dim=16)

    y = torch.randn(5, d)
    context = torch.randn(5, 2)

    # Forward
    z = layer(y, context=context)
    assert z.shape == y.shape

    # Inverse
    y_rec = layer.inverse(z, context=context)
    assert y_rec.shape == y.shape
    assert torch.allclose(y, y_rec, atol=1e-5)


def test_volume_preserving_flow_invertibility() -> None:
    d = 3
    flow = VolumePreservingFlow(d, context_dim=2, n_transforms=4, hidden_dim=16)

    y = torch.randn(10, d)
    context = torch.randn(10, 2)

    # Forward
    z = flow(y, context=context)
    assert z.shape == y.shape

    # Inverse
    y_rec = flow.inverse(z, context=context)
    assert torch.allclose(y, y_rec, atol=1e-5)


def test_mahalanobis_frontier_full_and_low_rank() -> None:
    # Full mode
    frontier_full = MahalanobisFrontier(d=2, context_dim=3, mode="full", hidden_dim=16)
    y = torch.randn(5, 2)
    context = torch.randn(5, 3)

    G, logdet = frontier_full(y, context)
    assert G.shape == (5,)
    assert logdet.shape == (5,)
    assert torch.all(G >= 0.0)

    # Low rank mode
    frontier_lr = MahalanobisFrontier(d=4, context_dim=3, mode="low_rank", rank=2, hidden_dim=16)
    y_4d = torch.randn(5, 4)
    G_lr, logdet_lr = frontier_lr(y_4d, context)
    assert G_lr.shape == (5,)
    assert logdet_lr.shape == (5,)
    assert torch.all(G_lr >= 0.0)


def test_union_frontier() -> None:
    frontier = UnionFrontier(d=2, K=3, context_dim=4, mode="full", hidden_dim=16)
    y = torch.randn(5, 2)
    context = torch.randn(5, 4)

    G, log_vol_term = frontier(y, context)
    assert G.shape == (5,)
    assert log_vol_term.shape == (5,)
    assert torch.all(G >= 0.0)


def test_sigmoidal_schedule() -> None:
    # Warmup phase
    val = sigmoidal_schedule(step=10, warmup_steps=50, init_val=0.5, min_val=0.05)
    assert val == 0.5

    # Post warmup decay
    val_post = sigmoidal_schedule(step=100, warmup_steps=50, init_val=0.5, min_val=0.05, t0=100.0)
    assert val_post < 0.5
    assert val_post >= 0.05


def test_sls_loss_warmup_and_forward() -> None:
    # 2D target, 3D context
    loss_fn = SLSLoss(d=2, context_dim=3, K=1, warmup_steps=5, reduction="mean")

    # Check that parameters are initialized
    assert len(list(loss_fn.parameters())) > 0

    x = torch.randn(10, 3, requires_grad=True)
    y = torch.randn(10, 2)

    # Step 1: Warmup phase forward and backward
    # Counter starts at 0. Calling forward increments to 1.
    loss_warmup = loss_fn(x, y)
    assert torch.isfinite(loss_warmup)
    loss_warmup.backward()
    assert x.grad is not None

    # Step 2: Push beyond warmup to trigger shrinking window
    loss_fn.step_counter = 10  # Manually advance counter
    x.grad = None
    loss_window = loss_fn(x, y)
    assert torch.isfinite(loss_window)
    loss_window.backward()
    assert x.grad is not None


def test_sls_loss_union_warmup_and_forward() -> None:
    # 2D target, 3D context, Union of K=2 flows
    loss_fn = SLSLoss(d=2, context_dim=3, K=2, warmup_steps=5, reduction="sum")

    x = torch.randn(10, 3, requires_grad=True)
    y = torch.randn(10, 2)

    loss_warmup = loss_fn(x, y)
    assert torch.isfinite(loss_warmup)
    loss_warmup.backward()
    assert x.grad is not None


# ---------------------------------------------------------------------------
# P0 (SLS counter fragility): explicit-step purity + bounded bookkeeping.
# Proposal: decouple step tracking from the loss module — pass ``step``
# explicitly from the training loop; the loss must not mutate internal state
# on that path, and internal bookkeeping must be bounded.
# ---------------------------------------------------------------------------


class TestSLSLossExplicitStepPurity:
    """With explicit ``step=``, SLSLoss must be side-effect free.

    ``forward`` / ``forward_frontier`` / ``forward_quantiles`` must not touch
    ``step_counter`` when the caller owns the schedule: repeated evaluation of
    the same step (validation loops, multi-GPU rank splitting, grad
    accumulation) must be deterministic and distributed-safe.
    """

    def test_explicit_step_does_not_mutate_counter(self) -> None:
        loss_fn = SLSLoss(d=2, context_dim=3, K=1, warmup_steps=5)
        torch.manual_seed(0)
        y_pred = torch.randn(4, 3)
        target = torch.randn(4, 2)
        for s in (0, 1, 7, 100):
            _ = loss_fn(y_pred, target, step=s)
            assert loss_fn.step_counter == 0, (
                f"explicit step={s} must not mutate step_counter; got {loss_fn.step_counter}"
            )

    def test_explicit_step_frontier_quantiles_pure(self) -> None:
        loss_fn = SLSLoss(d=2, context_dim=3, K=2, warmup_steps=5)
        torch.manual_seed(0)
        y_pred = torch.randn(4, 3)
        target = torch.randn(4, 2)
        before = loss_fn.step_counter
        _ = loss_fn.forward_frontier(y_pred, target, step=3)
        _ = loss_fn.forward_quantiles(y_pred, target, step=3)
        assert loss_fn.step_counter == before, (
            "forward_frontier/forward_quantiles must be pure in explicit step"
        )

    def test_explicit_step_deterministic(self) -> None:
        """Same explicit step + same inputs => bitwise-identical loss."""
        loss_fn = SLSLoss(d=2, context_dim=3, K=2, warmup_steps=5)
        loss_fn.eval()
        torch.manual_seed(0)
        y_pred = torch.randn(4, 3)
        target = torch.randn(4, 2)
        a = loss_fn(y_pred, target, step=42)
        b = loss_fn(y_pred, target, step=42)
        assert torch.equal(a, b), "repeated same-step evaluation must be deterministic"

    def test_stepless_fallback_advances_once_per_call(self) -> None:
        """B1 contract kept: step-less calls advance the counter once each."""
        loss_fn = SLSLoss(d=2, context_dim=3, K=1, warmup_steps=5)
        torch.manual_seed(0)
        y_pred = torch.randn(4, 3)
        target = torch.randn(4, 2)
        for expected in (1, 2, 3):
            _ = loss_fn(y_pred, target)
            assert loss_fn.step_counter == expected

    def test_stepless_counter_saturates(self) -> None:
        """The internal counter must be bounded (no int overflow on long runs)."""
        loss_fn = SLSLoss(d=2, context_dim=3, K=1, warmup_steps=5)
        loss_fn.step_counter = _SLS_STEP_CAP
        torch.manual_seed(0)
        y_pred = torch.randn(4, 3)
        target = torch.randn(4, 2)
        _ = loss_fn(y_pred, target)
        assert loss_fn.step_counter == _SLS_STEP_CAP, "counter must saturate at the cap"

    def test_forward_frontier_never_mutates_counter(self) -> None:
        """forward_frontier alone (step-less) must not advance the counter:
        only ``forward`` owns bookkeeping (single point of mutation)."""
        loss_fn = SLSLoss(d=2, context_dim=3, K=1, warmup_steps=5)
        torch.manual_seed(0)
        y_pred = torch.randn(4, 3)
        target = torch.randn(4, 2)
        _ = loss_fn.forward_frontier(y_pred, target)
        assert loss_fn.step_counter == 0, (
            "forward_frontier must not mutate the counter; ``forward`` is "
            "the single point of advancement"
        )


class TestUnionFrontierBetaBound:
    """K>1 frontier: geometric beta anneal must be clamped.

    Unbounded 1.01^n growth overflows float32 (~9k steps -> beta=inf ->
    logits -inf -> softmax NaN), silently poisoning long training runs.
    """

    def test_step_beta_clamped(self) -> None:
        frontier = UnionFrontier(d=2, K=2, context_dim=3)
        frontier.beta.fill_(frontier.beta_cap)
        frontier.step_beta()
        assert float(frontier.beta) == frontier.beta_cap, "beta must clamp at beta_cap"
        assert torch.isfinite(frontier.beta)

    def test_step_beta_geometric_then_clamped(self) -> None:
        frontier = UnionFrontier(d=2, K=2, context_dim=3, beta_init=1.0)
        for _ in range(50):
            frontier.step_beta()
        assert float(frontier.beta) == pytest.approx(1.01**50, rel=1e-5)
        for _ in range(100_000):
            frontier.step_beta()
        assert float(frontier.beta) == frontier.beta_cap
        assert torch.isfinite(frontier.beta)

    def test_sls_long_run_union_frontier_stays_finite(self) -> None:
        """Simulated long run past warmup: K>1 forward must stay finite.

        Pre-fix, ``forward`` multiplied the beta buffer by 1.01 on every
        post-warmup call — float32 overflow to inf after ~9k steps, softmax
        NaN. Now the effective beta is derived per-step (buffer untouched).
        """
        loss_fn = SLSLoss(d=2, context_dim=3, K=2, warmup_steps=5)
        torch.manual_seed(0)
        y_pred = torch.randn(4, 3)
        target = torch.randn(4, 2)
        for s in range(6, 20_000, 997):
            loss = loss_fn(y_pred, target, step=s)
            assert torch.isfinite(loss), f"loss diverged at step {s}"
        # Buffer is no longer mutated by forward (derived state).
        assert float(loss_fn.frontier.beta) == loss_fn.frontier.beta_init
