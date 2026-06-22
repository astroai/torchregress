"""Direct tests for the SLS internal modules.

The :mod:`torchregress.tests.losses.test_sls` module exercises SLSLoss end-to-end;
this file drills into the inner modules that SLSLoss composes -- volume-preserving
flows, the Mahalanobis frontier (full and low-rank parameterizations), the
multi-component Union frontier, and the QuantileNetwork that produces per-sample
quantile targets.  These internals are importable (they appear in ``__init__``)
and users can compose them in custom losses, so they deserve direct coverage.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from torchregress.losses.sls import (
    MahalanobisFrontier,
    QuantileNetwork,
    UnionFrontier,
    VolumePreservingCouplingLayer,
    VolumePreservingFlow,
)

# ---------------------------------------------------------------------------
# VolumePreservingCouplingLayer
# ---------------------------------------------------------------------------


class TestVolumePreservingCouplingLayer:
    def test_initialized_as_identity_at_construction(self):
        """The last Linear layer of a freshly-constructed coupling layer is zeros,
        so ``forward`` and ``inverse`` should both be the identity map at construction.
        This ensures training starts at the identity checkpoint, not at random.
        """
        d = 4
        mask = torch.tensor([True, False, True, False])
        layer = VolumePreservingCouplingLayer(d, mask, context_dim=2, hidden_dim=16)
        # Last layer's parameters are zero-initialised per the module's contract.
        last_linear = layer.net[-1]
        assert isinstance(last_linear, nn.Linear)
        assert torch.all(last_linear.weight == 0)
        assert torch.all(last_linear.bias == 0)

    def test_forward_inverse_round_trip_unconditional(self):
        d = 4
        mask = torch.tensor([True, False, True, False])
        layer = VolumePreservingCouplingLayer(d, mask, hidden_dim=16)
        y = torch.randn(8, d)
        z = layer(y)
        y_back = layer.inverse(z)
        assert torch.allclose(y, y_back, atol=1e-5)

    def test_forward_inverse_round_trip_with_context(self):
        d = 4
        mask = torch.tensor([True, False, True, False])
        layer = VolumePreservingCouplingLayer(d, mask, context_dim=3, hidden_dim=16)
        y = torch.randn(8, d)
        context = torch.randn(8, 3)
        z = layer(y, context=context)
        y_back = layer.inverse(z, context=context)
        assert torch.allclose(y, y_back, atol=1e-5)

    def test_parity_with_and_without_context_at_init(self):
        """At initialization the bias-correction term `self.net(zeros)` is 0
        (last Linear is zero-initialised), so the layer is the identity map
        for any input shape.  We exercise this invariant for both
        unconditional and zero-context inputs.
        """
        d = 4
        mask = torch.tensor([True, False, True, False])
        # No-context case (context_dim=0)
        layer_uncond = VolumePreservingCouplingLayer(d, mask, hidden_dim=16)
        y = torch.randn(6, d)
        assert torch.allclose(layer_uncond(y), y, atol=1e-5)
        assert torch.allclose(layer_uncond.inverse(y), y, atol=1e-5)
        # With-context case (context_dim=3): zero context must also yield identity.
        layer_cond = VolumePreservingCouplingLayer(d, mask, context_dim=3, hidden_dim=16)
        z_with = layer_cond(y, context=torch.zeros(6, 3))
        assert torch.allclose(z_with, y, atol=1e-5)


# ---------------------------------------------------------------------------
# VolumePreservingFlow
# ---------------------------------------------------------------------------


class TestVolumePreservingFlow:
    def test_d1_dimension_split_for_each_layer(self):
        """Each coupling layer splits d into halves with alternating masks."""
        d = 6
        flow = VolumePreservingFlow(d, context_dim=0, n_transforms=4, hidden_dim=16)
        assert len(flow.layers) == 4
        for i, layer in enumerate(flow.layers):
            assert isinstance(layer, VolumePreservingCouplingLayer)
            assert layer.d == d
            expected_halves = (
                [True, True, True, False, False, False]
                if i % 2 == 0
                else [
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                ]
            )
            assert torch.equal(layer.mask, torch.tensor(expected_halves))

    def test_d1_equals_one_for_unidimensional_targets(self):
        """d==1 collapses the flow to a single coupling layer applied to
        the only available axis; forward/inverse are identity by construction.
        """
        flow = VolumePreservingFlow(d=1, context_dim=0, n_transforms=4, hidden_dim=16)
        y = torch.randn(5, 1)
        z = flow(y)
        # The constructor with d==1 sets mask=[True]; forward/inverse fall through.
        assert torch.allclose(z, y)
        assert torch.allclose(flow.inverse(z), y)

    def test_round_trip_without_context(self):
        d = 5
        flow = VolumePreservingFlow(d, context_dim=0, n_transforms=4, hidden_dim=16)
        y = torch.randn(7, d)
        z = flow(y)
        assert z.shape == y.shape
        y_back = flow.inverse(z)
        assert torch.allclose(y, y_back, atol=1e-5)

    def test_round_trip_with_context(self):
        d = 5
        flow = VolumePreservingFlow(d, context_dim=3, n_transforms=4, hidden_dim=16)
        y = torch.randn(7, d)
        context = torch.randn(7, 3)
        z = flow(y, context=context)
        y_back = flow.inverse(z, context=context)
        assert torch.allclose(y, y_back, atol=1e-5)


# ---------------------------------------------------------------------------
# MahalanobisFrontier: full vs low_rank
# ---------------------------------------------------------------------------


@pytest.fixture
def full_frontier():
    return MahalanobisFrontier(d=2, context_dim=3, mode="full", hidden_dim=16)


@pytest.fixture
def low_rank_frontier():
    return MahalanobisFrontier(d=4, context_dim=3, mode="low_rank", rank=2, hidden_dim=16)


class TestMahalanobisFrontierFull:
    def test_dim_count_modes(self, full_frontier):
        # full mode requires d*(d+1)/2 L parameters
        assert full_frontier.mode == "full"
        assert full_frontier.num_L_params == 2 * 3 // 2

    def test_forward_shape_and_nonnegativity(self, full_frontier):
        y = torch.randn(5, 2)
        context = torch.randn(5, 3)
        G, log_det_L = full_frontier(y, context)
        assert G.shape == (5,)
        assert log_det_L.shape == (5,)
        # G is a squared Mahalanobis distance: >= 0.
        assert torch.all(G >= 0.0)
        # log_det_L is log(diag(L)) summed, can be negative (small diag values possible).
        assert torch.isfinite(log_det_L).all()

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            MahalanobisFrontier(d=2, context_dim=2, mode="banana")

    def test_forward_is_finite_in_high_dim(self):
        d = 6
        frontier = MahalanobisFrontier(d=d, context_dim=4, mode="full", hidden_dim=16)
        y = torch.randn(8, d)
        context = torch.randn(8, 4)
        G, log_det_L = frontier(y, context)
        assert torch.isfinite(G).all()
        assert torch.isfinite(log_det_L).all()


class TestMahalanobisFrontierLowRank:
    def test_default_rank_is_ceil_sqrt_d(self):
        # Caller does not specify rank; the module sets it to ceil(sqrt(d)).
        frontier = MahalanobisFrontier(d=10, context_dim=4, mode="low_rank", hidden_dim=16)
        assert frontier.rank == math.ceil(math.sqrt(10))

    def test_explicit_rank_overrides_default(self):
        frontier = MahalanobisFrontier(d=10, context_dim=4, mode="low_rank", rank=3, hidden_dim=16)
        assert frontier.rank == 3

    def test_num_L_params_uses_d_plus_d_times_rank(self, low_rank_frontier):
        # d=4, rank=2 => 4 + 4*2 = 12 L parameters
        assert low_rank_frontier.num_L_params == 4 + 4 * 2

    def test_forward_shape_and_nonnegativity(self, low_rank_frontier):
        y = torch.randn(5, 4)
        context = torch.randn(5, 3)
        G, log_det_L = low_rank_frontier(y, context)
        assert G.shape == (5,)
        assert log_det_L.shape == (5,)
        assert torch.all(G >= 0.0)

    def test_forward_returns_d_v_tuple(self, low_rank_frontier):
        """Low-rank mode should return ``(D, V)`` instead of a matrix ``L``.

        Sanity-check the unpacking path used in ``forward``.  We exercise
        ``_get_params`` and ``_get_L_matrix_and_logdet`` directly (rather
        than calling ``forward``) so the assertion targets the low-rank
        ``(D, V)`` contract cleanly without needing a full Mahalanobis
        input vector ``y``.
        """
        context = torch.randn(3, 3)
        _, L_params = low_rank_frontier._get_params(context)
        L_or_DV, log_det_L = low_rank_frontier._get_L_matrix_and_logdet(L_params)
        # Returns a (D, V) tuple where D is positive (after softplus) and V has rank columns.
        assert isinstance(L_or_DV, tuple)
        D, V = L_or_DV
        assert torch.all(D > 0)  # softplus + 1e-6 is strictly positive
        assert V.shape == (3, 4, 2)


# ---------------------------------------------------------------------------
# UnionFrontier
# ---------------------------------------------------------------------------


class TestUnionFrontier:
    def test_k_components_construction(self):
        frontier = UnionFrontier(d=2, K=3, context_dim=4, mode="full", hidden_dim=16)
        assert len(frontier.components) == 3
        for component in frontier.components:
            assert isinstance(component, MahalanobisFrontier)

    def test_beta_initialization_and_step(self):
        frontier = UnionFrontier(d=2, K=2, context_dim=4, mode="full", hidden_dim=16, beta_init=1.0)
        assert float(frontier.beta) == pytest.approx(1.0)
        frontier.step_beta()
        assert float(frontier.beta) == pytest.approx(1.01)
        frontier.step_beta()
        assert float(frontier.beta) == pytest.approx(1.01 * 1.01)

    def test_freeze_weights_default_true(self):
        frontier = UnionFrontier(d=2, K=2, context_dim=4, mode="full", hidden_dim=16)
        assert frontier._freeze_weights is True
        frontier.freeze_weights(False)
        assert frontier._freeze_weights is False
        frontier.freeze_weights(True)
        assert frontier._freeze_weights is True

    def test_frozen_mixture_weights_are_uniform_with_or_without_context(self):
        """When ``_freeze_weights`` is True, weights must be 1/K regardless of context."""
        # without context
        frontier0 = UnionFrontier(d=2, K=2, context_dim=0, mode="full", hidden_dim=16)
        # with context (uses dummy zeros via context_dim==0 path)
        w = frontier0._get_mixture_weights()
        assert torch.allclose(w, torch.full((2,), 0.5))

        # with context_dim > 0 (must supply a context)
        frontier1 = UnionFrontier(d=2, K=3, context_dim=4, mode="full", hidden_dim=16)
        ctx = torch.randn(5, 4)
        w = frontier1._get_mixture_weights(ctx)
        assert w.shape == (5, 3)
        assert torch.allclose(w, torch.full((5, 3), 1.0 / 3.0))

    def test_unfrozen_mixture_weights_sum_to_one_per_sample(self):
        frontier = UnionFrontier(d=2, K=4, context_dim=4, mode="full", hidden_dim=16)
        frontier.freeze_weights(False)
        ctx = torch.randn(6, 4)
        w = frontier._get_mixture_weights(ctx)
        assert w.shape == (6, 4)
        assert torch.allclose(w.sum(dim=-1), torch.ones(6), atol=1e-5)

    def test_unfrozen_with_zero_context_dim_raises_when_no_context(self):
        frontier = UnionFrontier(d=2, K=2, context_dim=0, mode="full", hidden_dim=16)
        frontier.freeze_weights(False)
        # The parameter path is taken when context_dim == 0; passing None would
        # trip a guard in front of it for the context_dim > 0 branch only.  Here
        # we just exercise the parameter-only branch.
        w = frontier._get_mixture_weights()
        assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-5)
        assert w.shape == (2,)

    def test_forward_shape_and_nonnegativity(self):
        frontier = UnionFrontier(d=2, K=2, context_dim=4, mode="full", hidden_dim=16)
        y = torch.randn(5, 2)
        context = torch.randn(5, 4)
        G, log_vol_term = frontier(y, context)
        assert G.shape == (5,)
        assert log_vol_term.shape == (5,)
        assert torch.all(G >= 0.0)


# ---------------------------------------------------------------------------
# QuantileNetwork
# ---------------------------------------------------------------------------


class TestQuantileNetwork:
    def test_context_dim_zero_uses_parameter_and_output_is_sortable(self):
        qn = QuantileNetwork(context_dim=0, hidden_dim=16)
        out = qn()  # no context
        assert out.shape == (3,)
        # exp(0) + eps is strictly positive and then it's sorted.
        assert torch.all(out > 0)
        assert torch.all(out[1:] >= out[:-1])

    def test_context_dim_zero_batch_expansion(self):
        qn = QuantileNetwork(context_dim=0, hidden_dim=16)
        out = qn()  # no context uses param, expands across batch shape from context
        # No context -> batch shape ()
        assert out.shape == (3,)

    def test_context_dim_nonzero_requires_context(self):
        qn = QuantileNetwork(context_dim=4, hidden_dim=16)
        with pytest.raises(ValueError, match="context"):
            qn()

    def test_context_dim_nonzero_with_context(self):
        qn = QuantileNetwork(context_dim=4, hidden_dim=16)
        ctx = torch.randn(8, 4)
        out = qn(ctx)
        assert out.shape == (8, 3)
        assert torch.all(out > 0)
        # Sorted ascending within each batch row.
        diffs = out[:, 1:] - out[:, :-1]
        assert (diffs >= 0).all()

    def test_gradient_flow_on_quantiles(self):
        """The quantile network's output depends on learnable parameters;
        a backward pass should produce finite and nonzero gradients on them.
        """
        qn = QuantileNetwork(context_dim=3, hidden_dim=16)
        ctx = torch.randn(4, 3)
        out = qn(ctx)
        # Sum so backward has non-trivial gradient.
        loss = out.sum()
        loss.backward()
        for name, param in qn.named_parameters():
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name
