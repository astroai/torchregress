"""
Unit tests for torchregress.inference.ratio.

All tests are deterministic (fixed tensors, no RNG) and use float64 where a tight
tolerance is required.
"""

from __future__ import annotations

import math

import pytest
import torch

from torchregress.inference import ratio
from torchregress.inference.ratio import (
    effective_sample_size,
    grid_log_posterior,
    maximum_likelihood_estimate,
    posterior_log_weights,
)

F64 = torch.float64


# ═══════════════════════════════════════════════════════════════════════════════
# posterior_log_weights
# ═══════════════════════════════════════════════════════════════════════════════


class TestPosteriorLogWeights:
    def test_probabilities_sum_to_one(self) -> None:
        """exp(log_weights) sums to one per bank; log weights are normalized."""
        log_ratio = torch.tensor([[0.0, 1.0, -2.0], [3.0, -1.0, 0.5]], dtype=F64)
        log_weights = posterior_log_weights(log_ratio)
        assert log_weights.shape == log_ratio.shape
        torch.testing.assert_close(log_weights.exp().sum(dim=-1), torch.ones(2, dtype=F64))
        torch.testing.assert_close(
            torch.logsumexp(log_weights, dim=-1),
            torch.zeros(2, dtype=F64),
            atol=1e-12,
            rtol=0.0,
        )

    def test_matches_manual_log_softmax(self) -> None:
        """Result equals log_softmax(log_ratio + log_prior - log_reference)."""
        log_ratio = torch.tensor([[0.2, -1.5, 0.0], [-0.3, 2.0, 0.1]], dtype=F64)
        torch.testing.assert_close(
            posterior_log_weights(log_ratio), torch.log_softmax(log_ratio, dim=-1)
        )
        log_prior = torch.tensor([0.0, 0.5, -1.0], dtype=F64)
        log_reference = torch.tensor([-0.5, 0.0, 2.0], dtype=F64)
        expected = torch.log_softmax(log_ratio + log_prior - log_reference, dim=-1)
        torch.testing.assert_close(
            posterior_log_weights(log_ratio, log_prior=log_prior, log_reference=log_reference),
            expected,
        )

    def test_constant_corrections_are_noops(self) -> None:
        """Constants cancel in the softmax (uniform prior/proposal corrections)."""
        log_ratio = torch.tensor([0.2, -1.0, 2.0], dtype=F64)
        base = posterior_log_weights(log_ratio)
        zero = torch.zeros(3, dtype=F64)
        torch.testing.assert_close(posterior_log_weights(log_ratio, log_prior=zero), base)
        torch.testing.assert_close(posterior_log_weights(log_ratio, log_reference=zero), base)
        constant = torch.full((3,), 2.5, dtype=F64)
        torch.testing.assert_close(
            posterior_log_weights(log_ratio, log_prior=constant, log_reference=constant), base
        )
        torch.testing.assert_close(
            posterior_log_weights(log_ratio, log_prior=1.7, log_reference=1.7), base
        )

    def test_prior_shift_moves_mass(self) -> None:
        """Flat likelihood + prior odds 1:3 gives posterior probabilities 1/4, 3/4."""
        log_prior = torch.tensor([0.0, math.log(3.0)], dtype=F64)
        probabilities = posterior_log_weights(torch.zeros(2, dtype=F64), log_prior=log_prior).exp()
        torch.testing.assert_close(probabilities, torch.tensor([0.25, 0.75], dtype=F64))

    def test_reference_shift_moves_mass_opposite(self) -> None:
        """A proposal that oversamples a candidate down-weights it."""
        log_reference = torch.log(torch.tensor([0.75, 0.25], dtype=F64))
        probabilities = posterior_log_weights(
            torch.zeros(2, dtype=F64), log_reference=log_reference
        ).exp()
        torch.testing.assert_close(probabilities, torch.tensor([0.25, 0.75], dtype=F64))

    def test_prior_and_reference_cancel_for_prior_candidates(self) -> None:
        """When candidates are drawn from the prior, q == p and the weights are flat."""
        log_prior = torch.log(torch.tensor([0.25, 0.75], dtype=F64))
        probabilities = posterior_log_weights(
            torch.zeros(2, dtype=F64), log_prior=log_prior, log_reference=log_prior
        ).exp()
        torch.testing.assert_close(probabilities, torch.tensor([0.5, 0.5], dtype=F64))

    def test_prior_over_reference_recovers_posterior(self) -> None:
        """Self-normalized IS recovers p/q reweighting with p != q."""
        target_prior = torch.tensor([0.25, 0.75], dtype=F64)
        proposal = torch.tensor([0.75, 0.25], dtype=F64)
        probabilities = posterior_log_weights(
            torch.zeros(2, dtype=F64),
            log_prior=target_prior.log(),
            log_reference=proposal.log(),
        ).exp()
        expected = target_prior / proposal
        torch.testing.assert_close(probabilities, expected / expected.sum())
        torch.testing.assert_close(probabilities, torch.tensor([0.1, 0.9], dtype=F64))

    def test_shapes_1d_2d_and_batching(self) -> None:
        """Single banks [K], batches [N, K], and extra batch dims are supported."""
        single = posterior_log_weights(torch.tensor([1.0, -1.0, 0.0], dtype=F64))
        assert single.shape == (3,)
        batch = torch.tensor([[0.0, 1.0], [2.0, -1.0], [-3.0, 0.5]], dtype=F64)
        batched = posterior_log_weights(batch)
        assert batched.shape == batch.shape
        torch.testing.assert_close(batched.exp().sum(dim=-1), torch.ones(3, dtype=F64))
        three_d = torch.zeros(2, 4, 3, dtype=F64)
        assert posterior_log_weights(three_d).shape == three_d.shape

    def test_broadcast_corrections(self) -> None:
        """Scalar, [K], and [N, 1] corrections broadcast like ordinary tensors."""
        log_ratio = torch.tensor([[0.0, 1.0], [2.0, -1.0]], dtype=F64)
        base = posterior_log_weights(log_ratio)
        torch.testing.assert_close(posterior_log_weights(log_ratio, log_prior=0.5), base)
        torch.testing.assert_close(
            posterior_log_weights(log_ratio, log_prior=torch.tensor(0.5, dtype=F64)), base
        )
        prior_k = torch.tensor([0.0, 1.0], dtype=F64)
        torch.testing.assert_close(
            posterior_log_weights(log_ratio, log_prior=prior_k),
            torch.log_softmax(log_ratio + prior_k, dim=-1),
        )
        prior_rows = torch.tensor([[0.0], [1.0]], dtype=F64)
        out = posterior_log_weights(log_ratio, log_prior=prior_rows)
        assert out.shape == log_ratio.shape
        torch.testing.assert_close(
            out,
            torch.log_softmax(log_ratio + prior_rows, dim=-1),
        )

    def test_non_last_dim(self) -> None:
        """dim selects the candidate axis of the softmax."""
        log_ratio = torch.tensor([[0.0, 1.0], [2.0, -1.0], [-3.0, 0.5]], dtype=F64)  # [K, N]
        log_weights = posterior_log_weights(log_ratio, dim=0)
        torch.testing.assert_close(log_weights, torch.log_softmax(log_ratio, dim=0))
        torch.testing.assert_close(log_weights.exp().sum(dim=0), torch.ones(2, dtype=F64))

    def test_large_scores_are_stable(self) -> None:
        """Huge scores and huge common shifts do not overflow or produce NaNs."""
        log_ratio = torch.tensor([1e4, 0.0, -1e4], dtype=F64)
        log_weights = posterior_log_weights(log_ratio)
        assert bool(torch.isfinite(log_weights).all())
        probabilities = log_weights.exp()
        torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0, dtype=F64))
        assert probabilities[0].item() > 1.0 - 1e-12
        shifted = posterior_log_weights(log_ratio + 1e6)
        torch.testing.assert_close(shifted, log_weights, atol=1e-12, rtol=0.0)
        # float32 path
        probabilities32 = posterior_log_weights(log_ratio.to(torch.float32)).exp()
        assert bool(torch.isfinite(probabilities32).all())
        torch.testing.assert_close(probabilities32.sum(), torch.tensor(1.0))

    def test_dtype_preserved_and_gradients_flow(self) -> None:
        """The utility is differentiable and does not detach its input."""
        x32 = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32, requires_grad=True)
        out32 = posterior_log_weights(x32)
        assert out32.dtype == torch.float32
        out32[0].backward()
        assert x32.grad is not None
        assert bool(torch.isfinite(x32.grad).all())
        x64 = torch.tensor([0.1, -0.2], dtype=F64)
        assert posterior_log_weights(x64).dtype == F64
        mixed = posterior_log_weights(x64, log_prior=torch.zeros(2, dtype=torch.float32))
        assert mixed.dtype == F64

    def test_validation_errors(self) -> None:
        with pytest.raises(ValueError, match="at least one dimension"):
            posterior_log_weights(torch.tensor(1.0))
        with pytest.raises(ValueError, match="non-empty"):
            posterior_log_weights(torch.zeros(0))
        with pytest.raises(ValueError, match="finite"):
            posterior_log_weights(torch.tensor([0.0, float("nan")]))
        with pytest.raises(ValueError, match="finite"):
            posterior_log_weights(torch.tensor([0.0, float("inf")]))
        with pytest.raises(ValueError, match="dim"):
            posterior_log_weights(torch.zeros(3), dim=1)
        with pytest.raises(ValueError, match="dim"):
            posterior_log_weights(torch.zeros(3), dim=-4)
        with pytest.raises(ValueError, match="broadcastable"):
            posterior_log_weights(torch.zeros(2, 4), log_prior=torch.zeros(3))
        with pytest.raises(ValueError, match="broadcastable"):
            posterior_log_weights(torch.zeros(1, 4), log_prior=torch.zeros(2, 4))
        with pytest.raises(ValueError, match="finite"):
            posterior_log_weights(torch.zeros(3), log_prior=torch.tensor([0.0, float("nan"), 0.0]))
        with pytest.raises(ValueError, match="finite"):
            posterior_log_weights(
                torch.zeros(3), log_reference=torch.tensor([0.0, float("inf"), 0.0])
            )


# ═══════════════════════════════════════════════════════════════════════════════
# effective_sample_size
# ═══════════════════════════════════════════════════════════════════════════════


class TestEffectiveSampleSize:
    def test_uniform_weights_give_k(self) -> None:
        """Uniform weights have ESS = K, normalized or not."""
        k = 5
        normalized = torch.full((k,), -math.log(k), dtype=F64)
        ess = effective_sample_size(normalized)
        assert ess.shape == ()
        torch.testing.assert_close(ess, torch.tensor(float(k), dtype=F64))
        unnormalized = torch.zeros(k, dtype=F64)
        torch.testing.assert_close(
            effective_sample_size(unnormalized), torch.tensor(float(k), dtype=F64)
        )

    def test_one_hot_weights_give_one(self) -> None:
        """A single dominant candidate leaves ESS = 1."""
        log_weights = torch.tensor([0.0, -700.0, -700.0], dtype=F64)
        torch.testing.assert_close(effective_sample_size(log_weights), torch.tensor(1.0, dtype=F64))

    def test_matches_direct_kish_formula(self) -> None:
        """ESS equals 1 / sum(w^2) for normalized weights."""
        probabilities = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=F64)
        expected = 1.0 / (probabilities**2).sum()
        torch.testing.assert_close(effective_sample_size(probabilities.log()), expected)

    def test_bounds_and_monotone_degradation(self) -> None:
        """Concentrating the weights decreases ESS within [1, K]."""
        concentration = torch.tensor([0.0, 0.5, 1.0, 2.0, 4.0], dtype=F64)
        scores = torch.stack([torch.zeros_like(concentration), concentration], dim=-1)
        ess = effective_sample_size(torch.log_softmax(scores, dim=-1))
        assert ess.shape == concentration.shape
        assert bool((ess >= 1.0 - 1e-12).all())
        assert bool((ess <= 2.0 + 1e-12).all())
        assert bool((ess[1:] < ess[:-1]).all())

    def test_batched_shape(self) -> None:
        """Per-bank ESS drops the candidate axis."""
        log_weights = torch.log_softmax(
            torch.tensor([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]], dtype=F64), dim=-1
        )
        assert effective_sample_size(log_weights).shape == (2,)
        single = torch.log_softmax(torch.zeros(4, dtype=F64), dim=0)
        assert effective_sample_size(single).shape == ()
        three_d = torch.log_softmax(torch.zeros(2, 3, 4, dtype=F64), dim=-1)
        assert effective_sample_size(three_d).shape == (2, 3)

    def test_shift_invariance(self) -> None:
        """A constant shift of all log weights does not change the ESS."""
        log_weights = torch.log_softmax(
            torch.tensor([[0.0, 1.0, -2.0], [3.0, 1.0, 0.0]], dtype=F64), dim=-1
        )
        torch.testing.assert_close(
            effective_sample_size(log_weights + 7.5),
            effective_sample_size(log_weights),
            rtol=1e-12,
            atol=0.0,
        )

    def test_extreme_log_weights_stay_finite(self) -> None:
        log_weights = torch.tensor([1e4, 0.0, -1e4], dtype=F64)
        ess = effective_sample_size(log_weights)
        assert bool(torch.isfinite(ess))
        assert 1.0 <= ess.item() <= 3.0
        ess32 = effective_sample_size(log_weights.to(torch.float32))
        assert bool(torch.isfinite(ess32))

    def test_end_to_end_with_posterior_weights(self) -> None:
        """Flat likelihood keeps every candidate; a peaked one collapses the bank."""
        flat = posterior_log_weights(torch.zeros(8, dtype=F64))
        torch.testing.assert_close(effective_sample_size(flat), torch.tensor(8.0, dtype=F64))
        peaked = posterior_log_weights(torch.tensor([0.0, -1000.0, -1000.0], dtype=F64))
        assert effective_sample_size(peaked).item() < 1.001

    def test_validation_errors(self) -> None:
        with pytest.raises(ValueError, match="at least one dimension"):
            effective_sample_size(torch.tensor(1.0))
        with pytest.raises(ValueError, match="non-empty"):
            effective_sample_size(torch.zeros(0))
        with pytest.raises(ValueError, match="finite"):
            effective_sample_size(torch.tensor([0.0, float("nan")]))
        with pytest.raises(ValueError, match="finite"):
            effective_sample_size(torch.tensor([0.0, float("-inf")]))
        with pytest.raises(ValueError, match="dim"):
            effective_sample_size(torch.zeros(3), dim=5)


# ═══════════════════════════════════════════════════════════════════════════════
# grid_log_posterior
# ═══════════════════════════════════════════════════════════════════════════════


class TestGridLogPosterior:
    def test_matches_posterior_log_weights(self) -> None:
        """The likelihood-scan wrapper has identical semantics."""
        log_scores = torch.tensor([[0.0, 1.0, 2.0], [-1.0, 0.0, 1.0]], dtype=F64)
        log_prior = torch.tensor([0.0, 1.0, 2.0], dtype=F64)
        log_reference = torch.tensor([-1.0, -0.5, 0.0], dtype=F64)
        torch.testing.assert_close(
            grid_log_posterior(log_scores), posterior_log_weights(log_scores)
        )
        torch.testing.assert_close(
            grid_log_posterior(log_scores, log_prior=log_prior),
            posterior_log_weights(log_scores, log_prior=log_prior),
        )
        torch.testing.assert_close(
            grid_log_posterior(log_scores, log_prior=log_prior, log_reference=log_reference),
            posterior_log_weights(log_scores, log_prior=log_prior, log_reference=log_reference),
        )

    def test_gaussian_scan_with_flat_prior(self) -> None:
        """A fine scan reproduces the analytic Gaussian posterior mean."""
        theta = torch.linspace(-6.0, 6.0, 241, dtype=F64)
        x, sigma = 0.5, 1.0
        log_like = -0.5 * ((x - theta) / sigma) ** 2
        weights = grid_log_posterior(log_like).exp()
        assert weights.shape == theta.shape
        torch.testing.assert_close(weights.sum(), torch.tensor(1.0, dtype=F64))
        torch.testing.assert_close(
            (weights * theta).sum(), torch.tensor(x, dtype=F64), atol=1e-6, rtol=0.0
        )

    def test_gaussian_scan_with_prior(self) -> None:
        """Likelihood N(0.5, 1) times N(0, 1) prior has posterior mean 0.25."""
        theta = torch.linspace(-6.0, 6.0, 241, dtype=F64)
        log_like = -0.5 * ((0.5 - theta) / 1.0) ** 2
        log_prior = -0.5 * theta**2
        weights = grid_log_posterior(log_like, log_prior=log_prior).exp()
        torch.testing.assert_close(
            (weights * theta).sum(), torch.tensor(0.25, dtype=F64), atol=1e-6, rtol=0.0
        )

    def test_reference_correction_reweights_non_uniform_grid(self) -> None:
        """Importance weights p/q recover the target distribution on the grid."""
        log_scores = torch.zeros(4, dtype=F64)  # flat likelihood
        target = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=F64)
        proposal = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=F64)
        weights = grid_log_posterior(
            log_scores, log_prior=target.log(), log_reference=proposal.log()
        ).exp()
        expected = target / proposal
        torch.testing.assert_close(weights, expected / expected.sum())

    def test_batched_scan(self) -> None:
        """Rows of a [N, K] scan are normalized independently."""
        theta = torch.linspace(-3.0, 3.0, 25, dtype=F64)
        observations = torch.tensor([[0.0], [1.5]], dtype=F64)
        scores = -0.5 * ((observations - theta) ** 2)  # [N, K]
        log_weights = grid_log_posterior(scores)
        assert log_weights.shape == scores.shape
        torch.testing.assert_close(log_weights.exp().sum(dim=-1), torch.ones(2, dtype=F64))

    def test_validation_errors(self) -> None:
        with pytest.raises(ValueError, match="at least one dimension"):
            grid_log_posterior(torch.tensor(1.0))
        with pytest.raises(ValueError, match="non-empty"):
            grid_log_posterior(torch.zeros(0))
        with pytest.raises(ValueError, match="finite"):
            grid_log_posterior(torch.tensor([0.0, float("nan")]))
        with pytest.raises(ValueError, match="dim"):
            grid_log_posterior(torch.zeros(3), dim=3)


# ═══════════════════════════════════════════════════════════════════════════════
# maximum_likelihood_estimate
# ═══════════════════════════════════════════════════════════════════════════════


class TestMaximumLikelihoodEstimate:
    def test_known_argmax_on_analytic_grid(self) -> None:
        """The grid linspace(-3, 3, 121) contains x = 0.7 and x = -1.4 exactly."""
        theta = torch.linspace(-3.0, 3.0, 121, dtype=F64)
        x = torch.tensor([0.7, -1.4], dtype=F64)
        scores = -((x[:, None] - theta[None, :]) ** 2)  # [N, K]
        estimates, best_index = maximum_likelihood_estimate(scores, theta)
        assert best_index.shape == (2,)
        assert best_index.dtype == torch.int64
        assert torch.equal(best_index, torch.tensor([74, 32]))
        assert estimates.shape == (2, 1)  # [K] hypotheses are [K, 1] coordinates
        torch.testing.assert_close(estimates.squeeze(-1), x)

    def test_2d_hypotheses_return_coordinates(self) -> None:
        """estimates are hypotheses[best_index], one coordinate row per bank."""
        theta = torch.linspace(-2.0, 2.0, 41, dtype=F64)
        hypotheses = torch.stack([theta, theta**2], dim=-1)  # [K, 2]
        scores = -((0.5 - theta) ** 2)[None, :]  # [1, K]
        estimates, best_index = maximum_likelihood_estimate(scores, hypotheses)
        expected_index = int((0.5 - theta).abs().argmin().item())
        assert int(best_index.item()) == expected_index
        assert estimates.shape == (1, 2)
        torch.testing.assert_close(estimates[0], hypotheses[expected_index])

    def test_flat_prior_matches_mle(self) -> None:
        """Zero/scalar priors leave the MLE unchanged."""
        theta = torch.linspace(-2.0, 2.0, 41, dtype=F64)
        scores = -((0.3 - theta) ** 2)[None, :]
        _, mle_index = maximum_likelihood_estimate(scores, theta)
        _, flat_index = maximum_likelihood_estimate(
            scores, theta, log_prior=torch.zeros_like(theta)
        )
        _, scalar_index = maximum_likelihood_estimate(scores, theta, log_prior=0.0)
        assert torch.equal(mle_index, flat_index)
        assert torch.equal(mle_index, scalar_index)

    def test_prior_changes_map(self) -> None:
        """Adding a prior turns the MLE into a MAP and can move the selected index."""
        scores = torch.tensor([[0.0, -0.1]], dtype=F64)
        hypotheses = torch.tensor([0.0, 1.0], dtype=F64)
        _, mle_index = maximum_likelihood_estimate(scores, hypotheses)
        assert int(mle_index.item()) == 0
        log_prior = torch.tensor([0.0, math.log(2.0)], dtype=F64)
        estimates, map_index = maximum_likelihood_estimate(scores, hypotheses, log_prior=log_prior)
        assert int(map_index.item()) == 1
        torch.testing.assert_close(estimates, torch.tensor([[1.0]], dtype=F64))

    def test_prior_can_override_strong_likelihood_preference(self) -> None:
        """MAP maximizes score + log_prior, not the likelihood alone."""
        scores = torch.tensor([[0.0, 1.0]], dtype=F64)
        hypotheses = torch.tensor([0.0, 1.0], dtype=F64)
        log_prior = torch.tensor([2.0, 0.0], dtype=F64)
        _, map_index = maximum_likelihood_estimate(scores, hypotheses, log_prior=log_prior)
        assert int(map_index.item()) == 0

    def test_1d_scores(self) -> None:
        """A single score vector returns a 0-dim index and a [D] estimate."""
        theta = torch.linspace(-1.0, 1.0, 21, dtype=F64)
        scores = -((0.25 - theta) ** 2)  # [K]
        estimates, best_index = maximum_likelihood_estimate(scores, theta)
        expected_index = int(scores.argmax().item())
        assert best_index.shape == ()
        assert int(best_index.item()) == expected_index
        assert estimates.shape == (1,)
        torch.testing.assert_close(estimates, theta[expected_index].reshape(1))

    def test_ties_break_to_first_index(self) -> None:
        scores = torch.zeros(4, dtype=F64)
        hypotheses = torch.arange(4, dtype=F64)
        estimates, best_index = maximum_likelihood_estimate(scores, hypotheses)
        assert int(best_index.item()) == 0
        torch.testing.assert_close(estimates, torch.tensor([0.0], dtype=F64))

    def test_integer_hypotheses_are_cast_to_float(self) -> None:
        scores = torch.tensor([[0.0, 1.0, 2.0]], dtype=F64)
        hypotheses = torch.tensor([10, 20, 30], dtype=torch.int64)
        estimates, best_index = maximum_likelihood_estimate(scores, hypotheses)
        assert estimates.dtype == torch.get_default_dtype()
        assert int(best_index.item()) == 2
        torch.testing.assert_close(estimates, torch.tensor([[30.0]]))

    def test_non_last_hypothesis_dim(self) -> None:
        """dim can select a non-trailing hypothesis axis."""
        scores = torch.tensor([[0.0, 3.0], [1.0, 2.0], [-1.0, 0.0]], dtype=F64)  # [K, N]
        hypotheses = torch.tensor([0.0, 1.0, 2.0], dtype=F64)
        estimates, best_index = maximum_likelihood_estimate(scores, hypotheses, dim=0)
        expected = scores.argmax(dim=0)
        assert torch.equal(best_index, expected)
        assert estimates.shape == (2, 1)
        torch.testing.assert_close(estimates.squeeze(-1), hypotheses[expected])

    def test_validation_errors(self) -> None:
        scores = torch.zeros(2, 3)
        hypotheses = torch.arange(3.0)
        with pytest.raises(ValueError, match="at least one dimension"):
            maximum_likelihood_estimate(torch.tensor(0.0), hypotheses)
        with pytest.raises(ValueError, match="non-empty"):
            maximum_likelihood_estimate(torch.zeros(0), torch.zeros(0))
        with pytest.raises(ValueError, match="finite"):
            maximum_likelihood_estimate(torch.tensor([[0.0, float("nan")]]), hypotheses)
        with pytest.raises(ValueError, match="finite"):
            maximum_likelihood_estimate(scores, torch.tensor([0.0, float("inf"), 2.0]))
        with pytest.raises(ValueError, match="1-D"):
            maximum_likelihood_estimate(scores, hypotheses.reshape(1, 3, 1))
        with pytest.raises(ValueError, match="row"):
            maximum_likelihood_estimate(scores, torch.arange(4.0))
        with pytest.raises(ValueError, match="broadcastable"):
            maximum_likelihood_estimate(scores, hypotheses, log_prior=torch.zeros(4))
        with pytest.raises(ValueError, match="dim"):
            maximum_likelihood_estimate(scores, hypotheses, dim=2)


# ═══════════════════════════════════════════════════════════════════════════════
# module exports
# ═══════════════════════════════════════════════════════════════════════════════


class TestModuleExports:
    def test_all_symbols_exported(self) -> None:
        assert set(ratio.__all__) == {
            "effective_sample_size",
            "grid_log_posterior",
            "maximum_likelihood_estimate",
            "posterior_log_weights",
        }
        for name in ratio.__all__:
            assert callable(getattr(ratio, name))
