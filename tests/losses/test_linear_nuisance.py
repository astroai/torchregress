"""Tests for the marginalised linear-nuisance Gaussian NLL."""

from __future__ import annotations

import math

import pytest
import torch
from torch.distributions import LowRankMultivariateNormal, MultivariateNormal

from torchregress.losses import (
    LinearNuisanceMarginal,
    MarginalizedMultiSourceGaussianNLLLoss,
    MarginalizedNuisanceGaussianNLLLoss,
    PackedObservations,
    marginalize_linear_nuisance,
)

F64 = torch.float64


def _problem(rows: int = 4, n: int = 25, m: int = 3, seed: int = 0, dtype: torch.dtype = F64):
    gen = torch.Generator().manual_seed(seed)
    t = torch.linspace(-1.0, 1.0, n, dtype=dtype)
    basis = torch.stack([t**k for k in range(m)], dim=-1)  # polynomial baseline
    y_pred = torch.randn(rows, n, generator=gen, dtype=dtype)
    target = torch.randn(rows, n, generator=gen, dtype=dtype)
    precision = torch.rand(rows, n, generator=gen, dtype=dtype) + 0.5
    return y_pred, target, precision, basis


def _spd(m: int, seed: int = 1) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(m, m, generator=gen, dtype=F64)
    return a @ a.T + 0.5 * torch.eye(m, dtype=F64)


def _dense_nll(y_pred, target, precision, basis, prior_cov):
    cov = torch.diag_embed(1.0 / precision).to(dtype=precision.dtype) + basis @ prior_cov @ basis.mT
    zero = torch.zeros(target.shape[-1], dtype=target.dtype)
    return -MultivariateNormal(zero, covariance_matrix=cov).log_prob(target - y_pred)


class TestDenseParity:
    def test_full_prior_covariance_matches_multivariate_normal(self) -> None:
        y_pred, target, precision, basis = _problem()
        prior_cov = _spd(3)
        res = marginalize_linear_nuisance(
            y_pred, target, precision, basis, prior_covariance=prior_cov
        )
        assert isinstance(res, LinearNuisanceMarginal)
        ref = _dense_nll(y_pred, target, precision, basis, prior_cov)
        torch.testing.assert_close(res.nll, ref, atol=1e-11, rtol=0)

    def test_diagonal_prior_precision_matches_multivariate_normal(self) -> None:
        y_pred, target, precision, basis = _problem()
        p = torch.tensor([0.5, 2.0, 7.0], dtype=F64)
        res = marginalize_linear_nuisance(y_pred, target, precision, basis, prior_precision=p)
        ref = _dense_nll(y_pred, target, precision, basis, torch.diag_embed(1.0 / p).to(dtype=F64))
        torch.testing.assert_close(res.nll, ref, atol=1e-11, rtol=0)
        scalar = marginalize_linear_nuisance(y_pred, target, precision, basis, prior_precision=2.0)
        ref2 = _dense_nll(y_pred, target, precision, basis, torch.eye(3, dtype=F64) / 2.0)
        torch.testing.assert_close(scalar.nll, ref2, atol=1e-11, rtol=0)

    def test_per_row_basis_and_prior_match_low_rank_normal(self) -> None:
        y_pred, target, precision, basis = _problem(rows=3)
        gen = torch.Generator().manual_seed(4)
        basis_rows = basis + 0.1 * torch.randn(3, *basis.shape, generator=gen, dtype=F64)
        prior_cov = torch.stack([_spd(3, seed=s) for s in range(3)])
        res = marginalize_linear_nuisance(
            y_pred, target, precision, basis_rows, prior_covariance=prior_cov
        )
        factor = basis_rows @ torch.linalg.cholesky(prior_cov)
        ref = -LowRankMultivariateNormal(y_pred, factor, 1.0 / precision).log_prob(target)
        torch.testing.assert_close(res.nll, ref, atol=1e-10, rtol=0)

    def test_posterior_mean_and_covariance_match_dense_formulas(self) -> None:
        y_pred, target, precision, basis = _problem(rows=2)
        prior_cov = _spd(3)
        res = marginalize_linear_nuisance(
            y_pred, target, precision, basis, prior_covariance=prior_cov
        )
        for i in range(2):
            lam = torch.diag(precision[i]).to(dtype=F64)
            post_prec = torch.linalg.inv(prior_cov) + basis.T @ lam @ basis
            post_cov = torch.linalg.inv(post_prec)
            mean = post_cov @ basis.T @ lam @ (target[i] - y_pred[i])
            torch.testing.assert_close(res.coefficients[i], mean, atol=1e-11, rtol=0)
            torch.testing.assert_close(res.coefficient_covariance[i], post_cov, atol=1e-11, rtol=0)
            # E[Ua | r] equals the GP-style conditional mean U S U^T C^{-1} r.
            cov = torch.diag(1.0 / precision[i]).to(dtype=F64) + basis @ prior_cov @ basis.T
            cond = basis @ prior_cov @ basis.T @ torch.linalg.solve(cov, target[i] - y_pred[i])
            torch.testing.assert_close(res.nuisance[i], cond, atol=1e-11, rtol=0)
        torch.testing.assert_close(res.residual, target - y_pred - res.nuisance)

    @pytest.mark.parametrize("method", ["direct", "information"])
    def test_matches_marginalized_multisource_with_diagonal_noise(self, method: str) -> None:
        y_pred, target, precision, basis = _problem(rows=4, n=25)
        mask = torch.rand(4, 25, generator=torch.Generator().manual_seed(6)) > 0.3
        prior_cov = _spd(3, seed=2)
        res = marginalize_linear_nuisance(
            y_pred, target, precision, basis, prior_covariance=prior_cov, mask=mask
        )
        obs = PackedObservations(
            values=target,
            mapped_mean=y_pred,
            jacobian=basis.expand(4, -1, -1),
            measurement_covariance=torch.diag_embed(1.0 / precision).to(
                device=precision.device, dtype=precision.dtype
            ),
            mask=mask,
        )
        ref = MarginalizedMultiSourceGaussianNLLLoss(reduction="none", method=method)(
            torch.zeros(4, 3, dtype=F64), torch.linalg.cholesky(prior_cov).expand(4, 3, 3), obs
        )
        torch.testing.assert_close(res.nll, ref, atol=1e-12, rtol=0)

    @pytest.mark.parametrize("rows", [1, 2, 5])  # rows <= (m + 1) / 2 takes the per-row path
    def test_shared_and_expanded_basis_agree(self, rows: int) -> None:
        y_pred, target, precision, basis = _problem(rows=rows, m=4)
        shared = marginalize_linear_nuisance(y_pred, target, precision, basis, prior_precision=1.0)
        expanded = marginalize_linear_nuisance(
            y_pred, target, precision, basis.expand(rows, *basis.shape), prior_precision=1.0
        )
        for a, b in zip(shared, expanded):
            torch.testing.assert_close(a, b, atol=1e-12, rtol=0)

    @pytest.mark.parametrize("layout", ["singleton", "expanded"])
    def test_effectively_shared_basis_takes_the_shared_path(self, layout: str, monkeypatch) -> None:
        import torchregress.losses.linear_nuisance as ln

        rows = 6
        y_pred, target, precision, basis = _problem(rows=rows, m=3)
        mask = torch.rand(rows, 25, generator=torch.Generator().manual_seed(4)) > 0.2
        kw = {"prior_precision": torch.tensor([0.0, 1.0, 2.0], dtype=F64), "mask": mask}

        def run(b: torch.Tensor):
            leaf = b.clone().requires_grad_(True)
            used = leaf if layout == "singleton" or leaf.dim() == 2 else leaf.expand(rows, -1, -1)
            pred = y_pred.clone().requires_grad_(True)
            res = marginalize_linear_nuisance(pred, target, precision, used, **kw)
            (res.nll.sum() + res.residual[mask].square().sum()).backward()
            return res, pred.grad, leaf.grad

        ref, ref_gp, ref_gu = run(basis)
        packed: list[int] = []
        real = ln._sym_index
        monkeypatch.setattr(ln, "_sym_index", lambda *a: packed.append(1) or real(*a))
        res, gp, gu = run(basis.unsqueeze(0))
        assert packed, "an effectively shared basis must use the packed shared-basis Gram"
        for got, want in zip(res, ref):
            torch.testing.assert_close(got, want, atol=1e-12, rtol=0)
        torch.testing.assert_close(gp, ref_gp, atol=1e-12, rtol=0)
        torch.testing.assert_close(gu.reshape(ref_gu.shape), ref_gu, atol=1e-12, rtol=0)

    def test_singleton_basis_dims_still_broadcast_the_output(self) -> None:
        y_pred, target, precision, basis = _problem(rows=1)
        res = marginalize_linear_nuisance(
            y_pred[0], target[0], precision[0], basis.unsqueeze(0), prior_precision=1.0
        )
        assert res.nll.shape == (1,) and res.coefficients.shape == (1, 3)
        assert res.nuisance.shape == (1, 25)
        ref = marginalize_linear_nuisance(
            y_pred[0], target[0], precision[0], basis, prior_precision=1.0
        )
        torch.testing.assert_close(res.nll[0], ref.nll)

    def test_masked_non_finite_singleton_basis_stays_per_row(self) -> None:
        y_pred, target, precision, basis = _problem(rows=4)
        mask = torch.ones_like(target, dtype=torch.bool)
        mask[:, 6] = False
        bad = basis.clone()
        bad[6] = float("nan")  # masked in every row: valid input for a per-row basis
        res = marginalize_linear_nuisance(
            y_pred, target, precision, bad.unsqueeze(0), prior_precision=1.0, mask=mask
        )
        ref = marginalize_linear_nuisance(
            y_pred, target, precision, basis, prior_precision=1.0, mask=mask
        )
        torch.testing.assert_close(res.nll, ref.nll)


class TestMasking:
    @pytest.mark.parametrize("shared", [True, False])
    def test_masking_equals_dropping_samples(self, shared: bool) -> None:
        y_pred, target, precision, basis = _problem(rows=4, n=30)
        mask = torch.rand(4, 30, generator=torch.Generator().manual_seed(3)) > 0.3
        b = basis if shared else basis.expand(4, *basis.shape).clone()
        target_bad = target.clone()
        target_bad[~mask] = float("nan")
        precision_bad = precision.clone()
        precision_bad[~mask] = float("inf")
        if not shared:
            b[~mask] = float("nan")
        prior = torch.tensor([1.0, 0.5, 0.0], dtype=F64)  # flat in one direction
        res = marginalize_linear_nuisance(
            y_pred, target_bad, precision_bad, b, prior_precision=prior, mask=mask
        )
        for i in range(4):
            keep = mask[i]
            ref = marginalize_linear_nuisance(
                y_pred[i, keep],
                target[i, keep],
                precision[i, keep],
                basis[keep],
                prior_precision=prior,
            )
            torch.testing.assert_close(res.nll[i], ref.nll, atol=1e-12, rtol=0)
            torch.testing.assert_close(res.coefficients[i], ref.coefficients, atol=1e-12, rtol=0)
            torch.testing.assert_close(
                res.coefficient_covariance[i], ref.coefficient_covariance, atol=1e-12, rtol=0
            )
        if shared:
            # The fitted nuisance is evaluated at masked samples too (interpolation).
            torch.testing.assert_close(res.nuisance, res.coefficients @ basis.T)

    def test_nan_at_masked_samples_does_not_leak_into_gradients(self) -> None:
        y_pred, target, precision, basis = _problem()
        mask = torch.ones_like(target, dtype=torch.bool)
        mask[:, 3:6] = False
        target = target.clone()
        target[~mask] = float("nan")
        pred = y_pred.clone().requires_grad_(True)
        loss = MarginalizedNuisanceGaussianNLLLoss()(pred, target, precision, basis, mask=mask)
        loss.backward()
        assert torch.isfinite(loss)
        assert pred.grad is not None and torch.isfinite(pred.grad).all()
        assert (pred.grad[~mask] == 0).all()

    @pytest.mark.parametrize("output", ["residual", "nuisance"])
    def test_nan_basis_at_masked_samples_does_not_poison_output_gradients(self, output) -> None:
        y_pred, target, precision, basis = _problem(rows=3, n=20)
        mask = torch.ones_like(target, dtype=torch.bool)
        mask[0, 4] = False
        mask[2, 10:12] = False
        u = basis.expand(3, *basis.shape).clone()
        u[0, 4, 1] = float("nan")  # documented as ignored at a masked sample
        u[2, 10] = float("inf")
        pred = y_pred.clone().requires_grad_(True)
        lam = precision.clone().requires_grad_(True)
        u = u.requires_grad_(True)
        res = marginalize_linear_nuisance(pred, target, lam, u, prior_precision=0.5, mask=mask)
        value = getattr(res, output)
        assert torch.isfinite(value[mask]).all()
        assert torch.isnan(res.nuisance[0, 4]) and torch.isfinite(res.nuisance[2, 11])
        objective = (value[mask] ** 2).sum() if output == "residual" else value[mask].sum()
        # The fitted nuisance at a masked sample with a finite basis keeps its gradient.
        objective = objective + res.nuisance[2, 11]
        objective.backward()
        for grad in (pred.grad, lam.grad):
            assert grad is not None and torch.isfinite(grad[mask]).all()
        assert u.grad is not None and torch.isfinite(u.grad[mask]).all()
        assert torch.isfinite(u.grad[2, 11]).all() and (u.grad[2, 11] != 0).any()
        assert (u.grad[0, 4] == 0).all() and (u.grad[2, 10] == 0).all()

    def test_empty_row_with_proper_prior_is_zero_and_dropped(self) -> None:
        y_pred, target, precision, basis = _problem(rows=3)
        mask = torch.ones_like(target, dtype=torch.bool)
        mask[1] = False
        res = marginalize_linear_nuisance(
            y_pred, target, precision, basis, prior_precision=2.0, mask=mask
        )
        torch.testing.assert_close(res.nll[1], torch.tensor(0.0, dtype=F64), atol=1e-14, rtol=0)
        torch.testing.assert_close(res.coefficients[1], torch.zeros(3, dtype=F64))
        torch.testing.assert_close(res.coefficient_covariance[1], torch.eye(3, dtype=F64) / 2.0)
        loss = MarginalizedNuisanceGaussianNLLLoss()(
            y_pred, target, precision, basis, prior_precision=2.0, mask=mask
        )
        torch.testing.assert_close(loss, res.nll[[0, 2]].mean())


class TestFlatPrior:
    def test_flat_limit_equals_profile_plus_log_det(self) -> None:
        y_pred, target, precision, basis = _problem(rows=3, n=40, m=4)
        flat = marginalize_linear_nuisance(y_pred, target, precision, basis)
        resid = target - y_pred
        for i in range(3):
            w = precision[i].sqrt()
            # Explicit profile: weighted least squares for the nuisance.
            coef = torch.linalg.lstsq(basis * w[:, None], resid[i] * w).solution
            torch.testing.assert_close(flat.coefficients[i], coef, atol=1e-11, rtol=0)
            chi2 = (precision[i] * (resid[i] - basis @ coef) ** 2).sum()
            profile = 0.5 * (40 * math.log(2 * math.pi) - precision[i].log().sum() + chi2)
            gram = basis.T @ (precision[i, :, None] * basis)
            # Improper p(a) = 1: + 1/2 log|U^T Lambda U| - m/2 log(2 pi).
            expected = profile + 0.5 * torch.logdet(gram) - 0.5 * 4 * math.log(2 * math.pi)
            torch.testing.assert_close(flat.nll[i], expected, atol=1e-11, rtol=0)
            torch.testing.assert_close(
                flat.coefficient_covariance[i], torch.linalg.inv(gram), atol=1e-11, rtol=0
            )

    def test_flat_is_the_limit_of_vanishing_prior_precision(self) -> None:
        y_pred, target, precision, basis = _problem()
        flat = marginalize_linear_nuisance(y_pred, target, precision, basis)
        errors = []
        for eps in (1e-2, 1e-4, 1e-6):
            proper = marginalize_linear_nuisance(
                y_pred, target, precision, basis, prior_precision=eps
            )
            # Improper p(a) = 1 is the limit of p_eps / (eps / 2 pi)^(m/2).
            limit = proper.nll + 0.5 * 3 * math.log(eps / (2 * math.pi))
            errors.append((limit - flat.nll).abs().max().item())
        assert errors[0] > errors[1] > errors[2]
        assert errors[2] < 1e-5

    @pytest.mark.parametrize("masked", [False, True])
    def test_flat_prior_equals_reml_on_error_contrasts(self, masked: bool) -> None:
        """Harville (1974): -log N(K^T r; 0, K^T Lambda^-1 K) + 1/2 log|U^T U|, K^T U = 0."""
        y_pred, target, precision, basis = _problem(rows=3, n=30, m=4)
        mask = torch.rand(3, 30, generator=torch.Generator().manual_seed(5)) > 0.2
        if not masked:
            mask = torch.ones_like(mask)
        flat = marginalize_linear_nuisance(y_pred, target, precision, basis, mask=mask)
        for i in range(3):
            keep = mask[i]
            u = basis[keep]
            q, _ = torch.linalg.qr(u, mode="complete")
            k = q[:, u.shape[1] :]  # orthonormal error contrasts, K^T U = 0
            r = (target[i] - y_pred[i])[keep]
            cov = k.T @ torch.diag(1.0 / precision[i, keep]).to(dtype=F64) @ k
            zero = torch.zeros(k.shape[1], dtype=F64)
            reml = -MultivariateNormal(zero, covariance_matrix=cov).log_prob(k.T @ r)
            reml = reml + 0.5 * torch.logdet(u.T @ u)
            torch.testing.assert_close(flat.nll[i], reml, atol=1e-10, rtol=0)

    def test_flat_nll_is_invariant_to_nuisance_shifts(self) -> None:
        y_pred, target, precision, basis = _problem()
        flat = marginalize_linear_nuisance(y_pred, target, precision, basis)
        a = torch.tensor([3.0, -2.0, 5.0], dtype=F64)
        shifted = marginalize_linear_nuisance(y_pred, target + basis @ a, precision, basis)
        torch.testing.assert_close(shifted.nll, flat.nll, atol=1e-10, rtol=0)
        torch.testing.assert_close(shifted.coefficients, flat.coefficients + a, atol=1e-10, rtol=0)

    def test_partially_flat_prior_matches_dense_limit(self) -> None:
        y_pred, target, precision, basis = _problem()
        mixed = torch.tensor([0.0, 3.0, 0.5], dtype=F64)
        res = marginalize_linear_nuisance(y_pred, target, precision, basis, prior_precision=mixed)
        eps = 1e-7
        cov = torch.diag_embed(1.0 / torch.tensor([eps, 3.0, 0.5], dtype=F64)).to(dtype=F64)
        ref = _dense_nll(y_pred, target, precision, basis, cov) + 0.5 * math.log(
            eps / (2 * math.pi)
        )
        torch.testing.assert_close(res.nll, ref, atol=1e-5, rtol=0)


class TestFailClosed:
    def test_empty_row_with_flat_prior_raises(self) -> None:
        y_pred, target, precision, basis = _problem()
        mask = torch.ones_like(target, dtype=torch.bool)
        mask[2] = False
        with pytest.raises(ValueError, match="1 row"):
            marginalize_linear_nuisance(y_pred, target, precision, basis, mask=mask)

    def test_unconstrained_flat_column_raises(self) -> None:
        y_pred, target, precision, basis = _problem(n=20)
        local = torch.zeros(20, 1, dtype=F64)
        local[:4] = 1.0  # a local term supported only on the first samples
        mask = torch.ones_like(target, dtype=torch.bool)
        mask[0, :4] = False
        with pytest.raises(ValueError, match="1 row"):
            marginalize_linear_nuisance(
                y_pred, target, precision, torch.cat([basis, local], -1), mask=mask
            )
        # The same masked row is fine once that column has a proper prior.
        prior = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=F64)
        res = marginalize_linear_nuisance(
            y_pred,
            target,
            precision,
            torch.cat([basis, local], -1),
            prior_precision=prior,
            mask=mask,
        )
        assert torch.isfinite(res.nll).all()

    def test_collinear_columns_raise(self) -> None:
        y_pred, target, precision, basis = _problem()
        with pytest.raises(ValueError, match="not numerically positive definite"):
            marginalize_linear_nuisance(
                y_pred, target, precision, torch.cat([basis, 2.0 * basis[:, :1]], -1)
            )

    @pytest.mark.parametrize(
        "kwargs,error",
        [
            ({"prior_precision": -1.0}, ValueError),
            ({"prior_covariance": -torch.eye(3, dtype=F64)}, ValueError),
            ({"prior_covariance": torch.eye(2, dtype=F64)}, ValueError),
            ({"prior_precision": 1.0, "prior_covariance": torch.eye(3, dtype=F64)}, ValueError),
            ({"mask": torch.ones(4, 25)}, TypeError),
        ],
    )
    def test_invalid_arguments(self, kwargs, error) -> None:
        y_pred, target, precision, basis = _problem()
        with pytest.raises(error):
            marginalize_linear_nuisance(y_pred, target, precision, basis, **kwargs)

    def test_negative_precision_raises_only_on_unmasked_samples(self) -> None:
        y_pred, target, precision, basis = _problem()
        bad = precision.clone()
        bad[1, 7] = -1.0
        with pytest.raises(ValueError, match="precision must be >= 0"):
            marginalize_linear_nuisance(y_pred, target, bad, basis, prior_precision=1.0)
        mask = torch.ones_like(target, dtype=torch.bool)
        mask[1, 7] = False
        res = marginalize_linear_nuisance(
            y_pred, target, bad, basis, prior_precision=1.0, mask=mask
        )
        ref = marginalize_linear_nuisance(
            y_pred, target, precision, basis, prior_precision=1.0, mask=mask
        )
        torch.testing.assert_close(res.nll, ref.nll)

    @pytest.mark.parametrize("transpose", [False, True])
    def test_asymmetric_prior_covariance_raises(self, transpose: bool) -> None:
        y_pred, target, precision, basis = _problem(m=2)
        cov = torch.tensor([[1.0, 50.0], [0.0, 1.0]], dtype=F64)
        with pytest.raises(ValueError, match="symmetric positive definite"):
            marginalize_linear_nuisance(
                y_pred, target, precision, basis, prior_covariance=cov.T if transpose else cov
            )

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"prior_precision": torch.tensor([1.0, -1.0, 0.0], dtype=F64)}, "prior_precision"),
            ({"prior_precision": float("nan")}, "prior_precision"),
            ({"prior_covariance": -torch.eye(3, dtype=F64)}, "prior_covariance"),
            ({"basis_nan": True}, "shared basis must be finite"),
        ],
    )
    def test_deferred_checks_keep_their_messages(self, kwargs, match) -> None:
        y_pred, target, precision, basis = _problem()
        if kwargs.pop("basis_nan", False):
            basis = basis.clone()
            basis[3, 1] = float("nan")
        with pytest.raises(ValueError, match=match):
            marginalize_linear_nuisance(y_pred, target, precision, basis, **kwargs)

    def test_per_row_basis_must_be_finite_on_unmasked_samples(self) -> None:
        y_pred, target, precision, basis = _problem(rows=3)
        mask = torch.ones_like(target, dtype=torch.bool)
        mask[0, 2] = False
        u = basis.expand(3, *basis.shape).clone()
        u[0, 2, 0] = float("nan")  # masked: fine
        u[1, 5, 2] = float("inf")  # observed: must raise
        with pytest.raises(ValueError, match="finite on unmasked samples"):
            marginalize_linear_nuisance(y_pred, target, precision, u, mask=mask)
        u[1, 5, 2] = 1e200  # huge but finite at a masked sample is ignored
        mask[1, 5] = False
        res = marginalize_linear_nuisance(y_pred, target, precision, u, mask=mask)
        ref = marginalize_linear_nuisance(y_pred, target, precision, basis, mask=mask)
        torch.testing.assert_close(res.nll, ref.nll)
        torch.testing.assert_close(res.coefficients, ref.coefficients)

    def test_shape_mismatch_raises(self) -> None:
        y_pred, target, precision, basis = _problem()
        with pytest.raises(ValueError):
            marginalize_linear_nuisance(y_pred, target, precision, basis[:-1])


class TestHostSync:
    @pytest.mark.parametrize(
        "case", ["flat_shared", "float_prior", "tensor_prior", "covariance", "per_row_masked"]
    )
    def test_successful_call_synchronises_once(self, case: str, monkeypatch) -> None:
        y_pred, target, precision, basis = _problem(rows=6, n=30, m=3)
        kw: dict = {}
        if case == "float_prior":
            kw["prior_precision"] = 2.0
        elif case == "tensor_prior":
            kw["prior_precision"] = torch.tensor([0.0, 1.0, 2.0], dtype=F64)
        elif case == "covariance":
            kw["prior_covariance"] = _spd(3)
        elif case == "per_row_masked":
            basis = basis + 0.1 * torch.randn(6, 30, 3, dtype=F64)
            kw["mask"] = torch.rand(6, 30, generator=torch.Generator().manual_seed(1)) > 0.2
            kw["prior_precision"] = 1.0
        syncs: list[str] = []
        for name in ("__bool__", "__int__", "__float__", "item", "tolist"):
            real = getattr(torch.Tensor, name)

            def spy(self, *a, _real=real, _name=name, **k):
                syncs.append(_name)
                return _real(self, *a, **k)

            monkeypatch.setattr(torch.Tensor, name, spy)
        marginalize_linear_nuisance(y_pred, target, precision, basis, **kw)
        assert syncs == ["__bool__"]


class TestGradients:
    def test_gradcheck_all_inputs_covariance_prior(self) -> None:
        y_pred, target, precision, basis = _problem(rows=2, n=8, m=2)
        gen = torch.Generator().manual_seed(9)
        basis_rows = (basis + 0.1 * torch.randn(2, 8, 2, generator=gen, dtype=F64)).requires_grad_(
            True
        )
        a = torch.randn(2, 2, generator=gen, dtype=F64)
        prior_cov = (a @ a.T + torch.eye(2, dtype=F64)).requires_grad_(True)
        mask = torch.ones(2, 8, dtype=torch.bool)
        mask[1, 2] = False
        inputs = (
            y_pred.clone().requires_grad_(True),
            precision.clone().requires_grad_(True),
            basis_rows,
            prior_cov,
        )

        def fn(yp, lam, u, cov):
            sym = 0.5 * (cov + cov.T)
            res = marginalize_linear_nuisance(yp, target, lam, u, prior_covariance=sym, mask=mask)
            return res.nll, res.coefficients, res.residual

        assert torch.autograd.gradcheck(fn, inputs)

    def test_gradcheck_diagonal_and_flat_prior(self) -> None:
        y_pred, target, precision, basis = _problem(rows=2, n=10, m=3)
        # p = 0 is a boundary of the prior's domain: differentiate the flat limit
        # separately from a strictly positive diagonal prior.
        p = torch.tensor([0.3, 0.7, 2.0], dtype=F64, requires_grad=True)
        pred = y_pred.clone().requires_grad_(True)
        shared = basis.clone().requires_grad_(True)

        def fn(yp, u, prior):
            res = marginalize_linear_nuisance(yp, target, precision, u, prior_precision=prior)
            return res.nll, res.coefficient_covariance

        def flat(yp, u):
            res = marginalize_linear_nuisance(yp, target, precision, u)
            return res.nll, res.coefficients

        assert torch.autograd.gradcheck(fn, (pred, shared, p))
        assert torch.autograd.gradcheck(flat, (pred, shared))

    def test_float32_is_close_to_float64(self) -> None:
        y_pred, target, precision, basis = _problem(rows=3, n=200, m=4)
        r64 = marginalize_linear_nuisance(y_pred, target, precision, basis, prior_precision=1.0)
        r32 = marginalize_linear_nuisance(
            y_pred.float(), target.float(), precision.float(), basis.float(), prior_precision=1.0
        )
        assert r32.nll.dtype == torch.float32
        torch.testing.assert_close(r32.nll.double(), r64.nll, rtol=1e-5, atol=1e-4)


class TestAutocast:
    @pytest.mark.parametrize("per_row", [False, True])
    def test_bfloat16_autocast_does_not_demote_the_solve(self, per_row: bool) -> None:
        rows = 16  # > (m + 1) / 2: the shared basis takes the packed-Gram path
        y_pred, target, precision, basis = _problem(rows=rows, n=200, m=4, dtype=torch.float32)
        b = basis.expand(rows, *basis.shape).clone() if per_row else basis
        mask = torch.rand(rows, 200, generator=torch.Generator().manual_seed(2)) > 0.1
        kw = {"prior_precision": torch.tensor([0.0, 1.0, 2.0, 0.5]), "mask": mask}
        ref = marginalize_linear_nuisance(y_pred, target, precision, b, **kw)
        loss_ref = MarginalizedNuisanceGaussianNLLLoss()(y_pred, target, precision, b, **kw)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            res = marginalize_linear_nuisance(y_pred, target, precision, b, **kw)
            loss = MarginalizedNuisanceGaussianNLLLoss()(y_pred, target, precision, b, **kw)
        for got, want in zip(res, ref):
            assert got.dtype == torch.float32
            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
        assert loss.dtype == torch.float32
        torch.testing.assert_close(loss, loss_ref, rtol=1e-5, atol=1e-5)

    def test_half_precision_inputs_compute_in_float32(self) -> None:
        y_pred, target, precision, basis = _problem(rows=3, n=50, m=3, dtype=torch.float32)
        args = [t.bfloat16() for t in (y_pred, target, precision, basis)]
        res = marginalize_linear_nuisance(*args, prior_precision=1.0)
        ref = marginalize_linear_nuisance(*[t.float() for t in args], prior_precision=1.0)
        for got, want in zip(res, ref):
            assert got.dtype == torch.float32
            torch.testing.assert_close(got, want)


class TestLossClass:
    def test_reductions_and_row_weights(self) -> None:
        y_pred, target, precision, basis = _problem()
        nll = marginalize_linear_nuisance(y_pred, target, precision, basis, prior_precision=1.0).nll
        args = (y_pred, target, precision, basis)
        torch.testing.assert_close(
            MarginalizedNuisanceGaussianNLLLoss(reduction="none")(*args, prior_precision=1.0), nll
        )
        torch.testing.assert_close(
            MarginalizedNuisanceGaussianNLLLoss(reduction="sum")(*args, prior_precision=1.0),
            nll.sum(),
        )
        w = torch.tensor([1.0, 2.0, 0.0, 1.0], dtype=F64)
        weighted = MarginalizedNuisanceGaussianNLLLoss()(*args, prior_precision=1.0, weights=w)
        torch.testing.assert_close(weighted, (w * nll).sum() / w.sum())

    @pytest.mark.parametrize("per_row", [False, True])
    def test_loss_equals_functional_nll_without_building_extras(
        self, per_row: bool, monkeypatch
    ) -> None:
        rows = 6
        y_pred, target, precision, basis = _problem(rows=rows, n=30, m=3)
        b = basis + 0.1 * torch.randn(rows, 30, 3, dtype=F64) if per_row else basis
        mask = torch.rand(rows, 30, generator=torch.Generator().manual_seed(8)) > 0.25
        kw = {"prior_precision": torch.tensor([0.0, 1.0, 3.0], dtype=F64), "mask": mask}
        w = torch.rand(rows, dtype=F64) + 0.5
        nll = marginalize_linear_nuisance(y_pred, target, precision, b, **kw).nll
        # The loss needs only the NLL: no posterior covariance (cholesky_inverse).
        calls: list[int] = []
        real = torch.cholesky_inverse
        monkeypatch.setattr(
            torch, "cholesky_inverse", lambda *a, **k: calls.append(1) or real(*a, **k)
        )
        for reduction, want in (
            ("none", nll),
            ("sum", (w * nll).sum()),
            ("mean", (w * nll).sum() / w.sum()),
        ):
            loss_fn = MarginalizedNuisanceGaussianNLLLoss(reduction=reduction)
            got = loss_fn(y_pred, target, precision, b, weights=w, **kw)
            torch.testing.assert_close(got, want if reduction != "none" else w * nll)
        assert calls == []

    def test_fitting_through_the_loss_ignores_the_baseline(self) -> None:
        """A model trained through the flat-prior loss recovers the signal, not the baseline."""
        n = 64
        t = torch.linspace(-1.0, 1.0, n, dtype=F64)
        basis = torch.stack([torch.ones_like(t), t], dim=-1)
        signal = torch.exp(-0.5 * (t / 0.1) ** 2)
        target = (2.0 * signal + 5.0 - 3.0 * t).unsqueeze(0)  # amplitude 2 on a linear baseline
        precision = torch.full_like(target, 100.0)
        amp = torch.zeros(1, dtype=F64, requires_grad=True)
        opt = torch.optim.LBFGS([amp], max_iter=50)
        loss_fn = MarginalizedNuisanceGaussianNLLLoss()

        def closure():
            opt.zero_grad()
            loss = loss_fn(amp * signal.unsqueeze(0), target, precision, basis)
            loss.backward()
            return loss

        opt.step(closure)
        torch.testing.assert_close(amp.detach(), torch.tensor([2.0], dtype=F64), atol=1e-8, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_matches_cpu() -> None:
    y_pred, target, precision, basis = _problem()
    mask = torch.rand(target.shape, generator=torch.Generator().manual_seed(0)) > 0.2
    cpu = marginalize_linear_nuisance(
        y_pred, target, precision, basis, prior_precision=1.0, mask=mask
    )
    gpu = marginalize_linear_nuisance(
        y_pred.cuda(),
        target.cuda(),
        precision.cuda(),
        basis.cuda(),
        prior_precision=1.0,
        mask=mask.cuda(),
    )
    for a, b in zip(cpu, gpu):
        torch.testing.assert_close(b.cpu(), a)
