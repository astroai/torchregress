"""Regression tests for the patches documented in the deep-loss review.

Each test class targets one BLOCKER / HIGH bug identified during the loss
review and pins down the post-patch behavior so the pre-patch regressions
cannot silently resurface.

Contract: each TestCase below is annotated with the pre-fix failure mode
that the corresponding test catches.  Tests in classes marked
``(DETECTS_PRE_FIX)`` will FAIL on pre-fix code; tests in classes
marked ``(HIGH-LEVEL SANITY)`` verify forward-looking invariants and may
not detect pre-fix in modern PyTorch (documented inline).

Affected fixes:
* B1 (``sls.py``): :class:`SLSLoss.forward` step_counter is no longer
  decremented at the end of ``forward``, so the curriculum actually advances.
* H1 (``base.py``): :class:`WeightedLossWrapper` no longer silently drops a
  non-default ``reduction`` already configured on a wrapped loss instance.
* H2 (``mdn.py``): :class:`MixtureDensityLoss` with
  ``covariance_type='full'`` builds its Cholesky factor without an
  in-place write on a tensor that carries gradient from ``y_pred``. The
  bug surface is reliably discriminated by the ``L_matrices._version``
  capture in
  ``TestMixtureDensityLossFullCovBackward::test_full_covariance_L_version_zero``,
  even under modern PyTorch's silent absorption of in-place writes.
* H3 (``conformal.py``): :class:`LocalConformal` and
  :class:`LocalConformalMAD` ``predict_interval`` append a sentinel True
  column to the cumulative-weight hit mask so ``argmax`` cannot silently
  return ``0`` when no entry satisfies ``cum_weights >= q_level``. See
  docstring on the H3 class for the detection caveat (modern PyTorch's
  ``torch.cumsum`` is non-trivial to monkeypatch from pytest).
* H4 (``conformal.py``): :class:`CQR.debias` docstring/code alignment and
  ``n<=0`` calibration set guard.

How we verified the test/pass-on-post-fix-on-pre-fixcontract
------------------------------------------------------------
We ran a revert/restore cycle per fix.  The four bugs that surface in
modern PyTorch:

* **B1** — all three SLS tests fail on the reverted (pre-fix) ``sls.py``.
  Counter stays pinned at 0; the K=2 union-frontier unfreeze branch
  never fires.
* **H1** — three of five WLW tests fail on the reverted ``base.py``
  (the bug only manifests for the *instance* path with a non-default
  reduction).  The two class-path tests are smoke tests.
* **H4** — four of six CQR debias tests fail on the reverted
  ``conformal.py`` (the direct ``(n+1)/n`` division raises
  ZeroDivisionError on empty calibration; the alpha factor mismatch
  surfaces in `test_debias_factor_shrinks_alpha` and
  `test_debias_alpha_is_restored`).
* **H2** — verified via ``L_matrices._version`` capture at forward-return
  time (recipe steps 1-3 of the original recipe).  Pre-fix builds
  ``L_matrices`` via two in-place ``__setitem__`` writes on
  ``L_offdiag`` (``_version >= 2``); post-fix composes it via
  ``torch.where`` (``_version == 0``).  See
  ``tests/losses/test_loss_fixes.py::TestMixtureDensityLossFullCovBackward::test_full_covariance_L_version_zero``.
* **H3** — modern PyTorch 2.x handles the pre-fix code paths gracefully, so
  the bare "raise / return inf" tests cannot reliably distinguish
  pre-fix from post-fix from a test.
"""

from __future__ import annotations

import re

import pytest
import torch
import torch.nn as nn

from torchregress.losses.base import WeightedLossWrapper
from torchregress.losses.conformal import (
    CQR,
    LocalConformal,
    LocalConformalMAD,
)
from torchregress.losses.mdn import MixtureDensityLoss
from torchregress.losses.sls import SLSLoss

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# B1 (DETECTS_PRE_FIX): ``SLSLoss.forward`` step_counter monotonicity
# ---------------------------------------------------------------------------


class TestSLSLossStepCounterAdvances:
    """``SLSLoss.forward`` must advance ``step_counter`` once per call so the
    warmup boundary and ``K>1`` union-frontier unfreeze can fire.

    Pre-fix regression: ``forward`` decremented the counter after
    ``forward_frontier`` incremented it, so the counter stayed at 0 across
    the entire training run.  Verified: all three tests below fail on the
    reverted ``sls.py``.
    """

    def test_step_counter_advances_after_single_forward(self) -> None:
        loss_fn = SLSLoss(d=2, context_dim=3, K=1, warmup_steps=10, reduction="mean")
        torch.manual_seed(0)
        y_pred = torch.randn(4, 3)
        target = torch.randn(4, 2)
        _ = loss_fn(y_pred, target)
        assert loss_fn.step_counter == 1, (
            f"step_counter not 1 after one forward() call; got "
            f"{loss_fn.step_counter}.  B1 regression: ``forward`` decremented "
            "the counter after ``forward_frontier`` incremented it."
        )

    def test_step_counter_advances_monotonically(self) -> None:
        loss_fn = SLSLoss(d=2, context_dim=3, K=1, warmup_steps=10, reduction="mean")
        torch.manual_seed(0)
        y_pred = torch.randn(4, 3)
        target = torch.randn(4, 2)

        counters = [loss_fn.step_counter]
        for _ in range(5):
            _ = loss_fn(y_pred, target)
            counters.append(loss_fn.step_counter)

        assert counters == [0, 1, 2, 3, 4, 5], (
            f"step_counter stuck; got {counters}.  "
            "B1 regression: ``forward`` decremented the counter."
        )

    def test_k_gt_1_unfreezes_union_weights_after_warmup(self) -> None:
        loss_fn = SLSLoss(
            d=2,
            context_dim=3,
            K=2,
            warmup_steps=3,
            reduction="mean",
        )
        torch.manual_seed(1)
        y_pred = torch.randn(8, 3)
        target = torch.randn(8, 2)

        frontier = loss_fn.frontier
        assert frontier._freeze_weights is True, (
            "K>1 union frontier should start frozen and only unfreeze past "
            "warmup; this is a regression in the unfreeze logic itself."
        )

        n_calls = loss_fn.warmup_steps + 2
        for _ in range(n_calls):
            _ = loss_fn(y_pred, target)

        assert loss_fn.step_counter == n_calls, (
            f"After {n_calls} forward passes step_counter should be "
            f"{n_calls} (got {loss_fn.step_counter})."
        )
        assert frontier._freeze_weights is False, (
            "K>1 union frontier weights never unfroze past warmup; the "
            "``step_counter > warmup_steps`` guard never fires because "
            "the counter was stuck at 0 (B1 regression)."
        )


# ---------------------------------------------------------------------------
# H1 (DETECTS_PRE_FIX): ``WeightedLossWrapper`` preserves a wrapped instance's
#                         reduction
# ---------------------------------------------------------------------------


class TestWeightedLossWrapperPreservesReduction:
    """Wrapping a pre-instantiated torch loss with non-default reduction must
    not silently overwrite that choice.

    Pre-fix regression: ``WeightedLossWrapper(nn.MSELoss(reduction='sum'))``
    produced a wrapper with ``reduction='mean'`` (overwritten).  Verified:
    the three instance-path tests below fail on the reverted ``base.py``.
    """

    def test_instance_with_sum_reduction_preserved(self) -> None:
        wrapped = WeightedLossWrapper(nn.MSELoss(reduction="sum"))
        assert wrapped.reduction == "sum", (
            f"H1 regression: weighted wrapper's reduction silently "
            f"overwritten to {wrapped.reduction!r}"
        )

    def test_instance_with_none_reduction_preserved(self) -> None:
        wrapped = WeightedLossWrapper(nn.MSELoss(reduction="none"))
        assert wrapped.reduction == "none", (
            f"H1 regression: 'none' overwritten -> {wrapped.reduction!r}"
        )

    def test_loss_value_matches_native_sum(self) -> None:
        """Behavioural cross-check: when ``reduction='sum'`` is configured on
        the wrapped loss, the wrapper's output must equal
        ``nn.MSELoss(reduction='sum')`` rather than the (pre-fix)
        'mean'-diverged value."""
        torch.manual_seed(2)
        y = torch.randn(16, 3)
        t = torch.randn(16, 3)
        wrapper = WeightedLossWrapper(nn.MSELoss(reduction="sum"))
        ref = nn.MSELoss(reduction="sum")(y, t)
        wrapper_out = wrapper(y, t)

        assert torch.isclose(wrapper_out, ref), (
            f"WeightedLossWrapper sum-reduction disagrees with native "
            f"torch loss: wrapper={wrapper_out.item():.4f} vs "
            f"ref={ref.item():.4f}.  Pre-fix silently reduces to 'mean'."
        )

    def test_class_default_reduction_mean(self) -> None:
        """Class path (no pre-existing setting -> falls back to 'mean')."""
        wrapped = WeightedLossWrapper(nn.MSELoss)
        assert wrapped.reduction == "mean"

    def test_explicit_override_still_wins(self) -> None:
        """Caller-supplied reduction must override any prior class fallback."""
        wrapped = WeightedLossWrapper(nn.MSELoss(reduction="sum"), reduction="mean")
        assert wrapped.reduction == "mean"


# ---------------------------------------------------------------------------
# H2 (HIGH-LEVEL SANITY): ``MixtureDensityLoss`` full-cov forward/backward
# ---------------------------------------------------------------------------


class TestMixtureDensityLossFullCovBackward:
    """``MixtureDensityLoss(covariance_type='full')`` must allow
    ``loss.backward()`` even when ``y_pred`` carries gradient, and must
    build its Cholesky factor functionally (i.e. without an in-place
    mutation that bumps the autograd version-counter).

    Pre-fix (deep review): the in-place write to ``L_matrices[..., diag]``
    on a tensor registered into autograd was reported to raise
    ``RuntimeError: a leaf Variable that requires grad is being used in an
    in-place operation``.  Modern PyTorch 2.x silently accommodates this
    pattern via version-counter autograd, so the bare ``RuntimeError`` test
    no longer fires.  The discriminator ``test_full_covariance_L_version_zero``
    below wraps ``_extract_distribution_parameters`` to capture
    ``L_matrices._version`` immediately after forward.  Post-fix code
    composes ``L_matrices`` via ``torch.where(diag_mask, L_diag,
    L_offdiag)`` (a brand-new tensor with ``_version == 0``); pre-fix code
    derived ``L_matrices`` via two ``__setitem__`` writes onto
    ``L_offdiag`` (a tensor whose ``_version >= 2`` after the writes).
    Asserting ``L_matrices._version == 0`` discriminates pre-fix from
    post-fix at runtime, surviving PyTorch's silent absorption.
    """

    @staticmethod
    def _expected_size(n_components: int, n_features: int) -> int:
        return (
            n_components
            + n_components * n_features
            + n_components * n_features * (n_features + 1) // 2
        )

    def test_full_covariance_backward_runs(self) -> None:
        torch.manual_seed(0)
        loss_fn = MixtureDensityLoss(
            n_components=2,
            n_features=3,
            covariance_type="full",
            reduction="mean",
        )
        y_pred = torch.randn(8, self._expected_size(2, 3), requires_grad=True)
        target = torch.randn(8, 3)
        loss = loss_fn(y_pred, target)
        loss.backward()
        assert y_pred.grad is not None
        assert torch.isfinite(y_pred.grad).all(), (
            "y_pred.grad has non-finite entries after full-cov backward"
        )

    def test_full_covariance_loss_finite(self) -> None:
        torch.manual_seed(0)
        loss_fn = MixtureDensityLoss(n_components=2, n_features=2, covariance_type="full")
        y_pred = torch.randn(4, self._expected_size(2, 2))
        target = torch.randn(4, 2)
        loss = loss_fn(y_pred, target)
        assert torch.isfinite(loss), f"non-finite forward loss: {loss.item()}"

    def test_full_covariance_param_update_step(self) -> None:
        """End-to-end: a single optimizer step on full-cov MDN with
        grad-enabled y_pred exercises the entire in-place / grad chain."""
        torch.manual_seed(0)
        loss_fn = MixtureDensityLoss(n_components=2, n_features=3, covariance_type="full")
        param = nn.Parameter(torch.randn(8, self._expected_size(2, 3)))
        target = torch.randn(8, 3)
        opt = torch.optim.SGD([param], lr=1e-3)

        opt.zero_grad()
        loss_fn(param, target).backward()
        opt.step()
        assert torch.isfinite(param).all(), "param diverged after step."

    def test_full_covariance_L_version_zero(self) -> None:
        """Runtime discriminator for the pre-fix H2 in-place bug surface,
        applying recipe steps 1-3 of the docs/loss_test_coverage.md 4-step
        recipe (``L._version`` snapshot before vs after the in-place
        scatter).

        We wrap ``MixtureDensityLoss._extract_distribution_parameters`` so
        that we can read ``L_matrices._version`` immediately after
        forward.  Post-fix code composes ``L_matrices`` purely functionally
        via ``torch.where(diag_mask, L_diag, L_offdiag)``, which yields a
        brand-new tensor with ``_version == 0`` at return time.  Pre-fix
        code path derived ``L_matrices`` via two ``__setitem__`` writes
        onto ``L_offdiag``, bumping its version to ``>= 2`` by the time it
        was returned.  Asserting ``L_matrices._version == 0`` discriminates
        pre-fix from post-fix at runtime, surviving PyTorch's silent
        absorption of the in-place pattern.

        Note on recipe step 4 (``L.grad`` parity vs manual Cholesky
        recompute):  This step is omitted here because it does NOT
        discriminate the H2 bug surface under modern PyTorch.  Both the
        pre-fix in-place scatter and the post-fix functional
        ``torch.where`` produce mathematically identical NLL output and
        yield identical ``y_pred.grad`` -- grad parity evaluates green on
        both the bug and the fix.  The ``L._version`` snapshot is the only
        reliable runtime signal for this bug class.
        """
        import torchregress.losses.mdn as mdn_mod

        torch.manual_seed(0)
        n_components, n_features = 2, 3
        out_dim = self._expected_size(n_components, n_features)

        loss_fn = MixtureDensityLoss(
            n_components=n_components,
            n_features=n_features,
            covariance_type="full",
            reduction="mean",
        )
        y_pred = torch.randn(8, out_dim, requires_grad=True)
        target = torch.randn(8, n_features)

        # Wrap ``_extract_distribution_parameters`` to capture
        # ``L_matrices._version`` immediately after construction.  The
        # wrapper is installed only for this test's lifetime and restored
        # in the ``finally`` block so it does not leak to other tests.
        captured_versions: list[int] = []
        _orig_extract = mdn_mod.MixtureDensityLoss._extract_distribution_parameters

        def _wrapped_extract(self_instance: "MixtureDensityLoss", y_pred_in: torch.Tensor):
            w, m, L_or_stds = _orig_extract(self_instance, y_pred_in)
            if self_instance.covariance_type == "full":
                captured_versions.append(int(L_or_stds._version))
            return w, m, L_or_stds

        mdn_mod.MixtureDensityLoss._extract_distribution_parameters = _wrapped_extract
        try:
            loss = loss_fn(y_pred, target)
            loss.backward()
        finally:
            mdn_mod.MixtureDensityLoss._extract_distribution_parameters = _orig_extract

        assert captured_versions, (
            "H2 setup: ``_extract_distribution_parameters`` wrapper did "
            "not capture any versions -- the test setup is broken."
        )
        assert captured_versions[0] == 0, (
            f"H2 regression: post-fix ``L_matrices`` must be a brand-new "
            f"tensor composed functionally with ``_version == 0`` at "
            f"forward-return time; observed "
            f"``_version == {captured_versions[0]}``.  Pre-fix derived "
            f"``L_matrices`` via in-place ``__setitem__`` mutations, "
            f"bumping the version to 2+ -- the documented pre-fix leaf-"
            f"tensor in-place-write runtime error surface."
        )

        # Note: empirical verification ran via
        # ``_archive/verify_historical_pre_fix_inject.py`` -- both the
        # synthesised post-fix-buffer-plus-extra-write form and the literal
        # historical recipe (``L = zeros; L[..., tril] = ...; L[..., diag,
        # diag] = F.softplus(...) + min_std``) produced ``_version >= 2``
        # and the discriminator rejected both (empirical observation
        # ranged from 2 to 3 depending on PyTorch's view-alias handling).

    def test_full_covariance_cholesky_no_inplace_graphnode(self) -> None:
        """Structural verification of the fix: post-fix the
        ``_extract_distribution_parameters`` for full covariance must
        build the Cholesky factor via scatter + functional diag
        composition (``L_matrices = torch.where(diag_mask, L_diag,
        L_offdiag)``) rather than the pre-fix in-place write ``
        L[..., diag, diag] = softplus(L[..., diag, diag]) + min_std``.

        This is a source-AST check that survives bytecode reloads and
        local ``torch.cumsum`` / autograd shadowing -- the test enforces
        the *pre-fix-incompatible pattern ban* rather than the specific
        runtime error.  Pre-fix code with the in-place line will FAIL
        this test."""
        from torchregress.losses import mdn as mdn_mod

        source = mdn_mod.__file__
        with open(source, "r") as f:
            code = f.read()

        # The in-place write pattern that was the original bug:
        bad_pattern = re.compile(
            r"L_matrices\[\.\.\.,\s*diag_indices,\s*diag_indices\]\s*="
            r"\s*\n?\s*F\.softplus\("
        )
        assert not bad_pattern.search(code), (
            "H2 regression: full-cov Cholesky factor rebuilt via the "
            "pre-fix in-place write pattern.  The fix should compose "
            "L_matrices functionally with `torch.where(diag_mask, ...)`."
        )

        # The post-fix functional composition pattern must be present:
        good_pattern = re.compile(r"L_matrices\s*=\s*torch\.where\(")
        assert good_pattern.search(code), (
            "H2 fix absence: full-cov Cholesky factor should be "
            "constructed functionally via `torch.where`."
        )


# ---------------------------------------------------------------------------
# H3 (DETECTS_PRE_FIX via source check; runtime patch is brittle — see note):
#     ``LocalConformal`` / ``LocalConformalMAD`` argmax sentinel fallback
# ---------------------------------------------------------------------------


class TestLocalConformalSentinelFallback:
    """When ``cum_weights`` cannot reach ``q_level`` (the float32-underflow
    edge case at extreme ``alpha``), ``predict_interval`` must surface an
    unbounded interval bound (``+inf``) via the sentinel column rather than
    silently collapsing to ``idx=0`` (which gives the smallest residual and
    would under-cover).

    Pre-fix behavior: ``argmax((cum_weights >= q_level).int())`` over an
    all-False mask returns 0 ⇒ ``q = resids[0]`` (smallest residual).

    Post-fix behavior: the appended sentinel ``True`` column guarantees
    ``argmax`` lands on the appended ``+inf`` calibration entry ⇒ ``q = inf``
    ⇒ unbounded interval (a safe conservative signal).

    Runtime detection note
    ----------------------
    Patching ``torch.cumsum`` from pytest's ``monkeypatch.setattr`` does
    not reliably take effect in modern PyTorch's C-extension-backed
    ``torch.cumsum`` attribute (verified empirically on Python 3.13 /
    PyTorch 2.x).  The cleanest discriminative test is therefore a
    **source-AST check**: the post-fix code must contain the
    ``torch.cat(..., sentinel)`` composition, which the pre-fix code
    does not.

    We additionally include a behavioural sanity test that verifies
    post-fix intervals are finite and well-formed (this test passes on
    both pre-fix and post-fix code, but guards future regressions).
    """

    @staticmethod
    def _build_calibrated_localcp(alpha: float, n_cal: int = 30) -> LocalConformal:
        cp = LocalConformal(alpha=alpha, bandwidth=1.0)
        torch.manual_seed(0)
        x_cal = torch.randn(n_cal, 2)
        # Linspace so the sorted residuals are non-trivial (0 != max).
        y_cal = torch.linspace(0.0, 5.0, n_cal).unsqueeze(1)
        cp.calibrate(y_cal, y_cal, x=x_cal)
        return cp

    def test_no_collapse_in_normal_alpha(self) -> None:
        """Sanity baseline: at typical alpha, intervals are finite and
        non-degenerate in both pre-fix and post-fix."""
        cp = self._build_calibrated_localcp(0.1)
        x_test = torch.randn(5, 2)
        y_test = torch.zeros(5, 1)
        lower, upper = cp.predict_interval(y_pred=y_test, x=x_test)
        assert torch.isfinite(lower).all() and torch.isfinite(upper).all(), (
            "LocalConformal interval produced non-finite bounds at normal "
            "alpha; this is a regression regardless of pre/post-fix state."
        )
        assert (upper >= lower).all(), f"Invalid interval width: lower={lower}, upper={upper}"

    def test_local_conformal_mad_no_collapse_in_normal_alpha(self) -> None:
        """Same baseline for the MAD variant."""
        mad_cp = LocalConformalMAD(alpha=0.1, bandwidth=1.0)
        torch.manual_seed(0)
        x_cal = torch.randn(30, 2)
        y_cal = torch.linspace(0.0, 5.0, 30).unsqueeze(1)
        mad_cal = torch.ones(30, 1) * 0.1
        mad_cp.calibrate(y_cal, y_cal, x=x_cal, mad=mad_cal)
        x_test = torch.randn(5, 2)
        y_test = torch.zeros(5, 1)
        mad_test = torch.ones(5, 1) * 0.1
        lower, upper = mad_cp.predict_interval(y_pred=y_test, x=x_test, mad=mad_test)
        assert torch.isfinite(lower).all() and torch.isfinite(upper).all()
        assert (upper >= lower).all()

    def test_local_conformal_predict_interval_uses_sentinel(self) -> None:
        """Source-string check for the sentinel column.

        Pre-fix without the sentinel column will FAIL this test.  We
        use a regex rather than AST because ``torch.cat([hits,
        sentinel], dim=1)`` flattens to a ``ast.List`` of ``ast.Name``
        children, which is awkward to verify via ``ast.walk`` and the
        string regex is more robust.
        """
        from torchregress.losses import conformal as conf_mod

        source = conf_mod.__file__
        with open(source, "r") as f:
            code = f.read()

        # Split the source at ``class `` boundaries so we can search
        # within each class body independently.
        classes = re.split(r"^class\s+(\w+)\b", code, flags=re.MULTILINE)
        # ``classes`` becomes ``['', 'ClassName', body1, 'ClassName', body2, ...]``.
        sentinel_per_class: dict[str, bool] = {}
        it = iter(classes[1:])
        for cls_name in it:
            body = next(it, "")
            sentinel_per_class[cls_name] = bool(
                # Post-fix pattern: torch.cat([..., sentinel], dim=...).
                re.search(
                    r"torch\.cat\(\[\s*hits\s*,\s*sentinel\s*\]\s*,",
                    body,
                )
                # The sentinel tensor itself is built via
                # ``sentinel = torch.ones((hits.shape[0], 1), ...)``.
                and re.search(
                    r"sentinel\s*=\s*torch\.ones\(\s*\(?\s*hits\.shape\[0\]",
                    body,
                )
            )

        assert sentinel_per_class.get("LocalConformal", False), (
            "H3 regression: LocalConformal.predict_interval does not "
            "include the sentinel-column fallback.  The post-fix path "
            "is missing (pre-fix: argmax of all-False hits returns 0)."
        )
        assert sentinel_per_class.get("LocalConformalMAD", False), (
            "H3 regression: LocalConformalMAD.predict_interval does not "
            "include the sentinel-column fallback."
        )


# ---------------------------------------------------------------------------
# H4 (DETECTS_PRE_FIX): ``CQR.debias`` correctly applies ``alpha * n / (n + 1)``
#     and guards ``n <= 0``
# ---------------------------------------------------------------------------


class TestCQRDebiasDocCodeAlignment:
    """Two facets of the H4 fix:

    - When ``debias=True``, the effective alpha used for calibration should
      be ``alpha * n / (n + 1)`` (shrinks alpha -> widens interval).
    - When the calibration set is empty (``n <= 0``), ``ValueError`` must be
      raised instead of silently applying the (now-degenerate) correction.

    Verified: four of the six tests below fail on the reverted
    ``conformal.py`` (the direct ``(n+1)/n`` division raises
    ``ZeroDivisionError`` on empty calibration; the alpha factor
    mismatch surfaces in :func:`test_debias_factor_shrinks_alpha` and
    :func:`test_debias_alpha_is_restored`).
    """

    def test_debias_factor_shrinks_alpha(self) -> None:
        """``debias=True`` with ``n=99`` must produce the same effective
        alpha as ``alpha * 99 / 100 = 0.099``.
        Pre-fix used ``alpha * (n+1)/n = 0.101⋯`` (which gives a different
        q_hat rank), so this test fails on pre-fix."""
        torch.manual_seed(4)
        n_cal = 99
        y_cal = torch.randn(n_cal, 1)
        y_pred = torch.cat(
            [
                y_cal - 0.4 * torch.rand(n_cal, 1),
                y_cal + 0.4 * torch.rand(n_cal, 1),
            ],
            dim=-1,
        )

        cp_debiased = CQR(alpha=0.1, debias=True)
        cp_explicit = CQR(alpha=0.1 * 99 / 100, debias=False)
        cp_debiased.calibrate(y_pred, y_cal)
        cp_explicit.calibrate(y_pred, y_cal)

        assert torch.isclose(cp_debiased.q_hat.float(), cp_explicit.q_hat.float(), atol=1e-6), (
            f"CQR.debias used an unexpected alpha factor: "
            f"q_debiased={float(cp_debiased.q_hat):.6f} vs "
            f"q_alpha_n/(n+1)={float(cp_explicit.q_hat):.6f}"
        )

    def test_debias_alpha_is_restored_after_calibrate(self) -> None:
        """The internal alpha mutation in calibrate must not leak.

        Pre-fix had no try/finally restoration, so
        ``debias=True`` with ``n=30`` left ``self.alpha = 0.1 * 31/30``.
        """
        torch.manual_seed(1)
        cp = CQR(alpha=0.1, debias=True)
        n_cal = 30
        y_cal = torch.randn(n_cal, 1)
        y_pred = torch.cat([y_cal - 0.3, y_cal + 0.3], dim=-1)
        cp.calibrate(y_pred, y_cal)
        assert cp.alpha == pytest.approx(0.1), f"alpha leaked during calibrate: {cp.alpha}"

    def test_debias_with_empty_calibration_raises(self) -> None:
        """An empty calibration set must raise a H4-specific ``ValueError``
        rather than a ``ZeroDivisionError`` from the unguarded alpha-factor
        computation in pre-fix code (``alpha * (n+1)/n`` with ``n=0``)."""
        cp = CQR(alpha=0.1, debias=True)
        with pytest.raises(ValueError, match="CQR\\.debias requires"):
            cp.calibrate(torch.empty(0, 2), torch.empty(0, 1))

    def test_debias_with_all_masked_samples_raises(self) -> None:
        """A calibration set whose mask filters every sample must also raise
        a H4-specific ValueError (not the parent-class "Calibration set is
        empty..." message whose ``match`` would not discriminate against
        pre-fix that didn't have the ``n<=0`` guard)."""
        cp = CQR(alpha=0.1, debias=True)
        n_cal = 5
        y_pred = torch.randn(n_cal, 2)
        y_cal = torch.randn(n_cal, 1)
        mask = torch.zeros(n_cal, dtype=torch.bool)
        with pytest.raises(ValueError, match="CQR\\.debias requires"):
            cp.calibrate(y_pred, y_cal, mask=mask)

    def test_debias_widens_q_hat(self) -> None:
        """``debias=True`` shrinks alpha -> widens q_hat."""
        torch.manual_seed(0)
        n_cal = 50
        y_cal = torch.randn(n_cal, 1)
        lo = y_cal - 0.3 * torch.rand(n_cal, 1) - 0.1
        hi = y_cal + 0.3 * torch.rand(n_cal, 1) + 0.1
        y_pred = torch.cat([lo, hi], dim=-1)

        cp_clean = CQR(alpha=0.1, debias=False)
        cp_debiased = CQR(alpha=0.1, debias=True)
        cp_clean.calibrate(y_pred, y_cal)
        cp_debiased.calibrate(y_pred, y_cal)

        q_clean = float(cp_clean.q_hat)
        q_debiased = float(cp_debiased.q_hat)
        assert q_debiased >= q_clean - 1e-6, (
            f"debias=True should widen: q_clean={q_clean}, q_debiased={q_debiased}"
        )

    def test_debias_false_does_not_mutate_alpha(self) -> None:
        """``debias=False`` is a pure pass-through and must not touch
        ``self.alpha``."""
        torch.manual_seed(5)
        cp = CQR(alpha=0.1, debias=False)
        n_cal = 30
        y_cal = torch.randn(n_cal, 1)
        y_pred = torch.cat([y_cal - 0.3, y_cal + 0.3], dim=-1)
        cp.calibrate(y_pred, y_cal)
        assert cp.alpha == pytest.approx(0.1)
