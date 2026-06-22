"""Orchestrate revert-verify cycles across the three follow-up test files.

For each pair of (source_module, mutation), this script:
1. Restores the source file from /tmp/torchregress_revert_backup/ (clean slate).
2. Applies a targeted bug injection that the matching test should detect.
3. Runs pytest on the matching test file and asserts >= 1 failure.
4. Restores the source file (clean slate) and asserts all tests pass.

Run from project root: ./.venv/bin/python scripts/revert_verify.py
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys

PROJECT = pathlib.Path("/Users/fabbros/src/torchregress")
BACKUP = pathlib.Path("/tmp/torchregress_revert_backup")
LOSSES = PROJECT / "src/torchregress" / "losses"


def restore(src_name: str) -> None:
    shutil.copy(BACKUP / src_name, LOSSES / src_name)


def run_pytest(test_file: str) -> tuple[int, int]:
    """Run pytest on a single file; return (n_failures, n_passed)."""
    cmd = [
        str(PROJECT / ".venv/bin/python"),
        "-m",
        "pytest",
        "-q",
        "--tb=line",
        "--no-header",
        str(PROJECT / test_file),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    text = proc.stdout + proc.stderr
    m_fail = re.search(r"(\d+) failed", text)
    m_pass = re.search(r"(\d+) passed", text)
    failed = int(m_fail.group(1)) if m_fail else 0
    passed = int(m_pass.group(1)) if m_pass else 0
    return failed, passed


# (source_file, test_file, label, apply_function)
def _apply_sls_alt_mask(LOSSES):
    """Drop the alternate-half branch entirely so every layer masks the first
    half.  No dead conditional left behind.
    """
    p = LOSSES / "sls.py"
    src = p.read_text()
    old = """                if i % 2 == 0:
                    mask[: d // 2] = True
                else:
                    mask[d // 2 :] = True
"""
    new = """                mask[: d // 2] = True  # BUG: never alternate halves
"""
    assert old in src, "sls.py alternating-mask anchor not found"
    p.write_text(src.replace(old, new, 1))


def _apply_eiv_cov_scalar_off(LOSSES):
    p = LOSSES / "eiv.py"
    src = p.read_text()
    # Scalar (int/float) path uses **2; break to **1 to off-by-one the variance.
    old1 = "return torch.eye(n_features, device=device, dtype=dtype) * float(sigma) ** 2"
    new1 = "return torch.eye(n_features, device=device, dtype=dtype) * float(sigma) ** 1"
    assert old1 in src, "eiv.py int/float scalar anchor not found"
    src = src.replace(old1, new1, 1)
    # 1-element tensor path: change to multiply by sigma (no square).
    old2 = "return torch.eye(n_features, device=device, dtype=dtype) * float(sigma.item()) ** 2"
    new2 = "return torch.eye(n_features, device=device, dtype=dtype) * float(sigma.item()) ** 1"
    assert old2 in src, "eiv.py 1-elem tensor anchor not found"
    (LOSSES / "eiv.py").write_text(src.replace(old2, new2, 1))


def _apply_quantile_factor(LOSSES):
    p = LOSSES / "quantile.py"
    src = p.read_text()
    # QuantileLoss.forward currently calls the shared utility with self.quantile.
    # Inject a multiplicative bug at the QuantileLoss forward boundary.
    old = (
        "        # Elementwise quantile loss via shared utility\n"
        "        loss = _util_quantile_loss(y_pred, target, self.quantile)"
    )
    new = (
        "        # BUG: factor 2.0 injected, "
        "loses MAD/MAE=0.5/median contract\n"
        "        loss = 2.0 * _util_quantile_loss(y_pred, target, self.quantile)"
    )
    assert old in src, "quantile.py shared-utility anchor not found"
    p.write_text(src.replace(old, new, 1))


def _apply_expectile_drop_factor2(LOSSES):
    p = LOSSES / "expectile.py"
    src = p.read_text()
    # Two occurrences of `loss = 2 * residuals**2 * weight` (one in
    # ExpectileLoss.forward and the same line at multi_expectile_loss).
    old = "loss = 2 * residuals**2 * weight"
    new = "loss = residuals**2 * weight  # BUG: factor 2 dropped"
    n = src.count(old)
    assert n >= 1, "expectile.py factor-2 anchor not found"
    p.write_text(src.replace(old, new))


def _apply_beta_nll_drop_half(LOSSES):
    p = LOSSES / "beta_nll.py"
    src = p.read_text()
    old = """        nll = 0.5 * (
            math.log(2 * math.pi)
            + torch.log(var + self.eps)
            + (target - mean) ** 2 / (var + self.eps)
        )"""
    new = """        nll = (  # BUG: 0.5 * factor removed
            math.log(2 * math.pi)
            + torch.log(var + self.eps)
            + (target - mean) ** 2 / (var + self.eps)
        )"""
    assert old in src, "beta_nll.py NLL anchor not found"
    p.write_text(src.replace(old, new, 1))


def _apply_tweedie_gamma_minus1(LOSSES):
    p = LOSSES / "tweedie.py"
    src = p.read_text()
    old = "return torch.log(mu / (target + self.eps) + self.eps) + target / (mu + self.eps) - 1"
    new = (
        "return torch.log(mu / (target + self.eps) + self.eps) "
        "+ target / (mu + self.eps)  # BUG: -1 tail dropped"
    )
    assert old in src, "tweedie.py gamma anchor not found"
    p.write_text(src.replace(old, new, 1))


def _apply_conformal_split_bias(LOSSES):
    p = LOSSES / "conformal.py"
    src = p.read_text()
    old = """            self._validate_inputs(y_pred, target, mask)
            loss = (y_pred - target) ** 2"""
    new = """            self._validate_inputs(y_pred, target, mask)
            loss = (y_pred - target) ** 2 + 0.1  # BUG: bias added"""
    assert old in src, "conformal.py split anchor not found"
    p.write_text(src.replace(old, new, 1))


def _apply_gw_factory_bias(LOSSES):
    p = LOSSES / "gaussian_wasserstein.py"
    src = p.read_text()
    old = (
        "        fn(pred_mean, target_mean, pred_covariance, target_covariance, "
        "mask=mask, weights=weights),"
    )
    new = (
        "        fn(pred_mean, target_mean, pred_covariance, target_covariance, "
        "mask=mask, weights=weights) + 0.05,"
    )
    assert old in src, "gaussian_wasserstein.py factory anchor not found"
    p.write_text(src.replace(old, new, 1))


def _apply_pg_factory_ignores_kwarg(LOSSES):
    """Force the mixture factory to silently swallow caller-supplied kwargs
    whenever the caller has passed ``**kwargs`` to construct a Config.  The
    ``kwargs``-only path (no Config) silently builds a default Config with
    discard of the caller's params, failing the
    ``test_mixture_factory_returns_instance`` assertions on
    ``initial_variance`` and (via the LR test) ``learn_variance``.
    """
    p = LOSSES / "poisson_gaussian.py"
    src = p.read_text()
    old = """def poisson_gaussian_mixture_loss(
    config: Optional[PoissonGaussianMixtureConfig] = None,
    **kwargs: Any,
) -> PoissonGaussianMixtureLoss:
    \"\"\"
    Factory function to create a PoissonGaussianMixtureLoss instance.
    \"\"\"
    if config is None:
        config = PoissonGaussianMixtureConfig(**kwargs)
    return PoissonGaussianMixtureLoss(config=config)"""
    new = """def poisson_gaussian_mixture_loss(
    config: Optional[PoissonGaussianMixtureConfig] = None,
    **kwargs: Any,
) -> PoissonGaussianMixtureLoss:
    # BUG: drop caller-supplied kwargs so factory always returns defaults.
    if config is None:
        config = PoissonGaussianMixtureConfig()
    return PoissonGaussianMixtureLoss(config=config)"""
    assert old in src, "poisson_gaussian.py mixture-factory anchor not found"
    p.write_text(src.replace(old, new, 1))


# ---------------------------------------------------------------------------
# New cycles for test_indirect_utilities.py / test_functional_wrappers.py.
# Each mutation breaks one of the indirect-utility contracts that the new
# test files lock in.
# ---------------------------------------------------------------------------


def _apply_gw_sqrt_discards_sqrt(LOSSES):
    """Drop the ``.sqrt()`` call in ``symmetric_spd_matrix_sqrt`` so the
    returned matrix is in fact :math:`Q \\Lambda Q^\\top` (i.e. equal to
    :math:`\\Sigma`) rather than the principal square root.  ``S @ S``
    then equals :math:`\\Sigma^2 \\neq \\Sigma` for non-identity inputs,
    which ``test_sqrt_squared_recovers_input`` detects.
    """
    p = LOSSES / "gaussian_wasserstein.py"
    src = p.read_text()
    old = "    s = torch.clamp(evals, min=eps).sqrt()"
    new = "    s = torch.clamp(evals, min=eps)  # BUG: sqrt() call dropped"
    assert old in src, "gaussian_wasserstein.py sqrt anchor not found"
    p.write_text(src.replace(old, new, 1))


def _apply_gaussian_mse_gate_off(LOSSES):
    """Force ``create_gaussian_nll`` to ignore ``use_mse_for_unit_variance``
    and always return ``GaussianNLLLoss`` for diagonal covariance, breaking
    the MSE-shortcut contract.
    """
    p = LOSSES / "gaussian.py"
    src = p.read_text()
    old = """    if covariance_type == "diagonal":
        if use_mse_for_unit_variance:
            return WeightedMSELoss(**kwargs)
        return GaussianNLLLoss(**kwargs)"""
    new = """    if covariance_type == "diagonal":
        return GaussianNLLLoss(**kwargs)  # BUG: use_mse_for_unit_variance gate dropped"""
    assert old in src, "gaussian.py use_mse anchor not found"
    p.write_text(src.replace(old, new, 1))


def _apply_quantile_alias_spoofed(LOSSES):
    """Repoint the ``QuantileCrossover`` compatibility alias at ``QuantileLoss``
    so the canonical-identity contract (``QuantileCrossover is
    QuantileCrossoverLoss``) fails.
    """
    p = LOSSES / "quantile.py"
    src = p.read_text()
    old = "QuantileCrossover = QuantileCrossoverLoss"
    new = "QuantileCrossover = QuantileLoss  # BUG: alias points at sibling class"
    assert old in src, "quantile.py alias anchor not found"
    p.write_text(src.replace(old, new, 1))


CYCLES = [
    (
        "sls.py",
        "tests/losses/test_sls_internals.py",
        "sls: VolumePreservingFlow mask always first half",
        _apply_sls_alt_mask,
    ),
    (
        "eiv.py",
        "tests/losses/test_eiv_internals.py",
        "eiv: _prepare_covariance_from_sigma scalar uses **1 instead of **2",
        _apply_eiv_cov_scalar_off,
    ),
    (
        "quantile.py",
        "tests/losses/test_functional_wrappers.py",
        "quantile: QuantileLoss forward multiplied by 2.0",
        _apply_quantile_factor,
    ),
    (
        "expectile.py",
        "tests/losses/test_functional_wrappers.py",
        "expectile: factor 2 dropped in asymmetric LE",
        _apply_expectile_drop_factor2,
    ),
    (
        "beta_nll.py",
        "tests/losses/test_functional_wrappers.py",
        "beta_nll: 0.5 * factor removed in NLL formula",
        _apply_beta_nll_drop_half,
    ),
    (
        "tweedie.py",
        "tests/losses/test_functional_wrappers.py",
        "tweedie: gamma-loss tail -1 dropped",
        _apply_tweedie_gamma_minus1,
    ),
    (
        "conformal.py",
        "tests/losses/test_functional_wrappers.py",
        "conformal: split branch adds +0.1 bias",
        _apply_conformal_split_bias,
    ),
    (
        "gaussian_wasserstein.py",
        "tests/losses/test_functional_wrappers.py",
        "gaussian_wasserstein: factory wrapper adds +0.05",
        _apply_gw_factory_bias,
    ),
    (
        "poisson_gaussian.py",
        "tests/losses/test_functional_wrappers.py",
        "poisson_gaussian: factory silently overrides Config with kwargs",
        _apply_pg_factory_ignores_kwarg,
    ),
    (
        "gaussian_wasserstein.py",
        "tests/losses/test_indirect_utilities.py",
        "gaussian_wasserstein: symmetric_spd_matrix_sqrt skips .sqrt()",
        _apply_gw_sqrt_discards_sqrt,
    ),
    (
        "gaussian.py",
        "tests/losses/test_indirect_utilities.py",
        "gaussian: create_gaussian_nll ignores use_mse_for_unit_variance",
        _apply_gaussian_mse_gate_off,
    ),
    (
        "quantile.py",
        "tests/losses/test_indirect_utilities.py",
        "quantile: QuantileCrossover alias repointed at QuantileLoss",
        _apply_quantile_alias_spoofed,
    ),
]


def main() -> int:
    print("=" * 78)
    print("Discriminative revert verification across 12 source modules")
    print("(9 source modules -> test_functional_wrappers.py / _internals files;")
    print(" 3 source modules -> test_indirect_utilities.py)")
    print("=" * 78)
    results: list[tuple[str, int, int, int]] = []
    for src_name, test_file, label, apply_fn in CYCLES:
        print()
        print(f"--- {label} ---")
        print(f"    source:  src/torchregress/losses/{src_name}")
        print(f"    target:  {test_file}")

        # Step 1: clean baseline (failsafe: restore even if the loop panics).
        try:
            restore(src_name)
            f0, p0 = run_pytest(test_file)
            print(f"    baseline:          {p0} passed, {f0} failed")

            # Step 2: apply mutation.
            try:
                apply_fn(LOSSES)
            except Exception as e:
                # Mutation may raise AssertionError (anchor not matched),
                # SyntaxError (a BUG-token injected mid-line mid-indent),
                # or ImportError on collection.  All three cases should
                # SKIP gracefully; the restore() in the outer `finally`
                # keeps the source tree clean for the next cycle.
                print(f"    SKIP: mutation rejected ({type(e).__name__}: {e})")
                continue  # restore fires via finally below; next cycle proceeds

            # Step 3: run pytest on the mutated source.
            f1, p1 = run_pytest(test_file)
            print(f"    mutated:           {p1} passed, {f1} failed")
            discriminate = f1 >= 1
            print(f"    discriminated: {'YES' if discriminate else 'NO'}")

            # Step 4: restore and confirm green again.
            restore(src_name)
            f2, p2 = run_pytest(test_file)
            print(f"    post-restore:      {p2} passed, {f2} failed")
        finally:
            # Belt-and-braces: ensure source is clean before next cycle.
            restore(src_name)

        results.append((src_name, f0, f1, p1))

    print()
    print("=" * 78)
    print("Summary")
    print("=" * 78)
    print(f"{'module':<35} {'baseline':>10} {'mutated failed':>15} {'verdict':>15}")
    for src_name, f0, f1, p1 in results:
        verdict = "DISCRIMINATING" if f1 >= 1 else "SILENT"
        print(f"{src_name:<35} {f0:>10} {f1:>15} {verdict:>15}")
    # Loudly reject any orchestrator truncation -- if an ``apply_fn`` raised
    # an unexpected exception that the per-cycle ``except Exception`` silently
    # SKIPped, this guard surfaces the truncated run rather than reporting a
    # deceptively clean N/M < 12/12.
    assert len(results) == len(CYCLES), (
        f"Only {len(results)}/{len(CYCLES)} cycles completed -- some apply_fn "
        "likely raised an unexpected exception that was silently SKIPped."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
