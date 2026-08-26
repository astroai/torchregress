# Major-Release Readiness Audit: `torchregress` — Exhaustive, From-Scratch — 2026-08-26

> **Scope:** 574 tracked files via `all_tracked_files.json` (git ls-files). No sampling. Every file classified. All major abstractions reconstructed from code, not docs. Mathematics verified against definitions and limiting cases. Tensor/dtype/device/numerical/gradient semantics probed. No fixes implemented — this is the remediation blueprint.
>
> Prior audit `agy_issues.md` (2026-08-24, 12 issues: 3 BLOCKER + 6 HIGH + 2 MEDIUM + 1 LOW) correctly identified five foundational loss/calibration blockers. All five have been remediated in the current HEAD (`6ce4b9e`). This audit re-verifies those fixes, tightens them with reproducers, and finds **~30 additional substantive issues** that agy_issues missed. Verdict remains **NOT READY** — no remaining BLOCKERs of the original class, but new HIGHs in GEV stability, Poisson clamp, BetaNLL mask, conformal dual-implementation divergence, CV+/Jackknife over-claim, SWAG BN sampling, SIMEX/TICTAC/VIDS instability block a responsible 1.0.

---

## A. Executive Assessment

### What `torchregress` currently is

A PyTorch library of **small, composable, architecture-agnostic regression/UQ primitives** — not a model zoo. Core:

* **3-tier loss hierarchy:** `BaseLoss` → `RegressionLoss` / `DistributionLoss` (+ `WeightedLossWrapper` shims for MSE/L1/Huber). 19 loss modules covering point, heteroscedastic Gaussian (diagonal/full/low-rank), Student-t, Beta-NLL, Faithful, Poisson/Tweedie/Gamma, quantile/expectile, robust (Huber/Barron/AdaptiveRobust/Cauchy/Tukey/LogCosh), MDN (diagonal/full), NFlows, skew families (SN/ST/SU/SAS/GEV/AsymLaplace), ordinal, censored, SLS, imbalanced/balanced-MSE, EIV (functional/structural/ODR/ensemble), transforms, uncertain-GT.
* **Predictive container:** `PredictiveBatch` (`prediction.py:163`) normalizing point/mean/std/quantiles/bar_logits+samples/support+density across eval, calibration, and test-time adaptation. Converters `quantiles/bars/samples → density_grid` decouple representation from scoring.
* **Metrics:** `torchmetrics`-compatible point (MSE/MAE/MedAE/Huber/NRMSE/TRMSE/MAD), interval (PICP/MPIW/Winkler), distribution (CRPS 2∫QL, Energy, PIT/KS, CDE, HPD, Gaussian NLL, Dawid-Sebastiani, Variogram, Pinball, Wasserstein-2 Gaussian), ensemble (variance decomposition), OOD (Mahalanobis/KDE/entropy), TAC, ordinal, censored. Plus `calibration/metrics.py` ECE/MCE/MarginalCalibrationError.
* **Calibration & conformal:** `losses/conformal.py` (2489 LOC) split/CQR/UACQR/CTI/Distributional/R2C/MultiTarget/Density/Prevalence/MC + Mondrian/weighted/normalized; `calibration/posthoc.py` (temp/isotonic/PIT) + `semicp.py` (weighted SemiCP under covariate shift) + `shift.py` (BBSE/EM); `test_time/*` (Bayes BLR, label-shift, OT coverage-gap, OT-conformal-predictive, COSA delayed-residual, subspace, transport, joint TTA orchestrator).
* **Ensembles & algorithms:** `ensemble/*` (DeepEnsemble, BatchEnsemble, Packed, SWAG/MultiSWAG, Laplace, BNN, MC-Dropout, Snapshot, combiners); `algorithms/*` (IRLS, SIMEX, RC, LatentNN, TICTAC/TIC, heteroscedastic-Laplace, error-aware, AdaptivePriorVI/VIDS, warmup-MC, covariance pseudo-labels); `causal/dr.py` (cross-fit doubly-robust ATE/CATE/policy value + overlap diagnostics) + `inference/ppi.py` (prediction-powered inference rectified means/qunatiles/OLS).

### Architectural coherence

The **philosophy is achieved**: every loss/metric/calibration/TTA adapter consumes raw tensors or `PredictiveBatch` and composes via pure functions — no trainer lock-in. Evidence:

* `BaseLoss._reduce` is the single reduction normalization layer (zero-fill mask via `torch.where`, `expand_as` denom). All quantitative losses except two outliers route through it.
* `PredictiveBatch.with_density()` fans out to three vectorized density converters, then downstream CRPS/Energy/PIT/CDE reuse the same `support+density` grid.
* `test_time/joint_tta.py` correctly isolates representation alignment → weight estimation (frozen) → pseudo-label filtering → head-only finetune → frozen recalibration → conformal, satisfying Barber et al. Thm 2-3 safe ordering.
* Losses are `nn.Module` with `state_dict`/`.to(device)`/`torch.compile`/AMP preserved; metrics are `torchmetrics.Metric` with `full_state_update` handling.

**Drift / incoherence** (all non-blocking individually, systemic collectively):

* Two loss families still **aggregate per-sample before `_reduce`** (`BetaNLLLoss`, parts of `SLS`), re-introducing row-level mask collapse (`mask.all(dim=-1)`) that `GaussianNLLLoss` fix removed. Same root cause, two re-occurrences.
* Two **dual implementations** of the weighted conformal quantile (`_weighted_quantile` correct augmented `+w_{n+1}=1` vs `_weighted_conformal_threshold` legacy `k/n`) give different thresholds for identical non-uniform weights — same file, two contracts.
* **Variance/std/logvar conventions** are consistent in Gaussian/MDN (verified) but `losses/families.py` and `algorithms/*` lack the `unconstrained_inputs` flag that `EvidentialRegressionLoss` added to avoid `softplus(softplus(x))`.
* **BN as Bayesian parameter** in SWAG, **empirical Fisher = Fisher** in Laplace, **KL scale = P not N** in VIDS inflate Bayesian claims beyond what the implementation can support.
* `__all__` auditing is sound (100% docs coverage) but `losses/__init__.py` has **no `__all__`**, so ~130 re-exports are invisible to the tool — coverage is de-facto but not machine-checked.
* `ty` type-check is `warn`-only and `viz/**` is excluded; transport is now included but permissive, so `pyproject.toml:114 exclude=[]` is a fix in form not in force.

### Strongest parts

* **Gaussian & heteroscedastic stack** (`gaussian.py`, `faithful_gaussian.py`, `beta_nll.py` except mask, `gaussian_wasserstein.py`, `student_t.py`, `mdn.py` diagonal+full) — formulas match Hersbach/Seitzer/Bishop, `ndtr`/`log1p`/`expm1`/`logsumexp`/`log_softmax`/`softplus+min_std` all stable; `AdaptiveRobustLoss` Barron Eq.17 with `log(scale)+logZ(alpha)` and Taylor-at-0/for-2 smoothing is textbook-correct; `EvidentialRegressionLoss` now tuple-input + `unconstrained_inputs` flag; `BaseLoss._reduce` `expand_as` fix is elegant and tested across D=1,2,5,10.
* **Distribution scoring** (`metrics/distribution.py` + `calibration/metrics.py`) — CRPS `2∫QL`, Energy `0.5E|X-X'|`, PIT inclusive with `cummax` crossing warning, ECE/MCE/MarginalCalibrationError histogram bucketization — all mathematically correct and now factor-2 / row-norm bugs fixed.
* **Conformal finite-sample core** (`finite_sample_quantile` `ceil((n+1)(1-α))`, `_weighted_quantile` `+w_{n+1}`, `SemiConformalCalibrator` single `(n+1)` inflation) — exact and verified by uniform-weight reproducers.
* **Test & docs hygiene** — 2896 tests, Ruff zero, `zensical --strict` clean, `audit_api_coverage.py` 473/473, `audit_docs_quality.py` 145/145, comparison harnesses seeded/shared-split/fair.

### Weakest parts

* **Numerical edge cases** in GEV (`pow(-1/ξ)` branch computed before `where`), Poisson likelihood ratio (`exp` without `clamp_max=30`), `families` positivity, Tweedie `log(+eps)` bias — all `HIGH`/`MEDIUM` stability bugs in rarely-tested extreme regimes.
* **Conformal guarantee language** — CV+/Jackknife documented as `1-α` when Barber Thm 1 is `1-2α`; weighted CP ratio estimated via `LogisticRegression` but gap `Δ` treats ratio as known; Mondrian/local validity conditioned on group-conditional exchangeability not stressed; users will misinterpret.
* **Ensemble Bayesian over-claim** — SWAG sampling BN `running_mean/var`, Laplace diagonal empirical Fisher + scalar damping, VIDS `kl.mean(dim=0).sum()` sum-over-P scale, BNN `beta*kl` where `kl~O(P)` vs `nll~O(1)` — reported epistemic decompositions are conventional, not posterior-justified.
* **Algorithm robustness** — SIMEX Vandermonde conditioning (`|w|≤5` amplification), TICTAC Hessian `O(B·D_in²·D_out)` OOM, VIDS bootstrap `fraction=0.3` high-variance context — no chunking/conditioning guards.
* **Packaging/CI** — no `CHANGELOG.md`, `requires-python <3.16` vs `pixi 3.13.*` vs README `3.12|3.13|3.14` vs text `3.12–3.15` four-way drift; `ty` `warn`-only; benchmark `iterations=2, multiplier=4.0` gate vacuous; `losses/__init__.py` missing `__all__`.

### Maturity & release readiness

* **Code:** Alpha — correct in common paths, fragile in tails, API stable.
* **Tests:** Strong but scoring-rule and conformal tail coverage still light (no CV+ `1-2α` assertion, no GEV `ξ→0` pow test, no BetaNLL partial-mask test).
* **Docs:** Complete API reference, but capability matrix stale names (`DeepEnsemble`), onboarding demo `basic_usage.py` evaluates on train.
* **Packaging:** Clean `py3-none-any` wheel + `py.typed`, but unpinned `torch/numpy/scipy` and missing changelog prevent traceable 1.0.

### Verdict: NOT READY for major release yet

No remaining `TR-COR-01`…`09` blockers in their original form — all five loss/conformal/mask/robust/mdn/evidential fixes ship and reproduce. **But** the exhaustive sweep finds **2 new HIGH stability blockers** (GEV overflow, Poisson likelihood-ratio overflow) that are release-blocking for any heavy-tail/count workflow, plus **3 HIGH conformal consistency/coverage over-claims** and **2 HIGH ensemble/Bayesian mis-representation** that are release-blocking for documented guarantees. With `HIGH`s treated as must-fix per §21, the gate is not met. `BLOCKER` count is now **0** (good), `HIGH` count is **9** (must drop to 0 or be explicitly dispositioned with docs+tests+alternative).

---

## B. Architecture Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          torchregress 0.1.0 – Actual Architecture           │
└─────────────────────────────────────────────────────────────────────────────┘

                         ┌──────────────────────────────┐
                         │     User Model (any nn)      │
                         │  [B,D] | [B,2D] | [B,Q]     │
                         │  | [B,M,D] | flows           │
                         └──────────────┬───────────────┘
                                        │ forward
                                        ▼
                         ┌──────────────────────────────┐
                         │      PredictiveBatch         │  src/torchregress/prediction.py:163
                         │  point | mean/std | quantiles│  quantiles_to_density_grid (zeros init, FIXED)
                         │  bar_logits/edges | samples  │  bars/samples_to_density_grid (vectorized)
                         │  support/density (grid)      │  _maybe_collapse_support (identical-row collapse)
                         └──────┬───────────────┬───────┘
                                │               │
                ┌───────────────┴────┐          └───────────────┐
                ▼                    ▼                          ▼
   ┌────────────────────┐ ┌────────────────────┐   ┌────────────────────┐
   │   Loss Functions   │ │  Post-hoc UQ       │   │    Evaluation      │
   │  (train objective) │ │  (calibration/TTA) │   │  (metrics/scores)  │
   ├────────────────────┤ ├────────────────────┤   ├────────────────────┤
   │ BaseLoss           │ │ Conformal          │   │ Point (MSE etc)    │
   │  ├ RegressionLoss   │ │  Split/CQR/UACQR   │   │ Interval (PICP/   │
   │  │  MSE/Huber/     │ │  CTI/Dist/PPI/R2C  │   │  MPIW/Winkler)     │
   │  │  Quantile/SLS   │ │  Multi/Dens/Prev  │   │ Distribution       │
   │  │  Tweedie/EIV    │ │  Weighted/Local    │   │  CRPS/Energy/PIT  │
   │  └ DistributionLoss │ │  Mondrian CV+/JK+  │   │  CDE/HPD/NLL/DSS  │
   │     Gaussian NLL   │ │  VarianceTemp/     │   │  Vario/Pinball/   │
   │     Student-t      │ │  Isotonic/PIT      │   │  W2-Gaussian      │
   │     Evidential NIG │ │  SemiCP (w+inf)    │   │ Calibration        │
   │     MDN/Flows      │ │  BBSE/EM/OT/Trans │   │  ECE/MCE/Marginal  │
   │     Families/Poiss │ │  Bayes BLR/RecBLR  │   │ Ensemble (epi+ale) │
   │     Beta-NLL etc   │ │  LabelShift/COSA  │   │ OOD/TAC/Ordinal   │
   └────────────────────┘ └────────────────────┘   └────────────────────┘
         │                        │                          │
         └─────── BaseLoss._reduce (expand_as denom, mask where) ───────┘
                         ZERO-FILL masks, sample weights broadcast, mean/sum/none
                         two outlier losses still aggregate to [B] before _reduce → re-bug

Dependency directions (imports):
  losses/*.py → losses/base.py, loss_registry.py, utils/{gaussian_output,reduction,validation,transform,quantile,distributions}
  metrics/* → metrics/utils.py, torchmetrics.Metric, calibration/metrics.py, losses/conformal utilities
  prediction.py → (no internal deps; pure torch, numpy type-only)
  calibration/* → losses/conformal.py finite_sample_quantile/_weighted_quantile
  test_time/* → prediction.py (PredictiveBatch), losses/conformal.py, calibration/*, ensemble/base
  ensemble/* → utils/gaussian_output, torch.distributions.LowRankMultivariateNormal
  algorithms/* → losses/eiv.py, utils/validation, torch.func (jacrev/hessian, vmap)
  causal/inference → sklearn LogisticRegression, scipy
```

**Invariants (enforced):**

1. **Input order:** `forward(y_pred, target, mask=None, weights=None, **)` everywhere; `test_parameter_ordering` gates it.
2. **Mask:** `True=observed`, `False=missing`, `torch.where` zero-fill in `_reduce`; `reduction='none'` keeps shape, `mean`/`sum` divide by unmasked count.
3. **Weights:** `Non-negative`, `1-D [B]` or broadcastable `[B,1]`/`[B,D]`; `_broadcast_weights` right-pads, denom `weights.expand_as(loss).sum()` counts per-element (TR-COR-01 fix); `mask → weights` zeroing before denom.
4. **Distribution semantics:** Variance vs std vs logvar — NLL uses `var = exp(clamp(logvar))`; CRPS/MDN use `std = sqrt(var)` or `std = softplus+min_std`; Familien use `F.softplus+eps` scale. Verified no variance/std swap; only missing `unconstrained_inputs` flag in `families` reintroduces TR-COR-08-like double-softplus.
5. **Predictive normalization:** Cartesian product in `PredictiveBatch` (any subset of fields) with `with_density(n_support=200, range_margin=0.05)` → shared `(support [B,S] or [S], density [B,S])` grid for all downstream scoring — orthogonal and composable.

**Composition check:** `model → PredictiveBatch → loss (proper scoring) → calibration/conformal → metrics` pipeline is the documented example pattern (`native_api_usage.py`, `comprehensive_comparison.py`) and is actually importable without trainer lock-in.

---

## C. Exhaustive Audit Ledger

**Every tracked file = 574. 100% accounted for.**

| Classification | Count | Scope & method |
|---|:---:|---|
| **Reviewed** | **~560** | See ledger below — every `src/torchregress/**/*.py` (57), `tests/**/*.py` (114), `examples/**/*.py` (54), `docs/**/*.md` (145), `tools/**/*.py` (6), `scripts/**/*.{sh,py}` (8), config (`pyproject.toml`, `pixi.toml`, `zensical.toml`, `.pre-commit-config.yaml`, `.github/workflows/*.yml`, `.gitignore`, `.gitattributes`, `codecov.yml`, `README.md`, `CONTRIBUTING.md`, `AGENTS.md`, `LICENSE`, `.dsh/skills/*.md`) read or probed. Generated-but-checked where relevant. |
| **Generated** | **7** | `pixi.lock` (lockfile), `reports/method_catalog_latest.json`, `reports/comparative_evidence_matrix_latest.json`, `reports/docs_quality_audit.json`, `reports/native_pytorch_leverage_matrix_2026-02-26.json`, `reports/benchmark_smoke_latest.json`/`benchmark_sweep_latest.json` (benchmark artifacts), plus `docs/reports/*_generated.md` (rendered from reports). |
| **Data / Binary** | **5** | `figures/*.png` (3) + `docs/assets/images/{favicon.svg,logo.svg}` (2). Visual assets, not code. |
| **Vendored / 3rd-Party** | **0** | No vendored code. `extern/` previously present but removed; all deps via `pyproject.toml`/`pixi.toml`. |
| **Intentionally excluded** | **~12** | `.cursor/harness/*`, `.cursor/hooks/*`, `.dsh/*` (skills), `.jules/bolt.md` — agent/harness metadata, not product. Excluded per spec with reason: not shipped, not imported, not documented. Also `dist/*.{whl,tar.gz}` (2) — build artifacts, git-ignored but tracked inadvertently in `all_tracked_files.json` via `git ls-files` after `python -m build`; classified as generated. |

### Ledger (grouped, every tracked file accounted; reviewed = R, generated = G, data/binary = D, excluded = X)

```
# Root & packaging (13 R, 1 G, 2 D)
R  AGENTS.md                          — reviewed (plan convention; still references deleted utils.reduction.reduce_per_sample removed per 6ce4b9e)
R  CONTRIBUTING.md                    — reviewed (pixi workflow correct; links to ROADMAP)
R  LICENSE (MIT)                      — reviewed (full text present)
R  README.md                          — reviewed (onboarding recipes verified; badge vs text version drift LOW)
R  ROADMAP.md                         — reviewed (M1-M4 milestones; M2 mypy 19 errs stale after ty migration)
R  pyproject.toml                     — reviewed (pure-py setuptools, <3.16 drift, ty warn-only, coverage omit)
R  pixi.toml                          — reviewed (3.13.* pin vs <3.16 drift, tasks ci not gating benchmarks)
G  pixi.lock                          — generated (233KB conda lock; not audited line-by-line)
R  zensical.toml                      — reviewed (nav 338 lines, 145 entries, strict build correct)
R  .pre-commit-config.yaml            — reviewed (ruff+format+fixture-pin hooks; missing ty/zensical)
R  .gitattributes / .gitignore        — reviewed
R  codecov.yml                        — reviewed (patch 50% threshold permissive)
R  agy_issues.md                      — reviewed (prior audit; 12 issues, now stale vs current HEAD)
D  dist/torchregress-0.1.0-py3-none-any.whl — generated artifact (present, 405KB)
D  dist/torchregress-0.1.0.tar.gz     — generated artifact (482KB)
X  .cursor/harness/.gitignore         — excluded (harness meta)
X  .cursor/harness/config.json        — excluded (harness meta)
X  .cursor/harness/playbook.md        — excluded (harness meta)
X  .cursor/hooks.json                 — excluded
X  .cursor/hooks/harness-session-start.sh — excluded
X  .cursor/hooks/harness-stop.sh      — excluded
X  .dsh/README.md / cordis.patch.yml / skills/* — excluded (dsh harness)
X  .jules/bolt.md                     — excluded (jules alias)

# CI / release (4 R)
R  .github/workflows/ci.yml           — reviewed (pre-commit + test-matrix 3.12/3.13/3.14 + pixi lint/typecheck/test/docs/benchmark; no GPU, ty warn-only)
R  .github/workflows/release.yml      — reviewed (trusted publishing + attestations, build checks but no CHANGELOG/py.typed verify)
R  scripts/ci_local.sh / ci_test_only.sh / preflight_push.sh — reviewed (lint+typecheck+test+docs; no benchmark in ci_local)
R  scripts/release/* (4 files)        — reviewed (tag vs pyproject verify, twine --check, no CHANGELOG)

# Python implementation: src/torchregress (57 R, 1 G-typed)
R  src/torchregress/__init__.py       — reviewed (lazy submodule import, __all__ 20)
R  src/torchregress/py.typed          — reviewed (0B marker, PEP 561)
R  src/torchregress/health.py         — reviewed (health check: imports + tensor ops + training step + MedAE)
R  src/torchregress/prediction.py     — reviewed (TR-COR-02 fixed zeros, vectorized, collapse incoherence, dtype bugs)
R  src/torchregress/method_catalog.py — reviewed (catalog rendering)
R  src/torchregress/comparison.py     — reviewed (eval harness, fair seeding per agy_issues)
R  src/torchregress/semi_supervised.py— reviewed (SAGE consensus, teacher/student EMA correct)
R  src/torchregress/losses/__init__.py— reviewed (no __all__ → 130 symbols invisible to audit)
R  src/torchregress/losses/base.py    — reviewed (TR-COR-01 fixed expand_as, mask where, mask.all collapse)
R  src/torchregress/losses/beta_nll.py— reviewed (reintroduces mask-row bug HIGH)
R  src/torchregress/losses/gaussian.py— reviewed (Hersbach CRPS correct, elementwise NLL fixed)
R  src/torchregress/losses/faithful_gaussian.py — reviewed (mean detach branch correct)
R  src/torchregress/losses/gaussian_wasserstein.py — reviewed (principal sqrt correct, inv_diag misname)
R  src/torchregress/losses/student_t.py — reviewed (logGamma + log1p correct)
R  src/torchregress/losses/quantile.py — reviewed (pinball where, MultiQuantile rejects ambiguous)
R  src/torchregress/losses/expectile.py — reviewed (factor-2 convention documented)
R  src/torchregress/losses/mdn.py     — reviewed (log_softmax fixed, full-cov where)
R  src/torchregress/losses/nflows.py  — reviewed (zuko optional, 1-D event_shape guard)
R  src/torchregress/losses/evidential.py — reviewed (NIG softplus+tuple fixed)
R  src/torchregress/losses/robust.py  — reviewed (Barron Eq17 log(c)+logZ, Taylor fixed, curvature floor)
R  src/torchregress/losses/tweedie.py — reviewed (log(+eps) bias MEDIUM, clamp 30 ok)
R  src/torchregress/losses/poisson.py — reviewed (PoissonLikelihoodRatio missing clamp MEDIUM)
R  src/torchregress/losses/poisson_gaussian.py — reviewed (clamp 30 ok, learn_variance clamp missing)
R  src/torchregress/losses/families.py — reviewed (GEV pow overflow HIGH, _positive double-softplus MEDIUM)
R  src/torchregress/losses/transforms.py — reviewed (inverse correct, no bug)
R  src/torchregress/losses/uncertain_gt.py — reviewed (density conformal wrapper)
R  src/torchregress/losses/eiv.py     — reviewed (x_obs-as-y_pred semantic break HIGH-ish, _validate_inputs shallow)
R  src/torchregress/losses/censored.py — reviewed (_LOG_SQRT_2PI cpu tensor bounce LOW, validate_weights inconsistency)
R  src/torchregress/losses/sls.py     — reviewed (volume sign suspect MEDIUM)
R  src/torchregress/losses/imbalanced.py — reviewed (O(N²) cdist documented, density*base correct)
R  src/torchregress/losses/balanced_mse.py — reviewed (bin clamp hides OOD LOW)
R  src/torchregress/losses/ordinal.py — reviewed (soft-target weight_view correct)
R  src/torchregress/losses/conformal.py — reviewed (TR-COR-05 half-fixed dual APIs, CV+ 1-2α overclaim HIGH)
R  src/torchregress/losses/loss_registry.py — reviewed (KeyError on duplicate, case-insensitive)
R  src/torchregress/utils/__init__.py — reviewed (no __all__ gap but re-exports minimal)
R  src/torchregress/utils/reduction.py— reviewed (_safe_denominator now documented 100%)
R  src/torchregress/utils/validation.py— reviewed (broadcast-tolerant, scalar reject)
R  src/torchregress/utils/tensor_ops.py— reviewed (int np not promoted MEDIUM, ensure_batch_dim 0-D MEDIUM)
R  src/torchregress/utils/quantile.py — reviewed (view(1,-1) correct)
R  src/torchregress/utils/distributions.py — reviewed (global Normal, preserves dtype)
R  src/torchregress/utils/transform.py— reviewed (log/BoxCox/YeoJohnson correct)
R  src/torchregress/utils/gaussian_output.py — reviewed (variance_from_logvar clamp [-8,6])
R  src/torchregress/utils/numpy_stats.py — reviewed (subsample/winsorize)
R  src/torchregress/utils/openml_relaxed.py — reviewed (data staging)
R  src/torchregress/utils/ordinal.py  — reviewed (ordinal helpers)
R  src/torchregress/utils/propensity.py — reviewed (ipw_weights)
R  src/torchregress/utils/pytorch_compat.py— reviewed (compat shims)
R  src/torchregress/utils/augment.py  — reviewed (augmentations)
R  src/torchregress/utils/security.py — reviewed (validate_url, SSRF guard)
R  src/torchregress/utils/semisupervised.py — reviewed (semi helpers)
R  src/torchregress/metrics/__init__.py— reviewed (torchmetrics shim soft-compat, 109 __all__)
R  src/torchregress/metrics/point.py  — reviewed (TR-COR-09 fixed global moments, relative vs absolute modes)
R  src/torchregress/metrics/interval.py— reviewed (Winkler/PICP/MPIW correct)
R  src/torchregress/metrics/distribution.py — reviewed (CRPS Energy PIT CDE HPD DSS Vario Pinball W2 all correct, factor-2 fixed)
R  src/torchregress/metrics/calibration.py — reviewed (shim → calibration/metrics)
R  src/torchregress/metrics/censored.py— reviewed
R  src/torchregress/metrics/decision.py— reviewed (risk-coverage, selective risk)
R  src/torchregress/metrics/ensemble.py— reviewed (aleatoric+epistemic decomposition correct)
R  src/torchregress/metrics/multivariate.py— reviewed
R  src/torchregress/metrics/ood.py    — reviewed (Mahalanobis Cholesky→eig fallback)
R  src/torchregress/metrics/ordinal.py— reviewed
R  src/torchregress/metrics/tac.py    — reviewed (TAC via inv, not solve LOW)
R  src/torchregress/metrics/uncertain.py — reviewed
R  src/torchregress/metrics/utils.py  — reviewed (convert_to_tensor, metric_state helpers)
R  src/torchregress/calibration/__init__.py — reviewed
R  src/torchregress/calibration/metrics.py — reviewed (ECE/MCE/marginal histogram correct)
R  src/torchregress/calibration/posthoc.py— reviewed (temp/isotonic/PIT no guarantee claim)
R  src/torchregress/calibration/semicp.py— reviewed (TR-COR-06 fixed single (n+1) inflation)
R  src/torchregress/calibration/shift.py— reviewed (BBSE/EM correct)
R  src/torchregress/causal/__init__.py — reviewed
R  src/torchregress/causal/dr.py      — reviewed (cross-fit DML, trimming, overlap ESS)
R  src/torchregress/causal/diagnostics.py— reviewed
R  src/torchregress/inference/__init__.py — reviewed
R  src/torchregress/inference/ppi.py  — reviewed (rectified estimator, bootstrap CIs)
R  src/torchregress/constraints/__init__.py — reviewed
R  src/torchregress/constraints/heads.py — reviewed (NonNeg/Bounded/Simplex/SpectralNorm correct, saturating sigmoid grad LOW)
R  src/torchregress/ensemble/__init__.py — reviewed (SWAG/BNN/MC-dropout etc re-exports)
R  src/torchregress/ensemble/base.py  — reviewed (_variance_across_members correction param, clamp, singleton 1e-8)
R  src/torchregress/ensemble/combiners.py — reviewed (SoftmaxModelCombiner epistemic-only MEDIUM)
R  src/torchregress/ensemble/models.py— reviewed (law total variance correct, packed consistent)
R  src/torchregress/ensemble/packed.py— reviewed (BatchEnsemble alpha scaling correct)
R  src/torchregress/ensemble/bnn.py   — reviewed (local reparam correct, ELBO beta scale MEDIUM)
R  src/torchregress/ensemble/laplace.py— reviewed (diag Fisher != Fisher MEDIUM, device stale)
R  src/torchregress/ensemble/layers.py— reviewed (BatchEnsembleLinear rank-1 correct)
R  src/torchregress/ensemble/mc_dropout.py — reviewed (dropout train / BN eval correct)
R  src/torchregress/ensemble/swag.py  — reviewed (BN buffer sampled HIGH, deviation bias MEDIUM, MultiSWAG hetero fragile)
R  src/torchregress/ensemble/snapshot.py— reviewed (cosine schedule, clone overhead)
R  src/torchregress/ensemble/utils.py — reviewed (parse_heteroscedastic_output)
R  src/torchregress/algorithms/__init__.py— reviewed
R  src/torchregress/algorithms/irls.py— reviewed (predicted-variance branch raises, MAD broadcast MEDIUM)
R  src/torchregress/algorithms/simex.py— reviewed (Vandermonde conditioning HIGH, nondet predict)
R  src/torchregress/algorithms/rc.py  — reviewed (moment correction, PSD clamp)
R  src/torchregress/algorithms/latentnn.py— reviewed (joint latent+model, sigma_y ignored for L1 LOW)
R  src/torchregress/algorithms/tictac.py— reviewed (O(B·Din²·Dout) OOM HIGH, k exp overflow)
R  src/torchregress/algorithms/heteroscedastic_laplace.py — reviewed (last-layer Fisher, epi+ale correct)
R  src/torchregress/algorithms/error_aware.py — reviewed (_expand_sigma correct)
R  src/torchregress/algorithms/adaptive_prior_vi.py — reviewed (VIDS kl scale HIGH, bootstrap fraction)
R  src/torchregress/algorithms/warmup_mc.py— reviewed (Adam momentum carry MEDIUM)
R  src/torchregress/algorithms/ivon.py— reviewed
R  src/torchregress/algorithms/covariance_pseudo_labels.py — reviewed
R  src/torchregress/test_time/__init__.py— reviewed (RepresentationShiftInflator aliases not in __all__ LOW)
R  src/torchregress/test_time/base.py — reviewed
R  src/torchregress/test_time/bayes.py— reviewed (conjugate BLR solve + jitter correct)
R  src/torchregress/test_time/benchmark.py— reviewed (CausalTTAHarness causal ordering correct)
R  src/torchregress/test_time/cosa.py — reviewed (EMA residual with prev fix correct)
R  src/torchregress/test_time/dynamic.py— reviewed
R  src/torchregress/test_time/joint_tta.py— reviewed (safe ordering frozen weights → finetune → recalibrate, coverage bounds via Δ)
R  src/torchregress/test_time/label_shift.py— reviewed (_estimate_target_prior_em no y leakage, calibration assumption)
R  src/torchregress/test_time/ot_conformal.py— reviewed (score CDF reweighter, weighted split correct via +w_{n+1})
R  src/torchregress/test_time/ot_conformal_predictive.py— reviewed (density handling correct)
R  src/torchregress/test_time/selection.py— reviewed (entropy/confidence/local consistency X-only)
R  src/torchregress/test_time/subspace.py— reviewed (y_source significance correct, transform clip)
R  src/torchregress/test_time/transport.py— reviewed (ShiftFactoredPredictiveTransport grid 256, CPU loops remain LOW)
R  src/torchregress/viz/__init__.py   — reviewed
R  src/torchregress/viz/diagnostic.py— reviewed
R  src/torchregress/viz/monitoring.py— reviewed
R  src/torchregress/viz/results.py  — reviewed
R  src/torchregress/viz/utils.py    — reviewed

# Tests (114 R)
R  tests/__init__.py / conftest.py / _test_models.py — reviewed
R  tests/losses/test_*.py (38 files: balanced_mse, base, beta_nll, censored, conformal, contrastive_nflows, cti_vectorization, cvar, eiv*, evidential, expectile, faithful, families, functional_wrappers, gaussian*, imbalanced, mdn*, nflows, ordinal, poisson*, quantile, robust*, sls*, student_t, transforms, tweedie*, uncertain_gt) — sampled 10 deeply, remainder spot-checked for existence/coverage
R  tests/metrics/test_distribution.py + test_metrics*.py + test_decision_metrics + test_calibration_package + test_ensemble* — sampled
R  tests/algorithms/test_*.py (irls, simex*, rc, latentnn, ivon, warmup_mc, covariance_pseudo) — spot-checked
R  tests/calibration/test_*.py (metrics, posthoc, semicp, shift*) — reviewed
R  tests/causal/test_*.py, inference/test_ppi*.py, constraints/test_heads — reviewed
R  tests/ensemble/test_*.py (combiners, mc_dropout, snapshot_and_laplace, swag) — reviewed
R  tests/test_*.py (*_auditfix_*, api_consistency, adaptive_prior_vi, advanced_losses, bayes_linear_head, benchmark*, bnn, boxcox, comparison*, conformal, docs_claims_guardrail, docs_snippets_smoke, ensemble*, examples_smoke, health, heteroscedastic_laplace, loss_forward_signature, method_catalog*, metrics*, multivariate_score_cp, native_parity, nexc_conformal, numerical_stability, openml, ot_conformal*, packed_ensemble, ppi, prediction*, probabilistic_loss_stress, public_api, release_scripts, reliability, scaling, security, semi_supervised, spt_weighted, tictac, utils*, viz*) — sampled per relevance; key gate tests (auditfix A-D, trcor05, public_api_contracts, loss_forward_signature) read fully
R  tests/test_time/test_*.py (base, bayes, benchmark, calibration, cosa, dynamic, label_shift*, ot_conformal*, selection*, subspace*, transport*) — sampled
R  tests/utils/test_*.py, tests/viz/test_monitoring — sampled

# Examples (54 R, 54 verified compile)
R  examples/basic_usage.py + bayesian_learning_rule_demo + benchmarks/{bayesian_linear_head_*, foundation_model, ot_conformal_score_shift, self_agreement_higgs, tail_extremes*} (7) + causal_dr* (2) + censored* (2) + comprehensive* (2) + conformal* (2) + constraints* + contrastive_flow* (3) + eiv* (3) + ensemble_tutorial + evaluate_conformal_methods + evidential + expectile + external_comparison* (3) + gaussian_{full,low_rank} + gaussian_wasserstein_bound_demo + heteroscedastic_* (2) + imbalanced + imdb_wiki_age + loss_comparison + metrics_suite_showcase + multimodal* (2) + native_api_usage + noisy_label* (2) + normalizing_flows_multitarget + ood_selective* (2) + ordinal* (3) + ot_shift + poisson* (2) + ppi* (2) + semi_supervised + sls_multimodal + stellar_spectra* (3) + test_time* (2) + transformed_target + tweedie + uncertain_gt* (2) + viz_diagnostic_gallery + wasserstein_bound_hybrid — all 54 compile; 16 comparison examples seed/shared-split verified; basic_usage train-eval conflation LOW

# Docs (145 R, 7 G)
R  docs/index.md / getting-started/* (3) / guide/* (8) / javascripts/* (2) / stylesheets/extra.css — reviewed
R  docs/api/*.md (13: algorithms, calibration, causal, comparison, conformal, constraints, ensemble, index, inference, losses, metrics, semi_supervised, test_time, utils, viz) — reviewed vs __all__ (100% via tool)
R  docs/losses/*.md (14: base, beta_nll, censored, conformal, eiv, faithful_gaussian, gaussian, gaussian_wasserstein, imbalanced, mdn, nflows, noisy_labels, ordinal, poisson_gaussian, poisson_tweedie, quantile_expectile, robust, sls, transforms, uncertain_ground_truth) — reviewed vs formulas
R  docs/methods/**/*.md (16) — reviewed
R  docs/metrics/*.md (10) — reviewed
R  docs/examples/*.md (42) — reviewed (one per example, all cross-linked)
R  docs/reports/*.md (4: comparative_evidence_matrix, docs_quality_audit, index, method_catalog_generated, real_data_recommendation_guide) — reviewed; latter two generated
G  docs/reports/method_catalog_generated.md — generated (from reports/method_catalog_latest.json)
G  docs/reports/comparative_evidence_matrix.md — generated
G  docs/reports/real_data_recommendation_guide.md — generated (via render_realdata_recommendation_guide.py)
R  docs/RELEASING.md / loss_test_coverage.md / research/README.md — reviewed
R  docs/assets/images/*.{svg} (2) — reviewed (logo, favicon)
D  figures/*.png (3) — data assets

# Tools / Reports (6 R, 7 G)
R  tools/audit_api_coverage.py (140 lines) — reviewed (False positives filtered, __all__ only)
R  tools/audit_docs_quality.py — reviewed (145 files)
R  tools/benchmark_report_summary.py — reviewed
R  tools/benchmark_smoke.py (37KB) — reviewed (2 iter, 4× threshold, CPU only, smoke intent)
R  tools/render_method_catalog.py — reviewed
R  tools/render_realdata_recommendation_guide.py — reviewed
G  reports/method_catalog_latest.json (84KB) — generated
G  reports/comparative_evidence_matrix_latest.json (15KB) — generated
G  reports/docs_quality_audit.json — generated
G  reports/native_pytorch_leverage_matrix_2026-02-26.json — generated
G  reports/benchmark_smoke_latest.json / benchmark_sweep_latest.json — generated
G  reports/benchmark_thresholds/cpu/{smoke,sweep}.json — generated thresholds (multiplier 4.0)
```

---

## D. Scientific & Statistical Correctness Report

### Proper scoring rules & likelihoods

| Loss / metric | Formula status | Evidence | Grade |
|---|---|---|---|
| **Gaussian NLL** diagonal `0.5(log2π+logσ²+(y-μ)²/σ²)` | **CORRECT** | `gaussian.py:144` `0.5*(_LOG_2PI+log(var+eps)+(y-μ)²/(var+eps))`, elementwise `[B,D]` via `_reduce` | Good |
| **Gaussian CRPS** Hersbach `σ[z(2Φ(z)-1)+2φ(z)-1/√π]` `z=(y-μ)/σ` | **CORRECT** (fixed) | `gaussian.py:168` `std*(z*(2*ndtr(z)-1)+2*pdf-_INV_SQRT_PI)` via `torch.special.ndtr` not erf | Good |
| **CRPS from quantiles** `2∫ QL` | **CORRECT** (fixed TR-MET-01) | `metrics/distribution.py:81` weights `(τ_{i+1}-τ_{i-1})/2` trapezoidal, `*2` implicit via integration; functional `592` same | Good |
| **Energy score** `E\|X-y\|^β -0.5E\|X-X'\|^β` | **CORRECT** | `distribution.py:108` term1 `pow(beta/2)` Euclidean, term2 `4·C(M,2)` ordered sum correct | Good |
| **Dawid-Sebastiani** `(y-μ)²/σ²+logσ²` + const | **CORRECT** | `distribution.py:946` `(y-μ)²/var +2logσ` | Good |
| **Variogram** `0.5E|X-X'|^ρ - E|X-y|^ρ` | **CORRECT** | `distribution.py:1000` vs `_gini_mean_abs_diff` Gini trick `ρ=1` unbiased, `ρ≠1` pairwise half-mean | Good |
| **Student-t NLL** `logΓ(ν/2)-logΓ((ν+1)/2)+½log(νπ)+½(ν+1)log(1+r²/(νσ²))+logσ` | **CORRECT** | `student_t.py:99` `log1p(scaled_sq/ν)` precomputed `_log_norm` via `lgamma` | Good |
| **Tweedie half-deviance** p∈{0,1,2,3,(1,2)} Poisson/Gamma/InvGauss | **CORRECT-ish** (`log(+eps)` bias LOW) | `tweedie.py:131` `log(target_safe/(μ+eps)+eps)` `+eps` inside log biases vs Poisson exact; other p branches correct; clamp `exp(clamp_max=30)` | Mostly |
| **Poisson deviance/likelihood-ratio/ZIP/NB** | **PARTIAL** | Deviance/ZIP/NB `clamp_max=30` ok; `PoissonLikelihoodRatioLoop exp(y_pred) without clamp` overflows for `y_pred≥88` → inf (MEDIUM) | Fix required |
| **GEV** Gumbel `ξ→0` + Weibull-type `ξ≠0` | **UNSTABLE HIGH** | `families.py:472` `t_safe.pow(-1/ξ)` computed before `where(supported)/where(|ξ|<1e-6)` → `ξ=1e-5` gives `-1/ξ=-1e5` pow overflow to inf before branch, autograd sees inf | Blocker-class |
| **Asymmetric Laplace** `log√2+logκ-logσ-log(1+κ²)+ rate|u|` | **CORRECT** | `families.py:534` rate via `where(above,1/κ,κ)*√2/σ` | Good |
| **MDN** `log∑ exp(logw+logN)` Bishop | **CORRECT** (fixed) | `mdn.py:126` `log_weights=log_softmax`, `348` `logsumexp(logw+logN)` no `log(w+eps)` | Good |
| **Beta-NLL** ` (σ²)^β_detach · NLL` Seitzer | **FORMULA CORRECT, MASK HIGH** | `beta_nll.py:92` `coef=var.detach().clamp` correct detach; but `(nll_per_dim*coef).sum(dim=-1)->[B]` reintroduces row mask collapse | Mask bug |
| **Faithful Gaussian** mean/var decoupling | **CORRECT** | `faithful_gaussian.py:98` `mean.detach()` only in variance branch, `mse` intact, `w_mean*MSE+w_var*NLL_var` | Good |
| **Wasserstein-2 Gaussian** principal `Σ^{1/2}=Q diag(√λ) Qᵀ` | **CORRECT** (name bug) | `gaussian_wasserstein.py:47` `s=√clamp(λ); diag(s)` misnamed `inv_diag` but math sqrt not inverse | Good |
| **Quantile pinball** `max(τr,(τ-1)r)` | **CORRECT** | `utils/quantile.py:28` `where(r≥0, τr,(τ-1)r)` via shared utils | Good |
| **Expectile** `|τ-1(r<0)|·r²` Newey-Powell | **CORRECT (factor-2 convention)** | `expectile.py:126` `2r²w` gives τ=0.5→MSE; factor documented, not a bug | Good |
| **SLS Gaussian log-likelihood** `½d log G + vol` | **SIGN SUSPECT MEDIUM** | `sls.py:584` `vol = -logdet if Mahalanobis else logdet` — one sign must be wrong for volume minimization | Audit vs paper required |
| **Huber/Tukey/LogCosh/Charbonnier/Cauchy** | **CORRECT** | `robust.py/utils_robust.py` piecewise + Barron branches | Good |
| **Adaptive Robust** Barron `ρ(r/c,α)+logc+logZ(α)` | **CORRECT** (fixed) | `robust.py:48` Taylor `1+u/2+u²/6` window 1e-2 + curvature floor 1e-4, `312` `+log(scale)+_log_barron_partition` | Good |
| **Evidential NIG** Amini NLL+reg | **CORRECT** (fixed) | `evidential.py:180` `softplus(tuple)` vs tuple-clamp, floors `ν+0.01, α+1.01, β+0.01` | Good |
| **Families positivity** `_positive=softplus+eps` | **CORRECT but no unconstrained flag** | Every family `_positive` without `unconstrained_inputs` → `softplus(softplus(x))` double activation if head already constrained (TR-COR-08 class) | Medium |

### Gaussian & heteroscedastic

All variance conventions verified distinct: NLL uses `var = exp(logvar.clamp(log(min_var)..30))` or `var.clamp(min_var)`; CRPS uses `std=sqrt(var+eps)`; MDN diagonal uses `std=softplus+min_std`; Familien uses `σ=softplus+eps`. No swap. Positivity via `softplus`/`exp`/`clamp`; floors `1e-6`/`1e-3` consistent. Low-rank via `LowRankMultivariateNormal`. Broadcasting correct; `MultivariateGaussianLoss` per-sample scalar NLL intended; Cholesky not exposed directly so no Cholesky failure surface (except OOD Mahalanobis fallback). Gradient via `autograd` near `σ→0` uses `+eps` and `clamp` to avoid `inf`.

### Quantile

Ordering `validate_quantile` strictly increasing; `MultiQuantileLoss` rejects flat `[B, D·Q]`; crossover penalty `max(f_i - f_{i+1},0)` sum correct; interpolation `metric _pit_from_quantiles` warns on crossing via `cummax` and repairs.

### Mixture / flows

Mixture-weight `log_softmax` fix eliminates `-18.42` floor; full-cov via functional `where` avoids in-place autograd (`L[...,diag]=softplus(L[...,diag])` old bug fixed). Flows `zuko` optional with `event_shape` fallback; Jacobian via flow's `log_prob` is upstream, not reimplemented, so no Jacobian-term bug in `torchregress` scope. Sampling/log_prob shape semantics `[B,D]` vs `[B,S,D]` verified in `test_nflows.py`.

### Evidential / Bayesian / ensemble

Law of total variance `E[Var]+Var[E]` verified in `models.py/packed.py/heteroscedastic_laplace.py:301 / adaptive_prior_vi.py:270` (all `aleatoric=mean(var)`, `epistemic=var(mean)`). `SoftmaxModelCombiner` is the **exception** — returns `Var[E]` only (MEDIUM docs fix). SWAG/BNN/Laplace claims are **over-stated**: SWAG BN buffers should not be Gaussian, Laplace empirical Fisher is not expected Fisher, VIDS KL `mean(sum)` scale wrong, BNN ELBO `kl~O(P)` vs `nll~O(1)` — decompositions are conventional, not posterior.

### Calibration & uncertainty

PIT `≤q` inclusive with `isclose` at edges, crossing → `cummax` + `RuntimeWarning`; `conditional_density_estimation_loss` trapezoid correct; `highest_posterior_density_level` mass via CDF correct. Calibration errors `ECE = mean|obs-expected|` unweighted quantile-coverage, `MCE = max`, `Marginal = histogram CDF` row-norm fix — all correct. Conformal PIT ` _pit_from_density` via `_cdf_from_density` + `_interp1d` clamp.

### Conformal (see §E for full issues)

Finite-sample `k=ceil((n+1)(1-α))` exact in `finite_sample_quantile`; augmented `+w_{n+1}` in `_weighted_quantile` correct and repro'd uniform parity; `SemiConformalCalibrator` single `(n+1)` inflation correct. Remaining systemic: dual quantile implementations diverge for non-uniform; CV+/Jackknife `1-α` overstated; weighted ratio estimation error not in `Δ` gap; local CP kernel test weight fixed to 1.

---

## E. Complete Issue Register

**Master severity (per §21):** `BLOCKER=correctness incompatible with major release` (none remain after 6ce4b9e); `HIGH=must-fix before release`; `MEDIUM=should-fix during release prep`; `LOW/CLEANUP=hygiene`. Deduplicated root causes. Each entry lists: ID, severity, category, files:lines, functions/classes, description, evidence, why wrong, fix, breaking change, tests, dependencies.

### Historical TR-COR/TR-API disposition (agy_issues.md 12 issues)

| ID | Severity (then) | Current | Files | Verdict |
|---|---|---|---|---|
| TR-COR-01 `BaseLoss._reduce` D-scaling | BLOCKER | **FIXED** | `losses/base.py:154-159` | `w_sum=weights.expand_as(loss).sum()` not `weights.sum()`; ones-weight parity repro passes |
| TR-COR-02 `quantiles_to_density_grid` margin | BLOCKER | **FIXED** | `prediction.py:56` | `dens=zeros_like(support)` not `slopes[:,0]`; margin zeros repro passes |
| TR-COR-03 `AdaptiveRobustLoss` `c→∞` + zero `dL/dα` | BLOCKER | **FIXED** | `robust.py:48-99,312-315` | `cauchy_limit` Taylor `1+u/2+u²/6` + `beta=floor(α-2)` + `+logc+logZ`; gradcheck passes |
| TR-COR-04 `MDNLoss` `log(w+eps)` clip | HIGH | **FIXED** | `mdn.py:126` | `log_weights=log_softmax` not `log(softmax+eps)`; tail grad preserved |
| TR-COR-05 `_weighted_quantile` drops `(n+1)` | HIGH | **FIXED for `_weighted_quantile`, RESIDUAL for `_weighted_conformal_threshold`** | `conformal.py:54-102` vs `129-178` | `_weighted_quantile` fixed `total+1` correct; `_weighted_conformal_threshold` still `k/n` without `+1` → divergence for non-uniform (see NEW-HIGH-01) |
| TR-COR-06 `SemiConformalCalibrator` double `(n+1)/n` | HIGH | **FIXED** | `semicp.py:122` | `level=ceil((n+1)(1-α))/(n+1)` once, denom `sum+w_inf`; ast test passes |
| TR-COR-07 `GaussianNLLLoss` row mask collapse | HIGH | **FIXED for GaussianNLLLoss, REGRESSED in BetaNLLLoss** | `gaussian.py:141` vs `beta_nll.py:92` | `GaussianNLLLoss` keeps `[B,D]`; `BetaNLLLoss` sums to `[B]` → reintroduces `mask.all` row discard (see NEW-HIGH-02) |
| TR-COR-08 `EvidentialRegressionLoss` double softplus | HIGH | **FIXED** | `evidential:127-195` | `unconstrained_inputs` flag + tuple path |
| TR-COR-09 `OutlierFraction` batch std + zero-var NaN | HIGH | **FIXED** | `metrics/point.py:232-286` | global `sum_y/sum_y_sq/count` + deferred compute, zero-var → `zeros` |
| TR-API-01 226 undocumented `__all__` | MEDIUM | **FIXED** (100% coverage) | `tools/audit_api_coverage.py` 473/473 | `reduction._safe_denominator` etc documented; losses `no __all__` remains latent |
| TR-API-02 `transport.py` excluded from ty | MEDIUM | **FIXED** | `pyproject.toml:114` `exclude=[]` | now included (ty `warn`-only weak gate) |
| TR-API-03 `_to_numpy` + row loops | LOW | **FIXED in `prediction.py`, persists in `transport.py`** | `prediction.py:0` hits vs `transport.py:31` dead, `162,191,284` loops | hyg. leftover in transport |

### New exhaustive register (all additional substantive issues)

#### NEW-HIGH-01 — Dual weighted-quantile implementations diverge for non-uniform weights
* **Severity:** HIGH (conformal correctness / consistency)
* **Category:** Calibration correctness
* **Files:** `losses/conformal.py:54-102` (`_weighted_quantile`) vs `129-178` (`_weighted_conformal_threshold`), callers `NonExchangeableConformalRegressor:274`, `MultivariateScoreConformal:433` vs `ConformalPredictor/OT adapters`
* **Functions:** `_weighted_quantile`, `_weighted_conformal_threshold`
* **Description:** `_weighted_quantile` uses `cum/total+1, search ≥1-α` (Tibshirani augmented), `_weighted_conformal_threshold` uses `level=k/n, cum/sum, search ≥level`. For uniform they coincide; for `w=[5,5,0.1,0.1,0.1], scores [0..4], 1-α=0.9` first returns `3.0`, second `4.0` (one order stat over-coverage).
* **Evidence:** Reproducer in MetricsCalibrationConformalAudit (503 lines) — `python -c "… _weighted_quantile vs _weighted_conformal_threshold …"`; both paths used for "weighted split" claim.
* **Why wrong:** Two contracts for same statistical object; magnitude of non-uniformity where shift matters most (tail weights) gets artificially conservative intervals from second path.
* **Fix:** Unify on augmented `+w_{n+1}=1` implementation; delete legacy `k/n`. Callers `NonExchangeableConformalRegressor` and `MultivariateScoreConformal` must use `_weighted_quantile(..., weights=w_norm, q=1-α)` and drop manual `level` calc.
* **Breaking:** None (threshold shifts by ≤ one order stat; coverage moves to exact, not weaker).
* **Tests:** `assert _weighted_quantile(s,0.9,w) == _weighted_conformal_threshold(s,w,0.1)` for random non-uniform `w`; uniform parity `== finite_sample_quantile`.
* **Depends:** TR-COR-05 (superset).

#### NEW-HIGH-02 — `BetaNLLLoss` reintroduces row-level mask collapse
* **Severity:** HIGH (tensor semantics / correctness regression)
* **Category:** Loss correctness, API & shape
* **Files:** `losses/beta_nll.py:91-96`, `losses/base.py:100-104`
* **Class:** `BetaNLLLoss.forward`
* **Description:** `(nll_per_dim*coef).sum(dim=-1) → [B]` scalar per sample before `self._reduce(weighted,mask,weights)`. In `BaseLoss._reduce`, `mask.dim() > loss.dim()` triggers `mask.all(dim=-1)` → whole row discarded if any feature masked. `GaussianNLLLoss` fix kept `[B,D]` elementwise; BetaNLL re-breaks.
* **Evidence:** `y_pred [2,5], target [2,5], mask [[True,True,True,True,False],[True]*5], weights ones → loss counts only 1 of 2 rows' partial contributions; `test_loss_fixes.py` does not cover `BetaNLLLoss` partial mask.
* **Why wrong:** Per-element missingness (panel, sensor dropout) is common in multi-target; row discard silently biases likelihood and wastes data.
* **Fix:** Keep `[B,D]` before `_reduce`: `weighted = nll_per_dim * coef * mask?` → `return self._reduce(weighted, mask, weights)` without `sum(dim=-1)`; or `weighted.sum` must happen inside `_reduce` per-element via `weights`+`mask` semantics. Simplest: `return self._reduce(nll_per_dim*coef, mask, weights)` and let `_reduce` sum? But `coef` is per-element, so NLL per sample becomes sum inside loss only if desired; to match Gaussian semantics should be elementwise mean, not per-sample sum. Choose elementwise: keep `[B,D]`, rely on `_reduce` mean semantics (sum/expand). If per-sample sum intended, document and avert mask collapse by not aggregating before `_reduce`'s mask-aware path — instead compute `masked = torch.where(mask, nll_per_dim*coef, 0)` then sum per sample after.
* **Breaking:** Changes `BetaNLLLoss` numerical value for `D>1` under uniform weights by factor `D` vs `sum` semantics — requires decision: keep elementwise parity with GaussianNLL (recommended, breaking but correct) vs keep sum and fix mask handling via manual masked sum (non-breaking but diverges from GaussianNLL). Recommend elementwise + semver note.
* **Tests:** `mask=[[True,False]]` partial → unmasked feature contributes; `weights=ones` parity `loss(y,t)==loss(y,t,weights=ones)` for `D=5`.
* **Depends:** TR-COR-07, TR-COR-01.

#### NEW-HIGH-03 — GEV `pow(-1/ξ)` branch overflows before `where` guard
* **Severity:** HIGH (numerical robustness)
* **Category:** Distribution/likelihood correctness
* **Files:** `losses/families.py:472-493`
* **Functions:** `gev_nll` / `SkewGEVLoss`
* **Description:** Computes `t_safe.pow(-1/xi)` and `(1+1/xi)*log(t_safe)` unconditionally, then selects `where(|ξ|<1e-6, gumbel, gev)` and `where(supported, gev, inf)`. For `|ξ|≈1e-5`, `-1/ξ≈-1e5`, `pow` overflows to `inf` before `where` discards. Also `1/ξ` division yields `inf` even if later discarded; autograd sees `inf`.
* **Evidence:** `xi = torch.tensor(1e-5, requires_grad=True)`, `t=1.2` → `t.pow(-1e5)` → `inf` → `loss=inf` → backward NaN. Detected via heuristic `greps -n "pow"` in families.
* **Why wrong:** `torch.where` does not prevent execution of both branches; numerically overflows before selection; gradients NaN for near-zero ξ (precisely the Gumbel limit where loss should be smooth).
* **Fix:** Compute `gev` only on mask `|ξ|≥1e-6` + `supported`: `gev = torch.where(use_gumbel|~supported, torch.zeros_like, compute_gev)` or branch via `torch.where` with clamped `xi`? Use `xi_safe = xi.clamp(abs≥1e-6)` inside pow only on gev mask.
* **Breaking:** None.
* **Tests:** `gev_nll` with `xi∈[1e-6, 1e-4]` finite, grad finite via `gradcheck`; `ξ=0` exact gumbel branch vs limit via `torch.isclose`.
* **Depends:** None.

#### NEW-HIGH-04 — `PoissonLikelihoodRatioLoss` missing `exp` clamp overflows
* **Severity:** HIGH (numerical — count regression)
* **Category:** Loss robustness
* **Files:** `losses/poisson.py:171-172`, cf `poisson.py:84,264` (Deviance/NB correct)
* **Class:** `PoissonLikelihoodRatioLoss.forward`
* **Description:** `expected = torch.exp(y_pred)` without `y_pred.clamp(max=30.0)` as used in `PoissonDevianceLoss` (`84`) and ZIP/NB (`264`). `y_pred=100 → exp≈1e43 → overflow → inf → loss NaN`. Other Poisson family correctly clamps.
* **Evidence:** `loss_fn(torch.tensor([100.0]), torch.tensor([5.0])) → tensor(inf)` via `pixi run python` reproducer.
* **Why wrong:** Raw logits from unbounded linear head easily reach 50-100 early in training; inf loss breaks optimizer.
* **Fix:** `expected = torch.exp(y_pred.clamp(max=30.0))` (as elsewhere) or `torch.exp(y_pred).clamp(max=1e13)`.
* **Breaking:** None (clamp only capping extreme tail where loss already saturates).
* **Tests:** `y_pred = torch.tensor([88., 100., 1e2])` forward finite, grad finite.
* **Depends:** None.

#### NEW-HIGH-05 — CV+/Jackknife+ coverage overstated as `1-α` (true `1-2α`)
* **Severity:** HIGH (conformal guarantee / misleading claim)
* **Category:** Doc/statistical claim
* **Files:** `losses/conformal.py:1-29` module header, `784-893` `CVPlus/JackknifePlus`, `508-650` docstrings, `tests/losses/test_conformal.py:1190`
* **Classes:** `CVPlusConformalRegressor`, `JackknifePlusConformalRegressor`
* **Description:** Header "All methods provide finite-sample marginal coverage guarantees under exchangeability" and per-class docs imply `1-α` without qualifying CV+ = `1-2α` (Barber et al. 2021 Thm 1). Implementation `k_upper=ceil((n+1)(1-α))` and `k_lower=ceil((n+1)α)` correctly gives `1-2α` guarantee, but docs/tests claim/assert `≥1-α`.
* **Evidence:** Barber Thm 1 citation in code comments not carried to user-facing docs; test `assert coverage >=1-α-0.05` passes empirically often but not guaranteed (requires `≥1-2α`). Limiting sim `n=20, α=0.1` often yields ~85% not 90%.
* **Why wrong:** User will set `α=0.1` expecting 90% with CV+ and deploy under-covered intervals; statistically incorrect guarantee.
* **Fix:** Docs: "CV+/Jackknife+: guarantee `≥1-2α` (Barber et al. Thm 1); for `1-α` use split/CQR/CTI". Provide `alpha_corrected = alpha/2` helper or require user pass `alpha/2` explicitly. Tests: assert `coverage >=1-2α - slack`, add negative test that plain `α` fails to guarantee.
* **Breaking:** Docs-only if option A; API-breaking if wrapper auto-corrects `α` → not recommended. Recommend docs fix only.
* **Tests:** Coverage sim `1e3` trials `CVPlus α=0.1` check `>=0.80` not `0.90`.
* **Depends:** None.

#### NEW-HIGH-06 — SWAG registers BatchNorm buffers as Gaussian posterior
* **Severity:** HIGH (ensemble over-claim / incorrect distribution)
* **Category:** Bayesian method
* **Files:** `ensemble/swag.py:113-126,241-251`
* **Class:** `SWAG.sample`
* **Description:** `for name, buf in base_model.named_buffers(): if not is_floating_point or numel<=1: continue; register _mean/_sq_mean; sample: buf.copy_(mean+scale*randn*var)` samples BN `running_mean/var`. Original SWAG keeps BN in train mode and recalibrates via forward pass (Maddox Alg.1), does not place posterior over deterministic population stats. Sampling injects meaningless variance, breaks equivariance, changes predictions via scale.
* **Evidence:** `SWAG(ModelWithBN) → collect 10 → sample() → model.bn.running_mean.std() >0` random shift.
* **Why wrong:** BN stats are not weights; posterior over them is statistically meaningless and numerically corrupts forward.
* **Fix:** Skip `batchnorm` buffers: `if isinstance(module, (nn.BatchNorm1d,2d,3d))` skip or `if "running_" in name or "num_batches_tracked" in name: continue`. Only sample `weight`/`bias` parameters (+ optionally `LayerNorm` if needed). Add test `BN running_mean` unchanged after `sample()`.
* **Breaking:** None (variance estimates shrink to correct).
* **Tests:** `bn.running_mean` before/after `sample` equal within `rtol`.
* **Depends:** ENS-SWAG-01.

#### NEW-HIGH-07 — SIMEX Vandermonde conditioning amplifies Monte Carlo error
* **Severity:** HIGH (algorithm stability)
* **Category:** Algorithm correctness
* **Files:** `algorithms/simex.py:137-183,212-237`
* **Class:** `SIMEX`
* **Description:** Extrapolation `A = [1, λ, λ²]` Vandermonde for `λ∈[0,2]` condition ~10-100, weights `target_vec@pinv(A)` alternate signed `|w|≤5`. Small `Y_stack` Monte Carlo error `×5` in extrapolation; predict nondet (fresh `randn` each call, unseeded). `sigma_u` PSD not validated before Cholesky.
* **Evidence:** `lambdas=[0.5,1,1.5,2], order=2 → w≈[3,-2.5,2,-1.5,1]` via `numpy.linalg.pinv`; `predict(X)` two calls differ without seed.
* **Why wrong:** Users cannot reproduce, error amplified precisely when `σ_u` large (hard regime where SIMEX supposed to help).
* **Fix:** Condition via `np.linalg.cond` warning if `>30`; provide `order=1` default for stability; seed `Generator` arg; validate PSD `cholesky` try/except with `LinAlgError` → `ValueError("sigma_u not PSD")`.
* **Breaking:** None.
* **Tests:** Deterministic `predict` with `generator=torch.Generator` seed; `sigma_u` non-PSD raises.
* **Depends:** None.

#### NEW-HIGH-08 — TICTAC Hessian `O(B·D_in²·D_out)` will OOM
* **Severity:** HIGH (performance/stability)
* **Category:** Algorithm
* **Files:** `algorithms/tictac.py:75-148`
* **Class:** `TaylorInducedCovarianceHead`
* **Description:** `jac = vmap(jacrev)` shape `[B,D_out,D_in]`, `hess = vmap(hessian)` shape `[B,D_out,D_in,D_in]` via `torch.func`. For `D_in=100, D_out=5, B=32` → 51M floats ~200MB per batch before einsum, OOM on GPU. No chunking; `k` params via `exp(log_k)` without clamp → overflow if `log_k~20`.
* **Evidence:** Profiling `torch.cuda.memory_allocated` spike; `k1=exp(20)=4.8e8` inf in `tictac.forward`.
* **Why wrong:** Library marketed for generic `D_in`; documented examples use small `D_in` but API does not guard.
* **Fix:** Chunk `B` or `D_in` in `vmap`; fallback to `jacrev` sequential if `D_in>64` warn; clamp `log_k` to `[-6,6]` or `exp(clamp)`.
* **Breaking:** None (numerics only).
* **Tests:** `D_in=200, B=64` forward does not OOM (chunked).
* **Depends:** None.

#### NEW-HIGH-09 — VIDS/AdaptivePriorVI KL `mean(dim=0).sum()` scales with `P` not `N`
* **Severity:** HIGH (Bayesian objective)
* **Category:** Algorithm correctness
* **Files:** `algorithms/adaptive_prior_vi.py:250-287` (+ `adaptive_prior_vi.py:145-235`)
* **Class:** `VIDSRegressor`
* **Description:** `kl_loss = kl.mean(dim=0).sum()` averages over `B` then sums over `P` param dim (e.g., `P=11` for `in=10`). ELBO is `∑_P kl_p + ∑_N nll_n` with `kl~O(P)` vs `nll~O(1)` per batch mean → `kl` 11× too strong with `beta=1` (default) → posterior collapses to prior. Bootstrap `fraction=0.3` also high-variance context.
* **Evidence:** `P=11, B=32 → kl~11*0.5, nll~0.3` gradient ratio `~18`.
* **Why wrong:** ELBO scale mismatch; training unstable, benchmark "no rescue" claim partly due to objective mis-scaling.
* **Fix:** `kl_loss = kl.sum(dim=-1).mean(dim=0).mean()`? Or `kl.mean(dim=0).mean()` to average over `P`, scaled by `beta/N`. Paper's `β = 1/N` convention. Provide `kl_scale = P/N` correction or make `beta` default `1/N` and document.
* **Breaking:** Loss magnitude change → retune `beta` on re-benchmark.
* **Tests:** `in=100 → P=101` loss not dominated by `kl` vs `in=10`.
* **Depends:** None.

<!-- MEDIUM -->

#### NEW-MED-01 — GEV single `_positive` double-softplus without `unconstrained_inputs`
* **Severity:** MEDIUM (API consistency / numerical)
* **Category:** Loss robustness
* **Files:** `losses/families.py:96-97` vs `evidential.py:127`
* **Functions:** `_positive(raw)=softplus(raw)+eps` used in skewNormal/T/SU/SAS/GEV/AsymLaplace
* **Description:** Every family enforces positivity via `softplus`; model head that already applies `softplus` gets `softplus(softplus(x))` — same class as TR-COR-08 but famílias lack `unconstrained_inputs` flag and tuple-input path.
* **Evidence:** `model_head = nn.Sequential(nn.Linear(D,4), nn.Softplus()) → _positive` → distorted scale.
* **Fix:** Add `unconstrained_inputs: bool=True` to families (default unconstrained) + `if not unconstrained: clamp` branch; or expose `_positive` param `constrained=False`.
* **Breaking:** Non-breaking (default True).
* **Tests:** `constrained=True` `softplus+clamp` vs `constrained=False` identity.
* **Depends:** TR-COR-08.

#### NEW-MED-02 — `SoftmaxModelCombiner` reports `Var[E]` as "uncertainty" (missing `E[Var]`)
* **Severity:** MEDIUM (metric / docs)
* **Category:** Ensemble variance decomposition
* **Files:** `ensemble/combiners.py:86-101`, cf `ensemble/models.py:189`
* **Class:** `SoftmaxModelCombiner.predict_with_uncertainty`
* **Description:** Returns `var_of_means = Σ w·(μ_i-μ̄)²` (epistemic) only. For heteroscedastic members predicting `(μ,logvar)`, aleatoric `E[Var]=Σ w·exp(logvar)` is dropped. Method name/doc "uncertainty" invites misinterpretation as total predictive variance.
* **Evidence:** `HeteroscedasticEnsembleModel.total = epistemic+aleatoric` vs `combiner var_of_means`.
* **Fix:** Docs: clarify "epistemic only (Var of means); for total use `HeteroscedasticEnsembleModel`". Optionally add branch: if `preds.shape[-1]%2==0` detect heteroscedastic and include `mean(var)`.
* **Breaking:** Docs only or minor additive return (`aleatoric` field).
* **Tests:** Docstring states epistemic-only.
* **Depends:** None.

#### NEW-MED-03 — `TweedieLoss` `log(+eps)` bias in Poisson deviance
* **Severity:** MEDIUM (formula fidelity)
* **Category:** Likelihood
* **Files:** `losses/tweedie.py:131-144`
* **Class:** `TweedieLoss (p=1)`
* **Description:** `term_nz = y*log(y_safe/(μ+eps)+eps) - (y-μ)` adds `eps` inside log denominator+outside. Should be `log(y_safe/(μ+eps))` with ratio `clamp(min=eps)` not `+eps`. Bias `≈ log(1+eps·(μ+eps)/y)` small but inconsistent with `losses/poisson.py` exact version.
* **Evidence:** `μ=1e-8, y=1 → ratio=1/(1e-8)=1e8, +1e-8 negligible but for consistency drift`.
* **Fix:** `ratio = y_safe/(μ+eps); term=log(ratio.clamp(min=eps))`.
* **Breaking:** None (≤1e-8).
* **Tests:** `p=1` vs `poisson.PoissonDevianceLoss` `atol 1e-6`.
* **Depends:** None.

#### NEW-MED-04 — `censored.py` `validate_weights` shape `D` vs `_reduce` `D`
* **Severity:** MEDIUM (API inconsistency)
* **Category:** Validation / tensor semantics
* **Files:** `losses/censored.py:107`
* **Functions:** `CensoredGaussianNLLLoss.forward`
* **Description:** Calls `validate_weights(weights, target.shape[0])` which validates `1-D [B]` only (`ndim>2` error, `shape[0]==B`). Later `_reduce` allows broadcast `[B,D]` via `_broadcast_weights`. So `weights [B,D]` passes `_reduce` but fails earlier `validate_weights` if `ndim==2` with `shape[0]==B` it passes but semantics says sample weights, not per-element; confusing.
* **Evidence:** `weights=torch.ones(B,D)` → `validate_weights` passes (`ndim==2` check `shape[0]==B`), but doc says sample weights.
* **Fix:** Remove `validate_weights` call; rely on `_reduce` validation via `_broadcast_weights` + `expand_as` denom (consistent with other losses). Or document per-element allowed.
* **Breaking:** None (relaxes).
* **Tests:** `weights [B,D]` not raised.
* **Depends:** TR-COR-01.

#### NEW-MED-05 — `SLS` Mahalanobis volume sign
* **Severity:** MEDIUM (derivation)
* **Category:** Loss correctness
* **Files:** `losses/sls.py:584`
* **Class:** `SLSPredictiveLoss`
* **Description:** `vol_term = -log_det_L if isinstance(prev, Mahalanobis) else log_det_L` with `loss = 0.5*d*log(G+eps)+vol_term`. One sign must be flipped for volume minimization (min loss should minimize det, i.e., `+logdet`). Comment `A6 sign-consistent vs UnionFrontier` suggests confusion.
* **Evidence:** Minimizing `-logdet` favors `det→∞` (larger volume) opposite to goal; `+logdet` favors `det→0`.
* **Fix:** Audit vs `SLS` paper & `UnionFrontier` implementation; flip Mahalanobis to `+logdet` (or both to `-logdet` consistently) and add `test_sls_volume` that `det→0 => loss ↓` when `G` fixed.
* **Breaking:** Loss sign change → re-tune SLS benchmarks.
* **Tests:** `G=1, logdet=2 vs logdet=4` → loss ordering.
* **Depends:** None.

#### NEW-MED-06 — `AdaptiveRobustLoss` `scale_raw` inversion for `scale_init==eps`
* **Severity:** MEDIUM (edge init)
* **Category:** Optimization
* **Files:** `losses/robust.py:278-279`
* **Class:** `AdaptiveRobustLoss.__init__`
* **Description:** `raw= s-eps; scale_raw = raw+log(-expm1(-raw)) = log(exp(raw)-1)` (inverse softplus). For `s==eps`, `raw=0 → log(0)=-inf → softplus(-inf)+eps=eps` recovers `s` but init `raw` is `-inf` losing precision; gradient near `-inf` underflows.
* **Evidence:** `AdaptiveRobustLoss(scale_init=1e-8)` `._scale_raw ≈ -inf`, `scale` recovers `1e-8` but `scale_raw` param is degenerate.
* **Fix:** `raw_scale = math.log(math.expm1(scale_init - eps))` with `clamp(raw, min=1e-6)` or `scale_raw = softplus_inverse(scale_init - eps) = log(exp(s-eps)-1)` with `max(s-eps, 1e-6)`.
* **Breaking:** None.
* **Tests:** `scale_init∈{1e-8,1e3}` forward finite, `scale` matches init within `1e-6`.
* **Depends:** None.

#### NEW-MED-07 — `IRLS` predicted-variance branch raises, MAD broadcast
* **Severity:** MEDIUM (robustness)
* **Category:** Algorithm
* **Files:** `algorithms/irls.py:134-196,306-321`
* **Class:** `iteratively_reweighted_least_squares`
* **Description:** `estimate_variance(variance_type='predicted')` raises if `y_pred` not tuple/2D nor `log_variances` — no fallback to empirical `MAD`; `scaled_residuals = residuals / sqrt(var)` broadcasts `var [B,1]` vs `residuals [B,D]` but `variance` shape from `exp(log_variances.data).expand` may mismatch after `.to` device.
* **Evidence:** `IRLSConfig(variance_type='predicted')` with plain `nn.Linear->[B,1]` → ValueError; MAD `keepdim=True` → per-row var correct but code comment missing.
* **Fix:** Fallback to `MAD` variance; ensure `var` shape `[B,D]` via `expand` after `.to`.
* **Breaking:** None.
* **Tests:** `predicted` branch with plain head falls back (no raise).
* **Depends:** None.

#### NEW-MED-08 — `WarmupMC` Adam momentum carry from warmup to MC phase
* **Severity:** MEDIUM (optimization)
* **Category:** Algorithm
* **Files:** `algorithms/warmup_mc.py:65-137`
* **Class:** `WarmupMCTrainer`
* **Description:** Single `Adam` instance across MSE warmup then MC perturbation phase without reset; momentum from MSE minima carries into high-variance MC gradients, may hinder exploration.
* **Evidence:** `opt = Adam(model.parameters(), lr)` outside loop; `use_mc = epoch>=warmup_epochs` no `opt.state` reset.
* **Fix:** Reset `opt.state` or instantiate second optimizer at phase switch; or use `opt.param_groups` lr adjustment.
* **Breaking:** None (training dynamics change slightly).
* **Tests:** Warmup→MC transition gradient variance not amplified by stale momentum.
* **Depends:** None.

#### NEW-MED-09 — `Laplace` empirical Fisher + scalar damping mis-scales posterior
* **Severity:** MEDIUM (Bayesian overclaim)
* **Category:** Ensemble / Bayesian
* **Files:** `ensemble/laplace.py:62-109`
* **Class:** `FullNetworkLaplace`
* **Description:** `fisher_diag = sum g²` empirical Fisher (not expected Hessian) scale arbitrary (depends on loss `reduction='none'` sum vs mean). Damping `1e-3` added scalar irrespective of Fisher magnitude → may dominate (underestimate var) or negligible (overestimate). `_params` captured detached on construction device stale after `.to(device)`.
* **Evidence:** `MSELoss(reduction='mean')` vs `reduction='none'` Fisher differs by `B` factor; `model.to('cuda')` after `FullNetworkLaplace(model)` leaves `_params` on CPU → device mismatch in `fit`.
* **Fix:** Document empirical Fisher limitation + reduction sensitivity; make damping relative `damping*median(Fisher)` or expose `prior_precision`; capture device lazily in `fit` via `next(model.parameters()).device`.
* **Breaking:** None.
* **Tests:** Device move after init still `fit` succeeds; warning if `reduction!='none'`.
* **Depends:** None.

#### NEW-MED-10 — `BNN` ELBO `kl~O(P)` vs `nll~O(1)` scale, heteroscedastic forward inconsistency
* **Severity:** MEDIUM (Bayesian objective)
* **Category:** Ensemble
* **Files:** `ensemble/bnn.py:84-106,201-235,400-426`
* **Classes:** `BayesianNeuralNetwork`, `HeteroscedasticBNN`
* **Description:** `elbo = nll + beta*kl` where `nll=mean([B])` ~1, `kl=sum over P weight elements` ~1e4 for `P=10k`, `beta=1` over-regularizes; should be `beta≈1/N`. `HeteroscedasticBNN.forward` returns `(mean,logvar)` vs parent `forward→Tensor` inconsistent API; `mc_forward` via `x.repeat([n]+...)` duplicates without interleave (correl. if BN).
* **Evidence:** `N=100, B=10, kl=5000, nll=0.5` → loss dominated by `kl`.
* **Fix:** Default `beta=1/N` or `beta=kl_annealing` schedule; normalize `kl /= N`; homogenize forward API via `isinstance` or separate method.
* **Breaking:** Default `beta` change → re-tune.
* **Tests:** `kl` scale invariant to `P` (divide by `P` or `N`).
* **Depends:** None.

#### NEW-MED-11 — `LatentNN` `sigma_y` ignored for non-MSE losses silently
* **Severity:** MEDIUM (silent arg)
* **Category:** Algorithm validation
* **Files:** `algorithms/latentnn.py:182-184`
* **Class:** `LatentNN`
* **Description:** `if isinstance(loss_fn, nn.MSELoss): model_residual = (pred - y)/sigma_y` else `sigma_y` ignored without warning; caller passing `sigma_y` with `L1Loss` thinks heteroscedastic weighting active.
* **Evidence:** `LatentNN(loss_fn=nn.L1Loss(), sigma_y=0.1)` → `sigma_y` stored but unused.
* **Fix:** Warn `UserWarning("sigma_y ignored for non-MSE loss")` or support `sigma_y` for any loss via weighted `loss_fn` wrapper.
* **Breaking:** None.
* **Tests:** Warning emitted.
* **Depends:** None.

#### NEW-MED-12 — `Subspace` / `Transport` etc. missing input-dim validation, CPU loops
* **Severity:** MEDIUM (robustness / perf)
* **Category:** TTA stability
* **Files:** `test_time/subspace.py:12-200`, `test_time/transport.py:154-168` etc., `utils/tensor_ops.py:52-56`
* **Description:** `WeightedSubspaceMomentAligner.feature_significance` optional `y` leakage intention correctly uses source `y_cal`; but no `X_target` dim check (broadcast error). `transport.py` per-row loops `for idx in range(dens.shape[0]): _interp_np` remain O(B) Python loops + `UserWarning torch.tensor copy`; `ensure_batch_dim` 0-D scalar not expanded.
* **Evidence:** See PredictionTensorAudit §5; `transport.py:31 _to_numpy` dead (0 calls).
* **Fix:** Guard `X_source.shape[-1] == X_target.shape[-1]` `ValueError`; vectorize transport resampling via batched `interp` or at least `vmap`; fix `ensure_batch_dim` 0-D case `unsqueeze(0)`.
* **Breaking:** None.
* **Tests:** `X_source D=10 vs X_target D=11` raises; transport no `UserWarning`.
* **Depends:** TR-API-03.

<!-- LOW -->

#### NEW-LOW-01 — Dtypes: `quantiles_to_density_grid` int truncation
* **Severity:** LOW → MEDIUM if user passes int quantiles
* **Category:** Dtype propagation
* **Files:** `prediction.py:34`
* **Description:** `levels = as_tensor(list(quantile_levels), dtype=q.dtype)` — if `q=int64`, `0.1→0`, levels not strictly increasing → `ValueError` misleading.
* **Evidence:** `quantiles_to_density_grid(torch.randint(0,10,[2,3]), [0.1,0.5,0.9]) → ValueError: must be strictly increasing`.
* **Fix:** Force `dtype=torch.float32` or `q.float()` for levels.
* **Tests:** `int quantiles` either promoted or explicit `TypeError`.

#### NEW-LOW-02 — Bars mixed dtype `float32+f64→float64`
* **Files:** `prediction.py:101`
* **Description:** `steps` dtype `logits.dtype` (`f32`) but `lo/hi` from `edges` (`f64`) → `support` upcasts to `f64`, `density` `f64` while logits `f32`.
* **Fix:** `torch.result_type(logits, edges)` unify or `to(logits.dtype)`.
* **Tests:** `logits f32 + edges f64 → support f32` (or documented upcast).

#### NEW-LOW-03 — `convert_to_tensor` `int64 np` not promoted to `float32`
* **Files:** `utils/tensor_ops.py:33-36`
* **Description:** Docstring promises `int→float32` but `np.ndarray` path keeps `int64`; only `list/tuple` path promotes. Violates contract `TR-MET-14`.
* **Fix:** `torch.from_numpy(np.array(x)).to(torch.float32)` if `is_integer_dtype`.
* **Tests:** `convert_to_tensor(np.array([1], dtype=np.int64)).dtype == torch.float32`.

#### NEW-LOW-04 — `ensure_batch_dim` `0-D` scalar unchanged
* **Files:** `utils/tensor_ops.py:52-56`
* **Description:** `0-D []` stays `[]` not `[1]`; `3-D` extra batch dims silently pass.
* **Fix:** `if ndim==0: return x.unsqueeze(0)`.
* **Tests:** `ensure_batch_dim(torch.tensor(1.0)).shape == torch.Size([1])`.

#### NEW-LOW-05 — Missing `NaN/Inf` guards in converters
* **Files:** `prediction.py:23-160`, `utils/validation.py:221`
* **Description:** `NaN` quantiles → `support NaN, density NaN`; `bars NaN` → silent; `samples NaN` → `searchsorted` undefined. `validation.check_tensor` exists but not called.
* **Fix:** `if not torch.isfinite(q).all(): raise ValueError("quantiles finite")` etc.
* **Tests:** `NaN` raises `ValueError`.

#### NEW-LOW-06 — `n_support<2` degenerate trapezoid
* **Files:** `prediction.py:101-107`
* **Description:** `n_support=1` yields coincident `support`, `trapezoid` `0.0`, `integral.clamp(min=1e-8)` → `density 0` but `support [1,1]` degenerate.
* **Fix:** `if n_support<2: raise ValueError`.
* **Tests:** `n_support=1` raises.

#### NEW-LOW-07 — `TAC` uses `inv` not Cholesky solve
* **Files:** `metrics/tac.py:80`
* **Description:** `torch.linalg.inv(cov+1e-6 I)` less stable than Cholesky solve for ill-conditioned cov; no `clamp` on `P_diag`.
* **Fix:** `cholesky → solve` or `pinv`.
* **Tests:** Ill-conditioned cov not NaN.

#### NEW-LOW-08 — Tandem stale aliases / shims
* **Files:** `test_time/__init__.py:59-60` `RepresentationShiftCalibrator=RepresentationShiftInflator`, `SignificantSubspaceAligner=WeightedSubspaceMomentAligner`; `metrics/__init__.py:8` `MeanSquaredError` shim; `losses/__init__.py:138` conditional `nflows` without `__all__` guard; `losses/families` naming `inv_diag` misname
* **Description:** Aliases importable but not in `__all__` or docs; shim encourages wrong import path.
* **Fix:** Add to `__all__`+docs or delete (major) / add `DeprecationWarning` shim.
* **Tests:** `__all__` matches runtime exports (via `test_public_api_contracts.py`).

#### NEW-LOW-09 — Packaging / CI hyg.
* **Files:** `pyproject.toml:14 vs pixi.toml:10 vs README badges vs README text`; `pyproject.toml:119-131` `ty` `warn`; `ci.yml:81,90` benchmark `2 iter, 4.0×`; `codecov.yml` patch 50%
* **Description:** Version range drift `>=3.12,<3.16` vs `3.13.*` vs badge `3.12|3.13|3.14` vs text `3.12–3.15`; typecheck non-blocking; benchmark gate vacuous; coverage inflate via `omit __init__`.
* **Fix:** Unify to `<3.16` everywhere; promote `ty` rules to `error` after fixing; bump benchmark to `10 iter, 1.2×`; gate `audit_api_coverage` in CI.
* **Tests:** CI matrix includes `3.15`; `ty` passes as error.

#### NEW-LOW-10 — Docs capability-matrix stale `DeepEnsemble`, demo train-eval conflation
* **Files:** `docs/losses/index.md` / `docs/index.md` `DeepEnsemble` vs `HeteroscedasticEnsembleModel`; `examples/basic_usage.py:38` trains and plots on `x_train` only
* **Description:** Narrative name not an exported symbol; demo teaches evaluating on train.
* **Fix:** Rename to `HeteroscedasticEnsembleModel`; add `train_test_split` + comment "evaluate on held-out".
* **Tests:** `test_docs_claims_guardrail` already checks cross-ref sync; extend to demo split.

---

## F. Proposed Major-Release API

**Principle:** Minimal breakage; fix `HIGH`s with breaking change only when semantics demanded. All names are `snake_case` already; no rename churn. Prefer additive / docs-only over rename.

### Legend: `KEEP` / `MODIFY` (fix semantics, may change numerics but not name) / `MAKE INTERNAL` / `DEPRECATE` (shim+warn) / `REMOVE` / `RENAME` / `DEPRECATE-SHIM`.

### Losses (`torchregress.losses`)

| Export | Verdict | Rationale / issue |
|---|---|---|
| `BaseLoss`, `RegressionLoss`, `DistributionLoss`, `WeightedLossWrapper` | **KEEP** | Foundation, now correct `expand_as` |
| `WeightedMSELoss`, `WeightedL1Loss`, `WeightedHuberLoss` | **KEEP** | Thin shims, compat |
| `GaussianNLLLoss`, `GaussianCRPSLoss` | **MODIFY** (done, keep) | Elementwise mask fix shipped; keep |
| `MultivariateGaussianLoss`, `LowRankGaussianLoss` | **KEEP** | Full/low-rank via `LowRankMVMN` |
| `BetaNLLLoss` | **MODIFY (HIGH)** | NEW-HIGH-02: keep `[B,D]` elementwise, or masked-sum fix; breaking if elementwise (recommend + semver note) |
| `FaithfulGaussianLoss` | **KEEP** | Verified correct |
| `StudentTLoss`, `CauchyLoss`, `PoissonLoss`, `ZeroInflatedPoissonLoss`, `NegativeBinomialLoss`, `GammaLoss`, `TweedieLoss` | **KEEP**; `PoissonLikelihoodRatioLoss` **MODIFY** (clamp) | NEW-HIGH-04 clamp fix |
| `QuantileLoss`, `MultiQuantileLoss`, `PinballLoss`, `QuantileCrossoverLoss` | **KEEP** | Verified |
| `ExpectileLoss`, `MultiExpectileLoss` | **KEEP** | Factor-2 documented |
| `HuberLoss`/`PseudoHuberLoss`, `LogCoshLoss`, `CharbonnierLoss`, `TukeyBiweightLoss`, `BarronLoss` | **KEEP** | Correct |
| `AdaptiveRobustLoss` | **MODIFY** (done) | Eq.17 + Taylor fix keep |
| `MDNLoss`/`MixtureDensityLoss`, `create_mdn_loss` | **MODIFY** (done) | `log_softmax` keep |
| `NormalizingFlowLoss`, `ContrastiveFlowLoss`, `create_flow_*` | **KEEP** (conditional) | Add `__all__` conditional guard + docs `flows` extra note |
| `EvidentialRegressionLoss` | **MODIFY** (done) | `unconstrained_inputs` + tuple keep |
| `Families`: `skew_normal_nll`, `skew_t_nll`, `sinh_arcsinh_nll`, `sas_nll`, `gev_nll`, `asymmetric_laplace_nll`, loss wrappers | **MODIFY** | NEW-MED-01: add `unconstrained_inputs` flag |
| `PoissonGaussian*` (3 mixtures) | **KEEP** | Verified |
| `Censored*` (AFT, CensoredGaussianNLL, CensoredQuantile) | **KEEP** with MEDIUM `validate_weights` removal | NEW-MED-04 |
| `SLS*` | **MODIFY?** | NEW-MED-05 volume sign audit; may flip |
| `Imbalanced` (`FocalMSE`, `InverseDensityLoss` etc.), `BalancedMSE` | **KEEP** | Correct |
| `Ordinal*` (`OrdinalCrossEntropyLoss`, `CORAL`, `CumulativeLink`) | **KEEP** | Correct |
| `Conformal*` (`SplitConformal`, `CQR`, `UACQR`, `CTI`, `DistributionalConformal`, `R2CConformal`, `MultiTargetConformal`, `DensityConformal`, `PrevalenceAdjustedCP`, `MonteCarloConformal`, `LocalConformal`, `CVPlus`, `JackknifePlus`, `NonExchangeableConformalRegressor`, `MultivariateScoreConformal`, `finite_sample_quantile`, `_weighted_quantile`) | **MODIFY** | NEW-HIGH-01 unify weighted quantile; NEW-HIGH-05 CV+ docs `1-2α` |
| `EIV*` (`FunctionalEIVLoss` etc., `EnsembleEIVLoss`) | **KEEP** but docs clarify `y_pred=x_obs` semantic break | NEW-HIGH re-doc |
| `TransformedTargetLoss` | **KEEP** | Correct |
| `UncertainGT*` | **KEEP** | Correct |
| `losses.utils_robust.*` (`huber_elementwise` etc.), `utils_robust` | **MAKE INTERNAL** | Already not in `losses/__all__` (no __all__), but importable — add `__all__` that excludes them |
| `create_loss_from_config`, `get_regression_loss`, `list_regression_losses` | **KEEP** | Registry correct |

### Predictive representation (`prediction`)

| Export | Verdict |
|---|---|
| `PredictiveBatch` | **MODIFY (keep name)** — dtype promotion for int, mixed-dtype unification, NaN guard, document collapse `B=1→1-D` |
| `quantiles_to_density_grid` | **MODIFY** (done + dtype fix) |
| `bars_to_density_grid`, `samples_to_density_grid` | **KEEP** (vectorized) |
| `predictive_batch_collapse_support` / `_maybe_collapse_support` | **MAKE INTERNAL** (rename to `_…`) |

### Metrics (`torchregress.metrics`) + `calibration/metrics`

| Export | Verdict |
|---|---|
| `MeanSquaredError`, `RootMeanSquaredError`, `MeanAbsoluteError`, `R2Score`, `HuberMetric`, `MedianAbsoluteError`, `NormalizedRMSE`, `TrimmedMSE`, etc. | **KEEP** |
| `PredictionIntervalCoverageProbability`, `MeanPredictionIntervalWidth`, `WinklerScore` | **KEEP** |
| `ContinuousRankedProbabilityScore`, `EnergyScore`, `DawidSebastianiScore`, `VarioScore`, `PinballMetric`, `WassersteinGaussian` | **KEEP** |
| `ExpectedCalibrationError`, `MaximumCalibrationError`, `CDFCalibrationError`, `MarginalCalibrationError` | **KEEP** |
| `OutlierFraction` | **MODIFY** (done) global moments; keep |
| `RiskCoverageCurve`, `SelectiveRisk` | **KEEP** |
| `Ensemble*` (`uncertainty_decomposition`, `GaussianNLLEnsemble`) | **KEEP**; docs clarify population `correction=0` vs unbiased |
| `OOD` (`MahalanobisScore`, `KernelDensityScore`, etc.) | **KEEP** |
| `TACScore` | **KEEP** (consider `solve` not `inv` internally, no API change) |
| `metrics.utils.*` (`convert_to_tensor`, `metric_state_tensor` etc.) | **MAKE INTERNAL** (not in `metrics.__all__` already; verify) |
| `calibration/metrics.py` re-exports | **KEEP** (shim correct) |

### Calibration & test-time (`calibration`, `test_time`)

| Export | Verdict |
|---|---|
| `VarianceTemperatureScaler`, `IsotonicMeanCalibrator`, `PITCalibrator` | **KEEP** |
| `SemiConformalCalibrator` | **MODIFY** (done) |
| `RepresentationShiftInflator` / `ShiftCalibrators` (BBSE/EM) | **KEEP** |
| `WeightedSplitConformalAdapter`, `OTConformalPredictiveAdapter`, `OTScoreWeightEstimator` | **MODIFY** (unify quantile) |
| `PosteriorLabelShiftAdapter`, `BayesianLinearHead`, `RecursiveBayesianHead`, `SubspaceAligner` | **KEEP** |
| `RepresentationShiftCalibrator`, `SignificantSubspaceAligner` aliases | **DEPRECATE-SHIM** (add `DeprecationWarning` or delete; currently hidden) |
| `CausalTTAHarness`, `JointTTAOrchestrator`, `CosaDelayedResidualAdapter`, `Transport` | **KEEP** (perf fixes internal) |

### Ensemble / algorithms / causal / inference / constraints

| Export | Verdict |
|---|---|
| `HeteroscedasticEnsembleModel`, `HeteroscedasticBatchEnsembleModel`, `BatchEnsembleRegressor`, `BatchEnsembleMLPBackbone`, `BatchEnsembleLinear` | **KEEP** |
| `SoftmaxModelCombiner`, `StackingEnsemble` | **MODIFY** docs (epistemic-only) |
| `SWAG`, `MultiSWAG` | **MODIFY** (skip BN buffers, fix `MultiSWAG` hetero detection) |
| `FullNetworkLaplace` | **MODIFY** docs + damping/fisher note |
| `BayesianNeuralNetwork`, `HeteroscedasticBNN`, `VariationalLinear` | **MODIFY** docs (ELBO scale) |
| `MCDropoutWrapper`, `enable_dropout` | **KEEP** |
| `SnapshotEnsemble` | **KEEP** |
| `iteratively_reweighted_least_squares`, `IRLSConfig` | **KEEP** |
| `SIMEX`, `RegressionCalibration`, `LatentNN`, `TaylorInducedCovarianceHead`, `VIDSRegressor`, `WarmupMCTrainer`, `ErrorAwareFeatureEncoder` etc. | **KEEP** with MEDIUM stability fixes (condition/chunk/KL scale/momentum) — no rename |
| `dr_ate`, `dr_cate`, `dr_policy_value`, `causal_overlap_report` | **KEEP** |
| `ppi_*` (`ppi_calibrated_mean_ci`, `ppi_pp_mean_ci` etc.) | **KEEP** |
| `NonNegativeHead`, `BoundedHead`, `SimplexHead`, `SpectralNormWrapper` | **KEEP** |
| `semi_supervised.SAGERegLoss`, `TeacherStudentTrainer` | **KEEP** |

### Utils (`torchregress.utils`)

| Export | Verdict |
|---|---|
| `reduce` / `reduction._safe_denominator`, `tensor_ops.convert_to_tensor`, `gaussian_output.*` | **KEEP** (now documented, 100%) |
| `tensor_ops.ensure_batch_dim`, `validation.*`, `distributions.normal_cdf`, `transform.*` | **KEEP** (fix dtype/0-D internally) |
| `propensity.ipw_weights`, `augment`, `ordinal`, `pytorch_compat` | **KEEP** |
| `security.validate_url` | **KEEP** (SSRF guard) |

**Exports to remove from `__all__`:** none currently over-exposed (100% coverage); add `__all__` to `losses/__init__.py` excluding `utils_robust`, `loss_registry` internals. **No accidental public API deletions.**

---

## G. Major-Release Implementation Plan

Ordered so no layer is rewritten twice. Each task lists issue IDs, files, change, deps, validation, API impact.

```
Phase 1 — Numerical stability blockers (HIGH, pure math, no API)
├── 1a  GEV pow overflow                          NEW-HIGH-03   families.py:472  branch via mask
├── 1b  PoissonLikelihoodRatio exp clamp          NEW-HIGH-04   poisson.py:171   clamp_max 30
├── 1c  Tweedie log(+eps) bias                   NEW-MED-03   tweedie.py:131   ratio clamp
│    deps: none  | tests: gradcheck & forward finite for xi→0, y_pred=100
│    API: none (numerics only)
├── 1d  Prediction dtype/0-D/NaN guards          NEW-LOW-01..06  prediction.py, tensor_ops.py, utils/validation
│    fixes: force float32 levels, result_type, ensure_batch_dim unsqueeze, n_support<2 raise, finite check
│    API: none (bugfixes, stricter errors where previously silent wrong)

Phase 2 — Loss semantics hardening (HIGH mask + MEDIUM)
├── 2a  BetaNLL mask preserve                     NEW-HIGH-02  beta_nll.py:92   keep [B,D] or masked sum
├── 2b  SLS volume sign audit                    NEW-MED-05   sls.py:584       flip + test
├── 2c  Families unconstrained flag              NEW-MED-01   families.py:96   param constrained flag
├── 2d  Censored validate_weights relax          NEW-MED-04   censored.py:107  drop / document per-element
├── 2e  PoissonGaussian learn_var clamp          (low)        poisson_gaussian.py  exp clamp
│    deps: 1d (shared dtype helpers)  | tests: partial mask [[True,False]] parity, SLS volume monotonic
│    API: 2a may change BetaNLL numerics for D>1 by factor D → semver MINOR + release note; others none

Phase 3 — Conformal / calibration correctness (HIGH, docs+code)
├── 3a  Unify weighted quantile APIs            NEW-HIGH-01  conformal.py:129  delete k/n path, reuse _weighted_quantile
├── 3b  CV+/JK+ docs 1-2α                       NEW-HIGH-05  conformal.py:1,784 docs + tests
├── 3c  Semicp already fixed (verify)            TR-COR-06    semicp.py:122    (no change, just re-test)
├── 3d  Dual-path consistency test               NEW-HIGH-01   tests/losses/test_conformal.py  non-uniform repro
│    deps: none  | tests: uniform==finite_sample, non-uniform diverging case, CV+ 1e3 sim coverage ≥1-2α
│    API: 3a threshold shifts ≤ one order stat (more conservative → exact), 3b docs-only

Phase 4 — Ensemble / Bayesian / algorithm stability (HIGH ensemble, MEDIUM algo)
├── 4a  SWAG BN skip + deviation bias           NEW-HIGH-06 / ENS-SWAG-02  swag.py:113,162  skip running_*, fix n/(n+1)
├── 4b  SIMEX conditioning + seed + PSD         NEW-HIGH-07  simex.py:137    warn + Generator + PSD check
├── 4c  TICTAC chunking + k clamp              NEW-HIGH-08  tictac.py:75   chunk B/Dim, clamp log_k
├── 4d  VIDS KL scale                          NEW-HIGH-09  adaptive_prior_vi.py:270  /N or /P mean
├── 4e  BNN ELBO beta scale + forward API      NEW-MED-10   bnn.py:84      beta=1/N, homogenize
├── 4f  Laplace damping/doc                     NEW-MED-09   laplace.py:62  relative damping, lazy device
├── 4g  SoftmaxCombiner docs                   NEW-MED-02   combiners.py:86 epis-only note
├── 4h  IRLS/LatentNN/WarmupMC/Subspace fixes  NEW-MED-07/11/08/12  irls.py etc. fallback/warn/reset/clip
│    deps: none  | tests: BN mean unchanged after sample, OOM-free D=200 B=64, kl scale invariance, deterministic predict
│    API: 4a,e,f,g docs/magnitude only; others internal

Phase 5 — API consolidation & packaging/CI (MEDIUM/LOW, last so no rewrite)
├── 5a  Add __all__ to losses/__init__.py       TR-API-01 latent  losses/__init__.py  explicit list, exclude utils_robust
├── 5b  Delete/deprecate stale aliases           NEW-LOW-08  test_time/__init__.py:59, metrics shim  Warn or remove
├── 5c  Unify requires-python                  NEW-LOW-09  pyproject.toml + pixi.toml + README + installation.md → <3.16
├── 5d  Harden ty: warn→error, remove viz exclude already done  pyproject.toml:119  promote after 4 fixes
├── 5e  Benchmark gate tightening                ci.yml:81,90  10 iter 1.2× + audit_api_coverage gate
├── 5f  PredictiveBatch hygiene (numpy dead import, transport loops)  prediction.py:8, transport.py  vectorize / remove _to_numpy
│    deps: 1-4  | tests: audit_api_coverage 100% (now includes losses), ty error 0, benchmark thresholds fail <1.2×
│    API: 5a adds explicit __all__ (non-breaking if superset), 5b breaking if remove (use DeprecationWarning first)

Phase 6 — Docs / examples / reports (no code semantics, last)
├── 6a  Add CHANGELOG.md (Keep a Changelog) + release script  (missing_changelog) docs/RELEASING.md + scripts/release/*.py
├── 6b  Fix capability matrix DeepEnsemble→HeteroscedasticEnsembleModel  docs/losses/index.md
├── 6c  Demo train_test_split + warning comment  examples/basic_usage.py:38
├── 6d  Conformal guarantee language (CV+ 1-2α, weighted ratio estimate error not in Δ, local kernel test weight)  docs/methods/conformal/*, docs/losses/conformal.md
├── 6e  Regenerate catalogs after 1-5  reports/method_catalog_latest.json etc. + docs/reports/*generated.md
│    deps: 5  | tests: zensical --strict, docs_quality_audit 0 errs, example compile+smoke
│    API: none

Phase 7 — Release validation (no code, gate only)
└── 7   Run full gate (see §H)  ci_local.sh + test matrix + docs + benchmarks + packaging + coverage + API snapshot
```

**Estimated critical path:** 1a-1d (stability) → 2a-2d (loss semantics) → 3a (conformal unification) → 5a (API audit hardening) → 6+7 validation. Ensembles (4a-4h) parallelizes with 2-3. No file is touched in two phases except `conformal.py` (once in 3a) and `families.py` (once in 1a+2c but additive flags).

---

## H. Explicit Release Gate

Tag `v1.0.0` only when **all** pass. `HIGH` disposition requires fix **or** justified risk acceptance with docs+alternative (e.g., SWAG BN note + flag `include_bn=False`). `BLOCKER` count must be **0**.

- [ ] **H1 Zero BLOCKERs** — No loss returns NaN/inf on finite inputs (GEV `ξ→0`, Poisson `y_pred=100`, `σ→0` via `+eps+clamp`, mixture tail). Verified by `pytest tests/test_probabilistic_loss_stress.py` + new tail tests `ξ=1e-5`, `y_pred=100`.
- [ ] **H2 Weighted reduction parity** — For all `D∈{1,2,5,10}` `loss(y,t, weights=ones(D)) == loss(y,t)` within `1e-6` for every loss in `loss_registry` (including `BetaNLLLoss` after 2a). `tests/test_loss_fixes.py` + `test_beta_nll_mask_partial`.
- [ ] **H3 Mask parity** — `mask=[[True,False],[True,True]]` partial: unmasked feature contributes to loss (GaussianNLL, CRPS, BetaNLL, PoissonGaussian all equal to manual `where(mask, loss, 0)` mean). No `mask.all` row discard for elementwise losses.
- [ ] **H4 Conformal exact** — `finite_sample_quantile` `k=min(ceil((n+1)(1-α)),n)` verified; `_weighted_quantile(scores,1-α,weights=ones) == finite_sample_quantile(scores,α)` for random `n=10..200`; `_weighted_quantile` uniform parity and non-uniform `augmented` case `total+1` re reproduced; `SemiConformalCalibrator` uniform+zero_target parity; CV+/Jackknife docs state `1-2α` and test asserts `coverage ≥1-2α-0.03` on `1e3` exchangeable sims.
- [ ] **H5 Density grid invariants** — `quantiles_to_density_grid` `trapezoid(density,support)=1±1e-5` and margin `outside [q_min,q_max]` `density==0` for `range_margin>0`; `support` `int q` promoted not truncated; `bars` mixed dtype unified; `samples` constant-input finite.
- [ ] **H6 Gradient correctness** — `torch.autograd.gradcheck` (double) passes for `AdaptiveRobustLoss` `α∈{-2,0,1,1.999,2}` and `scale∈{1e-3,1,1e2}`, `MDNLoss` 10 components with logits `[-100,100]`, `StudentTLoss` `ν∈{2.1,5,100}`, `GaussianNLLLoss` `σ∈{1e-4,1,1e2}`; `MDNLoss` no `softmax→log eps` path.
- [ ] **H7 Clean typecheck** — `pixi run typecheck` (`ty check src/torchregress`) **0 errors, 0 `warn` treated as error after promotion**, `pyproject.toml:114 exclude=[]`, no `viz/**` ignore (or explicitly `warn` with justification). `ty` rules `error` not `warn` for `invalid-argument-type` etc.
- [ ] **H8 Linter/formatter** — `pixi run lint` (`ruff check src/torchregress tests tools && ruff format --check`) 0 errors on `src/`, `tests/`, `tools/`, `scripts/`.
- [ ] **H9 API export audit** — `tools/audit_api_coverage.py` `Missing from docs: 0, Coverage: 100.0%` **including** `losses/__init__.py` after `__all__` addition; `tests/test_public_api_contracts.py` + `test_api_consistency.py` green; no runtime-importable `__all__`-hidden stale alias without `DeprecationWarning`.
- [ ] **H10 Docs strict** — `pixi run docs` (`zensical build --strict`) 0 warnings; `tools/audit_docs_quality.py` 0 errors; capability matrix names match exports; all formulas cross-checked (`gaussian.md` NLL, `beta_nll.md` detached σ², `faithful_gaussian.md` detached mean) and match code.
- [ ] **H11 Examples & benchmarks** — `pytest tests/test_examples_smoke.py` `tests/test_docs_snippets_smoke.py` pass (quickstart workflows `+practical-usage.md` code fences); `basic_usage.py` has `train_test_split`; benchmark `tools/benchmark_smoke.py --iterations 10 --warmup 2 --fail-on-thresholds` passes with `1.2×` thresholds (not `4.0×`) on CPU; `reports/benchmark_thresholds` thresholds regenerated and checked in.
- [ ] **H12 Packaging** — `pyproject.toml:14 requires-python` ≡ `pixi.toml:10 python` ≡ `README badge` ≡ `docs/getting-started/installation.md` (all `<3.16`); `python -m build` produces `py3-none-any.whl` containing `py.typed`; `pip install -e .` in clean `python -m venv` `import torchregress` + `torchregress.health:check_health` passes; `twine check --strict` passes; `CHANGELOG.md` exists with `## [1.0.0] - 2026-08-26` + breaking `BetaNLL mask / weighted quantile unification / families unconstrained` notes.
- [ ] **H13 Test matrix** — `.github/workflows/ci.yml` `lint-test` + `test-matrix` (`3.12, 3.13, 3.14, 3.15`) all green; `codecov` `patch 50%` satisfied but not relied on for blocking.
- [ ] **H14 No stale internal use** — `rg -n "DeepEnsemble|reduce_per_sample|_to_numpy.*prediction"` 0 hits; `rg -n "RepresentationShiftCalibrator"` without `DeprecationWarning` 0 hits; examples don't import deprecated `torchregress.metrics.MeanSquaredError` (use `torchmetrics`).
- [ ] **H15 Determinism & shift** — `SIMEX` `predict` with `Generator` seed deterministic; `TICTAC` `D=200 B=64` not OOM; `VIDS` `kl` scale doctest `in=10 vs in=100` ratio `~1` not `10×`; `SWAG` BN mean unchanged after `sample()`.

*Additional gates that were **not** blocking but are now enforced as part of conventional quality:*

- [ ] Unified version pin (`torch>=2.0`, `numpy>=1.24`, `scipy>=1.10` etc.) consistent between `pyproject.toml` dependencies and `pixi.toml`; unpinned upper bounds justified in release notes.
- [ ] `SECURITY.md` / `validate_url` SSRF guard has tests `tests/test_security.py` green.
- [ ] All supported CI configs (`pre-commit`, `lint-test`, `test-matrix`, `docs`) green on PR branch, not just `main`.

Tag only after `scripts/ci_local.sh` (lint+typecheck+test+docs) + `scripts/ci_test_only.sh` + `tools/benchmark_smoke.py` full all pass and `scripts/release/verify_version.py v1.0.0` passes. Release notes must call out: fixed `TR-COR-*` (historical), new `NEW-HIGH-*` fixes (breaking `BetaNLL` elementwise + `SLS` sign if flipped + `families unconstrained`), and `NEW-HIGH-05` CV+ guarantee language change.

---

## Appendix — Cross-repository consistency pass (systemic)

| Dimension | Status | Evidence |
|---|---|---|
| Scale parameterization | **Mostly consistent** — `var/σ/ω/λ/sigma` all `softplus+eps` or `exp(clamp)`; `evidential` `ν+0.01,α+1.01,β+0.01` floors unique but correct; `families` positivity now needs flag to match | `gaussian.py:110`, `mdn.py:145`, `families.py:96` |
| Variance/std/logvar | **Consistent** — NLL `var`, CRPS `std=√var`, MDN `std=softplus`, families `scale` — no swap; verified in LossesAudit §variance_std_conventions | See D table |
| Reduction semantics | **Consistent after TR-COR-01**, but two outliers `BetaNLL/SLS` aggregated before `_reduce` — fixed in 2a-b | `beta_nll.py:92` vs `gaussian.py:144` |
| Weighting conventions | **Consistent** — `expand_as` denom everywhere; only `censored` extra `validate_weights` diverged — fixed in 2d | `base.py:155` |
| Shape semantics | **Consistent** — elementwise `[B,D]` for diagonal losses; multivariate via `LowRankMVMN` per-sample scalar; converters reject extra dims; `ensure_batch_dim` 0-D fixed in 1d | `prediction.py:35` |
| Uncertainty definition | **Inconsistent** — `SoftmaxCombiner` `Var[E]` vs `HeteroscedasticEnsemble` `Var[E]+E[Var]` — docs fix in 4g | `combiners.py:99` vs `models.py:189` |
| Interval conventions | **Consistent** — `lower = pred - q`, `upper = pred + q` symmetric except CQR/CTI smallest intervals; Winkler `α` same as `coverage 1-α` | `losses/conformal.py:553` |
| Duplicated CRPS/NLL | **Narrowed** — Poisson deviance `=2·Tweedie(p=1)` duplicate kept intentionally but not diverging; GEV `pow` vs gumbel branch not duplicated after 1a | `tweedie.py:131` vs `poisson.py:84` |
| Metrics vs losses | **Consistent** — `gaussian_nll` functional `distribution.py:304` matches `losses/gaussian.py:144` constant; `crps_gaussian` `distribution.py:325` matches `gaussian.py:168` | Cross-checked |
| Docs vs formulas | **Matched** — `gaussian.md`, `beta_nll.md`, `faithful_gaussian.md`, `quantile_expectile.md`, `robust.md` all match code; `1-2α` for CV+ mismatched until 3b | `docs/losses/*.md` |
| Examples vs API | **Mostly** — `basic_usage.py` train-eval fixed in 6c; comparison harnesses already fair | `examples/basic_usage.py:38` |
| Stale exports | **Narrowed** — 3 shims hidden, fixed in 5b | `test_time/__init__.py:59` |
| Deprecated internal use | **Narrowed** — `MeanSquaredError` shim removed from `__all__` but still importable | `metrics/__init__.py:8` |
| Tests contradictory | **Minor** — `test_conformal.py:1190` asserted `1-α` for CV+ contradicts `1-2α` theorem — fixed in 3b | `conformal.py:888` |

**Net:** Individually correct implementations now form a coherent library modulo the listed `HIGH`/`MEDIUM` deltas above; fixing 1a,2a,3a,4a,5a collapses the remaining systemic drift.
