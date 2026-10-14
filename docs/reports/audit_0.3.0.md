# 0.3.0 release audit

Audit of the library before the first PyPI release (release plan Phase 2.2,
all five batches), run 2026-10-06: batches 1–3 at commit `5a334d3`, batches
4–5 at `d61ebb2`. **99 defects found, 99 fixed.** Batch C2 (code added afterwards) found and fixed 6 more.

**Evidence rule.** A finding counts only when a test reproduces it: every row
below had a test that failed on `5a334d3`, and that test now lives in
`tests/audit/` and passes. Tests are named `test_<ID>_<slug>`.

**Rubric** applied to every public export: maths against the cited paper,
float32/float64 numerics at extreme inputs, `gradcheck` in float64,
statistical guarantees checked by simulation (coverage inside the binomial
interval), the `reduction`/`mask`/`weights` contract of `losses/base.py`, and
dtype/device hygiene.

| Batch | Scope | Findings | Fix commit | Tests |
|:--|:--|--:|:--|:--|
| A | `losses`: Gaussian family, families, flows, quantile/expectile, robust, Poisson, censored, evidential, Wasserstein | 20 | `f6ebbbc` | `tests/audit/test_audit_losses_a.py` |
| B | `losses/conformal.py`, EIV, imbalanced, SLS, uncertain GT, `calibration/semicp.py` | 14 | `6c63316` | `tests/audit/test_audit_conformal_losses_b.py` |
| M | `metrics`, `calibration` (post-hoc) | 14 | `4f102ac` | `tests/audit/test_audit_metrics.py` |
| I | `inference`, `causal`, `ensemble`, `semi_supervised`, `prediction` | 13 | `39a8879` | `tests/audit/test_audit_inference.py` |
| 4 | `algorithms`, `test_time` | 19 | see below | `tests/audit/test_audit_batch4.py` |
| 5 | `utils`, `prediction`, `comparison`, `viz` | 19 | see below | `tests/audit/test_audit_batch5.py` |
| C2 | `models`, `estimators`, `inference.orthogonal` additions, `losses.nflows` | 6 | `c2ada98` | `tests/audit/test_audit_c2.py` |

Severity: **High** = wrong numbers or a broken guarantee in normal use;
**Medium** = edge cases or an API-contract violation; **Low** = minor.

## Batch A: losses

| ID | Sev | Export | Finding |
|:--|:--|:--|:--|
| A-LOSS-016 | High | `EvidentialRegressionLoss` | NIG NLL used Ω = 2β instead of 2β(1+ν) (Amini et al. 2020, Eq. 8): trained the Student-t of μ, not of y, inconsistent with `predict_interval` |
| A-LOSS-003 | High | `SkewTLoss` | Incomplete-beta symmetry branch divided by the wrong argument; Student-t CDF wrong, NLL off by up to 0.7 nat |
| A-LOSS-002 | High | `BetaNLLLoss` | 1-D `[B]` inputs summed over the batch; `'mean'` was B× too large, mask and weights ignored |
| A-LOSS-012 | High | `CVaRLoss` | 1-D inputs reduced over all dims or raised |
| A-LOSS-004 | High | `SkewNormalNLLLoss` | log Φ clamped at float32 tiny: NLL saturates, zero gradient on the short-tail side |
| A-LOSS-017 | High | `GaussianWassersteinBoundLoss` | `eigh` backward gave NaN gradients for repeated eigenvalues (e.g. isotropic init) |
| A-LOSS-001 | Medium | `BaseLoss._reduce` | `[B, D]` mask with `[B]` weights raised |
| A-LOSS-019 | Medium | `BaseLoss._reduce` | `[B, D]` weights on per-sample losses were summed over D |
| A-LOSS-015 | Medium | `CensoredGaussianNLLLoss`, `AFTLoss` | censored terms saturated at 18.42 nats beyond ~5.6σ |
| A-LOSS-009 | Medium | `NormalizingFlowLoss`, `ContrastiveFlowLoss` | bypassed `_reduce`: weighted mean not normalised, masked `'none'` not zero-filled |
| A-LOSS-010 | Medium | `MultiExpectileLoss`, `ExpectileCrossoverLoss` | own mean: weights not normalised, masked samples in the denominator |
| A-LOSS-011 | Medium | `ExpectileCrossoverLoss` | silently sorted levels while columns stayed put |
| A-LOSS-007 | Medium | `BetaRegressionNLLLoss`, `SQRLoss` | `[B, 1]` targets broadcast to a `[B, B]` loss |
| A-LOSS-013 | Medium | `ZeroInflatedPoissonNLLLoss` | probability-space eps saturation; vanishing gradient on confident logits |
| A-LOSS-005 | Medium | `GEVNLLLoss` | float32 error of 0.17 nat just above the Gumbel threshold |
| A-LOSS-008 | Medium | family losses (`unconstrained_inputs=False`) | inverse softplus overflowed for scales above ~88 |
| A-LOSS-020 | Low | `FaithfulGaussianLoss` | cached mean-term flag went stale after `load_state_dict` |
| A-LOSS-006 | Low | `GEVNLLLoss` | NaN gradient from the unused Gumbel branch |
| A-LOSS-014 | Low | `NegativeBinomialNLLLoss` | θ in float32 for float64 inputs |
| A-LOSS-018 | Low | `losses.__all__` | `StudentTLoss`, `ExpectileCrossover` missing |

## Batch B: conformal prediction and losses

| ID | Sev | Export | Finding |
|:--|:--|:--|:--|
| B-CONF-002 | High | weighted conformal, `NonExchangeableConformalRegressor`, `MultivariateScoreConformal` | test-point weight fixed at 1 on the raw scale and never +inf: coverage 0.77 vs ≥ 0.9 under covariate shift (Tibshirani et al. 2019) |
| B-CONF-003 | High | `SemiConformalCalibrator` | returned the max score instead of +inf; float32 CDF picked the wrong order statistic |
| B-CONF-004 | High | CV+ / Jackknife+ | lower rank `ceil` instead of `floor(α(n+1))` (Barber et al. 2021); no ±inf endpoints |
| B-CONF-005 | High | `DensityConformal` | calibration and prediction used different densities: scores not exchangeable, coverage 0 on a bimodal target |
| B-CONF-006 | High | Mondrian conformal | negative group ids read the wrong threshold; R2C, CTI, DCP, SLSConformal used the first group's threshold for all points |
| B-LOSS-002 | High | `LatentMarginalizationLoss`, `InputNoiseAugmentationLoss` | marginalisation silently skipped (base loss returned a scalar), weights ignored |
| B-LOSS-001 | High | Functional/Structural EIV | Jacobian detached: no gradient through J Σx Jᵀ |
| B-LOSS-003 | High | `OrthogonalDistanceRegressionLoss` | absolute 1e-3 jitter (loss 0.55 vs 1.15 closed form); batch-mean unrolled steps; masked rows counted |
| B-LOSS-005 | High | `BalancedMSELoss` | an empty bin collapsed the loss by ~1e-8 |
| B-CONF-001 | Medium | `finite_sample_quantile` and users | clamped to the max score when ⌈(n+1)(1−α)⌉ > n: coverage below 1−α at small n |
| B-LOSS-004 | Medium | `SLSLoss` | `forward` mutated frontier state; the two passes used different β |
| B-LOSS-006 | Medium | FocalR, BalancedMSE, BinReweightedMSE, DensityWeighted, LDS, PropensityWeighted | weights bypassed `_reduce` |
| B-LOSS-007 | Medium | `DensityWeightedLoss` | normalised over the batch without `sample_indices`; NaN targets poisoned the loss |
| B-LOSS-008 | Low | pseudo-label losses | NaN placeholder targets gave NaN |

## Batch M: metrics and calibration

| ID | Sev | Export | Finding |
|:--|:--|:--|:--|
| M-MET-001 | High | `distribution_metrics_report` | CRPS from 7 sample quantiles, 3–4% low |
| M-MET-006 | High | `median_absolute_error` (multi-output) | median of row means instead of mean of per-output medians (sklearn) |
| M-MET-002 | Medium | quantile PIT | out-of-range targets got PIT exactly 0/1: KS and χ² invalid for calibrated forecasts |
| M-MET-003 | Medium | `energy_score`, `kernel_density_score` | float32 `cdist` matmul mode lost up to 16% |
| M-MET-004 | Medium | `crps_from_samples`, `vario_score` | NaN for a single sample |
| M-MET-005 | Medium | `VarioScore` | `higher_is_better` had the wrong orientation |
| M-MET-008 | Medium | trimmed MSE | asymmetric trimming |
| M-MET-009 | Medium | `EnsembleIntervalMetrics` | `reset()` left child metrics stale |
| M-MET-010 | Medium | `gaussian_nll`, ensemble NLL/intervals | absolute variance floors distorted small-scale targets |
| M-MET-011 | Medium | Mahalanobis, TAC | absolute covariance jitter, not scale-equivariant |
| M-CAL-001 | Medium | `VarianceTemperatureScaler` | undocumented T ∈ [0.05, 20] clamp |
| M-CAL-002 | Medium | `VarianceTemperatureScaler` | default fit stopped 5% short of the MLE |
| M-CAL-003 | Medium | `IsotonicMeanCalibrator` | not isotonic regression (ties not pooled, centroid interpolation) |
| M-MET-007 | Low | MedAE, MAD, NMAD | lower median for even n |

## Batch I: inference, causal, ensembles

| ID | Sev | Export | Finding |
|:--|:--|:--|:--|
| I-INF-001 | High | `ppi_quantile_ci` | returned Q(f_u) + median residual, not the PPI quantile: 0% coverage (Angelopoulos et al. 2023) |
| I-CAU-001 | High | `dr_ate`/`dr_cate` (`fold_bootstrap`) | bootstrapped K fold means: SE 0.56× at K = 2, coverage 0.56–0.65 |
| I-ENS-001 | High | heteroscedastic/batch/packed/SWAG ensembles | predicted log-variance clamped to [−8, 6] vs the training range [log 1e-6, 30] |
| I-INF-002 | Medium | PPI functions | float64 downcast to float32 (SE 0 at large offsets) |
| I-CAU-002 | Medium | DR estimators | float32 downcast |
| I-CAU-003 | Medium | DR estimators | silent NaN when trimming removed every unit |
| I-ENS-002 | Medium | ensemble `predict` | ran in train mode after `fit` (dropout on, BatchNorm stats overwritten) |
| I-INF-004 | Medium | PPI bootstrap | `[n_boot, N]` index matrix: 15 GB at N = 1e6 |
| I-INF-003 | Medium | semi-supervised trust weights | documented options silently ignored or raised |
| I-ENS-003 | Medium | `bars_to_density_grid` | density outside the bin edges |
| I-INF-005 | Low | `orthogonal_partially_linear` | identification check not shift-invariant |
| I-ENS-004 | Low | ensembles with `base_seed` | reseeded the global RNG |
| I-ENS-005 | Low | ensemble docs | documented removed APIs |

## Checked and found correct

Reference implementations matched on: `GaussianNLLLoss`, `GaussianCRPSLoss`,
multivariate and low-rank Gaussian, Student-t, MDN, Johnson SU, beta
regression, GEV away from ξ = 0, Barron, Tweedie, Poisson, discrete
Wasserstein, quantile/expectile losses; split conformal, CQR, UACQR, Monte
Carlo, multi-target, Mahalanobis-score, R2C and CTI coverage; `crps_gaussian`,
`crps_from_samples`, `interval_score`, pinball, KS, Gaussian Wasserstein,
ordinal and point metrics; `ppi_mean_ci`, PPI++, calibrated PPI, PPI OLS,
DML (0.951 coverage at n = 500), AIPW; ensemble variance decomposition,
`BatchEnsembleLinear`, `VariationalLinear`, `MCDropoutWrapper`.

`tests/metrics/test_reference_parity.py` now checks every metrics export
against scoringrules, properscoring, scipy, scikit-learn or a closed form,
and fails when a new export has neither a test nor a stated reason.

## Batch 4: algorithms and test-time adaptation

Tests: `tests/audit/test_audit_batch4.py`.

| ID | Sev | Export | Finding |
|:--|:--|:--|:--|
| ALG-001 | High | `iteratively_reweighted_least_squares` | weights compounded every iteration (ψ^n instead of ψ; 0.001 instead of 0.5) |
| ALG-002 | High | IRLS `variance_type="robust"` | MAD over the output axis: zero for `(N, 1)` targets |
| ALG-004 | High | `SIMEX` | float64 inputs with scalar `sigma_u` raised |
| ALG-005 | High | `SIMEX` | absolute 1e-6 jitter: not scale-equivariant |
| ALG-006 | High | `SIMEX.predict` | one noise draw shared by all models |
| ALG-007 | High | `RegressionCalibration.posterior` | `sigma_u=` override ignored |
| TT-002 | High | `WeightedConformalRegressionAdapter` | clamped instead of infinite intervals: coverage 0.83 vs ≥ 0.90 |
| TT-004 | High | `DelayedLabelResidualAdapter` | first batch centred on itself: inflation collapsed to 1e-5 |
| TT-005 | High | transport conformal threshold | one order statistic too high; never infinite |
| ALG-003 | Medium | IRLS | `[mean, log_sigma]` outputs raised with the default config |
| ALG-008 | Medium | `RegressionCalibration` | absolute eigenvalue floors |
| ALG-009 | Medium | `HeteroscedasticLaplaceRegressor` | plain linear head failed in `fit` |
| ALG-011 | Medium | `IVON` | step count not checkpointed |
| TT-003 | Medium | `DomainClassifierRatioEstimator` | missing n_s/n_t prior correction (E[w] = 0.1 under no shift) |
| TT-006 | Medium | `BayesianLinearHead` | RBF bandwidth subsample ignored `generator=` |
| TT-007 | Medium | `ShiftFactoredPredictiveTransport` | outputs cast to float32 |
| ALG-010 | Low | `HeteroscedasticLaplaceRegressor` | NaN std with `n_samples=1` |
| ALG-012 | Low | `NeighborhoodCovariancePseudoLabeler` | float32 cast |
| TT-001 | Low | label-shift Gaussian moments | float32 output |

## Batch 5: utilities, prediction containers, comparison, viz

Tests: `tests/audit/test_audit_batch5.py`.

| ID | Sev | Export | Finding |
|:--|:--|:--|:--|
| PRD-003 | High | `comparison.compute_point_metrics` | `[N, 1]` vs `[N]` broadcast to N×N (MSE 2.24 vs 0.009) |
| VIZ-002 | High | `plot_causal_uplift_qini` | Qini area always NaN |
| VIZ-004 | High | `plot_qq_plot`, `plot_residual_histogram` | `[N, 1]` vs `[N]` gave N² residuals |
| PRD-004 | Medium | `compute_point_metrics` | multi-output R² pooled (0.9996 for a constant model) |
| UTL-001 | Medium | `masked_mean`, `masked_sum` | NaN at masked positions leaked |
| UTL-003 | Medium | `calculate_gaussian_nll` | per-sample variance misapplied |
| UTL-004 | Medium | `parse_heteroscedastic_output` | 3-D tensors split on the wrong axis |
| UTL-005 | Medium | Box-Cox, Yeo-Johnson | float32 cancellation at small λ |
| UTL-010 | Medium | OpenML relaxed loader | numeric nominal columns became NaN |
| VIZ-001 | Medium | PIT histogram | bins not on [0, 1] |
| VIZ-003 | Medium | calibration curve | RMSCE averaged empty bins |
| VIZ-005 | Medium | `plot_performance_comparison` | highlighted the worst model for several metrics |
| PRD-002 | Low | `quantiles_to_density_grid` | crossing quantiles fixed with `cummax` instead of sorting |
| UTL-002 | Low | `ipw_weights`, `labels_to_levels` | float32 hard-coded |
| UTL-006 | Low | `normal_cdf` | underflow in the lower tail |
| UTL-007 | Low | validators | NaN accepted |
| UTL-008 | Low | `BSplineDensityBasis.bin_integrals` | returned the shared float64 cache |
| UTL-009 | Low | `docs/api/utils.md` | 31 documented names did not exist; public names missing |

**PRD-001 (resolved).** `quantiles_to_density_grid` dropped the mass below the
first and above the last quantile and renormalised, so the CDF at knot k was
(τ_k − τ_0)/(τ_K − τ_0) rather than τ_k. The tail mass is now spread uniformly
over the grid margins, so the CDF is exact at every knot; `test_time.transport`
tests are unchanged.

## Batch C2: post-audit additions

Audit of the code added after the original audit (`models`, `estimators`, the
RFF / GCV / LOO additions to `inference.orthogonal`, `create_flow_model(bound=)`,
and the transport selection changes), run 2026-10-07 at `7167361b`. **6 defects
found, 6 fixed.** Tests: `tests/audit/test_audit_c2.py`.

| ID | Sev | Export | Finding |
|:--|:--|:--|:--|
| C2-003 | Medium | `fit_tabular`, `TabularFit.predict` | `GaussianNLLLoss(log_variance=False)` got the `"location"` layout: the variance head was returned as `mu + sigma * var` instead of `var * sigma^2` (new `"variance"` layout) |
| C2-004 | Medium | `ConformalRegressor`, `CalibratedRegressor`, `calibrated_deep_ensemble` | NaN or infinite calibration targets accepted: NaN intervals (conformal) or a temperature fitted on the remaining rows (calibrated) |
| C2-006 | Medium | `fit_tabular`, `calibrated_deep_ensemble` | `standardize_target=False` forced layout `"none"`, so a Gaussian fit could not be wrapped: `calibrated_deep_ensemble(..., standardize_target=False)` raised "no Gaussian head" |
| C2-002 | Low | `TabularPreprocessor` | smooth clip overflowed for \|z\| > 1e154 (huge outliers mapped to 0, the median) and gave NaN for infinite inputs |
| C2-001 | Low | `median_heuristic_bandwidth` | `cdist` matmul expansion lost all precision for offset data (1e9 offset gave the 1.0 fallback); NaN input returned 1.0 silently |
| C2-005 | Low | `select_high_confidence(scores=)` | NaN scores sorted last and were selected as the most confident rows |

**Checked and found correct (C2).** Periodic embedding and the RealMLP smooth
clip (forward and inverse); GCV and exact LOO closed forms (hat-matrix leverage
with an unpenalised intercept) against brute-force refits for the full penalty
grid; random Fourier features against the RBF kernel (max error 0.023 at
D = 20000); coverage by simulation within binomial tolerance for split
(0.905), normalized (0.906), CQR (0.907), CV+ (0.912; 0.927 at n = 12),
Jackknife+ (0.941) and calibrated normalized conformal on a trained ensemble
(0.905), with `+inf` intervals at n_cal < 1/alpha; weighted split, normalized
and CQR coverage under covariate shift (0.91 to 0.92, 0.90); no mutation of
the user's base estimator or inputs, no global RNG mutation, reproducibility
by seed, train/eval mode restored by the wrappers; no leakage of calibration
rows into the base or of the early-stopping carve into the calibration carve;
`BoundedNSF` densities integrate to 1; transport `in_grid_mass` agrees with the
exact Gaussian mass and with the density, sample and quantile paths.

## Also fixed after the audit

The CPU-vs-device parity suite (`tests/test_device_parity.py`) found 18 metrics
returning float32 for float64 inputs, and functional metric wrappers keeping
their state on the CPU; `import torchregress.test_time` needed scikit-learn.
All fixed (see CHANGELOG).
