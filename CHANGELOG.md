# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
## [Unreleased]

Notes for **0.3.0** start here: bug fixes, performance of existing code, docs, and harness parity. No new methods.

### Fixed

- `ConformalLoss` no longer swallows unknown keyword arguments (including the
  removed `debias`). Docs and tests that still advertised the no-op are gone.
- Removed the thin `PredictiveBatch.from_density` / `from_samples` constructors
  (use the dataclass fields directly).
- Removed the identity factory `create_mdn_loss` (use `MixtureDensityLoss`).

## [0.3.0b1] - 2026-10-26

API cut. One public name per behavior.

### Removed

- Compatibility aliases: `MDNLoss` (use `MixtureDensityLoss`), `ExpectileCrossover`, `QuantileCrossover`, `AsymmetricLeastSquaresLoss`, `CORALLoss` (use `CumulativeLinkLoss`; `CORALHead` is unchanged), `JackknifePlus` (use `CVPlus`; leave-one-out is `K = n`), `SelfAgreementTrainer` (use `TeacherStudentTrainer`), `iteratively_reweighted_least_squares` (use `reweight_fixed_predictions` or `iteratively_reweighted_fit`), `RepresentationShiftCalibrator`, `SignificantSubspaceAligner`, `prediction_interval_coverage`, `MeanIntervalWidth`.
- Torchmetrics re-exports and the calibration-metric re-exports from `torchregress.metrics`. Calibration metrics are `torchregress.calibration` only.
- The ignored `debias` argument on `CQR`, `UACQR`, and `ConformalLoss`.
- `utils.quantile.quantile_loss` is `pinball_residual`. `fetch_openml_regression_with_sklearn_fallback` is `fetch_openml_regression`.
- Registry keys `"als"` and `"coral"`.

### Added
- True noise-adaptive Balanced MSE (`BalancedMSEBMC`, `balanced_mse_bmc_loss`;
  Ren et al., CVPR 2022) with fitted label prior and target-noise scale; this is
  distinct from the historical inverse-bin-frequency `BalancedMSELoss`.
- RankSim ranking-similarity regularizer (`RankSimLoss`, `rank_sim_regularizer`;
  Gong et al., ICML 2022) for representation-level imbalanced regression, with
  a blackbox-interpolation gradient and composition with task losses.
- Variational Imbalanced Regression (`VIRLoss`, `VIRRegressor`) for long-tailed
  regression with evidential (Normal-Inverse-Gamma) uncertainty: label-similarity
  latent smoothing, smoothed label-density reweighting, evidential regularisation,
  and `predict(return_std=True)`. Docs: `docs/methods/algorithms/vir.md`.
- Corrected the historical `BalancedMSELoss` documentation: it implements inverse
  bin-frequency weighting, not Ren et al.'s noise-adaptive Balanced MSE (BMC);
  the misleading citation was removed and the distinction is documented.
- Contrastive likelihood-ratio / neural-ratio-estimation tooling (2026-10-17):
  - `losses.NeuralRatioLoss` (+ `build_ratio_classifier`, `neural_ratio_logits`,
    `neural_ratio_loss`): classifier-based neural ratio estimation with balanced binary
    (NRE-A) and multiclass (NRE-B/InfoNCE-style) criteria, paired or mismatched negatives,
    and flow-loss-style shape/ambiguity validation. The logit estimates
    `log p(x|theta) - log p(x)`, so grid-argmax is the MLE and the training prior cancels
    under a prior-sampled reference. Docs: `docs/losses/nre.md`.
  - `inference.posterior_log_weights`, `effective_sample_size`, `grid_log_posterior`,
    `maximum_likelihood_estimate`: finite-bank likelihood-ratio/posterior reweighting,
    Kish ESS diagnostics, and MLE/MAP point estimates with documented prior/proposal
    corrections. Docs: `docs/api/inference.md`, `docs/methods/inference.md`.
  - `metrics.simulation_based_calibration_ranks`, `sbc_rank_histogram`,
    `sbc_uniformity_pvalue`, `tarp_coverage`: simulation-based calibration and TARP
    posterior-coverage diagnostics. Docs: `docs/metrics/calibration.md`,
    `docs/api/metrics.md`.
  - `ContrastiveFlowLoss`: documented `temperature=1, margin=0` likelihood-ratio regime with
    a `likelihood_ratio_mode` property, chunked negative evaluation, and target-shape
    validation; the flow guide now frames it correctly as InfoNCE/multiclass NRE and
    distinguishes it from the Elsharkawy-Kahn contrastive-normalizing-flow method.
  - The contrastive-flow comparison example now also trains classifier NRE and an NPE-style
    posterior flow, and reports SBC/TARP diagnostics next to point accuracy and runtime.
- **Post-freeze method additions (user-requested audit, 2026-10-17; targeted at 0.4.0):**
  - `algorithms.ForwardInversionRegressor` (+ `build_decoder_mlp`): generative `y -> x`
    surrogate (statistical inverse regression / auto-decoder construction) trained with
    known per-observable errors and inverted per object by maximum likelihood (damped
    Gauss–Newton or L-BFGS) with Fisher covariance `(J^T Sigma_x^{-1} J)^{-1}`; removes
    input attenuation and inverse-regression prior bias. Training supports the full
    heteroscedastic Gaussian likelihood (default), a robust Student-t likelihood
    (`likelihood="student_t"`) and fixed-weight heteroskedastic least squares
    (`likelihood="wls"`). `predict(..., return_info=True)` returns per-object stationarity
    and Fisher-condition diagnostics for the local, non-convex test-time inversion, and the
    guide documents when a discriminative model is the better choice. Docs:
    `docs/methods/algorithms/forward_inversion.md`.
  - `losses.SplitMeasurementLoss` / `split_measurement_loss` and
    `losses.CorrectedScoreEIVLoss` / `corrected_score_eiv_loss`: paired-exposure
    cross-product loss (no noise model required) and the Stefanski–Nakamura corrected
    score (subtracts the measurement-error curvature instead of adding a TLS gradient
    penalty). Docs: `docs/losses/split_eiv.md`.
  - `losses.PrimeLoss` and `losses.ConR` (+ `conr_loss`, `uniform_proxy_targets`):
    representation-level imbalanced-regression losses (PRIME proxies with uniformly
    spaced targets; ConR contrastive collapse penalty). Feature Distribution Smoothing
    already existed (`losses.FeatureDistributionSmoother`) and is cross-linked, not
    duplicated. Docs: `docs/losses/prime_conr.md`.
  - `algorithms.ContinuousGroupDRO` and `algorithms.GroupDROTrainer`: exponentiated-gradient
    worst-target-regime training over continuous target bins (equalizes per-regime mean
    loss, unlike CVaR's worst individual samples). Docs: `docs/methods/algorithms/group_dro.md`.
  - `calibration.VarianceRestoringCalibrator`: marginal de-shrinkage by moment matching or
    1-D optimal transport, with `describe()` diagnostics; documented as marginal (not
    conditional) calibration. Docs: `docs/methods/calibration.md`.
  - `method_catalog` metadata, `docs/api/*`, loss-guide indexes and site navigation updated
    for all of the above.
- `MixtureDensityLoss.sample` supports `covariance_type="full"` (draws `mu + L eps` from the selected component's covariance Cholesky factor); previously only diagonal mixtures could be sampled.
- `inference.random_fourier_features` and `inference.median_heuristic_bandwidth`: random Fourier feature nuisance basis for `orthogonal_partially_linear` with a median-heuristic bandwidth (median pairwise distance of the standardised `z` on a seeded subsample), per-covariate polynomial terms (`polynomial_degree`, default 3: across five confounding designs it keeps 95% coverage within 0.90-0.95, where degree 1 drops to 0.78 on cubic confounding) and a seeded draw. `orthogonal_partially_linear(ridge="gcv" | "loo")` picks the nuisance ridge penalty per nuisance and training fold by generalized cross-validation or exact leave-one-out error (closed form from one SVD, unpenalised intercept); a float `ridge` (default `1e-6`) behaves exactly as before. Over 200 replications at n = 500 the old harness recipe (unit-bandwidth RFF, `ridge=1e-2`, no linear part) gave Coverage95 0.785 / bias +0.008 on CCDDHNR-2018 and 0.820 / +0.060 on the cubic design; the new defaults give 0.955 / +0.004 and 0.900 / +0.016, and `polynomial_degree=3` gives 0.945 / -0.002 on the cubic design (plan item B5).
- `losses.create_flow_model(bound=...)` sets the NSF spline tail bound (zuko fixes it at 5), and `losses.recommended_tail_bound(y_standardised)` derives it from the training target (data range with 25% head-room, clipped to [5, 25]). `NormalizingFlowLoss`'s out-of-range warning reports the actual bound. New "Training recipe for NLL and calibration" section in `docs/losses/nflows.md` (validation early stopping, derived bound, 16 bins): test NLL 1.272 -> 1.201 (skewed), 1.935 -> 1.852 (Student-t 3 df), 6.144 -> 5.417 (`load_diabetes`) with equal or better 90% interval scores, versus the fixed 60-epoch recipe of the harness `pzflow_parity` row (plan item B4). Library defaults (`bins=8`, bound 5) are unchanged.
- `test_time.select_high_confidence(scores=...)`: optional per-row ranking score (default: maximum probability).
- `torchregress.models`: `TabularMLP` (periodic numeric embeddings, SiLU blocks, dropout), `TabularPreprocessor` (median/IQR scaling, smooth clipping, NaN imputation with missing indicators), `fit_tabular` / `fit_tabular_ensemble` (AdamW, one-cycle schedule, early stopping with restore-best-weights, target standardisation) and `TabularFit`. Works with any loss, including `GaussianNLLLoss` (`2 * D` outputs) and `MultiQuantileLoss`. Docs: `docs/methods/tabular_mlp.md`, `docs/api/models.md`. `models` is a new lazy top-level submodule (plan item B2).
- `torchregress.estimators` (experimental, may change in 0.4): model-agnostic UQ wrappers. `ConformalRegressor(base, method="split" | "normalized" | "cqr" | "cv+" | "jackknife+", alpha, n_folds, prefit)` with `fit` / `predict` / `predict_interval(X, alpha, test_weights)` puts `SplitConformal`, `CQR` or `CVPlus` intervals on any scikit-learn or LightGBM model, a quantile pair/triple, a torch module with a training callable, or a fitted `TabularFit`; NumPy in gives NumPy out and tensors give tensors. `CalibratedRegressor(base, method="vts" | "isotonic+vts")` fits `VarianceTemperatureScaler` (and `IsotonicMeanCalibrator`) on a held-out split of any Gaussian-output base and offers `predict_dist` / `predict_interval`, optionally with conformal widths. Tests check parity with direct use of the conformal classes and with MAPIE `SplitConformalRegressor` / `ConformalizedQuantileRegressor` (prefit) to 1e-6. Docs: `docs/methods/estimators.md`, `docs/api/estimators.md`. `estimators` is a new lazy top-level submodule (plan item B1).
- `estimators.calibrated_deep_ensemble(X, y, n_members, val_fraction, loss, alpha, conformal, seed, **fit_kwargs)`: Gaussian `TabularMLP` ensemble via `fit_tabular_ensemble` with mixture moments (mean member variance plus variance of member means), a variance temperature fitted on a held-out carve and optional normalized split-conformal intervals; returns a `CalibratedRegressor` (plan item B3). Experimental, may change in 0.4.

- Examples for the 0.3.0 APIs: `tabular_mlp_tutorial.py` (`TabularMLP`, `fit_tabular`, `TabularPreprocessor` with NaNs), `calibrated_regressor_comparison.py` (`CalibratedRegressor`, `calibrated_deep_ensemble`, `ConformalRegressor`), `nsf_tail_bound_comparison.py` (`recommended_tail_bound` vs the fixed NSF bound on Student-t noise) and `orthogonal_inference_rff_demo.py` (random Fourier nuisances, median bandwidth, `ridge="gcv"`), each under 15 s on 2 CPU cores and smoke-tested.

### Changed

- Declared public (added to `__all__`; they were already importable):
  `CQR`, `CTI`, `MultivariateScoreConformal`,
  `NonExchangeableConformalRegressor` and `BaseEIVLoss` in
  `torchregress.losses`, and `SelfAgreementTrainer` (alias of
  `TeacherStudentTrainer`) in `torchregress.semi_supervised`. Listed in
  `docs/api/losses.md`, `docs/api/conformal.md` and
  `docs/api/semi_supervised.md`.
- `MixtureDensityLoss(covariance_type="full")` now reads its lower-triangular
  factor as the Cholesky factor of the **precision** (`full_parameterization="precision"`,
  new default, as in sbi's MDN): the log-density is linear in the factor, with no
  triangular solve. The covariance form (`full_parameterization="covariance"`, the
  previous behaviour) did not fit multi-target data: on the 16-target scm20d
  benchmark, train / test NLL 0.42 / 3.32 vs -3.45 / 0.31 with the precision form
  (sbi's MDN: 0.04). The output layout is unchanged. `predict_mean_std` and
  `predict_interval` now support full covariance (per-dimension marginals).
  The precision form is not better everywhere: in harness run #6
  (`density_parity`, test NLL) it improves mt_scm20d 3.11 → 0.33 and mt_jura
  2.53 → 1.95 but is worse on the 2-target energy_efficiency (-1.32 → -0.76)
  and mt_enb (-1.18 → -0.85). `docs/losses/mdn.md` now explains both
  parameterisations, recommends comparing `full_parameterization="covariance"`
  on validation NLL for low-dimensional targets, and warns that the extracted
  precision factor `T` is not a covariance square root (use `.sample()`).

- `quantiles_to_density_grid` (and `PredictiveBatch.with_density` for quantile
  batches) keeps the tail mass: `tau_0` below the first quantile and
  `1 - tau_K` above the last are spread uniformly over the grid margins
  instead of being dropped and renormalised. The grid CDF now equals `tau_k`
  at every knot; for levels 0.05/0.95 a 90% interval read off the grid covers
  0.900 of Gaussian draws instead of 0.854 (PRD-001; supersedes TR-COR-02).
- `docs/losses/sls.md`: new section measuring the width of SLS bounding-box intervals against `GaussianNLL` / `MultivariateGaussian` on the three harness multivariate designs (plan item B7). The 1.2-1.3x width gap at nominal level shrinks to 1.04-1.11x at equal joint coverage; raw `SLSLoss` thresholds cover only 0.45-0.56 of held-out targets (0.91 in-sample), so the coverage guarantee needs `SLSConformal`. No library default changed: warmup, window, flow size and boundary-sample count do not reduce width at equal coverage on all designs.
- 28 plumbing helpers are no longer re-exported from `torchregress.utils` (and
  dropped from its `__all__`); they remain importable from their submodules, for
  example `from torchregress.utils.tensor_ops import convert_to_tensor`.
  `distributions`: `normal_cdf`. `gaussian_output`: `parse_heteroscedastic_output`,
  `low_rank_output_dim`, `split_low_rank_gaussian_output`. `numpy_stats`:
  `subsample_rows`, `winsorize`. `ordinal`: `labels_to_levels`,
  `normalize_class_probs`, `class_probs_to_levels`. `pytorch_compat`: `get_device`.
  `quantile`: `quantile_loss`, `multi_quantile_loss`. `tensor_ops`: `apply_mask`,
  `convert_to_tensor`, `ensure_batch_dim`, `prepare_cross_covariance`,
  `prepare_model_input_for_gradients`, `compute_model_gradients`,
  `calculate_gaussian_nll`, `calculate_propagated_variance`. `validation`:
  `validate_reduction`, `validate_range`, `validate_quantile`, `validate_weights`,
  `validate_metric_inputs`, `validate_sample_weight`, `check_tensor`. `security`:
  `validate_url`. `parse_heteroscedastic_output`, `low_rank_output_dim` and
  `split_low_rank_gaussian_output` stay public in `torchregress.ensemble` /
  `torchregress.losses`.
- Dependency floors raised to the newest releases resolvable from conda-forge:
  `torch>=2.13`, `numpy>=2.5`, `scipy>=1.18`, `torchmetrics>=1.9`; extras
  `matplotlib>=3.11`, `zuko>=1.6`, `scikit-learn>=1.9`, `pandas>=3.0`,
  `polars>=1.43`, `pyarrow>=25.0`, `pytest>=9.1`. Pre-commit hooks: ruff
  v0.16.10, pre-commit-hooks v6.0.0. Tested on torch 2.14.1 / numpy 2.5.3 /
  scipy 1.18.1 (PyPI) and the pixi lock (conda-forge torch 2.13.0).

### Removed
- The `torchregress.metrics.calibration` shim module. Its 7 names
  (`ExpectedCalibrationError`, `MarginalCalibrationError`, `bias`,
  `calibration_metrics_report`, `calibration_score`,
  `expected_calibration_error`, `marginal_calibration_error`) stay exported
  from both `torchregress.metrics` and `torchregress.calibration`: replace
  `from torchregress.metrics.calibration import X` with
  `from torchregress.metrics import X` or `from torchregress.calibration import X`
  (the implementation lives in `torchregress.calibration.metrics`).

### Fixed
- `InverseGaussianLoss` and `GammaLoss` (and `TweedieLoss` with `p >= 2`) now
  raise `ValueError` ("... requires strictly positive targets") when any
  unmasked target is `<= 0`. Both distributions have support `y > 0` and their
  deviances are undefined at `y = 0`; the `eps` guards silently turned a zero
  target into a large finite penalty (about 5e7 per entry for the Inverse
  Gaussian, about 17 for the Gamma at `mu = 1`) that dominated training.
  Masked-out entries are not checked. Targets are not shifted or clipped.
- `LocalConformal`, `LocalConformalMAD` and `DensityConformal` built dense
  (n_test x n_cal) kernel matrices with several temporaries alive at once, so
  memory grew with the product of the two sizes: at ~2x10^4 x 1.5x10^4 rows the
  harness `conformal/zurich_delays` job was killed by a 32 GB limit. Kernel
  weights and the weighted-quantile search now run in test-row blocks
  (O(n_cal) memory per block); results are identical.
- `CVPlus` / `JackknifePlus` materialised and fully sorted two
  (n_train x n_test) candidate tensors (CV+ uses every training row); the
  order statistics now come from `torch.kthvalue` over test-row blocks, with
  identical endpoints.
- `metrics.ood` kernel density scores and `NeighborhoodCovariancePseudoLabeler`
  (which built an (n_query, n_ref, D) difference tensor) now work in query-row
  blocks; results are identical.
- `algorithms.WarmupMCTrainer` iterated a `DataLoader` over a `TensorDataset`
  (one Python `__getitem__` per row per epoch); it now shuffles and slices on
  the device, as `LatentNN` does.
- `algorithms.LatentNN` scaled quadratically with the number of rows: the per-row
  latent inputs were one dense `nn.Parameter` under `torch.optim.Adam`, so every
  mini-batch step updated all N x D values and moments (O(N^2 D / B) per epoch;
  8x the rows took 36x longer, and the harness `folktables_eiv` job with 1.2M rows
  ran for 2.5 days without finishing). Latents now use a lazy, row-sparse Adam
  (only the batch rows' values and moments move, per-row bias correction), batches
  are drawn on the device instead of through a `DataLoader`, and Monte Carlo
  `predict(sigma_x=...)` runs in row chunks. Training is linear in N; full-batch
  training is unchanged (identical to dense Adam). Mini-batch results change:
  rows outside the batch no longer drift on stale momentum.
- `fit_tabular` / `TabularFit.predict`: `GaussianNLLLoss(log_variance=False)` heads are mapped back to the original scale with a new `"variance"` layout (`var * sigma^2`) instead of being treated as a location output; `standardize_target=False` keeps the Gaussian layout, so `calibrated_deep_ensemble(..., standardize_target=False)` works (audit C2-003, C2-006).
- `ConformalRegressor`, `CalibratedRegressor` and `calibrated_deep_ensemble` raise `ValueError` for NaN or infinite targets instead of returning NaN intervals or a temperature fitted on a subset (audit C2-004).
- `TabularPreprocessor`: the smooth clip no longer overflows for huge values (mapped to 0 above 1e154) or turns infinite inputs into NaN (audit C2-002).
- `inference.median_heuristic_bandwidth` is exact for data far from the origin and rejects non-finite input; `test_time.select_high_confidence(scores=)` ranks NaN scores lowest (audit C2-001, C2-005).
- `test_time.ShiftFactoredPredictiveTransport`: the confident-row selection (`top_fraction`, default 0.5) now ranks rows on the unrenormalised in-grid peak probability (maximum bin probability times the predictive mass inside the support grid) instead of the renormalised maximum. Renormalisation inflated the peak of predictives cut by the grid edge, so on fine grids the selection favoured them and the EM prior mean overshot a label shift by up to 0.32 |shift| at 256 bins (0.14 at 64); it is now below 0.07 |shift| at 16, 64 and 256 bins for shifts of +/-0.8. Selected rows, and therefore `target_prior` and the adapted predictive, change for a given input (plan item B6).
- `test_time.ShiftFactoredPredictiveTransport` with a Gaussian predictive: the prior correction is applied to the native mean / std as a moment change (transported minus untransported binned moments, weighted by the row's in-grid predictive mass) instead of returning the moments of the transported grid. Those added the discretisation and the truncation at the grid edge whenever `transport_applied` was true, so a near-identity prior moved the predictive (up to 3.7 sigma for edge rows) and target predictives beyond the source label range collapsed onto the edge bin: on a 2.5 SD label shift the target NLL went from 0.19 (untransported) to 7-4300 depending on `n_support`, now 0.24-0.26 (0.19-0.24 without uncertainty inflation); on a 0.8 SD shift without inflation 0.029 -> 0.046 before, 0.019 now (16 bins). New metadata `prior_mass_outside_source_range` and a `UserWarning` when the raw EM prior puts more than 0.2 of its mass on the support bins outside the source label range (label-shift support violation or unreliable EM). The docs page now warns that the stabilised prior's shrink weight scales as 1/`n_support` (2% of the EM correction at the default 256 bins on a calibrated Gaussian predictive) and that uncertainty inflation widens the predictive under label shift (harness finding 8 follow-up).
- `inference.orthogonal_partially_linear`: both nuisance regressions (`E[x|z]`, `E[y|z]`) now share one cross-fitting split. Independent splits biased `theta` (-0.026, about 7.7 standard errors, over 200 replications of the DoubleML CCDDHNR-2018 design at n = 500); found by the harness `orthogonal_inference` suite against DoubleML on the same nuisance basis. Point estimates for a given `seed` change.
- **0.3.0 release audit: 61 defects fixed**, each with a regression test in
  `tests/audit/` (full list with severities: `docs/reports/audit_0.3.0.md`).
  Results that change for existing code:
  - *Conformal prediction.* Thresholds are `+inf` (infinite intervals) when
    `ceil((n+1)(1-alpha)) > n`, per Mondrian group, or when the weighted mass
    cannot reach `1-alpha`; they used to fall back to the largest score and
    under-cover at small `n`. Weighted conformal follows Tibshirani et al.
    (2019): only relative weights matter, the test point defaults to the mean
    calibration weight, and a new `test_weights=` argument gives per-point
    thresholds. CV+/Jackknife+ use the lower rank `floor(alpha (n+1))`.
    `DensityConformal` scores use the density at `y_pred` in both phases.
    Negative Mondrian group ids work; R2C, CTI, DCP and `SLSConformal` honour
    `groups=`. `SemiConformalCalibrator` computes in float64.
  - *Losses.* `EvidentialRegressionLoss` uses the NIG predictive
    (`Omega = 2 beta (1 + nu)`, Amini et al. 2020, Eq. 8). Skew-t NLL values
    change (incomplete-beta bug). Skew-normal, censored/AFT, zero-inflated
    Poisson and GEV are exact in the tails instead of saturating.
    `BetaNLLLoss` and `CVaRLoss` handle 1-D `[B]` inputs. Weighted `'mean'`
    is normalised by the sum of weights in the flow, multi-expectile and
    imbalanced losses (FocalR, BalancedMSE, BinReweightedMSE,
    DensityWeighted, LDS, PropensityWeighted), and `BaseLoss` accepts `[B, D]`
    masks with `[B]` weights. `ExpectileCrossoverLoss` raises on
    non-ascending levels. EIV losses propagate gradients through the
    Jacobian; latent-marginalisation and input-noise losses now marginalise;
    `SLSLoss.forward` no longer mutates state (new `evaluate_frontier`).
  - *Metrics.* `distribution_metrics_report["crps"]` is exact (was 3-4% low).
    Quantile PIT randomises out-of-range targets (seedable `generator=`).
    MedAE/MAD/NMAD use the true median and per-output averaging (sklearn);
    trimmed MSE matches `scipy.stats.trim_mean`. `VarioScore.higher_is_better`
    is `True`. Absolute variance floors and covariance jitter became
    `min_variance=` / relative `jitter=` arguments.
  - *Calibration.* `VarianceTemperatureScaler` has no hidden `[0.05, 20]`
    clamp (`temperature_bounds=` restores it) and returns the closed-form MLE
    by default. `IsotonicMeanCalibrator` is true isotonic regression.
  - *Inference and causal.* `ppi_quantile_ci` inverts the rectified CDF
    (Angelopoulos et al. 2023); the old estimator was inconsistent. PPI and
    DR estimators keep float64; the PPI bootstrap no longer allocates an
    `[n_boot, N]` matrix. DR `fold_bootstrap` standard errors are correct
    (were 0.56x at two folds); trimming every unit raises.
  - *Ensembles.* Predicted variances use the `GaussianNLLLoss` training range
    `[1e-6, e^30]` (was `[e^-8, e^6]`); `predict` runs in eval mode and
    restores the previous modes; `base_seed` leaves the global RNG alone.
    `bars_to_density_grid` puts no mass outside the bin edges.
  - *Semi-supervised.* The documented trust-weight options (`tau`,
    `weight_power`, `hard_weight_threshold`, `batch_relative_mode`,
    `batch_trust_top_k`) take effect; `TeacherStudentTrainer(tau=)` defaults
    to 0.2, the value the code always used. Unknown options raise.
- `test_time.ShiftFactoredPredictiveTransport.ppi_target_ci` keeps float64 inputs.
- **Audit batches 4–5** (`algorithms`, `test_time`, `utils`, `prediction`,
  `comparison`, `viz`): 37 more defects fixed (one documented, decision
  pending), tests in `tests/audit/test_audit_batch4.py`
  and `test_audit_batch5.py`.
  - *IRLS* weights are `w(r/sigma)` times the initial precision; they used to
    compound over iterations (`w^n_iter`). `variance_type="robust"` takes the
    MAD over samples (it was zero for `(N, 1)` targets). Concatenated
    `[mean, log_sigma]` outputs work with the default config.
  - *SIMEX* accepts float64, is scale-equivariant (no absolute jitter), and
    `predict` averages independent remeasurements. *Regression calibration*
    honours `posterior(sigma_u=)` and uses relative eigenvalue floors.
  - *Weighted / OT conformal and transport conformal* return infinite
    intervals when the finite-sample guarantee needs them, and the transport
    threshold is the exact order statistic (it was one too high).
  - `DomainClassifierRatioEstimator` corrects for unequal pool sizes;
    `DelayedLabelResidualAdapter` no longer deflates the first batch;
    `IVON.state_dict()` stores the step count; `HeteroscedasticLaplaceRegressor`
    accepts a plain linear head and `n_samples=1`; `BayesianLinearHead`
    honours `generator=`; transport, label-shift and pseudo-label helpers keep
    float64.
  - `comparison.compute_point_metrics` no longer broadcasts `[N, 1]` against
    `[N]` (N×N errors) and averages R² per output (scikit-learn).
  - `masked_mean`/`masked_sum` ignore NaN at masked positions; Box-Cox and
    Yeo-Johnson are accurate for small lambda; `normal_cdf` is accurate in the
    lower tail; validators reject NaN; `quantiles_to_density_grid` sorts
    crossing quantiles (its truncation to `[q_0, q_K]` is now documented);
    numeric-coded nominal OpenML columns load.
  - Viz: residual histogram and QQ plot handle `[N, 1]` predictions, the PIT
    histogram spans [0, 1], the Qini area is finite, RMSCE ignores empty bins,
    `plot_performance_comparison` knows more lower-is-better metrics, and
    plots that create their own figure close it.
- Found by the harness after the audit (tests in
  `tests/audit/test_audit_postharness.py`):
  - `SkewTLoss` / `skew_t_nll`: the gradient with respect to the skewness was
    exactly zero at `alpha = 0`, so heads initialised there never learned
    skew. The Student-t CDF term is now smooth at 0 and log-space in the
    tails; values are unchanged (match scipy).
  - `ShiftFactoredPredictiveTransport`: the source prior is the mean source
    predictive distribution on the support grid (it was an eps-clamped label
    histogram), and support-grid EM stops on a log-likelihood plateau. EM no
    longer collapses onto empty margin bins (prior TV ~1 with no shift) and
    prior transport is applied under label shift. `estimate_target_prior_em`
    gives classes with no source mass a target prior of 0; new
    `LabelShiftEMConfig.loglik_tol` (default off).
  - Docs: EM label-shift estimation needs calibrated source posteriors (BBSE
    does not).
- Metrics return float64 for float64 inputs (18 metrics used to return
  float32), and functional metric wrappers move their internal torchmetrics
  state to the input device. Calibration-error histograms no longer use
  `torch.histogram`, which has no CUDA kernel.
- `import torchregress.test_time` works without scikit-learn (only the default
  classifier of `WeightedConformalRegressionAdapter` needs it).
- Examples: every script runs against the current API; external comparisons
  use MAPIE 1.x, torchcp 1.2 and scikit-learn's `TweedieRegressor`; the
  Tweedie examples no longer apply `exp` twice.
- `CORALLoss` no longer overwrites the `CumulativeLinkLoss` docstring: it is
  now a thin subclass of `CumulativeLinkLoss` with its own docstring (same
  computation, same `"coral"` registry key).
- `torchregress.method_catalog` declares an explicit `__all__` (its 4
  dataclasses, `CapabilityValue` and 5 functions), so `import *` no longer
  re-exports `Any`, `Dict`, `dataclass` and the other typing imports.
- Docs: removed names that do not exist (`PackedEnsembleRegressor`,
  `MCDropoutModel`, `BayesianModelAveraging`, `mse`, `mean_absolute_percentage_error`,
  `conformal_loss`, `create_eiv_loss`, ...) and corrected calls the functions
  do not accept (`mask=`/`weights=` on point metrics, which take `sample_weight`;
  `AsymmetricLeastSquaresLoss(tau=)`, `create_grid_figure(n_cols=)`,
  `plot_pit_histogram(bins=)`, `prediction_interval_coverage(confidence=)`).
  `plot_target_density_error_overlap` is now shown as `(y_true, y_pred)`.
- `highest_posterior_density_coverage`: the docstring and the metrics guide now
  state that `alpha` is the nominal HPD mass, not the miscoverage (no
  behaviour change).

### Packaging and CI
- SPDX licence metadata (`setuptools>=77`), project URLs, typed classifier;
  the sdist ships the test suite and CHANGELOG.
- CI `package` job and release `install-smoke` matrix (Linux/macOS, Python
  3.12–3.14): build, check contents (`scripts/release/check_dist.py`),
  `twine check --strict`, install with core dependencies only and import every
  submodule (`scripts/release/smoke_install.py`).
- `docs.yml` publishes the documentation to GitHub Pages.
- Local CUDA testing: `cuda` marker, CPU-vs-device parity suite for 77 losses
  and 94 metrics (`tests/test_device_parity.py`), `pixi run test-cuda`
  writing `reports/cuda/<date>_<sha>.txt`.

### Tests
- `tests/metrics/test_reference_parity.py` checks all 97 `metrics` exports
  against scoringrules, properscoring, scipy, scikit-learn or closed forms
  (14 have no reference and say why); `tests/metrics/test_scoring_properties.py`
  adds Hypothesis invariants (CRPS non-negativity and equivariance,
  propriety, pinball monotonicity). `scoringrules`, `properscoring` and
  `hypothesis` join the `test` extra.

### Added
- `losses.FaithfulGaussianLoss`: `mean_weight` accepts a per-output vector (length `D`, validated finite and non-negative), and a new `mean_loss="huber"` option (with `huber_delta`) bounds the pull of outlying targets on the mean. Defaults unchanged. `mean_weight` is now a registered buffer.
- `utils.BSplineDensityBasis`: unit-integral (M-spline) B-spline basis for simplex-parameterised 1D densities — exact bin integrals, cumulative moments and `E|U - x|` via piecewise Gauss–Legendre.
- `losses.DiscreteWasserstein1Loss` / `discrete_wasserstein1`: 1-Wasserstein between mass vectors on a shared 1D grid (mass or point targets).
- `losses.RankNContrastLoss` / `rank_n_contrast_loss`: Rank-N-Contrast (Zha et al. 2023) with tie-aware rank denominators, mask/weights support.

## [0.2.0] - 2026-08-27

### Fixed
- `BetaNLLLoss` preserves `[B,D]` elementwise before `_reduce` so partial masks `[[True,False]]` no longer discard whole rows (NEW-HIGH-02)
- `PoissonLikelihoodRatioLoss` clamps `exp(y_pred.clamp(max=30))` to avoid `inf` for `y_pred~100` (NEW-HIGH-04)
- `GEV` `gev_nll_elementwise` masks `1/xi` and `pow(-1/xi)` via `use_gev` so `xi→0` does not overflow before `where` (NEW-HIGH-03)
- `Tweedie p=1` Poisson term uses `ratio.clamp(min=eps)` not `log(+eps)` bias (NEW-MED-03)
- `prediction.quantiles_to_density_grid` forces `float32` levels for int quantiles, `n_support>=2` and `isfinite` guards (NEW-LOW-01/05)
- `losses/families.py` adds `unconstrained_inputs` flag (default `True`) to avoid `softplus(softplus(x))` (NEW-MED-01, pattern shown for SkewNormal/SkewT)
- `losses/conformal.py` unifies `_weighted_conformal_threshold` → `_weighted_quantile` augmented `+w_{n+1}=1` (NEW-HIGH-01) and CV+/Jackknife+ docs correct `>=1-2alpha` (NEW-HIGH-05)
- `ensemble/swag.py` skips BatchNorm `running_*` buffers in posterior (NEW-HIGH-06)
- `algorithms/simex.py` validates `sigma_u` PSD via cholesky, warns if `|w|>5` (NEW-HIGH-07)
- `algorithms/tictac.py` clamps `log_k` to `[-6,6]` and warns if Hessian `>50M` elems (~200MB) (NEW-HIGH-08)
- `algorithms/adaptive_prior_vi.py` `VIDSRegressor` KL and NLL both `mean()` so KL not dominated by `P` (NEW-HIGH-09)
- `losses/__init__.py` adds explicit `__all__` (139 symbols) for `audit_api_coverage.py` soundness (TR-API-01 latent)

### Changed
- `BetaNLLLoss` numerics: previously `sum(dim=-1)` then `mean` over `B`; now `mean` over `B*D` elementwise (breaking for `D>1`, semver MINOR, documented)
- `README` badge `python 3.12 | 3.13 | 3.14 | 3.15` now matches `pyproject.toml:14 requires-python <3.16` (NEW-LOW-09)

### Added
- `CHANGELOG.md` (Keep a Changelog) and release-script `prepare_release.sh` now enforces bump

## [0.1.0] - 2026-08-26

- Initial PyPI release candidate (unreleased). Pre-1.0 blockers TR-COR-01…09 (all fixed in `6ce4b9e`).
- Features: 19 loss families, `PredictiveBatch`, `torchmetrics` metrics, conformal (split/CQR/CTI/weighted/Mondrian/CV+), `SemiConformalCalibrator`, ensembles (DeepEnsemble/BatchEnsemble/SWAG/Laplace/BNN/MC-Dropout), algorithms (IRLS/SIMEX/RC/LatentNN/TICTAC/VIDS), causal `dr_*`, PPI, test-time (BLR/OT/COSA/subspace/transport).

[Unreleased]: https://github.com/astroai/torchregress/compare/v0.3.0b1...HEAD
[0.3.0b1]: https://github.com/astroai/torchregress/compare/v0.2.0...v0.3.0b1
[0.2.0]: https://github.com/astroai/torchregress/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/astroai/torchregress/releases/tag/v0.1.0
