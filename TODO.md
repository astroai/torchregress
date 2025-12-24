# torchregress TODO List

This roadmap is prioritized by user impact for regression practitioners.
Each item includes candidate SOTA approaches as of today.

## P0 (Highest Impact)

### 1) Trustworthy uncertainty under shift
**Goal:** Calibrated, actionable uncertainty that remains useful when data drifts.
**SOTA candidates:**
- Conformal families for regression (CQR, ACI, Jackknife+/EnbPI) for coverage guarantees.
- Post-hoc calibration for regression: isotonic/quantile calibration, distributional calibration
  on PIT or CRPS, and temperature scaling for probabilistic heads.
- Epistemic methods with strong empirical performance: deep ensembles, SWAG, Laplace.
- Decision support: risk-coverage curves and abstain/reject policies tied to uncertainty.

### 2) Scalable multi-target regression with dependency structure
**Goal:** Accurate uncertainty when outputs are correlated at scale.
**SOTA candidates:**
- Low-rank + diagonal Gaussian heads with stable Cholesky parameterization.
- Structured covariance (Kronecker, block-diagonal, sparse precision) for large outputs.
- Copula-based post-hoc dependency modeling for non-Gaussian targets.
- Multi-output GP inspirations (LMC) as design references for loss/parameterization.

### 3) Measurement error + missingness as first-class
**Goal:** Robust learning with noisy features/labels and missing data.
**SOTA candidates:**
- Errors-in-variables (functional/structural) with heteroscedastic noise modeling.
- Total least squares / orthogonal regression variants for feature noise.
- SIMEX-inspired noise estimation and noise-level learning utilities.
- Masked losses + uncertainty-aware imputation (probabilistic heads).

### 4) Tail-risk and imbalance end-to-end
**Goal:** Better performance on rare/extreme targets, not just overall RMSE.
**SOTA candidates:**
- Quantile regression (pinball), expectiles, and CVaR-style losses.
- Density or difficulty reweighting (LDS, focal-style weighting for regression).
- Monotonic constraints for quantile crossing and interval consistency.
- Tail-centric evaluation (coverage, tail RMSE/MAE, extreme quantile calibration).

### 5) Actionable OOD/shift detection for regression
**Goal:** Move from OOD scores to decisions (alerting/abstention).
**SOTA candidates:**
- Energy, typicality, and Mahalanobis-style scores over learned representations.
- Density-ratio and ensemble disagreement for shift signals.
- Practical thresholding and alert policies tied to uncertainty reports.

### 6) Censored regression support
**Goal:** Native support for interval/right-censored targets.
**SOTA candidates:**
- Tobit-style losses, AFT losses, and interval-censored likelihoods.
- Survival-style baselines (Cox partial likelihood) as references.
- Censored quantile regression variants for robust intervals.

## P1 (High Impact Enablers)

### 7) Benchmark + recipe suite
**Goal:** Repeatable evaluation of robustness and uncertainty claims.
**SOTA candidates:**
- A fixed benchmark set (UCI + tabular OOD splits + synthetic stress tests).
- Standard metrics: NLL, CRPS, PICP, interval width, risk-coverage.
- Reproducible scripts for tail-risk and shift stress tests.

### 8) Documentation & examples refresh
**Goal:** Make advanced features approachable and correct.
**SOTA candidates:**
- Focused tutorials for uncertainty, robust regression, EIV, and multi-target.
- Minimal examples for each head type (Gaussian, low-rank, MDN, flows).

## P2 (Quality of Life + Ecosystem)

### 9) API extension ergonomics
**Goal:** Faster user iteration and fewer footguns.
**SOTA candidates:**
- Loss registration decorators and structured config-driven builders.
- Standardized input validation and actionable error messages.

### 10) Training integrations
**Goal:** Reduce boilerplate for real projects.
**SOTA candidates:**
- PyTorch Lightning templates for regression + uncertainty.
- TorchMetrics-aligned APIs for metrics consistency.

### 11) Performance & stability
**Goal:** Robustness at scale and production readiness.
**SOTA candidates:**
- Numerical stability tests (NaN/Inf, extreme values).
- GPU profiling for hot paths and memory scaling.

## Deferred / Revisit Later

### Conformal prediction stack choice
We may revisit conformal prediction later with a different dependency stack
or a native implementation. For now, conformal remains optional and external.
