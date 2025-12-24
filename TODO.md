# torchregress TODO List

This roadmap is prioritized by user impact for regression practitioners.
Each item lists representative baselines and promising research directions
(not a complete SOTA survey). References are provided for grounding.

## P0 (Highest Impact)

### 1) Trustworthy uncertainty under shift
**Goal:** Calibrated, actionable uncertainty that remains useful when data drifts.
**Representative approaches / recent directions:**
- Distribution-free coverage: CQR and Jackknife+ as practical baselines [R1, R2],
  with adaptive conformal variants emerging for nonstationary data [R3, R4, R5].
- Regression calibration: calibrated regression and PIT/CRPS-style calibration [R6],
  plus general neural calibration techniques as a starting point [R7].
- Epistemic baselines: deep ensembles, MC dropout, SWAG [R8, R9, R10].
- Decision support: risk-coverage curves and abstain/reject policies (tooling gap).

### 2) Scalable multi-target regression with dependency structure
**Goal:** Accurate uncertainty when outputs are correlated at scale.
**Representative approaches / recent directions:**
- Multi-output GP baselines and scaling ideas for correlated targets [R11, R12, R13].
- Low-rank + diagonal covariance heads with stable parameterization (practice-driven).
- Structured covariance (Kronecker, block-diagonal, sparse precision) for high-D outputs.
- Copula-based post-hoc dependency modeling for non-Gaussian targets (classical).

### 3) Measurement error + missingness as first-class
**Goal:** Robust learning with noisy features/labels and missing data.
**Representative approaches / recent directions:**
- Errors-in-variables (functional/structural) with heteroscedastic noise modeling.
- Total least squares / orthogonal regression variants for feature noise.
- SIMEX-style noise estimation (with recent regression variants) [R14].
- Masked losses + uncertainty-aware imputation (probabilistic heads).

### 4) Tail-risk and imbalance end-to-end
**Goal:** Better performance on rare/extreme targets, not just overall RMSE.
**Representative approaches / recent directions:**
- Quantile regression, expectiles, CVaR-style objectives (classic) [R15, R16, R17].
- Imbalanced regression baselines (LDS) and newer augmentation work [R18, R19].
- Monotonic constraints for quantile crossing and interval consistency.
- Tail-centric evaluation (coverage, tail RMSE/MAE, extreme quantile calibration).

### 5) Actionable OOD/shift detection for regression
**Goal:** Move from OOD scores to decisions (alerting/abstention).
**Representative approaches / recent directions:**
- Mahalanobis and typicality-style baselines [R20, R21].
- Energy-based OOD methods and recent refinements [R22, R23, R24].
- Density-ratio and ensemble disagreement for shift signals.
- Practical thresholding and alert policies tied to uncertainty reports (tooling gap).

### 6) Censored regression support
**Goal:** Native support for interval/right-censored targets.
**Representative approaches / recent directions:**
- Tobit/AFT-style losses and interval-censored likelihoods (classic).
- Censored regression with multi-output GP formulations [R25].
- Censored quantile regression variants for robust intervals (literature gap).

## P1 (High Impact Enablers)

### 7) Benchmark + recipe suite
**Goal:** Repeatable evaluation of robustness and uncertainty claims.
**Representative approaches / recent directions:**
- A fixed benchmark set (UCI + tabular OOD splits + synthetic stress tests).
- Standard metrics: NLL, CRPS, PICP, interval width, risk-coverage.
- Reproducible scripts for tail-risk and shift stress tests.

### 8) Documentation & examples refresh
**Goal:** Make advanced features approachable and correct.
**Representative approaches / recent directions:**
- Focused tutorials for uncertainty, robust regression, EIV, and multi-target.
- Minimal examples for each head type (Gaussian, low-rank, MDN, flows).

## P2 (Quality of Life + Ecosystem)

### 9) API extension ergonomics
**Goal:** Faster user iteration and fewer footguns.
**Representative approaches / recent directions:**
- Loss registration decorators and structured config-driven builders.
- Standardized input validation and actionable error messages.

### 10) Training integrations
**Goal:** Reduce boilerplate for real projects.
**Representative approaches / recent directions:**
- PyTorch Lightning templates for regression + uncertainty.
- TorchMetrics-aligned APIs for metrics consistency.

### 11) Performance & stability
**Goal:** Robustness at scale and production readiness.
**Representative approaches / recent directions:**
- Numerical stability tests (NaN/Inf, extreme values).
- GPU profiling for hot paths and memory scaling.

## Deferred / Revisit Later

### Conformal prediction stack choice
We may revisit conformal prediction later with a different dependency stack
or a native implementation. For now, conformal remains optional and external.

## References (selected, not exhaustive)

R1. Romano, Patterson, Candès (2019), "Conformalized Quantile Regression".
    arXiv:1905.03222
R2. Barber, Candès, Ramdas, Tibshirani (2019), "Predictive inference with the
    jackknife+". arXiv:1905.02928
R3. Podkopaev, Xu, Lee (2024), "Adaptive Conformal Inference by Betting".
    arXiv:2412.19318
R4. Blot, Angelopoulos, Jordan, Brunel (2024), "Automatically Adaptive Conformal
    Risk Control". arXiv:2406.17819
R5. Zaffran et al. (2022), "Adaptive Conformal Predictions for Time Series".
    arXiv:2202.07282
R6. Kuleshov, Fenner, Ermon (2018), "Accurate Uncertainties for Deep Learning
    Using Calibrated Regression". arXiv:1807.00263
R7. Guo et al. (2017), "On Calibration of Modern Neural Networks".
    arXiv:1706.04599
R8. Lakshminarayanan, Pritzel, Blundell (2016), "Simple and Scalable Predictive
    Uncertainty Estimation using Deep Ensembles". arXiv:1612.01474
R9. Gal, Ghahramani (2015), "Dropout as a Bayesian Approximation".
    arXiv:1506.02142
R10. Maddox et al. (2019), "A Simple Baseline for Bayesian Uncertainty in Deep
     Learning (SWAG)". arXiv:1902.02476
R11. Feinberg et al. (2017), "Large Linear Multi-output Gaussian Process Learning".
     arXiv:1705.10813
R12. Moreno-Munoz, Artes-Rodriguez, Alvarez (2018), "Heterogeneous Multi-output
     Gaussian Process Prediction". arXiv:1805.07633
R13. Joukov, Kulic (2020), "Fast Approximate Multi-output Gaussian Processes".
     arXiv:2008.09848
R14. Shi et al. (2019), "SIMEX Estimation in Parametric Modal Regression with
     Measurement Error". arXiv:1909.12331
R15. Koenker, Bassett (1978), "Regression Quantiles".
R16. Newey, Powell (1987), "Asymmetric least squares estimation and testing".
R17. Rockafellar, Uryasev (2000), "Optimization of Conditional Value-at-Risk".
R18. Yang et al. (2021), "Delving into Deep Imbalanced Regression".
     arXiv:2102.09554
R19. Stocksieker, Pommeret, Charpentier (2023), "Data Augmentation for Imbalanced
     Regression". arXiv:2302.09288
R20. Hendrycks, Gimpel (2016), "A Baseline for Detecting Misclassified and
     Out-of-Distribution Examples". arXiv:1610.02136
R21. Lee et al. (2018), "A Simple Unified Framework for Detecting OOD Samples
     and Adversarial Attacks". arXiv:1807.03888
R22. Liu et al. (2020), "Energy-based Out-of-distribution Detection".
     arXiv:2010.03759
R23. Wu et al. (2024), "Revisiting Energy-Based Model for OOD Detection".
     arXiv:2412.03058
R24. Hofmann et al. (2024), "Energy-based Hopfield Boosting for OOD Detection".
     arXiv:2405.08766
R25. Gammelli et al. (2020), "Generalized Multi-Output Gaussian Process Censored
     Regression". arXiv:2009.04822
