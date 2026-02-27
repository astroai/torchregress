# torchregress TODO

Prioritized by user impact for regression practitioners.
Items list representative baselines and research directions (not exhaustive).

---

## P0 — Core Gaps

### 1. Ordinal regression
**Goal:** Native losses and heads for ordered-but-discrete targets (grades, ratings, severity).
- Cumulative link models (proportional odds) [R1]
- CORAL: consistent rank logits with binary classifiers [R2]
- Ordinal-consistent losses (earth-mover / unimodal penalty variants)
- Rank-monotonic constraints to prevent incoherent predicted CDFs

### 2. Censored & interval-censored regression
**Goal:** Native support for detection limits, survival bounds, and interval targets.
- Tobit / Type-I censored Gaussian NLL (left, right, interval) [R3]
- Accelerated failure time (AFT) likelihoods (log-normal, Weibull) [R4]
- Interval-censored extensions for existing heads (Gaussian, quantile)
- Censored CRPS for proper scoring under censoring [R5]

### 3. Regression under uncertain / absent ground truth
**Goal:** Principled training and evaluation when labels are noisy, ambiguous, or absent.
- Monte Carlo conformal prediction with soft pseudo-labels [R6]
- Uncertainty-aware evaluation metrics (plausibility regions, annotation
  uncertainty scores) [R7]
- Semi-supervised regression: consistency regularization, pseudo-label NLL
- Propensity / inverse-probability weighting for selection-biased labels [R8]
- Teacher-student self-training losses for unlabeled data

### 4. Constraint layers & regularizers
**Goal:** Composable building blocks that enforce physical or structural priors.
- **Monotonicity:** unconstrained monotonic neural network (UMNN) layers [R9]
- **Non-negativity:** softplus / exp output heads (first-class, not ad-hoc)
- **Bounded outputs:** sigmoid-scaled heads, Beta distribution heads
- **Simplex / sum-to-constant:** Dirichlet heads, softmax projection layers
- **Convexity:** input-convex neural network (ICNN) blocks [R10]
- **Lipschitz:** spectral-norm wrappers on regression heads
- **Non-crossing quantiles:** monotonic sort/rearrangement layers [R11]

### 5. Multi-target multimodal conformal prediction
**Goal:** Distribution-free coverage for vector-valued, possibly multimodal outputs.
- Bonferroni-corrected marginal intervals (simple baseline)
- Copula-based joint conformal [R12]
- Density-based prediction sets (VSPS) for disjoint modes [R13]
- Long-tailed / class-conditional conformal via prevalence-adjusted
  score functions [R14]

### 6. Long-tailed & imbalanced regression (end-to-end)
**Goal:** Close the loop from reweighting → training → tail-specific evaluation.
- Propensity-score reweighting for continuous targets (IPW, CBPS) [R8]
- Feature/label distribution smoothing (LDS/FDS) — *exists, needs recipes*
- Tail-specific evaluation: tail RMSE/MAE, extreme quantile coverage,
  conditional calibration at percentile bins
- Prevalence-adjusted conformal for tail coverage [R14]
- Benchmark recipes with synthetic and real long-tailed data

---

## P1 — High-Impact Enablers

### 7. Heterogeneous multi-output heads
**Goal:** Mixed output types (continuous, counts, bounded) in a single model.
- Per-output distribution family selection (Gaussian, Poisson, Beta, …)
- Shared backbone → heterogeneous head architecture
- Joint NLL with per-output contributions

### 8. Calibration transforms (not just metrics)
**Goal:** Post-hoc recalibration that fixes model outputs, not just measures them.
- Isotonic regression recalibration of predicted mean + variance
- Platt-style variance scaling
- PIT-based recalibration for arbitrary predictive CDFs
- Composable with conformal wrappers

### 9. Concordance & ranking losses
**Goal:** Optimise for ordering correctness when exact values matter less.
- Differentiable concordance index loss (survival, recommendation)
- Spearman / rank-correlation surrogate losses
- Pairwise margin ranking losses for regression

### 10. Trustworthy uncertainty under shift
**Goal:** Calibrated, actionable uncertainty that remains useful when data drifts.
- Adaptive conformal variants for nonstationary data [R15, R16, R17]
- Regression calibration: PIT/CRPS-style [R18], neural calibration [R19]
- Epistemic baselines: deep ensembles, MC dropout, SWAG — *exist, need recipes*
- Risk-coverage curves and abstain/reject policies — *rc.py exists, needs integration*

### 11. Scalable multi-target dependency structure
**Goal:** Accurate uncertainty when outputs are correlated at scale.
- Multi-output GP baselines [R20, R21, R22]
- Structured covariance (Kronecker, block-diagonal, sparse precision)
- Copula-based post-hoc dependency for non-Gaussian targets
- Low-rank + diagonal covariance — *exists*

### 12. Actionable OOD/shift detection
**Goal:** Move from OOD scores to decisions (alerting/abstention).
- Mahalanobis, typicality, energy-based baselines — *exist*
- Density-ratio and ensemble disagreement for shift signals
- Practical thresholding and alert policies in uncertainty reports

---

## P2 — Quality of Life

### 13. Benchmark & recipe suite
- Fixed benchmark set (UCI + tabular OOD splits + synthetic stress tests)
- Standard metrics: NLL, CRPS, PICP, interval width, risk-coverage
- Reproducible scripts for tail-risk and shift stress tests

### 14. Documentation & examples refresh
- Focused tutorials: uncertainty, robust regression, EIV, multi-target
- Minimal examples for each head type (Gaussian, low-rank, MDN, flows)

### 15. API ergonomics
- Loss registration decorators and config-driven builders — *exists*
- Standardised input validation and actionable error messages — *exists*
- Unified `Distribution` return type (`.sample()`, `.log_prob()`, `.entropy()`)

### 16. Training integrations
- PyTorch Lightning templates for regression + uncertainty
- TorchMetrics-aligned APIs — *partially done*

### 17. Performance & stability
- Numerical stability tests (NaN/Inf, extreme values)
- GPU profiling for hot paths and memory scaling

---

## Deferred

- **Energy-based models / implicit regression:** Beautiful math, intractable
  inference for a fast library.
- **Profile likelihood solvers:** Require iterative retraining; domain-specific.
- **Joint causal MNAR modeling:** Requires user-specified DAGs; no generic API.
- **Time-series forecasting (EnbPI, copula CP, etc.):** Violates i.i.d.;
  needs entirely different pipelines.
- **Continuous diffusion / flow matching / MeanFlow:** Promising research
  but no clear path to production regression API yet.

---

## References

R1. McCullagh (1980), "Regression models for ordinal data".
R2. Cao, Mirjalili, Raschka (2020), "Rank consistent ordinal regression (CORAL)".
    arXiv:1901.07884
R3. Tobin (1958), "Estimation of relationships for limited dependent variables".
R4. Wei (1992), "The accelerated failure time model: a useful alternative to the
    Cox regression model".
R5. Berrisch, Ziel (2022), "CRPS learning". arXiv:2102.00968
R6. Stutz et al. (2023), "Conformal prediction under ambiguous ground truth".
    arXiv:2307.09302
R7. Stutz et al. (2023), "Evaluating AI systems under uncertain ground truth".
    arXiv:2307.02191
R8. Sarig, Galili, Eilat (2023), "balance – a Python package for balancing
    biased data samples". arXiv:2307.06024
R9. Wehenkel, Louppe (2019), "Unconstrained Monotonic Neural Networks".
    arXiv:1908.05164
R10. Amos, Xu, Kolter (2017), "Input Convex Neural Networks". arXiv:1609.07152
R11. Chernozhukov, Fernández-Val, Melly (2010), "Quantile and probability curves
     without crossing".
R12. Messoudi, Destercke, Rousseau (2021), "Copula-based conformal prediction".
     arXiv:2104.08438
R13. Izbicki, Shimizu, Stern (2022), "CD-split and HPD-split: efficient conformal
     regions in high dimensions". JMLR.
R14. Ding, Fermanian, Salmon (2025), "Conformal prediction for long-tailed
     classification". arXiv:2507.06867
R15. Podkopaev, Xu, Lee (2024), "Adaptive Conformal Inference by Betting".
     arXiv:2412.19318
R16. Blot, Angelopoulos, Jordan, Brunel (2024), "Automatically Adaptive Conformal
     Risk Control". arXiv:2406.17819
R17. Zaffran et al. (2022), "Adaptive Conformal Predictions for Time Series".
     arXiv:2202.07282
R18. Kuleshov, Fenner, Ermon (2018), "Accurate Uncertainties for Deep Learning
     Using Calibrated Regression". arXiv:1807.00263
R19. Guo et al. (2017), "On Calibration of Modern Neural Networks".
     arXiv:1706.04599
R20. Feinberg et al. (2017), "Large Linear Multi-output Gaussian Process Learning".
     arXiv:1705.10813
R21. Moreno-Munoz, Artes-Rodriguez, Alvarez (2018), "Heterogeneous Multi-output
     Gaussian Process Prediction". arXiv:1805.07633
R22. Joukov, Kulic (2020), "Fast Approximate Multi-output Gaussian Processes".
     arXiv:2008.09848
