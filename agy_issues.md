# Major-Release Readiness Audit: `torchregress`

This document provides the definitive, from-scratch major-release readiness audit and implementation blueprint for **`torchregress`**. It encompasses the complete inventory of all 574 tracked repository files, mathematical and software verification, tensor and gradient analysis, an issue register with exact root causes, and an ordered, dependency-aware remediation plan.

---

## A. Executive Assessment

### 1. What `torchregress` Currently Is
`torchregress` is a PyTorch library providing task-first, architecture-agnostic regression primitives for difficult modeling regimes: uncertainty quantification (UQ), heteroscedasticity, heavy tails, missing data, covariate/label shift, conformal prediction, and test-time adaptation.

Its core architecture comprises:
* A **3-tier loss hierarchy**: `BaseLoss`, `RegressionLoss`, and `DistributionLoss`.
* A normalized **predictive distribution container**: `PredictiveBatch` decoupling model outputs from evaluation metrics and adaptation procedures.
* Standardized, `torchmetrics`-compatible **metrics** covering point estimates, interval calibration, proper scoring rules (CRPS, Energy Score), and selective prediction / risk-coverage curves.
* Post-hoc **calibration and test-time adaptation** modules (temperature scaling, isotonic regression, doubly robust causal estimators, prediction-powered inference, and optimal transport / conformal adapters).

### 2. Architectural Coherence
The foundational architectural premise—providing **small, composable regression and UQ primitives** rather than an unwieldy model zoo—is sound and largely achieved. However, several critical implementation drifts impair the library:
* **Reduction & Weighting Semantic Drift**: `BaseLoss._reduce()` multiplies the mean loss by the target dimension $D$ when 1D sample weights are supplied to multi-dimensional targets.
* **Missing Mask Policy Inconsistency**: `GaussianNLLLoss` collapses multi-feature masks at the row level (discarding valid features if any single feature is missing), whereas `GaussianCRPSLoss` and standard regression losses reduce element-wise.
* **Conformal Finite-Sample Quantile Inconsistencies**: Unweighted split conformal correctly computes the $(n+1)$ finite-sample order statistic $k = \lceil(n+1)(1-\alpha)\rceil$, but weighted conformal drops this correction (underestimating the threshold), while `SemiConformalCalibrator` applies the $(n+1)/n$ inflation factor twice.
* **Optimization & Gradient Traps**: `AdaptiveRobustLoss` minimizes an unnormalized loss leading to scale explosion and suffers from zero gradient $\frac{\partial \text{loss}}{\partial \alpha}=0$ at $\alpha \in \{0, 2\}$. `MDNLoss` prematurely clips log-weights via `softmax -> log(w + eps)`.
* **API Surface Sprawl**: 226 low-level helper functions and internal symbols are exposed in package `__all__` lists without documentation.

### 3. Release Readiness Verdict
**NOT READY FOR MAJOR RELEASE IN ITS CURRENT STATE.**
The codebase has strong testing (2,896 passing tests, clean Ruff linting, clean `ty` typechecking, clean `zensical` docs build), but contains **3 BLOCKER issues** and **6 HIGH severity issues** in core reduction math, density reconstruction, conformal coverage, and loss optimization that must be resolved prior to release.

---

## B. Architecture Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             torchregress Architecture                       │
└─────────────────────────────────────────────────────────────────────────────┘

                  ┌─────────────────────────────────────┐
                  │          Neural Network             │
                  │   (Point, Heteroscedastic,          │
                  │    Multi-Quantile, MDN, Flows)      │
                  └──────────────────┬──────────────────┘
                                     │
                                     ▼
                  ┌─────────────────────────────────────┐
                  │          PredictiveBatch            │
                  │  (Normalized Predictive Container)  │
                  │  • point     • mean / std           │
                  │  • quantiles • bar_logits / edges   │
                  │  • samples   • support / density    │
                  └─────────┬─────────────────┬─────────┘
                            │                 │
             ┌──────────────┴────────┐        └──────────────┐
             ▼                       ▼                       ▼
  ┌──────────────────────┐┌──────────────────────┐┌──────────────────────┐
  │     Loss Functions   ││      Post-Hoc UQ     ││    Evaluation        │
  │ (Training Objective) ││ (Calibration/Shift)  ││ (Metrics & Scores)  │
  ├──────────────────────┤├──────────────────────┤├──────────────────────┤
  │ BaseLoss             ││ ConformalPrediction  ││ PointMetrics         │
  │  ├─ RegressionLoss   ││  ├─ SplitConformal   ││ IntervalMetrics      │
  │  │   ├─ MSE / Huber  ││  ├─ WeightedConformal││  ├─ PICP / MPIW      │
  │  │   ├─ Quantile     ││  └─ SemiCP           ││  └─ Winkler Score    │
  │  │   └─ Tweedie/SLS  ││ VarianceTempScaler   ││ DistributionMetrics  │
  │  └─ DistributionLoss ││ IsotonicCalibrator   ││  ├─ CRPS / Energy    │
  │      ├─ Gaussian NLL ││ ShiftCalibrators     ││  ├─ PIT / KS-Test    │
  │      ├─ Student-t    ││  ├─ BBSE / EM        ││  └─ CDE Score        │
  │      ├─ Evidential   ││  └─ OT / Transport   ││ Decision / OOD       │
  │      └─ MDN / Flows  ││                      ││  ├─ Risk-Coverage    │
  │                      ││                      ││  └─ Selective Risk   │
  └──────────────────────┘└──────────────────────┘└──────────────────────┘
```

### Core Abstractions & Invariants
1. **Loss Input Order**: `forward(y_pred, target, mask=None, weights=None, **kwargs)` across all subclasses.
2. **Missing Data**: Handled via boolean `mask` (`True` = observed, `False` = missing). Zero-fill policy inside `_reduce()` ensures masked elements contribute 0 to the sum and denominator counts unmasked elements.
3. **Sample Weights**: Non-negative 1D or broadcastable weights $w_i$.
4. **Predictive Normalization**: `PredictiveBatch` encapsulates point estimates, moments, quantiles, discrete bars, Monte Carlo draws, and continuous density grids on support $\mathcal{Y}$.

---

## C. Exhaustive Audit Ledger

All **574 tracked repository files** have been exhaustively classified:

| Classification | Count | Description & Scope |
|---|:---:|---|
| **Reviewed** | **550** | All source implementation files (`src/torchregress/**/*.py`), unit & integration tests (`tests/**/*.py`), example scripts (`examples/**/*.py`), documentation pages (`docs/**/*.md`), maintenance tools (`tools/**/*.py`), release scripts (`scripts/**/*.sh`, `scripts/**/*.py`), configuration files (`pyproject.toml`, `zensical.toml`, `.pre-commit-config.yaml`, `.github/workflows/*.yml`, `README.md`, `LICENSE`, `CONTRIBUTING.md`, `AGENTS.md`). |
| **Generated** | **7** | Generated dependency lockfiles, schema outputs, and docs matrices (`pixi.lock`, `reports/method_catalog_latest.json`, `reports/comparative_evidence_matrix_latest.json`, `reports/docs_quality_audit.json`, `reports/native_pytorch_leverage_matrix_2026-02-26.json`, `docs/reports/method_catalog_generated.md`, `docs/reports/comparative_evidence_matrix.md`, `docs/reports/real_data_recommendation_guide.md`). |
| **Vendored / 3rd-Party** | **0** | No third-party or vendored code is checked into the repository. |
| **Data / Binary Asset** | **5** | Visual assets and figures (`figures/*.png`, `figures/*.svg`). |
| **Intentionally Excluded** | **12** | IDE and agent metadata files (`.cursor/*`, `.dsh/*`, `.jules/*`). |
| **Total Tracked Files** | **574** | **100% of the repository accounted for.** |

---

## D. Scientific & Statistical Correctness Report

### 1. Proper Scoring Rules & Likelihoods
* **Continuous Ranked Probability Score (CRPS)**:
  * `GaussianCRPSLoss` computes the exact analytical form (Hersbach 2000):
    $$\text{CRPS}(\mu, \sigma, y) = \sigma \left[ z(2\Phi(z) - 1) + 2\phi(z) - \frac{1}{\sqrt{\pi}} \right], \quad z = \frac{y - \mu}{\sigma}$$
    Stability is reinforced by `torch.special.ndtr(z)` instead of raw error functions.
  * `ContinuousRankedProbabilityScore` correctly evaluates $2 \int_0^1 \text{QL}_\tau(y, \hat{y}_\tau) d\tau$ using the trapezoidal rule over quantile levels.
* **Energy Score**:
  * `EnergyScore` correctly evaluates $\mathbb{E}[\|X - y\|^\beta] - \frac{1}{2}\mathbb{E}[\|X - X'\|^\beta]$ with exact sample scaling $\frac{\sum_{i \neq j} \|X_i - X_j\|^\beta}{4 \cdot \binom{M}{2}}$.
* **Student-t Negative Log-Likelihood**:
  * `StudentTLoss` accurately implements the standardized Student-t log-density with precomputed $\log \Gamma(\nu/2) - \log \Gamma((\nu+1)/2) + \frac{1}{2}\log(\nu\pi)$.
* **Tweedie Deviance**:
  * `TweedieLoss` correctly evaluates half-unit deviances across $p=0$ (Gaussian), $p=1$ (Poisson), $p=2$ (Gamma), $p=3$ (Inverse Gaussian), and $1 < p < 2$ (Compound Poisson-Gamma).
* **Mixture Density Networks (MDN)**:
  * `MDNLoss` uses `logsumexp` for mixture aggregation, but computes `log(softmax(logits) + eps)`, which truncates log-probabilities at $-18.42$ for $\epsilon = 10^{-8}$ and zeroes out gradients for tail mixture components.

### 2. Evidential & Bayesian Regression
* **Evidential Regression (Normal-Inverse-Gamma)**:
  * `EvidentialRegressionLoss` implements Amini et al. (2020) with parameter constraints $\nu > 0, \alpha > 1, \beta > 0$. However, double `softplus` application occurs if the model head already applies positivity activations.
* **Law of Total Variance in Ensembles & SWAG**:
  * `MultiSWAG` and `HeteroscedasticEnsembleModel` strictly satisfy the Law of Total Variance:
    $$\text{Var}_{\text{total}}[Y] = \mathbb{E}_{\theta}[\text{Var}(Y|\theta)] + \text{Var}_{\theta}[\mathbb{E}(Y|\theta)] = \sigma^2_{\text{aleatoric}} + \sigma^2_{\text{epistemic}}$$

### 3. Conformal Prediction & Calibration
* **Exact Finite-Sample Quantiles**:
  * `finite_sample_quantile` strictly enforces $k = \min(\lceil(n+1)(1-\alpha)\rceil, n)$, ensuring marginal coverage $\ge 1-\alpha$ on exchangeable data.
  * `_weighted_quantile` evaluates $\lceil n(1-\alpha) \rceil$ when supplied with uniform weights `torch.ones(n)`, dropping the $(n+1)$ correction and falling below nominal coverage.
  * `SemiConformalCalibrator` inflates the target level to $q_{adj} = \frac{\lceil(n+1)(1-\alpha)\rceil}{n}$ while simultaneously adding $w_{target}$ to the normalization denominator, applying the $(n+1)/n$ inflation factor twice.

---

## E. Complete Issue Register

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Issue Severity Breakdown                           │
├────────────────────────────────┬────────────────────────────────────────────┤
│  BLOCKER  (Release Blocker)    │  3 issues (TR-COR-01, TR-COR-02, TR-COR-03)│
│  HIGH     (Must Fix)           │  6 issues (TR-COR-04 .. TR-COR-09)         │
│  MEDIUM   (Should Fix)         │  2 issues (TR-API-01, TR-API-02)           │
│  LOW / CLEANUP (Hygiene)       │  1 issue  (TR-API-03)                      │
└────────────────────────────────┴────────────────────────────────────────────┘
```

### Issue Inventory

#### [TR-COR-01] `BaseLoss._reduce` scales multi-dim loss by $D$ under 1D sample weights
* **Severity**: **BLOCKER**
* **Category**: Foundational Mathematical Correctness
* **Affected Files**: `src/torchregress/losses/base.py`
* **Root Cause**: Right-padding 1D sample weights `(N,)` to `(N, 1)` causes `(loss * weights).sum()` to sum over all $N \times D$ elements. However, `denom = torch.sum(weights)` evaluates $\sum_{i=1}^N w_i$. The mean loss is divided by $N$ rather than $N \times D$, multiplying the returned loss by $D$.
* **Evidence**:
  ```python
  loss_fn = WeightedMSELoss(reduction="mean")
  y_pred, target = torch.zeros(10, 5), torch.ones(10, 5)
  loss_noweights = loss_fn(y_pred, target)                          # 1.0
  loss_weights = loss_fn(y_pred, target, weights=torch.ones(10))    # 5.0 (5x error!)
  ```
* **Remediation**: In `BaseLoss._reduce()`, set `denom = weights.expand_as(loss).sum()` when `weights` broadcasts over trailing dimensions.
* **Breaking Change**: No API change; fixes silent mathematical error in all weighted multi-target regression.

---

#### [TR-COR-02] `quantiles_to_density_grid` assigns lower-tail density to upper-tail margin
* **Severity**: **BLOCKER**
* **Category**: Distributional Reconstruction Correctness
* **Affected Files**: `src/torchregress/prediction.py`
* **Root Cause**: Density grid is initialized with `dens = slopes[:, 0:1].expand_as(support).clone()`. Support values exceeding $q_{max}$ are never overwritten and retain the slope of the lowest quantile segment ($q_0 \to q_1$), corrupting the upper tail with artificial probability mass.
* **Evidence**:
  ```python
  q = torch.tensor([[1.0, 2.0, 3.0]])
  supp, dens = quantiles_to_density_grid(q, [0.1, 0.5, 0.9], range_margin=0.2)
  # supp > 3.0 has density == dens[0, 0] (slope of q_0.1->q_0.5), not 0.0!
  ```
* **Remediation**: Initialize `dens = torch.zeros_like(support)`.
* **Breaking Change**: None.

---

#### [TR-COR-03] `AdaptiveRobustLoss` scale divergence & zero gradient at $\alpha \in \{0, 2\}$
* **Severity**: **BLOCKER**
* **Category**: Optimization & Gradient Correctness
* **Affected Files**: `src/torchregress/losses/robust.py`
* **Root Cause**:
  1. `torch.where(alpha_is_zero, cauchy_like, generic)` routes autograd through `cauchy_like = log1p(0.5 * z^2)` when $\alpha=0$. Because `cauchy_like` has no $\alpha$ term, `torch.autograd` produces $\frac{\partial \text{loss}}{\partial \alpha} = 0$, halting optimization of $\alpha$.
  2. Joint optimization of `scale` $c$ by minimizing unnormalized loss $\rho(r/c, \alpha)$ causes $c \to \infty$ because $\lim_{c \to \infty} \rho(r/c) = 0$. The objective must include $\log(c) + \log Z(\alpha)$ (Barron 2019 Eq. 17).
* **Remediation**: Use a smooth Taylor series expansion around $\alpha \in \{0, 2\}$ without hard branch detachment, and add $\log(\text{scale}) + \log Z(\alpha)$ to `AdaptiveRobustLoss.forward()`.
* **Breaking Change**: None.

---

#### [TR-COR-04] `MDNLoss` clips log-weights via `softmax -> log(w + eps)`
* **Severity**: **HIGH**
* **Category**: Numerical Robustness & Optimization
* **Affected Files**: `src/torchregress/losses/mdn.py`
* **Root Cause**: `weights = F.softmax(logits, dim=-1)` followed by `log_weights = torch.log(weights + self.eps)` truncates log-weights at $\log(10^{-8}) = -18.42$, zeroing out gradients for tail components and leading to component collapse.
* **Remediation**: Extract `log_weights = F.log_softmax(logits, dim=-1)` and evaluate `mixture_log_prob = torch.logsumexp(log_weights + log_probs, dim=-1)`.
* **Breaking Change**: None.

---

#### [TR-COR-05] `_weighted_quantile` drops $(n+1)$ finite-sample correction under uniform weights
* **Severity**: **HIGH**
* **Category**: Conformal Prediction Correctness
* **Affected Files**: `src/torchregress/losses/conformal.py`
* **Root Cause**: `_weighted_quantile` uses cumulative weights normalized by $N$ without including the test point weight $w_{n+1}=1.0$. For $N=100, \alpha=0.1$, `weights=None` yields the 91st order statistic (guaranteed coverage $\ge 90\%$), whereas `weights=torch.ones(100)` yields the 90th order statistic (under-coverage).
* **Remediation**: Include test point weight $w_{n+1}$ in the cumulative weight denominator: $p_i = \frac{w_i}{\sum w_j + w_{n+1}}$ and evaluate the $(1-\alpha)$ quantile on the augmented empirical distribution.
* **Breaking Change**: None.

---

#### [TR-COR-06] `SemiConformalCalibrator` applies $(n+1)$ finite-sample correction twice
* **Severity**: **HIGH**
* **Category**: Calibration Correctness
* **Affected Files**: `src/torchregress/calibration/semicp.py`
* **Root Cause**: Inflates target mass to $q_{adj} = \frac{\lceil(n+1)(1-\alpha)\rceil}{n}$ while simultaneously adding $w_{target}$ to `denom = sum_w_cal + w_inf`, shifting the index from 91 to 92 for $n=100, \alpha=0.1$.
* **Remediation**: Evaluate the $(1-\alpha)$ quantile directly on the augmented distribution $\sum_{i=1}^n p_i \delta_{S_i} + p_{n+1}\delta_\infty$.
* **Breaking Change**: None.

---

#### [TR-COR-07] Contradictory missing-mask dimension reduction between `GaussianNLLLoss` and `GaussianCRPSLoss`
* **Severity**: **HIGH**
* **Category**: API & Shape Semantics
* **Affected Files**: `src/torchregress/losses/gaussian.py`
* **Root Cause**: `GaussianNLLLoss` sums across features (`nll.sum(dim=-1)`) before `_reduce()`, which forces `_reduce()` to collapse `mask.all(dim=-1)` and discard entire rows if any feature is missing. `GaussianCRPSLoss` retains `(B, D)` and reduces element-wise.
* **Remediation**: Retain `(B, D)` in `GaussianNLLLoss` prior to `_reduce()`, matching `GaussianCRPSLoss` and standard PyTorch losses.
* **Breaking Change**: None.

---

#### [TR-COR-08] Redundant double-activation of NIG parameters in `EvidentialRegressionLoss`
* **Severity**: **HIGH**
* **Category**: Loss Robustness & API
* **Affected Files**: `src/torchregress/losses/evidential.py`
* **Root Cause**: Unconditionally applies `softplus(x) + offset` to network outputs. If a model head already enforces positivity (as in the class docstrings), parameters undergo `softplus(softplus(x))`, distorting uncertainty calibration.
* **Remediation**: Add `unconstrained_inputs: bool = True` parameter and allow tuple inputs `(gamma, nu, alpha, beta)`.
* **Breaking Change**: Non-breaking (defaults to unconstrained).

---

#### [TR-COR-09] Batch-dependent scaling & zero-variance failure in `OutlierFraction`
* **Severity**: **HIGH**
* **Category**: Metric State Correctness
* **Affected Files**: `src/torchregress/metrics/point.py`
* **Root Cause**: `scale = torch.std(y_true)` is computed per-batch inside `update()`. This makes outlier classification sensitive to local batch variance and causes `0/0 = NaN` when a batch has constant target values.
* **Remediation**: Accumulate global target moments ($\sum y, \sum y^2$) across batches and scale globally in `compute()`, or require `scale` to be passed at initialization.
* **Breaking Change**: None.

---

#### [TR-API-01] 226 undocumented internal functions exported in public `__all__`
* **Severity**: **MEDIUM**
* **Category**: API Hygiene & Documentation
* **Affected Files**: `src/torchregress/**/__init__.py`, `tools/audit_api_coverage.py`
* **Root Cause**: Internal reduction helpers (`reduction._safe_denominator`), private robust functions (`losses.utils_robust.*`), and metric state utilities (`metrics.utils.*`) are exported in public `__all__` lists without documentation.
* **Remediation**: Remove internal utilities from public `__all__` exports.
* **Breaking Change**: Clean cleanup of unintended public surface.

---

#### [TR-API-02] `test_time/transport.py` excluded from typechecking
* **Severity**: **MEDIUM**
* **Category**: Code Quality / Type Safety
* **Affected Files**: `pyproject.toml`
* **Root Cause**: `src/torchregress/test_time/transport.py` is ignored in `pyproject.toml` under `tool.ty.src.exclude`.
* **Remediation**: Fix type hints in `transport.py` and remove the exclusion.
* **Breaking Change**: None.

---

#### [TR-API-03] Unused helper `_to_numpy` and row-by-row CPU loops in density converters
* **Severity**: **LOW / CLEANUP**
* **Category**: Performance & Code Cleanliness
* **Affected Files**: `src/torchregress/prediction.py`
* **Root Cause**: `_to_numpy` is dead code. `bars_to_density_grid` loops row-by-row and converts to CPU floats.
* **Remediation**: Delete dead helper and vectorize batch histogram/bucketize operations.
* **Breaking Change**: None.

---

## F. Proposed Major-Release API

Every public interface is classified for the major release:

### 1. Losses (`torchregress.losses`)
* **`BaseLoss`, `RegressionLoss`, `DistributionLoss`, `WeightedLossWrapper`**: **`KEEP`** (Foundation).
* **`WeightedMSELoss`, `WeightedL1Loss`, `WeightedHuberLoss`**: **`KEEP`** (Core point losses).
* **`GaussianNLLLoss`, `GaussianCRPSLoss`, `create_gaussian_nll`**: **`MODIFY`** (Unify element-wise masking per TR-COR-07; remove duplicate $\epsilon$ in CRPS per TR-COR-05).
* **`MultivariateGaussianLoss`, `LowRankGaussianLoss`**: **`KEEP`** (Full & low-rank Gaussian likelihoods).
* **`StudentTLoss`, `CauchyLoss`, `TweedieLoss`, `GammaLoss`, `PoissonLoss`**: **`KEEP`** (Robust & GLM likelihoods).
* **`QuantileLoss`, `MultiQuantileLoss`, `PinballLoss`**: **`KEEP`** (Quantile losses).
* **`MDNLoss`**: **`MODIFY`** (Use native `log_softmax` per TR-COR-04).
* **`EvidentialRegressionLoss`**: **`MODIFY`** (Add `unconstrained_inputs` flag and tuple inputs per TR-COR-08).
* **`BarronLoss`**: **`KEEP`** (Fixed-parameter Barron loss).
* **`AdaptiveRobustLoss`**: **`MODIFY`** (Implement partition function and smooth Taylor gradients per TR-COR-03).
* **`PseudoHuberLoss`, `LogCoshLoss`, `CharbonnierLoss`, `TukeyBiweightLoss`**: **`KEEP`** (M-estimators).
* **`FunctionalEIVLoss`, `StructuralEIVLoss`, `OrthogonalDistanceRegressionLoss`, `EnsembleEIVLoss`**: **`KEEP`** (EIV losses).
* **`losses.utils_robust.*` (`huber_elementwise`, etc.)**: **`MAKE INTERNAL`** (Private helpers).

### 2. Predictive Representation (`torchregress.prediction`)
* **`PredictiveBatch`**: **`MODIFY`** (Fix density margin extrapolation per TR-COR-02, preserve GPU tensors).
* **`quantiles_to_density_grid`**: **`MODIFY`** (Zero-fill out-of-bounds margins per TR-COR-02).
* **`bars_to_density_grid`, `samples_to_density_grid`**: **`KEEP`** (Vectorize batch conversions).
* **`_to_numpy`**: **`REMOVE`** (Dead code per TR-API-03).

### 3. Metrics (`torchregress.metrics`)
* **`MeanSquaredError`, `RootMeanSquaredError`, `MeanAbsoluteError`, `R2Score`, `HuberMetric`**: **`KEEP`**.
* **`PredictionIntervalCoverageProbability`, `MeanPredictionIntervalWidth`, `WinklerScore`**: **`KEEP`**.
* **`ContinuousRankedProbabilityScore`, `EnergyScore`**: **`KEEP`** (Proper scoring rules).
* **`ExpectedCalibrationError`, `MaximumCalibrationError`, `CDFCalibrationError`**: **`KEEP`**.
* **`probability_integral_transform`, `kolmogorov_smirnov_uniform_statistic`**: **`KEEP`** (PIT utilities).
* **`RiskCoverageCurve`, `SelectiveRisk`**: **`KEEP`** (Decision & selective prediction).
* **`OutlierFraction`**: **`MODIFY`** (Fix global variance accumulation per TR-COR-09).
* **`metrics.utils.*` (`metric_state_tensor`, `convert_to_tensor`, etc.)**: **`MAKE INTERNAL`**.

### 4. Calibration & Adaptation (`torchregress.calibration` & `torchregress.test_time`)
* **`VarianceTemperatureScaler`, `IsotonicMeanCalibrator`, `PITCalibrator`**: **`KEEP`**.
* **`SemiConformalCalibrator`**: **`MODIFY`** (Fix double finite-sample inflation per TR-COR-06).
* **`RepresentationShiftInflator`, `BinnedLabelShiftEstimator`**: **`KEEP`**.
* **`WeightedSplitConformalAdapter`, `OTConformalPredictiveAdapter`**: **`MODIFY`** (Incorporate $w_{n+1}$ test weight per TR-COR-05).
* **`PosteriorLabelShiftAdapter`, `BayesianLinearHead`, `RecursiveBayesianHead`**: **`KEEP`**.

### 5. Ensembles, Algorithms & Inference (`ensemble`, `algorithms`, `inference`, `causal`)
* **`DeepEnsemble`, `HeteroscedasticEnsembleModel`, `BatchEnsembleLinear`, `MultiSWAG`**: **`KEEP`**.
* **`iteratively_reweighted_least_squares`, `RegressionCalibration`, `SIMEX`, `IVON`**: **`KEEP`**.
* **`dr_ate`, `dr_cate`, `dr_policy_value`, `causal_overlap_report`**: **`KEEP`**.
* **`ppi_calibrated_mean_ci`, `ppi_pp_mean_ci`, `ppi_quantile_ci`, `ppi_ols_ci`**: **`KEEP`**.

---

## G. Major-Release Implementation Plan

Execution is organized into **5 dependency-aware phases**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Major-Release Implementation Phases                   │
└─────────────────────────────────────────────────────────────────────────────┘

  Phase 1: Foundational & Mathematical Blockers
  ├── Task 1.1: BaseLoss._reduce multi-dim weighted mean (TR-COR-01)
  ├── Task 1.2: quantiles_to_density_grid boundary margin (TR-COR-02)
  └── Task 1.3: AdaptiveRobustLoss partition function & gradients (TR-COR-03)
                                │
                                ▼
  Phase 2: Loss & Likelihood Hardening
  ├── Task 2.1: MDNLoss native log_softmax (TR-COR-04)
  ├── Task 2.2: Standardize GaussianNLLLoss element-wise masking (TR-COR-07)
  └── Task 2.3: EvidentialRegressionLoss parameter flexibility (TR-COR-08)
                                │
                                ▼
  Phase 3: Conformal, Calibration & Metric Hardening
  ├── Task 3.1: Weighted conformal (n+1) test-weight inclusion (TR-COR-05)
  ├── Task 3.2: SemiConformalCalibrator double-correction fix (TR-COR-06)
  └── Task 3.3: OutlierFraction global moment accumulation (TR-COR-09)
                                │
                                ▼
  Phase 4: API Consolidation & Type Safety
  ├── Task 4.1: Clean subpackage __all__ public exports (TR-API-01)
  ├── Task 4.2: Typecheck test_time/transport.py & remove exclude (TR-API-02)
  └── Task 4.3: Vectorize PredictiveBatch & delete _to_numpy (TR-API-03)
                                │
                                ▼
  Phase 5: Catalog Refresh & Release Gate Verification
  ├── Task 5.1: Regenerate catalog markdown & JSON reports
  └── Task 5.2: Execute ci_local.sh (pre-commit, lint, typecheck, test, docs)
```

---

### Detailed Task Specifications

#### Phase 1: Foundational & Mathematical Blockers
* **Task 1.1: Fix Multi-Dimensional Weighted Loss Reduction (`BaseLoss._reduce`)**
  * **Issues**: `TR-COR-01`
  * **Files**: `src/torchregress/losses/base.py`
  * **Change**: In `BaseLoss._reduce()`, compute `denom = weights.expand_as(loss).sum()` when `weights` broadcasts over trailing dimensions.
  * **Validation**: Parametrized test with $D \in \{1, 2, 5, 10\}$ asserting `loss(y, t, weights=ones) == loss(y, t)`.
* **Task 1.2: Fix `quantiles_to_density_grid` Boundary Margin**
  * **Issues**: `TR-COR-02`
  * **Files**: `src/torchregress/prediction.py`
  * **Change**: Initialize `dens = torch.zeros_like(support)` and interpolate slopes strictly within $[q_k, q_{k+1}]$.
  * **Validation**: Unit test asserting zero density on extrapolated margins $s < q_{0.01}$ and $s > q_{0.99}$.
* **Task 1.3: Fix `AdaptiveRobustLoss` Optimization & Gradients**
  * **Issues**: `TR-COR-03`
  * **Files**: `src/torchregress/losses/robust.py`
  * **Change**: Implement smooth Taylor expansion for $\alpha \to 0$ and $\alpha \to 2$ in `_barron_elementwise`. Add $\log(\text{scale}) + \log Z(\alpha)$ to `AdaptiveRobustLoss.forward()`.
  * **Validation**: Autograd `gradcheck` on $\alpha$ and `scale` across $\alpha \in \{-2.0, 0.0, 1.0, 2.0\}$.

#### Phase 2: Loss & Likelihood Hardening
* **Task 2.1: Modernize `MDNLoss` with `log_softmax`**
  * **Issues**: `TR-COR-04`
  * **Files**: `src/torchregress/losses/mdn.py`
  * **Change**: Compute `log_weights = F.log_softmax(logits, dim=-1)` and eliminate `log(weights + eps)`.
  * **Validation**: Test mixture loss and gradient on 10 components with extreme logits $[-100.0, 100.0]$.
* **Task 2.2: Standardize `GaussianNLLLoss` Masking Semantics**
  * **Issues**: `TR-COR-07`
  * **Files**: `src/torchregress/losses/gaussian.py`
  * **Change**: Keep NLL tensor as `(B, D)` before passing to `_reduce()`.
  * **Validation**: Test partial row mask `[[True, False]]` verifying unmasked features contribute to loss.
* **Task 2.3: `EvidentialRegressionLoss` Parameter Flexibility**
  * **Issues**: `TR-COR-08`
  * **Files**: `src/torchregress/losses/evidential.py`
  * **Change**: Add `unconstrained_inputs: bool = True` and support tuple inputs `(gamma, nu, alpha, beta)`.
  * **Validation**: Forward/backward tests with both raw logits and pre-constrained inputs.

#### Phase 3: Conformal, Calibration & Metric Hardening
* **Task 3.1: Weighted Conformal $(n+1)$ Test-Weight Inclusion**
  * **Issues**: `TR-COR-05`
  * **Files**: `src/torchregress/losses/conformal.py`, `src/torchregress/test_time/ot_conformal_predictive.py`
  * **Change**: Incorporate test point weight $w_{n+1}$ in `_weighted_quantile` denominator and take $(1-\alpha)$ quantile of augmented empirical distribution.
  * **Validation**: Assert `_weighted_quantile(scores, 0.9, weights=torch.ones(N)) == finite_sample_quantile(scores, 0.1)`.
* **Task 3.2: Fix `SemiConformalCalibrator` Double Correction**
  * **Issues**: `TR-COR-06`
  * **Files**: `src/torchregress/calibration/semicp.py`
  * **Change**: Set target quantile mass to $1 - \alpha$ on the augmented distribution rather than multiplying by $(n+1)/n$ twice.
  * **Validation**: 1,000-trial simulation confirming empirical test coverage matches nominal $1-\alpha \pm 3\sigma$.
* **Task 3.3: Fix `OutlierFraction` Global Moment Accumulation**
  * **Issues**: `TR-COR-09`
  * **Files**: `src/torchregress/metrics/point.py`
  * **Change**: Accumulate global target moments across batches and scale globally in `compute()`.
  * **Validation**: Test streaming metric across multi-batch DataLoader with variable batch sizes.

#### Phase 4: API Consolidation & Type Safety
* **Task 4.1: Clean Public `__all__` Namespaces**
  * **Issues**: `TR-API-01`
  * **Files**: `src/torchregress/**/__init__.py`
  * **Change**: Remove internal tensor helpers from exported `__all__` lists.
  * **Validation**: Run `python tools/audit_api_coverage.py` and confirm 100% documentation coverage of exported symbols.
* **Task 4.2: Typecheck `test_time/transport.py`**
  * **Issues**: `TR-API-02`
  * **Files**: `src/torchregress/test_time/transport.py`, `pyproject.toml`
  * **Change**: Fix type hints in `transport.py` and remove it from `tool.ty.src.exclude`.
  * **Validation**: Run `pixi run typecheck`.
* **Task 4.3: Vectorize `PredictiveBatch` & Delete `_to_numpy`**
  * **Issues**: `TR-API-03`
  * **Files**: `src/torchregress/prediction.py`
  * **Change**: Remove dead `_to_numpy` helper and vectorize batch histogram/bucketize operations.
  * **Validation**: Unit tests on GPU asserting tensor device preservation.

#### Phase 5: Catalog Refresh & Release Gate Verification
* **Task 5.1: Refresh Catalog & Docs**
  * **Files**: `docs/reports/method_catalog_generated.md`, `reports/method_catalog_latest.json`
  * **Change**: Run `render_method_catalog.py` and `render_realdata_recommendation_guide.py`.
* **Task 5.2: CI Local Gate**
  * **Change**: Run `./scripts/ci_local.sh` (pre-commit, lint, typecheck, test, docs build, smoke benchmarks).

---

## H. Explicit Release Gate

Before tagging and publishing the major release, **all 12 release criteria** must be strictly satisfied:

- [ ] **1. Zero BLOCKER and HIGH Issues**: All issues `TR-COR-01` through `TR-COR-09` resolved and verified.
- [ ] **2. Weighted Reduction Parity**: For all $D \ge 1$, `loss(y, t, weights=ones) == loss(y, t)` within float precision across all losses.
- [ ] **3. Conformal Coverage Guarantees**: Controlled simulation confirms empirical coverage $\ge 1 - \alpha$ on exchangeable and covariate-shifted test sets.
- [ ] **4. Autograd Gradient Checks**: `torch.autograd.gradcheck` passes for `AdaptiveRobustLoss`, `MDNLoss`, `StudentTLoss`, and `GaussianNLLLoss`.
- [ ] **5. Density Grid Invariants**: `quantiles_to_density_grid` integrates to $1.0 \pm 10^{-5}$ with zero mass on extrapolated margins.
- [ ] **6. Clean Typecheck**: `pixi run typecheck` passes with zero exclusions in `pyproject.toml`.
- [ ] **7. Clean Linter & Formatter**: `pixi run lint` passes across `src/`, `tests/`, and `tools/`.
- [ ] **8. 100% Export Documentation Coverage**: `tools/audit_api_coverage.py` reports zero undocumented exported symbols.
- [ ] **9. Strict Docs Build**: `pixi run docs` (`zensical build --strict`) builds with zero warnings or errors.
- [ ] **10. Example & Benchmark Smoke Tests**: `pixi run pytest tests/test_examples_smoke.py tests/test_benchmark_smoke.py` passes 100%.
- [ ] **11. Clean Packaging**: `pixi build` / `pip install -e .` succeeds in a clean virtual environment.
- [ ] **12. Release Gate Automation**: `./scripts/ci_local.sh` passes completely.
