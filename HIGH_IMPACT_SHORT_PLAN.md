# High-Impact Short Plan (v1)

Date: 2026-02-12
Scope: Short execution plan aligned to library focus: uncertainty, robustness to shift, and practical scalability.

## Quick Evaluation of Current Direction

What to keep:
1. Emphasis on uncertainty quality, calibration, and distribution shift.
2. Practical scalability patterns (AMP, batching, compile guidance).
3. Strong real-world example strategy (photo-z already exists and is rich).
4. Existing metrics foundation as a first-class asset (`torchregress.metrics`) with good compatibility potential with `torchmetrics`.


## Product Decision

`torchregress` stays focused on regression uncertainty + robustness, with scalability.

Time-series stance:
1. Not a primary modeling target now.
2. Keep APIs compatible with time-indexed/streamed evaluation so future expansion is easy.

## Plan (Short and High-Impact)

### Track A: Debugging and Reliability Hardening

Goal: Reduce hidden failure modes and improve trust in uncertainty outputs.

Deliverables:
1. Add a targeted "numerical and calibration sanity" test suite:
- finite outputs under extreme targets/predictions
- monotonic behavior for key losses where expected
- mask/weight invariants across all major losses
- interval coverage smoke checks for quantile/conformal pipelines

2. Add metrics-system hardening tests:
- `torchregress.metrics` functional vs class-based consistency checks
- `torchmetrics`-style stateful `update/compute/reset` behavior checks where applicable
- parity checks for key uncertainty/calibration metrics in batched evaluation

3. Add a "bug triage matrix" in docs:
- symptom -> likely root cause -> metric to inspect -> fix path

4. Add one command for quick health checks (lint + focused tests + metrics-focused tests + example smoke).

Acceptance criteria:
1. New tests fail on seeded regressions and pass on fixed code.
2. Health-check command runs locally and in CI.
3. At least 5 common failure patterns documented with concrete diagnostics.

### Track B: One Flagship Example at Practical Scale

Goal: Provide a single, credible end-to-end workflow scientists can reuse.

Choice: Expand `examples/photoz.py` into a structured "stress suite" (preferred, because it already covers the right failure modes).

Enhancements:
1. Standardized experiment modes:
- `quick` (fast local smoke)
- `full` (publication-quality run)
2. Controlled stress scenarios:
- covariate shift (bright -> faint regime split)
- heteroscedastic noise increase
- outlier contamination
- missing-label / masked-target scenario
3. Unified report artifact:
- point metrics (RMSE, MAE, NMAD)
- uncertainty metrics (PICP, interval score, calibration error)
- robustness metrics by subgroup/bin
- runtime + memory summary
4. Metrics execution pattern in example:
- use `torchregress.metrics` as primary evaluation layer
- demonstrate `torchmetrics.MetricCollection` integration for tracking and logging
- show clear mapping from metric outputs to model-selection decisions
5. Clear model ladder:
- baseline point model
- GaussianNLL, MDN and friends
- conformal predictions
- heteroscedastic ensemble
- eiv, simex, rc

Acceptance criteria:
1. Single command generates a reproducible comparison report + plots.
2. Metrics are reported both overall and by shift subgroup.
3. Example is documented as the canonical "how to evaluate uncertainty under shift" workflow.
4. Example includes at least one `torchmetrics`-compatible evaluation path for users already using torchmetrics logging stacks.

### Track C: Thin Scalability Layer (No New Trainer Framework)

Goal: Improve scale readiness without expanding scope.

Deliverables:
1. Add lightweight utilities/recipes only:
- safe gradient accumulation helper
- modern AMP recipe (`torch.amp`)
- `torch.compile` opt-in recipe with caveats
2. Validate compatibility with existing APIs and examples.
3. Add concise "performance playbook" docs page.

Acceptance criteria:
1. Photo-z stress suite runs in both fp32 and AMP modes.
2. No new high-level trainer abstraction added to public API.
3. Throughput/memory changes documented with a repeatable method.

## Time-Series Ready (Without Entering Time-Series Scope)

Implement now:
1. Evaluation utilities that accept optional `timestamp` and `group_id` columns.
2. Rolling-window calibration and drift metrics that are generic to tabular streams.
3. Data split helpers supporting time-based splits (without sequence models).

Defer:
1. RNN/Transformer forecasting model classes.
2. Multi-horizon forecasting losses.

## Minimal Backlog (First 6 PRs)

1. Reliability suite + health-check command.
2. Bug triage/debugging guide.
3. Metrics interoperability pass (`torchregress.metrics` + `torchmetrics` integration checks and docs).
4. Photo-z stress suite refactor (`quick`/`full`, scenario toggles, unified report).
5. AMP + accumulation + compile recipes.
6. Time-index-ready evaluation interfaces (`timestamp`/rolling metrics support).

## Success Criteria

1. Reliability: materially fewer numerical/calibration regressions caught post-merge.
2. Usability: a new user can run one flagship example and understand uncertainty tradeoffs quickly.
3. Scientific value: robust uncertainty metrics are available under controlled shift, not only IID.
4. Scope control: no drift into full time-series modeling before core roadmap goals are complete.


## Extra: Photo-z Metrics
The RAIL framework implements a suite of metrics. We should also ensure we can produce the same so that we can compare. This is the evaluation of the RAIL (LSST) and qp.

* Typical Photo-z Metrics (community-standard)

- Point residual family: (e_z=(z_{phot}-z_{spec})/(1+z_{spec})), bias, scatter, robust scatter (sigma_MAD/NMAD), IQR-based width, outlier/catastrophic outlier fraction.
- PDF calibration family: PIT distribution and PIT goodness-of-fit tests (KS, CvM, AD), QQ-style calibration checks.
- Proper scoring rules / probabilistic accuracy: CDE loss, Brier score, RMSE/KLD between estimated and reference distributions, sometimes CRPS (common in literature, not in the core RAIL list shown below).
- Tomographic/bin metrics: overlap and cross-bin contamination diagnostics, bin-wise redshift bias and width.

* Implemented in RAIL (notably via rail.evaluation + qp metrics backend)

- Point-to-point (PointToPointEvaluator): point_stats_ez, point_stats_iqr, point_bias, point_outlier_rate, point_stats_sigma_mad.
- Dist-to-point (DistToPointEvaluator): cdeloss, pit, brier.
- Dist-to-dist (DistToDistEvaluator): cvm, ks, rmse, kld, ad.
- Additional available metric names exposed in evaluator flow: moment, outlier, rbpe (risk-based point estimate).
- RAIL metrics namespace classes: PointBias, PointOutlierRate, PointSigmaIQR,  PointSigmaMAD, PointStatsEz, CDELoss, KDEBinOverlap.

* RAIL-specific details worth keeping in mind

- point_outlier_rate is defined with the Science Book style threshold using max(0.06, 3σ) in e_z space.
- PIT support includes meta-metrics (AD/CvM/KS behavior through PIT utilities).
- rbpe in qp is explicitly tied to risk-based point estimation (Tanaka et al. 2018 approach).
    